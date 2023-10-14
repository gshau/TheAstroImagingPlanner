import time
import queue
import threading
import json
import flask
import warnings
import datetime
import yaml
import os
import webbrowser
import pytz
import requests
import shutil
import plotly.io as pio


import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
from config import PlannerConfig, VoyagerConfig, SwitchConfig, InspectorThresholds


import dash_leaflet as dl

from dash.exceptions import PreventUpdate
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns

import plotly.graph_objects as go


from dash.dependencies import Input, Output, State, ALL
from flask import request, jsonify


from layout import disable_options
from watcher import Watcher
from fits_processing import RunFileProcessor
from auto_preproc.src.run import run_auto_preproc
from astro_planner.update_voyager_rating import main, asyncio

from data_utils import (
    pull_data,
    pull_target_data,
    add_rejection_criteria,
    target_df_to_data,
    duration,
    update_target_status_data,
    ThreadWithReturnValue,
)
from planner_utils import (
    filter_targets,
    store_target_coordinate_data,
    get_mpsas_from_lat_lon,
    get_time_limits,
    get_target_ephemeris_data_for_plotly,
    update_weather,
)


from astro_planner.utils import timer
from astro_planner.site import get_utc_offset, get_site
from astro_planner.contrast import add_contrast
from astro_planner.data_merge import (
    compute_ra_order,
    merge_targets_with_stored_metadata,
)
from image_grading.frame_viz import (
    show_inspector_image,
    show_frame_analysis,
)

from layout import serve_layout, yaxis_map, make_options

from astropy.utils.exceptions import AstropyWarning
from astro_planner.globals import (
    DATA_DIR,
    INSTRUMENT_COL,
    EXPOSURE_COL,
    FOCALLENGTH_COL,
    BINNING_COL,
    EXC_INFO,
    FILTER_LIST,
    FILTER_MAP,
    COLORS,
    NB_FILTERS,
    BB_FILTERS,
    TABLE_EXPORT_FORMAT,
    ROUTE_PREFIX,
    IS_PROD,
)


from astro_planner.utils import (
    get_config,
    save_config,
    get_db_conn,
)


from astro_planner.logger import log

warnings.simplefilter("ignore", category=AstropyWarning)


server = flask.Flask(__name__)


theme = dbc.themes.COSMO

app = dash.Dash(
    __name__,
    external_stylesheets=[theme],
    server=server,
    title="The AstroImaging Planner",
    update_title=None,
    routes_pathname_prefix=ROUTE_PREFIX,
)


all_target_coords = pd.DataFrame()
all_targets = []

GLOBAL_DICT = {}


def clear_from_global_store(key, global_dict=GLOBAL_DICT):
    if key in global_dict:
        global_dict.pop(key)


def push_object_to_global_store(obj, key, global_dict=GLOBAL_DICT):
    global_dict[key] = obj


def get_object_from_global_store(key, global_dict=GLOBAL_DICT, prevent_update=True):
    obj = global_dict.get(key)
    if obj is None and prevent_update:
        raise PreventUpdate
    return obj


def set_date_cols(df, utc_offset):
    df["date"] = df["DATE-OBS"].values
    df["date_night_of"] = (
        pd.to_datetime(df["DATE-OBS"], errors="coerce")
        + pd.Timedelta(hours=utc_offset - 12)
    ).dt.date

    return df


def get_target_data(config):
    df_targets = get_object_from_global_store("df_targets")
    return target_df_to_data(df_targets)


def use_inspector(config):
    log.info(config.get("switch_config", {}).get("inspector_switch", False))
    return config.get("switch_config", {}).get("inspector_switch", False)


def use_planner(config):
    log.info(config.get("switch_config", {}).get("planner_switch", False))
    return config.get("switch_config", {}).get("planner_switch", False)


def check_use_planner(config):
    if not use_planner(config):
        raise PreventUpdate


def check_use_inspector(config):
    if not use_inspector(config):
        raise PreventUpdate


@timer
def update_data(conn, config, targets=[], global_dict=GLOBAL_DICT):
    log.info("***** CALLING UPDATE_DATA *****")

    t0 = time.time()
    log.debug(f"Time elapsed: {time.time() - t0:.2f}")
    t0 = time.time()
    df_data = None
    log.debug(f"Time elapsed: {time.time() - t0:.2f}")
    t0 = time.time()

    if use_inspector(config):
        t0 = time.time()
        df_data = pull_data(conn, config)
        push_object_to_global_store(df_data, "df_data", global_dict=global_dict)
        log.info(f"Time elapsed: {time.time() - t0:.2f}")

    if use_planner(config):
        target_data, df_targets, df_target_status = pull_target_data(conn, config)
        if (
            target_data is not None
            and df_targets is not None
            and df_target_status is not None
        ):
            push_object_to_global_store(
                df_targets, "df_targets", global_dict=global_dict
            )
            push_object_to_global_store(
                df_target_status, "df_target_status", global_dict=global_dict
            )
            push_object_to_global_store(
                target_data, "target_data", global_dict=global_dict
            )
        else:
            clear_from_global_store("df_targets", global_dict=global_dict)
            clear_from_global_store("df_target_status", global_dict=global_dict)
            clear_from_global_store("target_data", global_dict=global_dict)

    log.debug(f"Time elapsed: {time.time() - t0:.2f}")
    t0 = time.time()
    if (
        use_planner(config)
        and use_inspector(config)
        and df_data is not None
        and df_targets is not None
        and df_data.shape[0] > 0
        and df_targets.shape[0] > 0
    ):
        df_merged_exposure, df_merged = merge_targets_with_stored_metadata(
            df_data, df_targets
        )

        push_object_to_global_store(
            df_merged_exposure, "df_merged_exposure", global_dict=global_dict
        )
        push_object_to_global_store(df_merged, "df_merged", global_dict=global_dict)
    else:
        clear_from_global_store("df_merged_exposure", global_dict=global_dict)
        clear_from_global_store("df_merged", global_dict=global_dict)

    log.debug(f"Time elapsed: {time.time() - t0:.2f}")
    t0 = time.time()
    return config


def invert_filter_map(filter_map):
    d = {}
    for filter_name, filter_list in filter_map.items():
        for mapped_filter_name in filter_list:
            d[mapped_filter_name] = filter_name
    return d


@timer
def get_progress_graph(
    df,
    date_string,
    days_ago,
    targets=[],
    apply_rejection_criteria=True,
    config={},
    set_inspector_progress=False,
):
    selection = df["DATE-OBS"] < "1970-01-01"
    if days_ago > 0:
        selection |= df["DATE-OBS"] > str(
            datetime.datetime.today() - datetime.timedelta(days=days_ago)
        )
    targets_in_last_n_days = list(df[selection]["OBJECT"].unique())
    targets += targets_in_last_n_days
    if targets:
        selection |= df["OBJECT"].isin(targets)
    df0 = df[selection]
    df0 = df0.reset_index()
    filter_map = invert_filter_map(config.get("filter_map", FILTER_MAP))
    df0["FILTER"] = df0["FILTER"].replace(filter_map)

    df0 = df0.dropna(
        subset=[
            "OBJECT",
            "FILTER",
            BINNING_COL,
            FOCALLENGTH_COL,
            INSTRUMENT_COL,
            "is_ok",
            EXPOSURE_COL,
            "OBJCTRA",
        ]
    )
    if set_inspector_progress:
        push_object_to_global_store(df0, "df_inspector_progress")

    p = go.Figure()
    if df0.shape[0] == 0:
        return dcc.Graph(figure=p), pd.DataFrame()
    exposure_col = EXPOSURE_COL
    if "is_ok" in df0.columns and apply_rejection_criteria:
        exposure_col = "exposure_ok"
        df0[exposure_col] = df0[EXPOSURE_COL] * (2 * df0["is_ok"].astype(int) - 1)

    df_summary = (
        df0.groupby(
            ["OBJECT", "FILTER", BINNING_COL, FOCALLENGTH_COL, INSTRUMENT_COL, "is_ok"]
        )
        .agg({exposure_col: "sum"})
        .dropna()
    )

    exposure = df_summary[exposure_col] / 3600
    accepted_exposure = exposure[exposure > 0].sum()
    rejected_exposure = -exposure[exposure < 0].sum()
    total_exposure = accepted_exposure + rejected_exposure
    n_frames = df0.shape[0]

    df_summary = df_summary.unstack(1).fillna(0)
    df_summary = df_summary[exposure_col] / 3600

    cols = ["OBJECT"] + FILTER_LIST
    df_summary = df_summary[[col for col in cols if col in df_summary.columns]]

    df0["ra_order"] = df0["OBJCTRA"].apply(
        lambda ra: compute_ra_order(ra, date_string=date_string)
    )

    objects_sorted = (
        df0.dropna(subset=["OBJECT", "ra_order"])
        .groupby("OBJECT")
        .agg({"ra_order": "mean"})
        .sort_values(by="ra_order")
        .index
    )

    df0 = df_summary.reset_index()
    bin = df0[BINNING_COL].astype(int).astype(str)
    fl = df0[FOCALLENGTH_COL].astype(int).astype(str)
    df0["text"] = df0[INSTRUMENT_COL] + " @ " + bin + "x" + bin + " FL = " + fl + "mm"

    df0["text"] = df0.apply(
        lambda row: "<br>Sensor: "
        + str(row[INSTRUMENT_COL])
        + f"<br>BINNING: {row[BINNING_COL]}"
        + f"<br>FOCAL LENGTH: {row[FOCALLENGTH_COL]} mm",
        axis=1,
    )

    barmode = config.get("progress_mode", "group")
    if barmode == "stack":
        df0["object_with_status"] = df0.apply(
            lambda row: f'{row["OBJECT"]} {row["is_ok"]}', axis=1
        )
        df_summary = df0[df0["OBJECT"].isin(objects_sorted)].set_index(
            ["object_with_status"]
        )
    else:
        df_summary = df0[df0["OBJECT"].isin(objects_sorted)].set_index(["OBJECT"])
        df_summary = df_summary.loc[objects_sorted]
    cols = [col for col in COLORS if col in df_summary.columns]
    for filter in cols:
        p.add_trace(
            go.Bar(
                name=f"{filter}",
                x=df_summary.index,
                y=df_summary[filter],
                hovertext=df_summary["text"],
                marker_color=COLORS[filter],
                text=np.abs(df_summary[filter].round(2)),
                textposition="auto",
            )
        )

    p.update_layout(
        barmode=barmode,
        yaxis_title="Total Exposure (hr) <br> Negative values are auto-rejected",
        xaxis_title="Object",
        title={
            "text": f"Acquired Data (N={n_frames}), Exposure Total = {total_exposure:.2f} hr<br> \
                Accepted = {accepted_exposure:.2f} hr, Rejected = {rejected_exposure:.2f} hr",
            "y": 0.97,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        font=dict(size=12, color="Black"),
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.02),
        title_x=0.5,
        transition={"duration": 250},
    )

    update_template = False
    if update_template:
        layout = pio.templates["plotly_dark"]["layout"]
        layout["plot_bgcolor"] = "rgb(68, 68, 68)"
        layout["paper_bgcolor"] = "rgb(68, 68, 68)"
        template = dict(layout=layout)

        p.update_layout(template=template)
    graph = dcc.Graph(
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d"]},
        figure=p,
    )

    return graph, df_summary


@timer
def get_rejection_criteria_and_status(
    df_data,
    z_score_thr,
    iqr_scale,
    eccentricity_median_thr,
    star_trail_strength_thr,
    min_star_reduction,
    gradient_thr,
):
    if df_data.shape[0] == 0:
        push_object_to_global_store(None, "df_reject_criteria")
        push_object_to_global_store(None, "df_reject_criteria_all")

    df_reject_criteria_all = add_rejection_criteria(
        df_data,
        z_score_thr=z_score_thr,
        iqr_scale=iqr_scale,
        eccentricity_median_thr=eccentricity_median_thr,
        star_trail_strength_thr=star_trail_strength_thr,
        min_star_reduction=min_star_reduction,
        gradient_thr=gradient_thr,
    )

    cols = [
        "filename",
        "star_count_iqr_outlier",
        "star_count_z_score",
        "fwhm_iqr_outlier",
        "fwhm_z_score",
        "high_ecc",
        "high_fwhm",
        "high_gradient",
        "star_count_fraction",
        "bad_star_count_z_score",
        "bad_fwhm_z_score",
        "trailing_stars",
        "low_star_count_fraction",
        "low_star_count",
        "bad_star_shape",
        "is_ok",
        "fwhm_status",
        "ecc_status",
        "star_trail_status",
        "iqr_status",
        "star_z_score_status",
        "fwhm_z_score_status",
        "gradient_status",
        "star_count_fraction_status",
    ]

    df_reject_criteria = df_reject_criteria_all[cols]

    push_object_to_global_store(df_reject_criteria, "df_reject_criteria")
    push_object_to_global_store(df_reject_criteria_all, "df_reject_criteria_all")

    return ""


def preprocess_targets(config, df_inspector_progress, df_data):
    df_reject_criteria = df_inspector_progress.copy()
    df_data_ = df_data.copy()

    filter_map = invert_filter_map(config.get("filter_map", FILTER_MAP))
    df_reject_criteria["FILTER"] = df_reject_criteria["FILTER"].replace(filter_map)
    df_data_["FILTER"] = df_data_["FILTER"].replace(filter_map)
    df_data_ = df_data_.query('not filename.str.contains("_master_")')

    target_names = df_reject_criteria["OBJECT"].unique()

    is_ok = df_reject_criteria["is_ok"] == 1

    dirs = config.get("directories", {})
    lights_dir = dirs.get("data_dirs")[0]
    output_dir = dirs.get("preproc_out_dirs")[0]
    calibration_dir = dirs.get("calibration_dirs")[0]
    kwarg_list = []
    all_records = []
    for target_name in target_names:
        selection = df_reject_criteria["OBJECT"] == target_name
        df_ok = df_reject_criteria[selection & is_ok]
        files = df_ok["filename"].values

        kwargs = dict(
            lights_dir=lights_dir,
            calibration_dir=calibration_dir,
            df_header=df_data_,
            master_cal_dir=f"{output_dir}/master_calibration",
            output_dir=output_dir,
            matching_files=files,
            app=app,
        )

        log.info(kwargs)
        kwarg_list.append(kwargs)

        records = run_auto_preproc(target_name, **kwargs)
        log.info(records)
        all_records += records
    app.preproc_list = []
    app.preproc_progress = 0
    app.preproc_count = 0
    app.preproc_status = ""
    return all_records, output_dir


@timer
@app.server.route("/shutdown")
def shutdown_dash():
    app.queue.put_nowait(None)
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()
    return "OK"


# Callbacks
@app.callback(
    Output("dummy-id-3", "children"),
    [Input("themes", "value")],
    prevent_initial_call=True,
)
def update_theme_callback(theme):
    app.config["external_stylesheets"] = [dbc.themes.__dict__.get(theme)]
    return ""


@app.callback(
    [Output("voyager-inputs", "style"), Output("sync-ratings-col", "style")],
    [Input(SwitchConfig.VOYAGER_SWITCH, "on")],
)
def show_voyager_inputs_callback(switch):
    if switch:
        return {}, {"textAlign": "center"}
    return {"display": "none"}, {"display": "none"}


@app.callback(
    Output("loading-output-rating-sync", "children"),
    [Input("sync-ratings", "n_clicks")],
    [
        State("progress-graph-files", "children"),
        State(VoyagerConfig.HOSTNAME, "value"),
        State(VoyagerConfig.PORT, "value"),
        State(VoyagerConfig.USER, "value"),
        State(VoyagerConfig.PASSWORD, "value"),
        State("target-matches", "value"),
    ],
    prevent_initial_call=True,
)
def sync_ratings_callback(
    n1, filenames, server_url, server_port, auth_user, auth_passwd, target_names
):
    df = get_df_for_status(filenames=filenames)
    auth_token = f"{auth_user}:{auth_passwd}"
    asyncio.run(
        main(
            server_url=server_url,
            server_port=server_port,
            auth_token=auth_token,
            target_names=target_names,
        )
    )

    return ""


@app.callback(
    Output("date-picker", "date"),
    [Input(PlannerConfig.LAT, "value"), Input(PlannerConfig.LON, "value")],
    [State("date-picker", "date"), State("store-config", "data")],
)
def set_date_callback(lat, lon, current_date, config):
    check_use_planner(config)
    df_data = get_object_from_global_store("df_data")

    utc_offset = get_utc_offset(lat, lon, current_date)

    date = datetime.datetime.now() + datetime.timedelta(hours=utc_offset)
    df_data = set_date_cols(df_data, utc_offset=utc_offset)

    push_object_to_global_store(df_data, "df_data")

    return date


def get_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    [
        Output("alert-preproc", "children"),
        Output("alert-preproc", "duration"),
        Output("alert-preproc", "color"),
        Output("alert-preproc", "is_open"),
        Output("loading-output-preproc", "children"),
    ],
    [Input("preprocess", "n_clicks")],
    [State("store-config", "data")],
    prevent_initial_call=True,
)
def preprocess_targets_callback(n1, config):
    q = queue.Queue()
    df_inspector_progress = get_object_from_global_store("df_inspector_progress")
    conn = get_db_conn(config)
    df_header = pull_data(conn, config, join_type="left")
    df_header["filename"] = df_header["full_file_path"]

    p = ThreadWithReturnValue(
        target=preprocess_targets,
        args=(config, df_inspector_progress, df_header),
    )
    p.start()
    records, output_dir = p.join()

    log.info(records)

    record_strings = []
    for record in records:
        record["full_path"] = (Path(output_dir) / Path(record["out_file"])).resolve()
        record_string = ", ".join(
            [
                f"{k}: {str(v)}"
                for k, v in record.items()
                if k in ["OBJECT", "FILTER", FOCALLENGTH_COL, BINNING_COL, "full_path"]
            ]
        )
        record_strings.append(f"Preprocessing: {record_string}")
        record_strings.append(html.Br())
    if len(record_strings) == 0:
        raise PreventUpdate
    return record_strings, 60000, "primary", True, ""


@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal_callback(n1, n2, is_open):
    return get_modal(n1, n2, is_open)


@app.callback(
    [
        Output(PlannerConfig.PROFILES, "options"),
        Output(PlannerConfig.PROFILES, "value"),
        Output(PlannerConfig.PROFILES, "disabled"),
    ],
    [
        Input("upload-data", "contents"),
    ],
    [
        State("upload-data", "filename"),
        State("upload-data", "last_modified"),
        State(PlannerConfig.PROFILES, "value"),
        State("store-config", "data"),
    ],
)
def update_output_callback(
    list_of_contents,
    list_of_names,
    list_of_dates,
    profiles_selected,
    config,
):
    check_use_planner(config)
    target_data = get_target_data(config)
    i = "None"
    profile_dropdown_is_disabled = True
    if not target_data:
        return [{"label": i, "value": i}], [], profile_dropdown_is_disabled
    if not target_data.profiles:
        return [{"label": i, "value": i}], [], profile_dropdown_is_disabled
    if len(target_data.profiles) == 0:
        return [{"label": i, "value": i}], [], profile_dropdown_is_disabled

    profile_dropdown_is_disabled = False
    first_profile = target_data.profiles[0]

    inactive_profiles = config.get("inactive_profiles", [])
    default_profiles = config.get("planner_config", {}).get("profiles")

    options = [
        {"label": profile, "value": profile}
        for profile in target_data.profiles
        if profile not in inactive_profiles
    ]
    if profiles_selected is None:
        profiles_selected = [first_profile]
    default_options = profiles_selected
    if default_profiles:
        default_options = default_profiles

    return options, default_options, profile_dropdown_is_disabled


@app.callback(
    [
        Output("weather-graph", "children"),
        Output("clear-outside", "href"),
        Output("nws-weather", "href"),
        Output("goes-satellite", "href"),
        Output("clear-outside-img", "src"),
    ],
    [
        Input(PlannerConfig.LAT, "value"),
        Input(PlannerConfig.LON, "value"),
        Input(PlannerConfig.UTC_OFFSET, "value"),
    ],
    [State("store-config", "data")],
)
def update_weather_data_callback(lat, lon, utc_offset, config):
    check_use_planner(config)
    site = get_site(
        lat=lat,
        lon=lon,
        utc_offset=utc_offset,
    )

    (
        weather_graph,
        clear_outside_link,
        nws_link,
        goes_link,
        clear_outside_forecast_img,
    ) = update_weather(site, config)

    return (
        weather_graph,
        clear_outside_link,
        nws_link,
        goes_link,
        clear_outside_forecast_img,
    )


@timer
def filter_targets_for_matches_and_filters(
    targets, status_matches, priority_matches, filters, profile_list, config
):
    check_use_planner(config)

    df_target_status = get_object_from_global_store("df_target_status")
    target_data = get_target_data(config)

    targets = []
    if len(target_data.target_list) > 0:
        for profile in profile_list:
            if profile in target_data.target_list:
                targets += list(target_data.target_list[profile].values())

        if filters:
            targets = filter_targets(targets, filters)

        if status_matches:
            matching_targets = df_target_status[
                df_target_status["status"].isin(status_matches)
            ]["TARGET"].values
            targets = [target for target in targets if target.name in matching_targets]

            log.debug(f"Target matching status {status_matches}: {targets}")

        if priority_matches:
            matching_targets = df_target_status[
                df_target_status["priority"].isin(priority_matches)
            ]["TARGET"].values
            targets = [target for target in targets if target.name in matching_targets]

            log.debug(f"Target matching priority {priority_matches}: {targets}")

    return targets


@app.callback(
    [
        Output("location-marker", "children"),
        Output("map", "center"),
    ],
    [
        Input(PlannerConfig.LAT, "value"),
        Input(PlannerConfig.LON, "value"),
    ],
    [
        State("store-config", "data"),
    ],
)
def set_marker_callback(lat, lon, config):
    check_use_planner(config)
    x = [lat, lon]
    return (
        dl.Marker(
            position=x,
            children=[dl.Tooltip("Default Location")],
        ),
        x,
    )


@app.callback(
    Output("modal-location", "is_open"),
    [Input("open-location", "n_clicks"), Input("close-location", "n_clicks")],
    [State("modal-location", "is_open"), State("store-config", "data")],
)
def toggle_location_modal_callback(n1, n2, is_open, config):
    check_use_planner(config)

    return get_modal(n1, n2, is_open)


@app.callback(
    Output(PlannerConfig.MPSAS, "value"),
    [
        Input(PlannerConfig.LAT, "value"),
        Input(PlannerConfig.LON, "value"),
    ],
    [
        State(PlannerConfig.MPSAS, "value"),
    ],
)
def update_mpsas_data_callback(lat, lon, mpsas):
    atlas_available = os.path.exists(
        f"{DATA_DIR}/data/sky_atlas/World_Atlas_2015_compressed.tif"
    )

    lat = np.round(lat, 4)
    lon = (np.round(lon, 4) + 180) % 360 - 180
    if atlas_available:
        mpsas = np.round(get_mpsas_from_lat_lon(lat, lon), 2)
    return mpsas


@app.callback(Output("download-lpmap", "style"), Input("dummy-id", "children"))
def update_download_lpmap_style_callback(x):
    atlas_available = os.path.exists(
        f"{DATA_DIR}/data/sky_atlas/World_Atlas_2015_compressed.tif"
    )

    if not atlas_available:
        return {}
    return {"display": "none"}


@app.callback(
    Output("aip-profile-load-button", "n_clicks"),
    [Input("download-lpmap", "n_clicks")],
    prevent_initial_call=True,
)
def download_file(n_clicks):
    if n_clicks is not None and n_clicks > 0:
        # URL of the file to be downloaded
        file_url = "https://github.com/gshau/TheAstroImagingPlanner/releases/download/lp-map-v1.0/World_Atlas_2015_compressed.tif"

        # Path where the file will be saved
        if not os.path.exists(f"{DATA_DIR}/data/sky_atlas"):
            os.makedirs(f"{DATA_DIR}/data/sky_atlas")
        save_path = f"{DATA_DIR}/data/sky_atlas/World_Atlas_2015_compressed.tif"

        # Download the file and save it to the specified path
        response = requests.get(file_url)
        with open(save_path, "wb") as file:
            file.write(response.content)

        # Reset the button click counter
        return 0
    else:
        return 0


@app.callback(
    [
        Output("location-text", "children"),
        Output("location-tab-text", "children"),
        Output("bortle-tab-badge", "children"),
    ],
    [
        Input(PlannerConfig.MPSAS, "value"),
    ],
    [
        State(PlannerConfig.LAT, "value"),
        State(PlannerConfig.LON, "value"),
    ],
)
def update_bortle_location_callback(mpsas, lat, lon):
    bortle_bins = [21.99, 21.89, 21.69, 20.49, 19.5, 18.94, 18.38]
    bortle_colors = [
        "dark",
        "secondary",
        "primary",
        "success",
        "warning",
        "danger",
        "danger",
        "light",
        "light",
    ]
    bortle_value = np.digitize(mpsas, bortle_bins) + 1
    bortle_color = bortle_colors[bortle_value - 1]

    bortle_badge = dbc.Badge(
        f"Bortle: {bortle_value}",
        color=bortle_color,
        className="mr-1",
    )

    card_body = [
        html.H3("Location data", className="card-title"),
        html.H4(f"Latitude: {lat}"),
        html.H4(f"Longitude: {lon}"),
    ]
    tab_text = f"Lat: {lat:.2f} Lon: {lon:.2f}"

    atlas_available = os.path.exists(
        f"{DATA_DIR}/data/sky_atlas/World_Atlas_2015_compressed.tif"
    )

    if atlas_available:
        card_body += [
            html.H4(f"Sky Brightness: {mpsas} magnitudes/arc-second^2"),
            html.H4(bortle_badge),
        ]
        tab_text = f"{tab_text} Sky Brightness: {mpsas:.2f} mpsas"

    text = dbc.Card(dbc.CardBody(card_body))

    return text, tab_text, bortle_badge


@app.callback(
    [
        Output(PlannerConfig.LAT, "value"),
        Output(PlannerConfig.LON, "value"),
    ],
    [
        Input("map", "click_lat_lng"),
    ],
    [
        State("date-picker", "date"),
    ],
    prevent_initial_call=True,
)
def update_lat_lon_data_callback(click_lat_lon, date_string=None):
    lat, lon = click_lat_lon
    lat = np.round(lat, 4)
    lon = (np.round(lon, 4) + 180) % 360 - 180
    return lat, lon


@app.callback(
    Output(PlannerConfig.UTC_OFFSET, "value"),
    [
        Input(PlannerConfig.LAT, "value"),
        Input(PlannerConfig.LON, "value"),
        Input("date-picker", "date"),
    ],
    prevent_initial_call=True,
)
def update_time_location_data_callback(lat, lon, date_string=None):
    utc_offset = get_utc_offset(lat, lon, date_string)
    return utc_offset


@app.callback(
    [
        Output("target-status-match", "options"),
        Output("target-status-match", "value"),
        Output(PlannerConfig.TARGET_STATUS, "value"),
        Output(PlannerConfig.TARGET_PRIORITIES, "value"),
    ],
    [Input(PlannerConfig.PROFILES, "value")],
    [
        State(PlannerConfig.TARGET_STATUS, "value"),
        State(PlannerConfig.TARGET_PRIORITIES, "value"),
        State("store-config", "data"),
    ],
)
def update_target_for_status_callback(
    profile_list, status_match, priority_match, config
):
    check_use_planner(config)

    df_target_status = get_object_from_global_store("df_target_status")
    df_target_status = df_target_status[df_target_status["GROUP"].isin(profile_list)]
    targets = sorted(list(df_target_status["TARGET"].values))
    if status_match is None or len(status_match) == 0:
        status_match = config.get("planner_config", {}).get("target_status", None)
    if priority_match is None or len(priority_match) == 0:
        priority_match = config.get("planner_config", {}).get("target_priorities", None)
    return make_options(targets), "", status_match, priority_match


@app.callback(
    [
        Output("target-status-selector", "value"),
        Output("target-priority-selector", "value"),
    ],
    [Input("target-status-match", "value")],
    [
        State("store-target-status", "data"),
        State(PlannerConfig.PROFILES, "value"),
        State("store-config", "data"),
    ],
)
def update_radio_status_for_targets_callback(
    targets, target_status_store, profile_list, config
):
    check_use_planner(config)

    df_target_status = get_object_from_global_store("df_target_status")
    status = None
    priority = None
    if targets and profile_list:
        log.info(f"TARGETS: {targets}")
        selection = df_target_status["TARGET"].isin(targets)
        selection &= df_target_status["GROUP"].isin(profile_list)
        status_priority_set = df_target_status[selection][
            ["status", "priority"]
        ].drop_duplicates()

        if len(set(status_priority_set["status"])) == 1:
            status = status_priority_set["status"].values[0]
        if len(set(status_priority_set["priority"])) == 1:
            priority = status_priority_set["priority"].values[0]

    return status, priority


@app.callback(
    [Output("store-target-status", "data"), Output("dummy-radio-change", "children")],
    [
        Input("target-status-selector", "value"),
        Input("target-priority-selector", "value"),
    ],
    [
        State("target-status-match", "value"),
        State(PlannerConfig.PROFILES, "value"),
        State("store-config", "data"),
    ],
)
def update_target_with_status_callback(status, priority, targets, profile_list, config):
    check_use_planner(config)

    df_target_status = get_object_from_global_store("df_target_status")
    if len(targets) == 0:
        raise PreventUpdate
    try:
        log.info(targets)
        selection = df_target_status["TARGET"].isin(targets)
        selection &= df_target_status["GROUP"].isin(profile_list)
        if status is not None:
            df_target_status.loc[selection, "status"] = status
        if priority is not None:
            df_target_status.loc[selection, "priority"] = priority
    except:
        log.info("issue", exc_info=True)
        raise PreventUpdate
    conn = get_db_conn(config)
    update_status_data(conn, df_target_status.to_dict("records"))
    push_object_to_global_store(df_target_status, "df_target_status")
    return "", ""


@timer
def update_status_data(conn, table_data):
    df = pd.DataFrame.from_records(table_data)
    if df.shape[0] > 0:
        push_object_to_global_store(df, "df_target_status")
        with conn:
            df.to_sql("target_status", conn, if_exists="replace", index=False)
    return df


@app.callback(
    [
        Output("tab-target-div", "style"),
        Output("tab-data-table-div", "style"),
        Output("tab-files-table-div", "style"),
        Output("tab-settings-div", "style"),
        Output("tab-help-div", "style"),
    ],
    [Input("tabs", "value"), Input("dummy-id-2", "children")],
)
def render_content_callback(tab, n1):
    tab_names = [
        "tab-target",
        "tab-targets-table",
        "tab-inspector",
        "tab-settings",
        "tab-help",
    ]

    styles = [{"display": "none"}] * len(tab_names)
    indx = tab_names.index(tab)

    styles[indx] = {}
    return styles


@app.callback(
    [
        Output("bulk-status", "options"),
        Output("bulk-priority", "options"),
        Output("goal-dropdown", "options"),
        Output(PlannerConfig.TARGET_STATUS, "options"),
        Output(PlannerConfig.TARGET_PRIORITIES, "options"),
        Output("target-status-selector", "options"),
        Output("target-priority-selector", "options"),
    ],
    [Input("dummy-id-2", "children")],
    [State("store-config", "data")],
)
def set_bulk_options_callback(n, config):
    check_use_planner(config)
    priority_options = []
    for valid_priority_name in config.get("valid_priorities"):
        label = str(valid_priority_name).replace("_", " ").title()
        priority_options.append({"value": label, "label": label})

    status_options = []
    for valid_status_name in config.get("valid_status"):
        label = str(valid_status_name).replace("_", " ").title()
        status_options.append({"value": valid_status_name, "label": label})

    with open(f"{DATA_DIR}/data/_template/equipment/image_goals.yml", "r") as f:
        image_goals = yaml.load(f, Loader=yaml.SafeLoader)
    image_goal_names = list(image_goals.keys())
    image_goal_options = make_options(image_goal_names)

    return (
        status_options,
        priority_options,
        image_goal_options,
        status_options,
        priority_options,
        status_options,
        priority_options,
    )


@app.callback(
    [
        Output("table-progress", "data"),
        Output("table-progress", "columns"),
        Output("table-progress", "selected_rows"),
        Output("table-progress", "dropdown"),
        Output("goal-table", "children"),
    ],
    [
        Input("bulk-status", "value"),
        Input("bulk-priority", "value"),
        Input("goal-dropdown", "value"),
        Input("select-all", "n_clicks"),
        Input("select-none", "n_clicks"),
        Input("save-changes", "n_clicks"),
        Input("dummy-radio-change", "children"),
    ],
    [
        State("table-progress", "data"),
        State("table-progress", "columns"),
        State("table-progress", "selected_rows"),
        State("table-progress", "derived_virtual_data"),
        State("table-progress", "dropdown"),
        State("goal-table", "children"),
        State("store-config", "data"),
    ],
)
def bulk_update_status_priority_callback(
    status,
    priority,
    goal_dropdown,
    n_all,
    n_none,
    n_save,
    n_radio,
    table_data,
    table_cols,
    selected_rows,
    derived_rows,
    dropdowns,
    goal_table,
    config,
):
    ctx = dash.callback_context
    if ctx.triggered:
        conn = get_db_conn(config)
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "bulk-status" and status is not None:
            for selected_row in selected_rows:
                table_data[selected_row]["status"] = status
            df = update_target_status_data(conn, table_data)
            push_object_to_global_store(df, "df_target_status")
        elif button_id == "bulk-priority" and priority is not None:
            for selected_row in selected_rows:
                table_data[selected_row]["priority"] = priority
            df = update_target_status_data(conn, table_data)
            push_object_to_global_store(df, "df_target_status")
        elif button_id == "select-all":
            selected_rows = []
            for derived_row in derived_rows:
                row_index = table_data.index(derived_row)
                log.info(f"Row index: {row_index}")
                selected_rows.append(row_index)
        elif button_id == "select-none":
            selected_rows = []
        elif button_id == "goal-dropdown" and goal_dropdown is not None:
            with open(f"{DATA_DIR}/data/_template/equipment/image_goals.yml", "r") as f:
                image_goals = yaml.load(f, Loader=yaml.SafeLoader)
            goal_data = image_goals.get(goal_dropdown)
            for selected_row in selected_rows:
                table_data[selected_row]["exposure_goal"] = json.dumps(goal_data)
            df = pd.DataFrame(goal_data)
            goal_table = dbc.Table.from_dataframe(
                df, striped=True, bordered=True, hover=True
            )
        elif button_id == "save-changes":
            df = update_target_status_data(conn, table_data)
            push_object_to_global_store(df, "df_target_status")
            raise PreventUpdate

        if button_id != "dummy-radio-change":
            return table_data, table_cols, selected_rows, dropdowns, goal_table

    df = get_object_from_global_store("df_target_status")
    table_data = df.to_dict("records")
    dropdown_cols = ["status", "priority"]
    dropdowns = {}
    table_cols = []
    selected_rows = []
    for col in df.columns:
        column_property = {"name": col, "id": col, "editable": False}
        if col in dropdown_cols:
            column_property["presentation"] = "dropdown"
            column_property["editable"] = True
            if col == "status":
                values = config.get("valid_status")
            if col == "priority":
                values = config.get("valid_priorities")
            dropdowns[col] = {
                "options": [{"label": str(i), "value": str(i)} for i in values]
            }
        table_cols.append(column_property)

    return table_data, table_cols, selected_rows, dropdowns, goal_table


@app.callback(
    Output("dummy-id-2", "children"),
    [
        Input("table-progress", "data"),
        Input("table-progress", "columns"),
        Input("table-progress", "selected_rows"),
    ],
)
def update_table_callback(rows, columns, selected_rows):
    df = pd.DataFrame(rows, columns=[c["name"] for c in columns])
    push_object_to_global_store(df, "df")
    return None


@app.callback(
    [
        Output("profile-list-div", "style"),
        Output("button-download-target-status-table", "style"),
        Output("button-download-target-data-table", "style"),
    ],
    [Input("store-config", "data")],
)
def update_ui_vis_callback(config):
    if use_planner(config):
        profile_list_style = {}
    else:
        profile_list_style = {"display": "none"}
    return profile_list_style, profile_list_style, profile_list_style


@app.callback(
    [
        Output("tabs", "value"),
        Output("tab-target-review", "style"),
        Output("tab-target-review", "disabled"),
        Output("tab-targets-table", "style"),
        Output("tab-targets-table", "disabled"),
        Output("tab-inspector", "style"),
        Output("tab-inspector", "disabled"),
    ],
    [Input("dummy-id", "children"), Input("store-config", "data")],
    [
        State("tabs", "active_tab"),
    ],
)
def set_target_review_status_callback(
    dummy_id,
    config,
    current_tab,
):
    disabled = False
    planning_disabled = disabled
    table_disabled = disabled
    inspector_disabled = disabled

    if current_tab is None:
        current_tab = "tab-inspector"
    if not use_planner(config):
        planning_disabled = True
        table_disabled = True
        if current_tab == "tab-target" or current_tab == "tab-targets-table":
            current_tab = "tab-settings"
    if not use_inspector(config):
        inspector_disabled = True
        if current_tab == "tab-inspector":
            current_tab = "tab-settings"

    if inspector_disabled and planning_disabled:
        current_tab = "tab-settings"

    review_tab_style = {}
    tab_review_label_style = "text-primary"
    if planning_disabled:
        review_tab_style = {}
        tab_review_label_style = None
    table_tab_style = {}
    tab_table_label_style = "text-success"
    if table_disabled:
        table_tab_style = {}
        tab_table_label_style = None
    inspector_tab_style = {}
    tab_inspector_label_style = "text-info"
    if inspector_disabled:
        inspector_tab_style = {}
        tab_inspector_label_style = None

    if disabled:
        current_tab = "tab-settings"
    return (
        current_tab,
        review_tab_style,
        planning_disabled,
        table_tab_style,
        table_disabled,
        inspector_tab_style,
        inspector_disabled,
    )


@app.callback(
    [
        Output("all-download-data", "data"),
        Output("alert-file-download", "children"),
        Output("alert-file-download", "is_open"),
        Output("alert-file-download", "duration"),
        Output("alert-file-download", "color"),
    ],
    [
        Input("button-download-target-status-table", "n_clicks"),
        Input("button-download-target-data-table", "n_clicks"),
        Input("button-download-data-table", "n_clicks"),
        Input("button-download-log", "n_clicks"),
        Input("button-download-database", "n_clicks"),
        Input("button-download-zip", "n_clicks"),
    ],
    [State("store-config", "data")],
    prevent_initial_call=True,
)
def download_data_callback(n1, n2, n3, n4, n5, n6, config):
    is_open = True
    duration = 5000
    color = "primary"

    ctx = dash.callback_context
    if ctx.triggered:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        log.info(button_id)
        mapping = {
            "button-download-target-status-table": "df_target_status",
            "button-download-target-data-table": "df_targets",
            "button-download-data-table": "df_data",
            "button-download-log": "log",
            "button-download-database": "db",
            "button-download-zip": "zip",
        }
        object_name = mapping.get(button_id)
        if object_name == "zip" or object_name == "db":
            env = config.get("env", "primary")
            log.info(env)
            timestamp = str(datetime.datetime.now())
            log.info(timestamp)
            if object_name == "zip":
                import zipfile

                def zip(src, dst_file):
                    zf = zipfile.ZipFile(dst_file, "w", zipfile.ZIP_DEFLATED)
                    abs_src = os.path.abspath(src)
                    for dirname, subdirs, files in os.walk(src):
                        for filename in files:
                            absname = os.path.abspath(os.path.join(dirname, filename))
                            arcname = absname[len(abs_src) + 1 :]
                            log.info(
                                f"zipping {os.path.join(dirname, filename)} as {arcname}"
                            )
                            zf.write(absname, arcname)
                    zf.close()

                zip(f"{DATA_DIR}/data/user/{env}", f"{DATA_DIR}/data/user/{env}.zip")
                children = ["zip file saved to Downloads folder"]
                return (
                    dcc.send_file(
                        f"{DATA_DIR}/data/user/{env}.zip", f"{env}_{timestamp}.zip"
                    ),
                    children,
                    is_open,
                    duration,
                    color,
                )
            if object_name == "db":
                children = ["database saved to Downloads folder"]
                return (
                    dcc.send_file(
                        f"{DATA_DIR}/data/user/{env}/data.db",
                        f"data_{env}_{timestamp}.db",
                    ),
                    children,
                    is_open,
                    duration,
                    color,
                )

        conn = get_db_conn(config)
        update_data(conn, config)

        if object_name == "log":
            log.info(f"{DATA_DIR}/data/logs/planner.log")
            children = ["log saved to Downloads folder"]
            return (
                dcc.send_file(f"{DATA_DIR}/data/logs/planner.log"),
                children,
                is_open,
                duration,
                color,
            )
        df = get_object_from_global_store(object_name)
        if object_name is not None and isinstance(df, pd.DataFrame):
            timestamp = str(datetime.datetime.now())
            filename = f"{object_name}_{timestamp}.csv"
            log.info(filename)
            children = [f"{filename} saved to Downloads folder"]
            return (
                dcc.send_data_frame(df.to_csv, filename),
                children,
                is_open,
                duration,
                color,
            )
    raise PreventUpdate


@app.callback(
    Output("dummy-id-target-data", "children"),
    [
        Input("date-picker", "date"),
        Input(PlannerConfig.LAT, "value"),
        Input(PlannerConfig.LON, "value"),
        Input(PlannerConfig.UTC_OFFSET, "value"),
    ],
    [State("store-config", "data")],
)
def get_target_data_callback(
    date_string,
    lat,
    lon,
    utc_offset,
    config,
):
    global all_target_coords, all_targets

    check_use_planner(config)

    target_data = get_target_data(config)

    all_target_coords, all_targets = store_target_coordinate_data(
        target_data, date_string, lat, lon, utc_offset, config
    )

    return ""


@app.callback(
    Output("dummy-id-contrast-data", "children"),
    [
        Input("dummy-id-target-data", "children"),
        Input(PlannerConfig.MPSAS, "value"),
        Input(PlannerConfig.K_EXTINCTION, "value"),
    ],
    [State("store-config", "data")],
)
def update_contrast_callback(dummy_input, local_mpsas, k_ext, config):
    global all_target_coords

    check_use_planner(config)
    if local_mpsas is None:
        local_mpsas = config.get("mpsas")
    filter_bandwidth = config.get("bandwidth")
    if k_ext is None:
        k_ext = config.get("k_extinction")
    try:
        all_target_coords = add_contrast(
            all_target_coords,
            n_thread=1,
            filter_bandwidth=filter_bandwidth,
            mpsas=local_mpsas,
            include_airmass=True,
            k_ext=k_ext,
        )
    except KeyError:
        raise PreventUpdate

    push_object_to_global_store(all_target_coords, "all_target_coords")

    return ""


@app.callback(
    [
        Output("store-target-planner-data", "data"),
        Output("store-target-metadata", "data"),
        Output("store-dark-sky-duration", "data"),
        Output("loading-output", "children"),
    ],
    [
        Input("dummy-id-contrast-data", "children"),
        Input("store-target-status", "data"),
        Input("y-axis-type", "value"),
        Input("filter-seasonal-targets", "on"),
        Input(PlannerConfig.TARGET_STATUS, "value"),
        Input(PlannerConfig.TARGET_PRIORITIES, "value"),
        Input("filter-match", "value"),
        Input(PlannerConfig.MIN_MOON_DISTANCE, "value"),
        Input("save-changes", "n_clicks"),
        Input(PlannerConfig.SOLAR_ALTITUDE_FOR_NIGHT, "value"),
        Input(PlannerConfig.COLOR_PALETTE, "value"),
    ],
    [State(PlannerConfig.PROFILES, "value")],
    [State("store-config", "data")],
)
def store_data_callback(
    dummy_input,
    target_status_store,
    value,
    filter_seasonal_targets,
    status_matches,
    priority_matches,
    filters,
    min_moon_distance,
    n_save,
    sun_alt_for_twilight,
    color_palette,
    profile_list,
    config,
):
    check_use_planner(config)
    global all_target_coords, all_targets

    df_targets = get_object_from_global_store("df_targets")

    if min_moon_distance is None:
        min_moon_distance = config.get("min_moon_distance")
    if profile_list is None:
        profile_list = []

    targets = filter_targets_for_matches_and_filters(
        all_targets, status_matches, priority_matches, filters, profile_list, config
    )
    target_names = [t.name for t in targets]
    target_names.append("sun")
    target_names.append("moon")

    target_coords = dict(
        [[k, v] for k, v in all_target_coords.items() if k in target_names]
    )

    sun_down_range = get_time_limits(target_coords, sun_alt=sun_alt_for_twilight + 10)

    df_target_status = get_object_from_global_store("df_target_status")

    data, duration_sun_down_hrs = get_target_ephemeris_data_for_plotly(
        df_target_status,
        target_coords,
        df_targets,
        profile_list,
        config,
        value=value,
        filter_targets=filter_seasonal_targets,
        min_moon_distance=min_moon_distance,
        sun_alt_for_twilight=sun_alt_for_twilight,
        color_palette=color_palette,
    )

    dark_sky_duration_text = (
        f"Length of sky darkness: {duration_sun_down_hrs:.1f} hours"
    )

    metadata = dict(date_range=sun_down_range, value=value)
    filtered_targets = []

    push_object_to_global_store(data, "target_data")
    push_object_to_global_store(filtered_targets, "filtered_targets")

    return data, metadata, dark_sky_duration_text, ""


def get_current_time_data(utc_offset):
    t_now = datetime.datetime.now(pytz.timezone("UTC")) + datetime.timedelta(
        seconds=3600 * utc_offset
    )
    current_time = dict(
        x=[t_now, t_now],
        y=[0, 90],
        mode="lines",
        line=dict(color="blue", width=2),
        showlegend=True,
        name="Current Time",
    )
    return current_time


@app.callback(
    Output("target-graph", "figure"),
    [
        Input("target-graph-interval", "n_intervals"),
    ],
    [
        State("store-target-planner-data", "data"),
        State(PlannerConfig.UTC_OFFSET, "value"),
        State("target-graph", "figure"),
    ],
    prevent_initial_call=True,
)
def update_target_graph_callback(n, target_data, utc_offset, figure):
    if target_data is None or figure is None:
        raise PreventUpdate
    target_data.append(get_current_time_data(utc_offset=utc_offset))
    figure["data"] = target_data
    figure["layout"]["uirevision"] = True
    return figure


@app.callback(
    [
        Output("target-graph-div", "children"),
        Output("target-graph-div", "style"),
        Output("progress-graph", "children"),
        Output("data-table", "children"),
    ],
    [
        Input("store-target-planner-data", "data"),
        Input("dummy-rejection-criteria-id", "children"),
        Input("new-data-available-trigger", "children"),
        Input(PlannerConfig.MIN_FRAME_OVERLAP_FRACTION, "value"),
    ],
    [
        State("store-target-metadata", "data"),
        State("store-dark-sky-duration", "data"),
        State(PlannerConfig.PROFILES, "value"),
        State(PlannerConfig.TARGET_STATUS, "value"),
        State("date-picker", "date"),
        State(PlannerConfig.UTC_OFFSET, "value"),
        State("monitor-mode-interval", "disabled"),
        State("store-config", "data"),
    ],
)
@timer
def update_target_graph_visibility_callback(
    target_data,
    rejection_criteria_change_input,
    refresh_plots,
    min_frame_overlap_fraction,
    metadata,
    dark_sky_duration,
    profile_list,
    status_list,
    date,
    utc_offset,
    monitor_mode_off,
    config,
):
    check_use_planner(config)

    ctx = dash.callback_context
    if ctx.triggered and not monitor_mode_off:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "new-data-available-trigger":
            log.info("Checking...")
            if refresh_plots != "new":
                raise PreventUpdate
            log.info("Doing update")

    df_target_status = get_object_from_global_store("df_target_status")
    df_reject_criteria = get_object_from_global_store("df_reject_criteria_all")

    log.debug("Calling update_target_graph")
    if not metadata:
        metadata = {}
    try:
        value = metadata["value"]
        date_range = metadata["date_range"]
    except KeyError:
        return None, None, None, None

    date_string = str(date.split("T")[0])
    title = (
        "Imaging Targets For Night of {date_string} <br> {dark_sky_duration}".format(
            date_string=date_string, dark_sky_duration=dark_sky_duration
        )
    )

    yaxis_type = "linear"
    if value == "alt":
        y_range = [0, 90]
    elif value == "airmass":
        y_range = [0, 1]
        yaxis_type = "log"
    elif value == "sky_mpsas":
        y_range = [16, 22]
    elif value == "contrast":
        y_range = [0, 1]

    selection = df_target_status["GROUP"].isin(profile_list)
    if status_list:
        selection &= df_target_status["status"].isin(status_list)
    targets = list(df_target_status[selection]["TARGET"].values)

    target_data.append(get_current_time_data(utc_offset=utc_offset))

    target_graph = dcc.Graph(
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d"]},
        figure={
            "data": target_data,
            "layout": dict(
                xaxis={"title": "", "range": date_range},
                yaxis={"title": yaxis_map[value], "range": y_range, "type": yaxis_type},
                title=title,
                margin={"l": 50, "b": 50, "t": 50, "r": 50},
                height=600,
                plot_bgcolor="#ccc",
                paper_bgcolor="#fff",
                hovermode="closest",
                transition={"duration": 50},
                uirevision=True,
            ),
        },
        id="foo",
    )

    progress_days_ago = int(config.get("progress_days_ago", 0))

    df_merged_exposure = get_object_from_global_store("df_merged_exposure")
    df_merged = get_object_from_global_store("df_merged")

    if not min_frame_overlap_fraction:
        min_frame_overlap_fraction = config.get("min_frame_overlap_fraction")
    target_object_matchup = df_merged.loc[
        df_merged["frame_overlap_fraction"] > min_frame_overlap_fraction
    ][["TARGET", "OBJECT"]].drop_duplicates()

    df_merged_exposure_targets = df_merged_exposure.explode("matching_targets").rename(
        {"matching_targets": "TARGET"}, axis=1
    )
    df_merged_exposure_targets = pd.merge(
        df_merged_exposure_targets,
        df_target_status,
        on=["GROUP", "TARGET"],
        how="outer",
    )

    matching_objects = list(
        target_object_matchup.loc[
            target_object_matchup["TARGET"].isin(targets), "OBJECT"
        ]
    )

    progress_graph, df_summary = get_progress_graph(
        df_reject_criteria.query('not IMAGETYP.str.contains("MASTER")'),
        date_string=date_string,
        days_ago=progress_days_ago,
        targets=matching_objects,
        apply_rejection_criteria=True,
        config=config,
    )

    cols = ["OBJECT", "TARGET", "GROUP", "status", "priority"]

    cols += [f for f in df_merged_exposure_targets.columns if f in FILTER_LIST]
    cols += [
        "Instrument",
        "Focal Length",
        "Binning",
        "NOTE",
        "RAJ2000",
        "DECJ2000",
    ]

    columns = []
    for col in cols:
        if col in df_merged_exposure_targets.columns:
            entry = {"name": col, "id": col, "deletable": False, "selectable": True}
            columns.append(entry)

    # target table
    data = df_merged_exposure_targets.round(3).to_dict("records")
    target_table = html.Div(
        [
            dash_table.DataTable(
                id="table-dropdown",
                columns=columns,
                data=data,
                editable=False,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                selected_columns=[],
                selected_rows=[],
                page_action="native",
                page_current=0,
                page_size=50,
                style_cell={"padding": "5px"},
                style_as_list_view=True,
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "rgb(248, 248, 248)",
                    }
                ],
                style_header={
                    "backgroundColor": "rgb(230, 230, 230)",
                    "fontWeight": "bold",
                },
                export_format=TABLE_EXPORT_FORMAT,
            )
        ]
    )

    return target_graph, {}, progress_graph, target_table


@app.callback(
    Output("img-type-matches", "value"),
    [
        Input("dummy-id-4", "children"),
    ],
)
def update_image_type_matches(
    x,
):
    values = []
    df_data = get_object_from_global_store("df_data")
    image_types = df_data["IMAGETYP"].unique()
    for image_type in image_types:
        if "light" in image_type.lower() and "master" not in image_type.lower():
            values.append(image_type)
    return values


@app.callback(
    [
        Output("files-table", "children"),
        Output("summary-table", "children"),
        Output("header-col-match", "options"),
        Output("target-matches", "options"),
        Output("fl-matches", "options"),
        Output("sensor-matches", "options"),
        Output("px-size-matches", "options"),
        Output("filter-matches", "options"),
        Output("img-type-matches", "options"),
        Output("x-axis-field", "options"),
        Output("y-axis-field", "options"),
        Output("scatter-size-field", "options"),
        Output("summary-table-group-keys", "value"),
    ],
    [
        Input("store-target-planner-data", "data"),
        Input("header-col-match", "value"),
        Input("target-matches", "value"),
        Input("inspector-dates", "value"),
        Input("fl-matches", "value"),
        Input("sensor-matches", "value"),
        Input("px-size-matches", "value"),
        Input("filter-matches", "value"),
        Input("img-type-matches", "value"),
        Input("summary-table-group-keys", "value"),
        Input("dummy-rejection-criteria-id", "children"),
        Input("new-data-available-trigger", "children"),
    ],
    [
        State("store-config", "data"),
    ],
)
def update_inspector_options_callback(
    target_data,
    header_col_match,
    target_matches,
    inspector_dates,
    fl_matches,
    sensor_matches,
    px_size_matches,
    filter_matches,
    image_type_matches,
    group_keys,
    dummy_reject_criteria_update,
    dummy_new_data_available_trigger,
    config,
):
    df_data = get_object_from_global_store("df_data")
    df_reject_criteria = get_object_from_global_store("df_reject_criteria")

    targets = sorted(df_data["OBJECT"].unique())

    target_options = make_options(targets)

    df_merged = pd.merge(df_data, df_reject_criteria, on="filename", how="left")

    filter_map = invert_filter_map(config.get("filter_map", FILTER_MAP))
    df_merged["FILTER"] = df_merged["FILTER"].replace(filter_map)

    df0 = df_merged.copy()
    if inspector_dates:
        log.info("Selecting dates")
        df0 = df0[df0["date_night_of"].astype(str).isin(inspector_dates)]

    if target_matches:
        log.info("Selecting target match")
        df0 = df0[df0["OBJECT"].isin(target_matches)]

    df1 = df0.copy()

    fl_match_options = make_options(sorted(df_merged[FOCALLENGTH_COL].unique()))
    if fl_matches:
        log.info("Selecting focal length match")
        df0 = df0[df0[FOCALLENGTH_COL].isin(fl_matches)]

    sensor_match_options = make_options(sorted(df_merged[INSTRUMENT_COL].unique()))
    if sensor_matches:
        log.info("Selecting sensor match")
        df0 = df0[df0[INSTRUMENT_COL].isin(sensor_matches)]

    px_size_options = make_options(sorted(df_merged["XPIXSZ"].unique()))
    if px_size_matches:
        log.info("Selecting pixel size match")
        df0 = df0[df0["XPIXSZ"].isin(px_size_matches)]

    unique_filters = df_merged["FILTER"].unique()
    filter_options = make_options(
        sorted(
            unique_filters,
            key=lambda d: FILTER_LIST.index(d) if d in FILTER_LIST else 1000,
        )
    )
    if filter_matches:
        log.info("Selecting Filter match")
        df0 = df0[df0["FILTER"].isin(filter_matches)]

    image_type_options = make_options(sorted(df_merged["IMAGETYP"].unique()))
    if image_type_matches:
        log.info("Selecting Image type match")
        df0 = df0[df0["IMAGETYP"].isin(image_type_matches)]
    allow_disable = False
    if allow_disable:
        fl_match_options = disable_options(
            fl_match_options, df0[FOCALLENGTH_COL].unique()
        )
        sensor_match_options = disable_options(
            sensor_match_options, df0[INSTRUMENT_COL].unique()
        )
        px_size_options = disable_options(px_size_options, df1["XPIXSZ"].unique())
        filter_options = disable_options(filter_options, df1["FILTER"].unique())
        image_type_options = disable_options(
            image_type_options, df1["IMAGETYP"].unique()
        )

    log.info("Done with filters")

    columns = []
    default_cols = [
        "OBJECT",
        "DATE-OBS",
        "FILTER",
        EXPOSURE_COL,
        "XPIXSZ",
        FOCALLENGTH_COL,
        "arcsec_per_pixel",
        "CCD-TEMP",
        "fwhm_median",
        "eccentricity_mean",
        "is_ok",
    ]
    if "rejected" in df0.columns:
        default_cols += ["rejected"]
    fits_cols = [
        col
        for col in df0.columns
        if col in header_col_match and col not in default_cols
    ]
    fits_cols = default_cols + fits_cols
    other_header_cols = [col for col in df0.columns if col not in default_cols]
    header_options = make_options(other_header_cols)

    for col in fits_cols:
        entry = {"name": col, "id": col, "deletable": False, "selectable": True}
        columns.append(entry)
    df0["FILTER_indx"] = df0["FILTER"].map(
        dict(zip(FILTER_LIST, range(len(FILTER_LIST))))
    )
    df0 = df0.sort_values(by=["OBJECT", "FILTER_indx", "DATE-OBS"]).drop(
        "FILTER_indx", axis=1
    )
    data = df0.round(2).to_dict("records")
    files_table = html.Div(
        [
            dash_table.DataTable(
                columns=columns,
                data=data,
                editable=False,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                selected_columns=[],
                selected_rows=[],
                page_action="native",
                page_current=0,
                page_size=50,
                style_cell={"padding": "5px"},
                style_as_list_view=True,
                style_cell_conditional=[
                    {"if": {"column_id": c}, "textAlign": "left"}
                    for c in ["Date", "Region"]
                ],
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "rgb(248, 248, 248)",
                    }
                ],
                style_header={
                    "backgroundColor": "rgb(230, 230, 230)",
                    "fontWeight": "bold",
                },
                export_format=TABLE_EXPORT_FORMAT,
            )
        ]
    )
    df0["DATE-OBS"] = pd.to_datetime(df0["DATE-OBS"], errors="coerce")
    df_numeric = df0.select_dtypes(
        include=[
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
            "datetime64[ns]",
        ]
    )

    numeric_cols = [col for col in df_numeric.columns if "corr__" not in col]
    scatter_field_options = make_options(numeric_cols)

    if len(group_keys) == 0:
        group_keys = ["OBJECT"]
    df_agg = df0.groupby(group_keys).agg(
        {
            EXPOSURE_COL: "sum",
            "CCD-TEMP": "std",
            "DATE-OBS": "count",
            "fwhm_median": "mean",
            "fwhm_median_arcsec": "mean",
            "eccentricity_median": "mean",
            "star_orientation_score": "mean",
        }
    )
    df_agg[EXPOSURE_COL] = df_agg[EXPOSURE_COL] / 3600
    col_map = {
        "DATE-OBS": "n_subs",
        "CCD-TEMP": "CCD-TEMP Dispersion",
        EXPOSURE_COL: "EXPOSURE (hour)",
        "fwhm_median": "Avg. FWHM (px)",
        "fwhm_median_arcsec": "Avg. FWHM (arc-sec)",
        "eccentricity_median": "Avg. Eccentricity",
    }
    df_agg = df_agg.reset_index().rename(col_map, axis=1)
    if "FILTER" in df_agg.columns and "OBJECT" in df_agg.columns:
        df_agg["FILTER_indx"] = df_agg["FILTER"].map(
            dict(zip(FILTER_LIST, range(len(FILTER_LIST))))
        )
        df_agg = df_agg.sort_values(by=["OBJECT", "FILTER_indx"]).drop(
            "FILTER_indx", axis=1
        )
    data = df_agg.round(2).to_dict("records")

    columns = []
    for col in df_agg.columns:
        entry = {"name": col, "id": col, "deletable": False, "selectable": True}
        columns.append(entry)
    summary_table = html.Div(
        [
            dash_table.DataTable(
                columns=columns,
                data=data,
                editable=False,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                selected_columns=[],
                selected_rows=[],
                page_action="native",
                page_current=0,
                page_size=50,
                style_cell={"padding": "5px"},
                style_as_list_view=True,
                style_cell_conditional=[
                    {"if": {"column_id": c}, "textAlign": "left"}
                    for c in ["Date", "Region"]
                ],
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "rgb(248, 248, 248)",
                    }
                ],
                style_header={
                    "backgroundColor": "rgb(230, 230, 230)",
                    "fontWeight": "bold",
                },
                export_format=TABLE_EXPORT_FORMAT,
            )
        ]
    )

    return (
        files_table,
        summary_table,
        header_options,
        target_options,
        fl_match_options,
        sensor_match_options,
        px_size_options,
        filter_options,
        image_type_options,
        scatter_field_options,
        scatter_field_options,
        scatter_field_options,
        group_keys,
    )


@app.callback(
    [Output("x-axis-field", "value"), Output("y-axis-field", "value")],
    [Input("scatter-radio-selection", "value")],
)
def update_scatter_axes_callback(value):
    x_col = "fwhm_median"
    y_col = "eccentricity_median"
    if value:
        x_col, y_col = value.split(" vs. ")
    return x_col, y_col


@app.callback(
    [Output("target-matches", "value"), Output("inspector-last-n-days", "value")],
    [
        Input("all-data-curr-targets", "n_clicks"),
    ],
    [
        State("inspector-dates", "value"),
        State("target-matches", "value"),
    ],
    prevent_initial_call=True,
)
def set_all_data_callback(
    n_click,
    selected_dates,
    input_target_matches,
):
    if selected_dates is None or len(selected_dates) == 0:
        raise PreventUpdate

    df_data = get_object_from_global_store("df_data")
    df0 = df_data[df_data["date_night_of"].astype(str).isin(selected_dates)]
    target_matches = list(df0["OBJECT"].unique())
    if input_target_matches is not None and len(input_target_matches) != 0:
        target_matches = list(
            set(input_target_matches).intersection(set(target_matches))
        )

    if len(target_matches) == 0:
        raise PreventUpdate

    return target_matches, 0


@app.callback(
    [
        Output("inspector-dates", "options"),
        Output("inspector-dates", "value"),
        Output("monitor-mode-interval", "disabled"),
        Output("monitor-mode-indicator", "color"),
        Output("session-avail-count", "children"),
    ],
    [
        Input("store-target-planner-data", "data"),
        Input("target-matches", "value"),
        Input("monitor-mode", "on"),
        Input("inspector-last-n-days", "value"),
        Input("new-data-available-trigger", "children"),
    ],
    [State("inspector-dates", "value"), State("scatter-radio-selection", "value")],
)
def update_inspector_dates_callback(
    target_data,
    target_matches,
    monitor_mode,
    last_n_days,
    new_data,
    selected_dates,
    radio_selection,
):
    df_data = get_object_from_global_store("df_data")

    df0 = df_data.copy()

    if target_matches:
        df0 = df_data[df_data["OBJECT"].isin(target_matches)]

    all_dates = list(sorted(df0["date_night_of"].dropna().unique(), reverse=True))
    n_sessions = len(all_dates)
    options = make_options(all_dates)
    default_dates = None
    if all_dates:
        default_dates = [all_dates[0]]

    interval_disabled = False
    monitor_mode_indicator_color = "#cccccc"
    if monitor_mode:
        monitor_mode_indicator_color = "#00cc00"
        default_dates = [all_dates[0]]
        interval_disabled = False
    if selected_dates:
        default_dates = selected_dates
    ctx = dash.callback_context
    if ctx.triggered:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "target-matches":
            default_dates = None
    if last_n_days is None or last_n_days == 0:
        default_dates = None
    elif last_n_days > 0:
        default_dates = all_dates[:last_n_days]
    elif last_n_days < 0:
        default_dates = all_dates[last_n_days:]

    return (
        options,
        default_dates,
        interval_disabled,
        monitor_mode_indicator_color,
        f"Recent Sessions ({n_sessions} total)",
    )


def get_called_button_id():
    ctx = dash.callback_context
    if ctx.triggered:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        return button_id


@app.callback(
    Output("dummy-rejection-criteria-id", "children"),
    [
        Input(InspectorThresholds.ECC_THR, "value"),
        Input(InspectorThresholds.STAR_FRAC_THR, "value"),
        Input(InspectorThresholds.Z_SCORE, "value"),
        Input(InspectorThresholds.IQR_SCALE, "value"),
        Input(InspectorThresholds.TRAIL_THR, "value"),
        Input(InspectorThresholds.GRADIENT_THR, "value"),
        Input("new-data-available-trigger", "children"),
    ],
)
def rejection_criteria_callback(
    eccentricity_median_thr,
    min_star_reduction,
    z_score_thr,
    iqr_scale,
    star_trail_strength_thr,
    gradient_thr,
    n,
):
    df_data = get_object_from_global_store("df_data")
    return get_rejection_criteria_and_status(
        df_data,
        z_score_thr,
        iqr_scale,
        eccentricity_median_thr,
        star_trail_strength_thr,
        min_star_reduction,
        gradient_thr,
    )


@app.callback(
    [
        Output("subframe-scatter-graph", "figure"),
        Output("single-target-progress-graph", "children"),
        Output("progress-graph-files", "children"),
    ],
    [
        Input("store-target-planner-data", "data"),
        Input("inspector-dates", "value"),
        Input("target-matches", "value"),
        Input("fl-matches", "value"),
        Input("px-size-matches", "value"),
        Input("filter-matches", "value"),
        Input("sensor-matches", "value"),
        Input("img-type-matches", "value"),
        Input("x-axis-field", "value"),
        Input("y-axis-field", "value"),
        Input("scatter-size-field", "value"),
        Input("dummy-rejection-criteria-id", "children"),
        Input("new-data-available-trigger", "children"),
        Input("show-text-in-scatter", "on"),
        Input("disable-scatter-graph", "on"),
    ],
    [
        State("sync-ratings", "n_clicks"),
        State(InspectorThresholds.ECC_THR, "value"),
        State(InspectorThresholds.STAR_FRAC_THR, "value"),
        State(InspectorThresholds.Z_SCORE, "value"),
        State(InspectorThresholds.IQR_SCALE, "value"),
        State(InspectorThresholds.TRAIL_THR, "value"),
        State(InspectorThresholds.GRADIENT_THR, "value"),
        State("monitor-mode-interval", "disabled"),
        State("store-config", "data"),
    ],
)
@timer
def update_scatter_plot_callback(
    target_data,
    inspector_dates,
    target_matches,
    fl_matches,
    px_size_matches,
    filter_matches,
    sensor_matches,
    image_type_matches,
    x_col,
    y_col,
    size_col,
    dummy,
    refresh_plots,
    show_text,
    disable_scatter,
    sync_ratings_btn,
    eccentricity_median_thr,
    min_star_reduction,
    z_score_thr,
    iqr_scale,
    star_trail_strength_thr,
    gradient_thr,
    monitor_mode_off,
    config,
):
    df_data = get_object_from_global_store("df_data")

    ctx = dash.callback_context
    sync_ratings = False
    if ctx.triggered and not monitor_mode_off:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "new-data-available-trigger":
            log.info("Checking...")
            if refresh_plots != "new":
                raise PreventUpdate
            log.info("Doing update")
            get_rejection_criteria_and_status(
                df_data,
                z_score_thr,
                iqr_scale,
                eccentricity_median_thr,
                star_trail_strength_thr,
                min_star_reduction,
                gradient_thr,
            )
        if button_id == "sync-ratings":
            sync_ratings = True

    df_reject_criteria = get_object_from_global_store("df_reject_criteria")
    df_reject_criteria_all = get_object_from_global_store("df_reject_criteria_all")
    p = go.Figure()
    df0 = df_data.copy()
    filter_map = invert_filter_map(config.get("filter_map", FILTER_MAP))
    df0["FILTER"] = df0["FILTER"].replace(filter_map)
    df0["FILTER_indx"] = df0["FILTER"].map(
        dict(zip(FILTER_LIST, range(len(FILTER_LIST))))
    )

    df0 = pd.merge(df0, df_reject_criteria, on="filename", how="left")
    if inspector_dates:
        df0 = df0[df0["date_night_of"].astype(str).isin(inspector_dates)]
        df_reject_criteria_all = df_reject_criteria_all[
            df_reject_criteria_all["date_night_of"].astype(str).isin(inspector_dates)
        ]

    if not target_matches:
        target_matches = sorted(df0["OBJECT"].unique())
    df0 = df0[df0["OBJECT"].isin(target_matches)]

    if sensor_matches:
        log.info("Selecting sensor match")
        df0 = df0[df0[INSTRUMENT_COL].isin(sensor_matches)]

    if fl_matches:
        log.info("Selecting focal length match")
        df0 = df0[df0[FOCALLENGTH_COL].isin(fl_matches)]

    if px_size_matches:
        log.info("Selecting pixel size match")
        df0 = df0[df0["XPIXSZ"].isin(px_size_matches)]

    if filter_matches:
        log.info("Selecting filter match")
        df0 = df0[df0["FILTER"].isin(filter_matches)]

    if image_type_matches:
        log.info("Selecting image type match")
        df0 = df0[df0["IMAGETYP"].isin(image_type_matches)]

    if df0.shape[0] == 0:
        progress_graph = dcc.Graph(
            config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d"]},
            figure=p,
        )
        return set_bkg_color(p), set_bkg_color(progress_graph), df0["filename"].values

    target_matches = sorted(df0["OBJECT"].unique())

    filters = df0["FILTER"].unique()
    if not x_col:
        x_col = "fwhm_median"
    if not y_col:
        y_col = "eccentricity_median"

    progress_graph, df_summary = get_progress_graph(
        df0,
        date_string="2020-01-01",
        days_ago=0,
        targets=target_matches,
        apply_rejection_criteria=True,
        set_inspector_progress=True,
        config=config,
    )

    df_eff = df_reject_criteria_all.groupby(["date_night_of", FOCALLENGTH_COL]).agg(
        {EXPOSURE_COL: ["sum", "last", "count"], "DATE-OBS": duration}
    )
    df_eff.columns = ["_".join(col).strip() for col in df_eff.columns.values]
    relative_efficiency = (
        df_eff[f"{EXPOSURE_COL}_sum"]
        / (df_eff["DATE-OBS_duration"] + df_eff[f"{EXPOSURE_COL}_last"])
    ).mean()

    df_eff = df_reject_criteria_all.groupby(["date_night_of"]).agg(
        {EXPOSURE_COL: ["sum", "last", "count"], "DATE-OBS": duration}
    )
    df_eff.columns = ["_".join(col).strip() for col in df_eff.columns.values]
    absolute_efficiency = (
        df_eff[f"{EXPOSURE_COL}_sum"]
        / (df_eff["DATE-OBS_duration"] + df_eff[f"{EXPOSURE_COL}_last"])
    ).mean()

    progress_graph.figure.layout.title.text = f"{progress_graph.figure.layout.title.text}<br>Acquisition / Total Efficiency: {100 * relative_efficiency:.1f}% / {100 * absolute_efficiency:.1f}%"  # noqa E501

    progress_graph.figure.layout.height = 600
    progress_graph.figure = set_bkg_color(progress_graph.figure)

    if disable_scatter:
        return set_bkg_color(p), progress_graph, df0["filename"].values

    df0["text"] = df0.apply(
        lambda row: "<br>Object: "
        + str(row["OBJECT"])
        + f"<br>Date: {row['DATE-OBS']}"
        + f"<br>Star count: {row['n_stars']}"
        + (f"<br>{size_col}: {row[size_col]:.2f}" if size_col else "")
        + f"<br>{row['fwhm_status']} FWHM: {row['fwhm_median']:.2f}"
        + f"<br>{row['fwhm_z_score_status']} FWHM Z-score: {row['fwhm_z_score']:.2f}"
        + f"<br>{row['ecc_status']} Eccentricity: {row['eccentricity_median']:.2f}"
        + f"<br>{row['star_trail_status']} Star trail metric: {row['star_trail_strength']:.2f}"
        + f"<br>{row['star_count_fraction_status']} Star count fraction: {row['star_count_fraction']:.2f}"
        + f"<br>{row['iqr_status']} Star IQR reject: {row['star_count_iqr_outlier']}"
        + f"<br>{row['star_z_score_status']} Star Z-score: {row['star_count_z_score']:.2f}"
        + f"<br>{row['gradient_status']} Relative Gradient Strength: {row['relative_gradient_strength']:.2f}"
        + f"<br>Accept frame: {row['is_ok']==1}",
        axis=1,
    )

    group_cols = [
        "FILTER_indx",
        "FILTER",
        "is_ok",
        "low_star_count",
        "high_fwhm",
        "high_gradient",
        "IMAGETYP",
    ]
    inputs = (
        (df0[group_cols].drop_duplicates())
        .sort_values(
            by=group_cols, ascending=[True, True, False, False, False, False, False]
        )
        .values
    )

    i_filter = 0

    size_ref_global = True
    default_marker_size = 10
    df0["marker_size"] = default_marker_size
    if size_col:
        default_marker_size = df0[size_col].median()
        if np.isnan(default_marker_size):
            default_marker_size = 10

        df1 = df0[[size_col]].fillna(default_marker_size)
        values = 8 * (df1 - df1.min()) / (df1.max() - df1.min()) + 7
        df0["marker_size"] = values

    for (
        filter_index,
        filter,
        status_is_ok,
        low_star_count,
        high_fwhm,
        high_gradient,
        image_type,
    ) in inputs:
        selection = df0["FILTER"] == filter
        selection &= df0["is_ok"] == status_is_ok
        selection &= df0["low_star_count"] == low_star_count
        selection &= df0["high_fwhm"] == high_fwhm
        selection &= df0["high_gradient"] == high_gradient
        selection &= df0["IMAGETYP"] == image_type
        df1 = df0[selection].reset_index()

        if size_col and not size_ref_global:
            default_size = 10
            df1["marker_size"] = MinMaxScaler(feature_range=(7, 15)).fit_transform(
                df1[[size_col]].fillna(default_size)
            )

        legend_name = filter
        if filter in NB_FILTERS:
            symbol = "square"
        elif filter in BB_FILTERS:
            symbol = "circle"
        else:
            symbol = "diamond"

        if status_is_ok:
            legend_name = f"{legend_name} &#10004; "
        else:
            symbol = "x"
            if low_star_count:
                symbol = f"{symbol}-open"
                legend_name = f"{legend_name} &#10006; - count"
            elif high_fwhm:
                symbol = f"{symbol}-open-dot"
                legend_name = f"{legend_name} &#10006; - bloat"
            elif high_gradient:
                symbol = "triangle-nw"
                symbol = f"{symbol}-open-dot"
                legend_name = f"{legend_name} &#10006; - gradient"
            else:
                symbol = "diamond-wide"
                symbol = f"{symbol}-open-dot"
                legend_name = f"{legend_name} &#10006; - shape"

        if "master" in image_type.lower():
            legend_name = f"MASTER {legend_name}"
            symbol = "star"

        if filter in COLORS:
            color = COLORS[filter]
        else:
            color = sns.color_palette(n_colors=len(filters)).as_hex()[i_filter]
            i_filter += 0

        mode = "markers"
        if show_text:
            mode = "markers+text"

        p.add_trace(
            go.Scatter(
                x=df1[x_col],
                y=df1[y_col],
                mode=mode,
                name=legend_name,
                hovertext=df1["text"],
                hovertemplate="<b>%{hovertext}</b><br>"
                + f"{x_col}: "
                + "%{x:.2f}<br>"
                + f"{y_col}: "
                + "%{y:.2f}<br>",
                text=df1["OBJECT"],
                textposition="bottom right",
                textfont=dict(color=color, size=8),
                marker=dict(color=color, size=df1["marker_size"], symbol=symbol),
                customdata=df1["filename"],
                cliponaxis=False,
            )
        )
    max_targets_in_list = 5
    target_list = ", ".join(target_matches[:max_targets_in_list])
    if len(target_matches) > max_targets_in_list:
        target_list = f"{target_list} and {len(target_matches) - max_targets_in_list} other targets"

    p.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=600,
        title=f"Subframe data for {target_list}<br>Click points to view in frame and inspector",
        font=dict(size=12, color="Black"),
        legend=dict(orientation="v"),
        transition={"duration": 250},
    )

    return set_bkg_color(p), progress_graph, df0["filename"].values


@app.callback(
    [
        Output("frame-gradient-graph", "figure"),
        Output("xy-frame-graph", "figure"),
        Output("xy-frame-graph", "style"),
        Output("inspector-frame", "figure"),
        Output("loading-output-click", "children"),
    ],
    [
        Input("subframe-scatter-graph", "clickData"),
        Input("aberration-preview", "on"),
        Input("frame-heatmap-dropdown", "value"),
    ],
    [State("store-config", "data")],
)
@timer
def inspect_frame_analysis_callback(
    data, as_aberration_inspector, frame_heatmap_col, config
):
    df_data = get_object_from_global_store("df_data")

    p0 = set_bkg_color(go.Figure())
    p1 = p0
    p2 = p0

    if data is None:
        return p0, p1, p1, p2, ""
    base_filename = data["points"][0]["customdata"]
    if not base_filename:
        log.info("No base filename found")
        return p0, p1, p1, p2, ""

    full_file_path = df_data[df_data["filename"] == base_filename]
    if full_file_path.shape[0] != 0:
        filename = full_file_path["full_file_path"].values[0]
    else:
        log.info(f"No full path found for {base_filename}")
        return p0, p1, p1, p2, ""

    xy_query = f"""select * from xy_frame_metrics where filename = '{base_filename}';"""
    conn = get_db_conn(config)
    with conn:
        t0 = time.time()
        df_xy = pd.read_sql(xy_query, conn)
        log.info(f"Time elapsed: {time.time() - t0:.2f}")
    df_xy.drop_duplicates(subset=["filename", "x_bin", "y_bin"], inplace=True)
    aspect_ratio = df_xy["x_bin"].max() / df_xy["y_bin"].max()
    p1_style = {"width": "35vw", "height": f"{int(35 / aspect_ratio * 1.15)}vw"}
    p2, canvas = show_inspector_image(
        filename,
        as_aberration_inspector=as_aberration_inspector,
        with_overlay=False,
        n_cols=3,
        n_rows=3,
        border=5,
    )

    p1 = show_frame_analysis(df_xy, filename=filename, feature_col=frame_heatmap_col)
    log.info(f"Time elapsed: {time.time() - t0:.2f}")

    return set_bkg_color(p0), set_bkg_color(p1), p1_style, set_bkg_color(p2), ""


def set_bkg_color(p):
    # p.update_layout(paper_bgcolor="rgba(0,0,0,0.01)", plot_bgcolor="rgba(0,0,0,0.)")
    return p


@app.callback(
    [
        Output("alert-auto", "children"),
        Output("alert-auto", "is_open"),
        Output("alert-auto", "duration"),
        Output("alert-auto", "color"),
        Output("new-data-available-trigger", "children"),
        Output("monitor-mode-interval", "interval"),
    ],
    [
        Input("monitor-mode-interval", "n_intervals"),
        Input("force-update-button", "n_clicks"),
    ],
    [
        State("alert-auto", "children"),
        State("store-config", "data"),
        State(SwitchConfig.SILENCE_ALERTS_SWITCH, "on"),
    ],
)
def toggle_alert_callback(
    n,
    n2,
    response,
    config,
    silence_alerts,
):
    app.rfp.config = config
    (
        n_total,
        n_processed,
        n_removed,
        processed_file_list,
        pending_file_list,
    ) = app.rfp.get_update(clear=False)
    n_debug = 0
    ctx = dash.callback_context
    if ctx.triggered:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "force-update-button":
            n_debug = 1

    log.debug(
        f"total={n_total}, proc={n_processed}, removed={n_removed}, manual={n_debug}"
    )
    if n_total + n_processed + n_removed + n_debug == 0:
        raise PreventUpdate

    if n_processed >= n_total:
        app.rfp.reset()
    log.info(f"Updates detected for {n_total + n_removed} files!")
    df_header_old = get_object_from_global_store("df_header", prevent_update=False)
    if df_header_old is None:
        df_header_old = pd.DataFrame(columns=["filename"])

    conn = get_db_conn(config)
    update_data(conn, config)
    df_header = pd.read_sql(
        """select filename from fits_headers fh where is_valid_header = True""", conn
    )
    push_object_to_global_store(df_header, "df_header")

    old_filenames = set(df_header_old["filename"].unique())
    curr_filenames = set(df_header["filename"].unique())

    new_filenames = curr_filenames.difference(old_filenames)
    rm_filenames = old_filenames.difference(curr_filenames)

    removed_row_count = len(rm_filenames)

    total_row_count = len(curr_filenames)
    new_row_count = len(new_filenames)

    has_new_files = new_row_count > 0
    has_removed_files = removed_row_count > 0

    files_changed = has_new_files | has_removed_files

    update_frequency = config.get("monitor_mode_update_frequency", 15) * 1000

    color = "primary"
    is_open = not silence_alerts
    duration = 10000

    n_file_limit = 8
    if files_changed:
        filenames = list(new_filenames)[:n_file_limit]
        removed_filenames = list(rm_filenames)[:n_file_limit]
        response = []

        if has_new_files:
            response.append(html.Br())
            response += [
                f"Recently reviewed {new_row_count} new file{'s' if new_row_count>1 else ''}:"
            ]
            for filename in filenames:
                response.append(html.Br())
                response.append(filename)
            if len(new_filenames) > n_file_limit:
                response.append(html.Br())
                response.append(f"and {len(new_filenames) - n_file_limit} others...")
                response.append(html.Br())
        if has_removed_files:
            color = "danger"
            response.append(html.Br())
            response.append(
                f"Recently removed {removed_row_count} file{'s' if removed_row_count>1 else ''}:"
            )
            for filename in removed_filenames:
                response.append(html.Br())
                response.append(filename)
            if len(rm_filenames) > n_file_limit:
                response.append(html.Br())
                response.append(f"and {len(rm_filenames) - n_file_limit} others...")
                response.append(html.Br())
        response += [
            html.Br(),
            f"Total file count: {total_row_count}",
        ]
    return response, is_open, duration, color, "new", update_frequency


@app.callback(
    [
        Output("config-save", "color"),
        Output("config-save", "disabled"),
    ],
    [
        Input("config-save", "n_clicks"),
        Input({"type": "dir", "sub-type": "target", "index": ALL}, "value"),
        Input({"type": "dir", "sub-type": "data", "index": ALL}, "value"),
        Input({"type": "dir", "sub-type": "calibration", "index": ALL}, "value"),
        Input({"type": "dir", "sub-type": "preproc-out", "index": ALL}, "value"),
        Input("thread-count-slider", "value"),
        Input("process-batch-size-slider", "value"),
        Input("profile-list", "value"),
        Input("profile-list", "options"),
        Input(InspectorThresholds.ECC_THR, "value"),
        Input(InspectorThresholds.STAR_FRAC_THR, "value"),
        Input(InspectorThresholds.Z_SCORE, "value"),
        Input(InspectorThresholds.IQR_SCALE, "value"),
        Input(InspectorThresholds.TRAIL_THR, "value"),
        Input(InspectorThresholds.GRADIENT_THR, "value"),
        Input(PlannerConfig.LAT, "value"),
        Input(PlannerConfig.LON, "value"),
        Input(PlannerConfig.MPSAS, "value"),
        Input(PlannerConfig.MIN_FRAME_OVERLAP_FRACTION, "value"),
        Input(PlannerConfig.MIN_MOON_DISTANCE, "value"),
        Input(PlannerConfig.TIME_RESOLUTION, "value"),
        Input(PlannerConfig.SOLAR_ALTITUDE_FOR_NIGHT, "value"),
        Input(PlannerConfig.K_EXTINCTION, "value"),
        Input(PlannerConfig.PROFILES, "value"),
        Input(PlannerConfig.TARGET_PRIORITIES, "value"),
        Input(PlannerConfig.TARGET_STATUS, "value"),
        Input(SwitchConfig.SILENCE_ALERTS_SWITCH, "on"),
        Input(SwitchConfig.SIRIL_SWITCH, "on"),
        Input(SwitchConfig.PLANNER_SWITCH, "on"),
        Input(SwitchConfig.INSPECTOR_SWITCH, "on"),
        Input(SwitchConfig.CULL_DATA_SWITCH, "on"),
        Input(SwitchConfig.VOYAGER_SWITCH, "on"),
    ],
)
def change_save_button_color_callback(n_save_clicks, *wargs):
    ctx = dash.callback_context
    if ctx.triggered:
        button_id = ctx.triggered[0]["prop_id"]
        if button_id == "config-save.n_clicks":
            color = "success"
            disabled = True
        else:
            color = "warning"
            disabled = False
        return color, disabled
    raise PreventUpdate


@app.callback(
    [
        Output("table-collapse", "is_open"),
        Output("collapse-button", "children"),
    ],
    [Input("collapse-button", "n_clicks")],
    [State("table-collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    button_text = "Show Tables"
    if n:
        button_text = "Hide Tables"
        if is_open:
            button_text = "Show Tables"
        return not is_open, button_text
    return is_open, button_text


@app.callback(
    [
        Output({"type": "dir", "sub-type": "target", "index": ALL}, "disabled"),
        Output({"type": "dir", "sub-type": "data", "index": ALL}, "disabled"),
        Output({"type": "dir", "sub-type": "calibration", "index": ALL}, "disabled"),
        Output({"type": "dir", "sub-type": "preproc-out", "index": ALL}, "disabled"),
        Output("preprocess-col", "style"),
    ],
    [
        Input(SwitchConfig.SIRIL_SWITCH, "on"),
        Input(SwitchConfig.PLANNER_SWITCH, "on"),
        Input(SwitchConfig.INSPECTOR_SWITCH, "on"),
    ],
)
def toggle_siril_callback(siril_on, planner_on, inspector_on):
    style = {"display": "none"}
    if siril_on:
        style = {}
    cal_disabled = not siril_on
    preproc_disabled = not siril_on
    target_disabled = not planner_on
    data_disabled = not inspector_on

    return [target_disabled], [data_disabled], [cal_disabled], [preproc_disabled], style


@app.callback(
    Output("config-text", "style"),
    [Input("config-show", "n_clicks")],
    [State("config-text", "style")],
)
def show_config_callback(n, state):
    if state == {"height": 600, "width": "80%"}:
        state = {"display": "none", "height": 600, "width": "80%"}
    else:
        state = {"height": 600, "width": "80%"}
    return state


@app.callback(
    Output("aip-profile-modal", "is_open"),
    [
        Input("aip-profile-edit-button", "n_clicks"),
        Input("aip-profile-close-button", "n_clicks"),
    ],
    [State("aip-profile-modal", "is_open")],
    prevent_initial_call=True,
)
def aip_profile_edit_callback(n1, n2, is_open):
    return get_modal(n1, n2, is_open)


@app.callback(
    Output("aip-profile-list", "options"),
    [
        Input("aip-profile-modal", "is_open"),
        Input("dummy-id-5", "children"),
    ],
    prevent_initial_call=True,
)
def aip_profile_add_callback(is_open, n1):
    profiles = get_aip_profiles()
    profiles_options = make_options(profiles)
    return profiles_options


@app.callback(
    [
        Output("new-aip-profile-group", "style"),
        Output("aip-profile-status", "children"),
        Output("aip-profile-status", "color"),
        Output("aip-profile-status", "style"),
        Output("aip-profile-selected", "value"),
        Output("dummy-id-5", "children"),
    ],
    [
        Input("new-aip-profile-button", "n_clicks"),
        Input("save-aip-profile-button", "n_clicks"),
        Input("aip-profile-delete-button", "n_clicks"),
    ],
    [State("aip-profile-selected", "value"), State("aip-profile-list", "value")],
    prevent_initial_call=True,
)
def toggle_new_aip_profile_group_visible_callback(
    n_new, n_save, n_delete, new_profile_name, profile_to_delete
):
    button_id = get_called_button_id()
    group_style = {"display": "none"}
    status_style = {"display": "none"}
    status = ""
    status_color = "primary"

    if button_id == "new-aip-profile-button":
        group_style = {}

    elif button_id == "aip-profile-delete-button":
        status_style = {}
        profile_dir = f"{DATA_DIR}/data/user/{profile_to_delete}"
        if not os.path.exists(profile_dir):
            status = f"This profile does not exist: {profile_to_delete}"
            status_color = "warning"
        else:
            shutil.rmtree(profile_dir)
            status = f"Profile {new_profile_name} in {profile_dir} has been deleted"
            status_color = "success"
    elif button_id == "save-aip-profile-button":
        status_style = {}
        profile_dir = f"{DATA_DIR}/data/user/{new_profile_name}"
        if os.path.exists(profile_dir):
            status = "This profile already exists"
            status_color = "warning"
        else:
            os.makedirs(profile_dir)
            status = f"New profile {new_profile_name} created in {profile_dir}"
            status_color = "success"

    return group_style, status, status_color, status_style, None, []


@app.callback(
    [
        Output("dummy-profile", "children"),
        Output("page-content", "children"),
        Output("force-update-button", "n_clicks"),
        Output("config-save", "n_clicks"),
    ],
    [
        Input("aip-profile-load-button", "n_clicks"),
    ],
    [
        State("aip-profile", "value"),
        State("force-update-button", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def change_env(n, env, n_force_update):
    app.env = env
    config = get_config(env=env)
    GLOBAL_DICT = {}
    startup(config)
    conn = get_db_conn(config)
    # time.sleep(1)
    config = update_data(conn, config)

    q = queue.Queue()
    q.put_nowait(config)

    app.rfp = RunFileProcessor(config)
    app.rfp.reset()
    if True:
        log.info("Using processor")
        debug = False
        p = threading.Thread(target=start_watcher, args=(config, app.rfp, q))
        p.daemon = True
        p.start()

    print(f"n_force_update: {n_force_update}")
    return (
        f"Environ: {env}",
        html.Div(
            [
                html.H3("Loading new profile"),
                html.Meta(httpEquiv="refresh", content="0"),
            ]
        ),
        n_force_update + 1,
        n_force_update + 1,
        # config,
    )


def get_aip_profiles():
    import glob

    profiles = [
        os.path.basename(path)
        for path in glob.glob(f"{DATA_DIR}/data/user/*")
        if os.path.isdir(path)
    ]
    profiles = sorted(
        [
            p
            for p in profiles
            # if p in ["testing", "demo", "secondary", "primary", "all_raw"]
        ]
    )
    return profiles


@app.callback(
    [
        Output("aip-profile", "value"),
        Output("aip-profile", "options"),
    ],
    [
        Input("config-save", "n_clicks"),
        Input("dummy-id-5", "children"),
    ],
    [State("aip-profile", "value")],
)
def set_aip_profile_and_options(n, dummy_id, aip_profile):
    profiles = get_aip_profiles()
    profile_options = make_options(profiles)

    return aip_profile, profile_options


@app.callback(
    [
        Output("store-config", "data"),
        Output("config-text", "value"),
    ],
    [
        Input("config-save", "n_clicks"),
    ],
    [
        State("store-config", "data"),
        State({"type": "dir", "sub-type": "target", "index": ALL}, "value"),
        State({"type": "dir", "sub-type": "data", "index": ALL}, "value"),
        State({"type": "dir", "sub-type": "calibration", "index": ALL}, "value"),
        State({"type": "dir", "sub-type": "preproc-out", "index": ALL}, "value"),
        State({"type": "dir", "sub-type": "target", "index": ALL}, "valid"),
        State({"type": "dir", "sub-type": "data", "index": ALL}, "valid"),
        State({"type": "dir", "sub-type": "calibration", "index": ALL}, "valid"),
        State({"type": "dir", "sub-type": "preproc-out", "index": ALL}, "valid"),
        State("thread-count-slider", "value"),
        State("process-batch-size-slider", "value"),
        State("profile-list", "value"),
        State("profile-list", "options"),
        State(PlannerConfig.TARGET_STATUS, "value"),
        State(PlannerConfig.TARGET_PRIORITIES, "value"),
        State(PlannerConfig.PROFILES, "value"),
        State(PlannerConfig.MIN_MOON_DISTANCE, "value"),
        State(PlannerConfig.SOLAR_ALTITUDE_FOR_NIGHT, "value"),
        State(PlannerConfig.MIN_FRAME_OVERLAP_FRACTION, "value"),
        State(PlannerConfig.K_EXTINCTION, "value"),
        State(PlannerConfig.TIME_RESOLUTION, "value"),
        State(PlannerConfig.LAT, "value"),
        State(PlannerConfig.LON, "value"),
        State(PlannerConfig.UTC_OFFSET, "value"),
        State(PlannerConfig.MPSAS, "value"),
        State(VoyagerConfig.HOSTNAME, "value"),
        State(VoyagerConfig.PORT, "value"),
        State(VoyagerConfig.USER, "value"),
        State(VoyagerConfig.PASSWORD, "value"),
        State("themes", "value"),
        State(InspectorThresholds.ECC_THR, "value"),
        State(InspectorThresholds.STAR_FRAC_THR, "value"),
        State(InspectorThresholds.Z_SCORE, "value"),
        State(InspectorThresholds.IQR_SCALE, "value"),
        State(InspectorThresholds.TRAIL_THR, "value"),
        State(InspectorThresholds.GRADIENT_THR, "value"),
        State(SwitchConfig.VOYAGER_SWITCH, "on"),
        State(SwitchConfig.SIRIL_SWITCH, "on"),
        State(SwitchConfig.SILENCE_ALERTS_SWITCH, "on"),
        State(SwitchConfig.PLANNER_SWITCH, "on"),
        State(SwitchConfig.INSPECTOR_SWITCH, "on"),
        State(SwitchConfig.CULL_DATA_SWITCH, "on"),
    ],
    prevent_initial_call=True,
)
def update_config_callback(
    n,
    config_data,
    target_dirs,
    data_dirs,
    calibration_dirs,
    preproc_out_dirs,
    target_dirs_valid,
    data_dirs_valid,
    calibration_dirs_valid,
    preproc_out_dirs_valid,
    n_thread_count,
    n_batch_size,
    values,
    options,
    status_match,
    priority_match,
    profile_selection,
    min_moon_distance,
    sun_altitude_night,
    min_frame_overlap_frac,
    k_extinction,
    time_resolution,
    lat,
    lon,
    utc_offset,
    mpsas,
    voyager_hostname,
    voyager_port,
    voyager_user,
    voyager_password,
    theme,
    ecc_thr,
    star_frac_thr,
    z_score,
    iqr_scale,
    trail_thr,
    gradient_thr,
    voyager_switch,
    siril_switch,
    silence_alerts_switch,
    planner_switch,
    inspector_switch,
    cull_data_switch,
):
    try:
        log.debug(f"app.env = {app.env}")
        config_data = get_config(env=app.env)

        if all(target_dirs_valid):
            config_data["directories"]["target_dirs"] = target_dirs
        if all(data_dirs_valid):
            config_data["directories"]["data_dirs"] = data_dirs
        if all(calibration_dirs_valid):
            config_data["directories"]["calibration_dirs"] = calibration_dirs
        if all(preproc_out_dirs_valid):
            config_data["directories"]["preproc_out_dirs"] = preproc_out_dirs
        if options is not None and len(options) > 0:
            config_data["inactive_profiles"] = [
                option["value"] for option in options if option["value"] not in values
            ]
        if values is None:
            values = []

        if profile_selection is None:
            profile_selection = []

        config_data["voyager_config"] = {}
        config_data["voyager_config"][VoyagerConfig.HOSTNAME] = voyager_hostname
        config_data["voyager_config"][VoyagerConfig.PORT] = voyager_port
        config_data["voyager_config"][VoyagerConfig.USER] = voyager_user
        config_data["voyager_config"][VoyagerConfig.PASSWORD] = voyager_password

        config_data["planner_config"] = {}
        config_data["planner_config"][PlannerConfig.PROFILES] = list(profile_selection)
        config_data["planner_config"][PlannerConfig.TARGET_PRIORITIES] = priority_match
        config_data["planner_config"][PlannerConfig.TARGET_STATUS] = status_match
        config_data["planner_config"][PlannerConfig.LAT] = lat
        config_data["planner_config"][PlannerConfig.LON] = lon
        config_data["planner_config"][PlannerConfig.UTC_OFFSET] = utc_offset
        config_data["planner_config"][PlannerConfig.MPSAS] = mpsas
        config_data["planner_config"][PlannerConfig.TIME_RESOLUTION] = time_resolution
        config_data["planner_config"][
            PlannerConfig.MIN_MOON_DISTANCE
        ] = min_moon_distance
        config_data["planner_config"][
            PlannerConfig.SOLAR_ALTITUDE_FOR_NIGHT
        ] = sun_altitude_night
        config_data["planner_config"][
            PlannerConfig.MIN_FRAME_OVERLAP_FRACTION
        ] = min_frame_overlap_frac
        config_data["planner_config"][PlannerConfig.K_EXTINCTION] = k_extinction

        config_data["switch_config"] = {}
        config_data["switch_config"][
            SwitchConfig.SILENCE_ALERTS_SWITCH
        ] = silence_alerts_switch
        config_data["switch_config"][SwitchConfig.SIRIL_SWITCH] = siril_switch
        config_data["switch_config"][SwitchConfig.PLANNER_SWITCH] = planner_switch
        config_data["switch_config"][SwitchConfig.INSPECTOR_SWITCH] = inspector_switch
        config_data["switch_config"][SwitchConfig.CULL_DATA_SWITCH] = cull_data_switch
        config_data["switch_config"][SwitchConfig.VOYAGER_SWITCH] = voyager_switch

        config_data["inspector_thresholds"] = {}
        config_data["inspector_thresholds"][InspectorThresholds.ECC_THR] = ecc_thr
        config_data["inspector_thresholds"][
            InspectorThresholds.STAR_FRAC_THR
        ] = star_frac_thr
        config_data["inspector_thresholds"][InspectorThresholds.Z_SCORE] = z_score
        config_data["inspector_thresholds"][InspectorThresholds.IQR_SCALE] = iqr_scale
        config_data["inspector_thresholds"][InspectorThresholds.TRAIL_THR] = trail_thr
        config_data["inspector_thresholds"][
            InspectorThresholds.GRADIENT_THR
        ] = gradient_thr

        config_data["theme"] = theme

        config_data["n_threads"] = n_thread_count
        config_data["process_file_batch_size"] = n_batch_size
        app.queue.put_nowait(config_data)
        app.rfp.stop_loop = True
        save_config(config_data, app.env)

    except:
        log.warning("Config data could not be loaded from text", exc_info=EXC_INFO)
        log.warning(f"Config data parsed: {config_data}")
    return config_data, yaml.dump(config_data)


def get_df_for_status(target_names=None, filenames=None):
    df = get_object_from_global_store("df_reject_criteria_all")
    df = df.round(3)

    if df is not None:
        if target_names is not None:
            selection = df["OBJECT"].isin(target_names)
            df = df[selection]
        if filenames is not None:
            selection = df["filename"].isin(filenames)
            df = df[selection]

    return df


@app.callback(
    [
        Output("processing-progress", "value"),
        Output("processing-progress", "max"),
        Output("processing-progress", "label"),
        Output("processing-label", "children"),
        Output("processing-label", "style"),
        Output("preproc-progress", "value"),
        Output("preproc-progress", "max"),
        Output("preprocessing-label", "children"),
        Output("preprocessing-label", "style"),
        Output("preproc-progress", "style"),
    ],
    [Input("processing-interval", "n_intervals")],
)
def toggle_processing_callback(n):
    (
        n_total,
        n_processed,
        n_removed,
        processed_file_list,
        pending_file_list,
    ) = app.rfp.get_update(clear=False)
    progress_label = None
    progress_label_style = {}
    if len(processed_file_list) > 0:
        progress_label = f"File progress: {n_processed} / {n_total} ({n_processed / n_total*100:.0f}%)"
    else:
        n_processed = 0
        progress_label = None
        n_total = 1

    pp_progress_label = None
    pp_progress_label_style = {"display": "none"}
    pp_n_new = app.preproc_progress
    pp_n_count = app.preproc_count
    status = app.preproc_status
    if pp_n_count > 0:
        pp_progress_label = ", ".join(
            [
                f"{k}: {str(v)}"
                for k, v in app.preproc_list[-1].items()
                if k in ["OBJECT", "FILTER", FOCALLENGTH_COL, BINNING_COL, "count"]
            ]
        )
        pp_progress_label_style = {"height": "40px"}
        pp_progress_label = f"{status}: {pp_progress_label}"
    return (
        n_processed,
        n_total,
        "",
        progress_label,
        progress_label_style,
        pp_n_new,
        pp_n_count,
        pp_progress_label,
        {},
        pp_progress_label_style,
    )


@app.callback(
    [Output("profile-list", "options"), Output("profile-list", "value")],
    [Input("store-config", "data"), Input("new-data-available-trigger", "children")],
)
def profile_update_callback(config, x):
    target_data = get_target_data(config)
    options = [{"label": profile, "value": profile} for profile in target_data.profiles]
    values = [
        profile
        for profile in target_data.profiles
        if profile not in config.get("inactive_profiles", [])
    ]
    return options, values


@app.callback(
    [
        Output({"type": "dir", "sub-type": "target", "index": ALL}, "value"),
    ],
    [Input("store-config", "data")],
)
def get_target_dirs_callback(config):
    dirs = config.get("directories", {}).get("target_dirs", [])
    if dirs is None or len(dirs) == 0:
        raise PreventUpdate
    return [dirs]


@app.callback(
    [
        Output({"type": "dir", "sub-type": "data", "index": ALL}, "value"),
    ],
    [Input("store-config", "data")],
)
def get_data_dirs_callback(config):
    dirs = config.get("directories", {}).get("data_dirs", [])
    if dirs is None or len(dirs) == 0:
        raise PreventUpdate
    return [dirs]


@app.callback(
    [
        Output({"type": "dir", "sub-type": "calibration", "index": ALL}, "value"),
    ],
    [Input("store-config", "data")],
)
def get_cal_dirs_callback(config):
    dirs = config.get("directories", {}).get("calibration_dirs", [])
    if dirs is None or len(dirs) == 0:
        raise PreventUpdate
    return [dirs]


@app.callback(
    [
        Output({"type": "dir", "sub-type": "preproc-out", "index": ALL}, "value"),
    ],
    [Input("store-config", "data")],
)
def get_preproc_dirs_callback(config):
    dirs = config.get("directories", {}).get("preproc_out_dirs", [])
    if dirs is None or len(dirs) == 0:
        raise PreventUpdate
    return [dirs]


@app.callback(
    [
        Output({"type": "dir", "sub-type": ALL, "index": ALL}, "valid"),
        Output({"type": "dir", "sub-type": ALL, "index": ALL}, "invalid"),
    ],
    [Input({"type": "dir", "sub-type": ALL, "index": ALL}, "value")],
)
def check_validity_callback(dir_list):
    is_valid = [
        os.path.exists(dir) or (dir == "") if dir is not None else True
        for dir in dir_list
    ]
    return is_valid, [not x for x in is_valid]


@app.server.route("/status/<target_name>")
def serve_status_route(target_name):
    df = get_df_for_status(target_names=[target_name])
    if df is not None:
        cols = [
            "filename",
            "is_ok",
            "fwhm_median",
            "eccentricity_median",
            "star_trail_strength",
            "star_count_fraction",
            "relative_gradient_strength",
            "bad_star_shape",
            "low_star_count",
            "high_fwhm",
            "high_gradient",
        ]
        json_string = json.dumps(
            df[cols].to_dict(orient="records"),
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
        )
        return json_string
    return "[{}]"


@app.server.route("/stats")
def get_stats():
    group_key_string = request.args.get("group_keys", default="date_night_of")
    group_keys = group_key_string.split(",")
    df_data = get_object_from_global_store("df_data")
    df0 = df_data.copy()
    if "year" in group_keys:
        df0["year"] = df_data["date_night_of"].apply(lambda s: s.year)
    if "month" in group_keys:
        df0["month"] = df_data["date_night_of"].apply(lambda s: s.month)
    for key in group_keys:
        df0[key] = df0[key].astype(str)
    df_stats = (
        (df0.groupby(group_keys).agg({EXPOSURE_COL: "sum"}) / 3600)
        .round(3)
        .sort_index()
        .reset_index()
    )
    return jsonify(df_stats.to_dict(orient="records"))


def open_browser(port):
    webbrowser.open_new(f"http://localhost:{port}")


def run_dash(
    port,
    config,
    debug=True,
    monitor_mode_on=True,
    rfp=None,
    threaded=True,
    queue=None,
    ready_queue=None,
):
    env = config.get("env")

    save_config(config, env)
    app.env = env
    theme_name = config.get("theme", "LITERA")
    theme = dbc.themes.__dict__.get(theme_name, "LITERA")
    log.info(f"Theme: {theme}")
    app.config["external_stylesheets"] = [theme]
    log.info(f"Serving layout with monitor mode on = {monitor_mode_on}")

    app.layout = serve_layout(
        app,
        monitor_mode_on=monitor_mode_on,
    )
    log.info("Running server")
    app.rfp = rfp
    app.queue = queue
    app.preproc_list = []
    app.preproc_progress = 0
    app.preproc_count = 0
    app.preproc_status = ""

    if IS_PROD:
        open_browser(port)

    app.run(
        debug=debug,
        host="127.0.0.1",
        port=port,
        dev_tools_serve_dev_bundles=debug,
        threaded=threaded,
        use_reloader=debug,
        dev_tools_ui=debug,
    )


def startup(config):
    log.info("=" * 80)
    lines = open(f"{DATA_DIR}/data/banner.txt", "r").read().split("\n")
    for line in lines:
        log.info(line)
    log.info("https://github.com/gshau/AIP".center(80, " "))
    log.info("-" * 80)
    log.info("-" * 80)
    env = config.get("env")
    log.info(f"Using ENV = {env}".center(80, " "))
    log.info("=" * 80)


def start_watcher(config, rfp, q):
    data_dirs = config.get("directories", {}).get("data_dirs", [])
    if data_dirs is None or len(data_dirs) == 0:
        data_dirs = []
    target_dirs = config.get("directories", {}).get("target_dirs", [])
    if target_dirs is None or len(target_dirs) == 0:
        target_dirs = []
    calibration_dirs = config.get("directories", {}).get("calibration_dirs", [])
    if calibration_dirs is None or len(calibration_dirs) == 0:
        calibration_dirs = []
    preproc_out_dirs = config.get("directories", {}).get("preproc_out_dirs", [])
    if preproc_out_dirs is None or len(preproc_out_dirs) == 0:
        preproc_out_dirs = []
    watch_dirs = data_dirs + target_dirs + calibration_dirs + preproc_out_dirs
    w = Watcher(watch_dirs, target=rfp, queue=q)
    w.run()


def run_app(
    port=5678,
    debug=False,
    threaded=True,
    no_processor=False,
    disable_monitor_mode=False,
    env="primary",
    ready_queue=None,
):
    config = get_config(env=env)

    log.debug(config)
    startup(config)

    conn = get_db_conn(config)
    config = update_data(conn, config)

    q = queue.Queue()
    q.put_nowait(config)
    use_processor = not no_processor
    rfp = RunFileProcessor(config)
    if use_processor:
        log.info("Using processor")
        debug = False
        p = threading.Thread(target=start_watcher, args=(config, rfp, q))
        p.daemon = True
        p.start()
    log.info(f"Starting dash with debug = {debug}")

    run_dash(
        port,
        config,
        debug=debug,
        threaded=threaded,
        monitor_mode_on=not disable_monitor_mode,
        rfp=rfp,
        queue=q,
        ready_queue=ready_queue,
    )

    if use_processor:
        p.join()
