import base64
import time
import io
import os
import sqlalchemy
from distutils.util import strtobool

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

import pandas as pd
import numpy as np

import plotly.graph_objects as go

import warnings

import datetime
import yaml

import paho.mqtt.client as mqtt

from dash.dependencies import Input, Output, State
from sqlalchemy import exc
from flask import request

from scipy.interpolate import interp1d
from astro_planner.weather import NWS_Forecast
from astro_planner.target import object_file_reader, normalize_target_name, Objects
from astro_planner.contrast import add_contrast, add_moon_distance
from astro_planner.site import update_site
from astro_planner.ephemeris import get_coordinates
from astro_planner.data_parser import (
    INSTRUMENT_COL,
    EXPOSURE_COL,
    FOCALLENGTH_COL,
    BINNING_COL,
)

from image_grading.preprocessing import clear_tables, init_tables

from astro_planner.data_merge import (
    compute_ra_order,
    merge_targets_with_stored_metadata,
    get_targets_with_status,
    dump_target_status_to_file,
    load_target_status_from_file,
    set_target_status,
    update_targets_with_status,
)

from image_grading.frame_analysis import (
    show_inspector_image,
    show_frame_analysis,
    show_fwhm_ellipticity_vs_r,
)

from layout import serve_layout, yaxis_map
import seaborn as sns

from astropy.utils.exceptions import AstropyWarning

import flask

from astro_planner.logger import log
from pathlib import Path


warnings.simplefilter("ignore", category=AstropyWarning)


server = flask.Flask(__name__)

BS = "https://stackpath.bootstrapcdn.com/bootswatch/4.4.1/cosmo/bootstrap.min.css"
BS = dbc.themes.FLATLY
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO], server=server)

app.title = "The AstroImaging Planner"


mqtt_client = mqtt.Client()
mqtt_client.connect("mqtt", 1883, 60)


base_dir = Path(__file__).parents[1]
with open(f"{base_dir}/conf/config.yml", "r") as f:
    CONFIG = yaml.safe_load(f)

HORIZON_DATA = CONFIG.get("horizon_data", {})

with open(f"{base_dir}/conf/equipment.yml", "r") as f:
    EQUIPMENT = yaml.safe_load(f)
    if EQUIPMENT is None:
        EQUIPMENT = {}


ROBOCLIP_FILE = os.getenv("ROBOCLIP_FILE", "/roboclip/VoyRC.mdb")

DEFAULT_LAT = CONFIG.get("lat", 43.37)
DEFAULT_LON = CONFIG.get("lon", -88.37)
DEFAULT_UTC_OFFSET = CONFIG.get("utc_offset", -5)
DEFAULT_MPSAS = CONFIG.get("mpsas", 20.1)
DEFAULT_BANDWIDTH = CONFIG.get("bandwidth", 120)
DEFAULT_K_EXTINCTION = CONFIG.get("k_extinction", 0.2)
DEFAULT_TIME_RESOLUTION = CONFIG.get("time_resolution", 300)
DEFAULT_MIN_MOON_DISTANCE = CONFIG.get("min_moon_distance", 30)

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
PGPORT = os.getenv("PGPORT")
PGHOST = os.getenv("PGHOST", "0.0.0.0")

POSTGRES_ENGINE = sqlalchemy.create_engine(
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{PGHOST}:{PGPORT}/{POSTGRES_DB}"
)


USE_CONTRAST = strtobool(os.getenv("USE_CONTRAST", "True")) == 1
log.info(f"use contrast: {USE_CONTRAST}")
styles = {}
if not USE_CONTRAST:
    styles["k-ext"] = {"display": "none"}
    styles["local-mpsas"] = {"display": "none"}


L_FILTER = "L"
R_FILTER = "R"
G_FILTER = "G"
B_FILTER = "B"
HA_FILTER = "Ha"
OIII_FILTER = "OIII"
SII_FILTER = "SII"
BAYER = "OSC"
BAYER_ = "** BayerMatrix **"
NO_FILTER = "NO_FILTER"

FILTER_LIST = [
    L_FILTER,
    R_FILTER,
    G_FILTER,
    B_FILTER,
    HA_FILTER,
    OIII_FILTER,
    SII_FILTER,
    BAYER,
    BAYER_,
    NO_FILTER,
]

FILTER_MAP = {
    "Red": R_FILTER,
    "Green": G_FILTER,
    "Blue": B_FILTER,
    "Lum": L_FILTER,
    "Luminance": L_FILTER,
}


COLORS = {
    L_FILTER: "black",
    R_FILTER: "red",
    G_FILTER: "green",
    B_FILTER: "blue",
    HA_FILTER: "crimson",
    SII_FILTER: "maroon",
    OIII_FILTER: "teal",
    BAYER: "gray",
    BAYER_: "gray",
    NO_FILTER: "gray",
}


TRANSLATED_FILTERS = {
    "ha": ["ho", "sho", "hoo", "hos", "halpha", "h-alpha"],
    "oiii": ["ho", "sho", "hoo", "hos"],
    "sii": ["sho", "hos"],
    "nb": ["ha", "oiii", "sii", "sho", "ho", "hoo", "hos", "halpha", "h-alpha"],
    "bb": ["luminance", "lrgb"],
    "rgb": ["osc", "bayer", "dslr", "slr", "r ", " g ", " b "],
    "lum": ["luminance"],
}


# load main data
object_data = None
df_combined = pd.DataFrame()
df_target_status = pd.DataFrame()
df_stored_data = pd.DataFrame()
df_stars_headers = pd.DataFrame()
all_target_coords = pd.DataFrame()
all_targets = []


def pull_inspector_data():
    global df_stars_headers

    query = "select fh.*, asm.* from fits_headers fh left join aggregated_star_metrics asm on fh.filename  = asm.filename ;"

    try:
        df_stars_headers_ = pd.read_sql(query, POSTGRES_ENGINE)
        df_stars_headers_ = df_stars_headers_.loc[
            :, ~df_stars_headers_.columns.duplicated()
        ]
    except exc.SQLAlchemyError:
        log.info(
            f"Issue with reading tables, waiting for 15 seconds for it to resolve..."
        )
        time.sleep(15)
        return None

    df_stars_headers_["fwhm_mean_arcsec"] = (
        df_stars_headers_["fwhm_mean"] * df_stars_headers_["arcsec_per_pixel"]
    )
    df_stars_headers_["fwhm_std_arcsec"] = (
        df_stars_headers_["fwhm_std"] * df_stars_headers_["arcsec_per_pixel"]
    )

    df_stars_headers_["frame_snr"] = (
        10 ** df_stars_headers_["log_flux_mean"] * df_stars_headers_["n_stars"]
    ) / df_stars_headers_["bkg_val"]

    root_name = df_stars_headers_["file_full_path"].apply(lambda f: f.split("/")[1])
    df_stars_headers_["OBJECT"] = df_stars_headers_["OBJECT"].fillna(root_name)

    df_stars_headers = df_stars_headers_.copy()


def pull_stored_data():
    global df_stored_data

    log.debug("Checking for new data")

    stored_data_query = """
    select file_full_path as filename,
        "OBJECT",
        "DATE-OBS",
        cast("CCD-TEMP" as float),
        "FILTER",
        "OBJCTRA",
        "OBJCTDEC",
        "OBJCTALT",
        "INSTRUME" as "Instrument",
        cast("FOCALLEN" as float) as "Focal Length",
        cast("EXPOSURE" as float) as "Exposure",
        cast("XBINNING" as float) as "Binning",
        date("DATE-OBS") as "date"
        from fits_headers
    """

    try:
        df_stored_data = pd.read_sql(stored_data_query, POSTGRES_ENGINE)
    except exc.SQLAlchemyError:
        log.info(
            f"Issue with reading tables, waiting for 15 seconds for it to resolve..."
        )
        time.sleep(15)
        return None

    df_stored_data["date"] = pd.to_datetime(df_stored_data["date"])

    root_name = df_stored_data["filename"].apply(lambda f: f.split("/")[1])
    df_stored_data["OBJECT"] = df_stored_data["OBJECT"].fillna(root_name)
    df_stored_data["OBJECT"] = df_stored_data["OBJECT"].apply(normalize_target_name)


def pull_target_data():
    global object_data
    global df_combined
    global df_target_status
    global df_stored_data

    target_query = """select filename,
        "TARGET",
        "GROUP",
        "RAJ2000",
        "DECJ2000",
        "NOTE"
        FROM targets
    """

    try:
        df_objects = pd.read_sql(target_query, POSTGRES_ENGINE)
    except exc.SQLAlchemyError:
        log.info(
            f"Issue with reading tables, waiting for 15 seconds for it to resolve..."
        )
        time.sleep(15)
        return None

    object_data = Objects()
    object_data.load_from_df(df_objects)


def merge_target_with_stored_data():
    global df_stored_data
    global object_data
    global df_target_status
    global df_combined

    default_status = CONFIG.get("default_target_status", "")
    df_combined = merge_targets_with_stored_metadata(
        df_stored_data,
        object_data.df_objects,
        EQUIPMENT,
        default_status=default_status,
    )

    df_target_status = load_target_status_from_file()
    df_combined = set_target_status(df_combined, df_target_status)
    if df_combined["status"].isnull().sum() > 0:
        df_combined["status"] = df_combined["status"].fillna(default_status)
        dump_target_status_to_file(df_combined)


def update_data():
    pull_stored_data()
    pull_target_data()
    pull_inspector_data()
    merge_target_with_stored_data()


update_data()


def load_custom_horizon_function(
    horizon_file, sep=" ", header=3,
):
    flat_horizon_alt = HORIZON_DATA.get("flat_horizon_alt", 0)
    flat_horizon = interp1d([0, 360], [flat_horizon_alt, flat_horizon_alt])
    if not horizon_file:
        return flat_horizon
    try:
        df_horizon = pd.read_csv(horizon_file, sep=sep, header=header)
        df_horizon.columns = ["az", "alt"]
        df_horizon = df_horizon.append(
            pd.Series(dict(az=360, alt=df_horizon.iloc[-1]["alt"])), ignore_index=True
        )
        df_horizon = df_horizon.append(
            pd.Series(dict(az=0, alt=df_horizon.iloc[0]["alt"])), ignore_index=True
        )
        df_horizon = df_horizon.drop_duplicates(["az"])
        df_horizon = df_horizon.sort_values(by="az")
        f_horizon = interp1d(df_horizon["az"], df_horizon["alt"])
    except:
        log.warning(
            f"Issue with setting custom horizon from file: {horizon_file}, using flat horizon at {flat_horizon_alt} degrees"
        )
        return flat_horizon
    return f_horizon


f_horizon = load_custom_horizon_function(
    horizon_file=HORIZON_DATA.get("horizon_file", ""),
    sep=HORIZON_DATA.get("alt_az_seperator", " "),
    header=HORIZON_DATA.get("header_size", 3),
)


def make_options(elements):
    return [{"label": element, "value": element} for element in elements]


def get_time_limits(targets, sun_alt=-5):
    sun = targets["sun"]
    # Get sun up/down
    sun_up = np.where(np.gradient((sun.alt > sun_alt).astype(int)) > 0)[0][0]
    sun_dn = np.where(np.gradient((sun.alt > sun_alt).astype(int)) < 0)[0][0]
    return sun.index[sun_dn], sun.index[sun_up]


def get_data(
    target_coords,
    df_combined,
    value="alt",
    sun_alt_for_twilight=-18,
    filter_targets=True,
    min_moon_distance=30,
):

    target_names = [
        name for name in (list(target_coords.keys())) if name not in ["sun", "moon"]
    ]

    moon_data = dict(
        x=target_coords["moon"].index,
        y=target_coords["moon"]["alt"],
        text="Moon",
        opacity=1,
        line=dict(color="Gray", width=4),
        name="Moon",
    )

    sun_data = dict(
        x=target_coords["sun"].index,
        y=target_coords["sun"]["alt"],
        text="Sun",
        opacity=1,
        line=dict(color="Orange", width=4),
        name="Sun",
    )

    # Get sun up/down
    sun = target_coords["sun"]
    sun_up = np.where(np.gradient((sun.alt > sun_alt_for_twilight).astype(int)) > 0)[0][
        0
    ]
    sun_dn = np.where(np.gradient((sun.alt > sun_alt_for_twilight).astype(int)) < 0)[0][
        0
    ]

    log.info(f"Sun down: {sun.index[sun_dn]}")
    log.info(f"Sun up: {sun.index[sun_up]}")

    duraion_sun_down = sun.index[sun_up] - sun.index[sun_dn]
    duraion_sun_down_hrs = duraion_sun_down.total_seconds() / 3600.0

    log.info(f"Sun down duration: {duraion_sun_down_hrs:.2f} hours")

    sun_up_data = dict(
        x=[sun.index[sun_up], sun.index[sun_up], sun.index[-1], sun.index[-1]],
        y=[0, 90, 90, 0],
        mode="lines",
        line=dict(color="Orange", width=1),
        showlegend=False,
        fill="toself",
        name="Sun up",
    )
    sun_dn_data = dict(
        x=[sun.index[sun_dn], sun.index[sun_dn], sun.index[0], sun.index[0]],
        y=[0, 90, 90, 0],
        mode="lines",
        line=dict(color="Orange", width=1),
        showlegend=False,
        fill="toself",
        name="Sun down",
    )
    data = [sun_data, sun_up_data, sun_dn_data, moon_data]
    if (value == "contrast") or (value == "airmass") or (value == "sky_mpsas"):
        data = [sun_up_data, sun_dn_data]
    n_targets = len(target_coords)
    colors = sns.color_palette(n_colors=n_targets).as_hex()

    if target_names:
        # Sort targets by transit time
        records = []
        for target_name in target_names:
            max_alt_time = target_coords[target_name]["alt"].idxmax()
            records.append(dict(target_name=target_name, transit=max_alt_time))
        df_transit = pd.DataFrame(records).sort_values(by="transit")
        sorted_target_names = df_transit["target_name"].values

        for i_target, (color, target_name) in enumerate(
            zip(colors, sorted_target_names)
        ):
            df = target_coords[target_name]

            if filter_targets:
                meridian_at_night = (df["alt"].idxmax() > sun.index[sun_dn]) & (
                    df["alt"].idxmax() < sun.index[sun_up]
                )
                high_at_night = (
                    df.loc[sun.index[sun_dn] : sun.index[sun_up], "alt"].max() > 60
                )
                if not (meridian_at_night or high_at_night):
                    continue
            render_target = True
            notes_text = df_combined.loc[target_name, "NOTE"]
            skip_below_horizon = True
            for horizon_status in ["above", "below"]:
                if (horizon_status == "below") and skip_below_horizon:
                    continue
                if render_target:
                    df0 = df.copy()
                    show_trace = df["alt"] >= f_horizon(np.clip(df["az"], 0, 360))
                    show_trace &= df["moon_distance"] >= min_moon_distance

                    in_legend = True
                    opacity = 1
                    width = 3
                    if horizon_status == "below":
                        show_trace = df["alt"] > -90
                        in_legend = False
                        opacity = 0.15
                        width = 1

                    if show_trace.sum() == 0:
                        render_target = False
                        continue

                    df0.loc[~show_trace, value] = np.nan

                    text = df0.apply(
                        lambda row: f"Notes: {notes_text}<br>Moon distance: {row['moon_distance']:.1f} degrees<br>Local sky brightness (experimental): {row['sky_mpsas']:.2f} mpsas",
                        axis=1,
                    )

                    data.append(
                        dict(
                            x=df0.index,
                            y=df0[value],
                            mode="lines",
                            line=dict(color=color, width=width),
                            showlegend=in_legend,
                            name=target_name,
                            connectgaps=False,
                            legend_group=target_name,
                            customdata=np.dstack(
                                (df0["moon_distance"].values, df0["sky_mpsas"].values)
                            ),
                            hovertext=text,
                            opacity=opacity,
                        )
                    )

    return data, duraion_sun_down_hrs


def parse_loaded_contents(contents, filename, date):
    global object_data
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if ".mdb" in filename:
            file_root = filename.replace(".mdb", "")
            local_file = f"./data/uploads/{file_root}.mdb"
            with open(local_file, "wb") as f:
                f.write(decoded)
            object_data = object_file_reader(local_file)
            log.debug(object_data.df_objects.head())
            log.debug(local_file)
        elif ".sgf" in filename:
            out_data = io.StringIO(decoded.decode("utf-8"))
            file_root = filename.replace(".sgf", "")
            local_file = f"./data/uploads/{file_root}.sgf"
            with open(local_file, "w") as f:
                f.write(out_data.read())
            log.info("Done!")
            object_data = object_file_reader(local_file)
        elif ".xml" in filename:
            out_data = io.StringIO(decoded.decode("utf-8"))
            file_root = filename.replace(".xml", "")
            local_file = f"./data/uploads/{file_root}.xml"
            with open(local_file, "w") as f:
                f.write(out_data.read())
            log.info("Done!")
            object_data = object_file_reader(local_file)
        else:
            return None
    except Exception as e:
        log.warning(e)
        return None
    return object_data


def update_weather(site):
    log.debug("Trying NWS")
    nws_forecast = NWS_Forecast(site.lat, site.lon)
    df_weather = nws_forecast.parse_data()

    data = []
    for col in df_weather.columns:
        data.append(
            dict(
                x=df_weather.index,
                y=df_weather[col],
                mode="lines",
                name=col,
                opacity=1,
            )
        )

    graph_data = [
        dcc.Graph(
            config={
                "displaylogo": False,
                "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
            },
            figure={
                "data": data,
                "layout": dict(
                    title=df_weather.index.name,
                    margin={"l": 50, "b": 100, "t": 50, "r": 50},
                    legend={"x": 1, "y": 0.5},
                    yaxis={"range": [0, 100]},
                    height=600,
                    plot_bgcolor="#ddd",
                    paper_bgcolor="#fff",
                    hovermode="closest",
                    transition={"duration": 150},
                ),
            },
        )
    ]

    clear_outside_link = (
        f"http://clearoutside.com/forecast/{site.lat}/{site.lon}?view=current",
    )
    nws_link = (
        f"http://forecast.weather.gov/MapClick.php?lon={site.lon}&lat={site.lat}#.U1xl5F7N7wI",
    )

    return graph_data, clear_outside_link[0], nws_link[0]


def target_filter(targets, filters):
    targets_with_filter = []
    for filter in filters:
        for target in targets:
            if target.info["notes"]:
                if filter in target.info["notes"].lower():
                    targets_with_filter.append(target)
        if filter.lower() in TRANSLATED_FILTERS:
            for t_filter in TRANSLATED_FILTERS[filter.lower()]:
                targets_with_filter += [
                    target
                    for target in targets
                    if t_filter.lower() in target.info["notes"].lower()
                ]
    return list(set(targets_with_filter))


def get_progress_graph(df, date_string, profile_list, days_ago, targets=[]):

    selection = df["date"] < "1970-01-01"
    if days_ago > 0:
        selection |= df["date"] > str(
            datetime.datetime.today() - datetime.timedelta(days=days_ago)
        )
    targets_in_last_n_days = list(df[selection]["OBJECT"].unique())
    targets += targets_in_last_n_days
    if targets:
        selection |= df["OBJECT"].isin(targets)
    df0 = df[selection].reset_index()
    df0["FILTER"] = df0["FILTER"].replace(FILTER_MAP)

    p = go.Figure()
    if df0.shape[0] == 0:
        return dcc.Graph(figure=p), pd.DataFrame()

    df_summary = (
        df0.groupby(["OBJECT", "FILTER", BINNING_COL, FOCALLENGTH_COL, INSTRUMENT_COL])
        .agg({EXPOSURE_COL: "sum"})
        .dropna()
    )
    df_summary = df_summary.unstack(1).fillna(0)[EXPOSURE_COL] / 3600
    cols = ["OBJECT"] + FILTER_LIST
    df_summary = df_summary[[col for col in cols if col in df_summary.columns]]

    df0["ra_order"] = df0["OBJCTRA"].apply(
        lambda ra: compute_ra_order(ra, date_string=date_string)
    )

    objects_sorted = (
        df0.dropna()
        .groupby("OBJECT")
        .agg({"ra_order": "mean"})
        .sort_values(by="ra_order")
        .index
    )

    df0 = df_summary.reset_index()
    bin = df0[BINNING_COL].astype(int).astype(str)
    fl = df0[FOCALLENGTH_COL].astype(int).astype(str)
    df0["text"] = df0[INSTRUMENT_COL] + " @ " + bin + "x" + bin + " FL = " + fl + "mm"
    df_summary = df0.set_index("OBJECT").loc[objects_sorted]
    for filter in [col for col in COLORS if col in df_summary.columns]:
        p.add_trace(
            go.Bar(
                name=f"{filter}",
                x=df_summary.index,
                y=df_summary[filter],
                hovertext=df_summary["text"],
                marker_color=COLORS[filter],
            )
        )
    p.update_layout(
        barmode=CONFIG.get("progress_mode", "group"),
        yaxis_title="Total Exposure (hr)",
        title="Acquired Data",
        title_x=0.5,
    )
    graph = dcc.Graph(
        config={"displaylogo": False, "modeBarButtonsToRemove": ["pan2d", "lasso2d"]},
        figure=p,
    )

    return graph, df_summary


# Set layout
app.layout = serve_layout


# Callbacks
@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal_callback(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    [Output("profile-selection", "options"), Output("profile-selection", "value")],
    [Input("upload-data", "contents")],
    [
        State("upload-data", "filename"),
        State("upload-data", "last_modified"),
        State("profile-selection", "value"),
    ],
)
def update_output_callback(
    list_of_contents, list_of_names, list_of_dates, profiles_selected
):
    global object_data

    if not object_data:
        return [{}], [{}]
    if not object_data.profiles:
        return [{}], [{}]

    profile = object_data.profiles[0]

    inactive_profiles = CONFIG.get("inactive_profiles", [])
    default_profiles = CONFIG.get("default_profiles", [])

    options = [
        {"label": profile, "value": profile}
        for profile in object_data.profiles
        if profile not in inactive_profiles
    ]

    default_options = profiles_selected
    if default_profiles:
        default_options = default_profiles

    if list_of_contents is not None:
        for (c, n, d) in zip(list_of_contents, list_of_names, list_of_dates):
            object_data = parse_loaded_contents(c, n, d)
            if object_data:
                for profile in object_data.profiles:
                    if profile not in inactive_profiles:
                        options.append({"label": profile, "value": profile})
        return options, default_options
    return options, default_options


@app.callback(
    [
        Output("weather-graph", "children"),
        Output("clear-outside", "href"),
        Output("nws-weather", "href"),
    ],
    [Input("store-site-data", "data")],
)
def update_weather_data_callback(site_data):
    site = update_site(
        site_data,
        default_lat=DEFAULT_LAT,
        default_lon=DEFAULT_LON,
        default_utc_offset=DEFAULT_UTC_OFFSET,
    )
    weather_graph, clear_outside_link, nws_link = update_weather(site)
    return weather_graph, clear_outside_link, nws_link


@app.callback(
    Output("store-site-data", "data"),
    [
        Input("input-lat", "value"),
        Input("input-lon", "value"),
        Input("input-utc-offset", "value"),
    ],
)
def update_time_location_data_callback(lat=None, lon=None, utc_offset=None):

    site_data = dict(lat=DEFAULT_LAT, lon=DEFAULT_LON, utc_offset=DEFAULT_UTC_OFFSET)
    if lat:
        site_data["lat"] = lat
    if lon:
        site_data["lon"] = lon
    if utc_offset:
        site_data["utc_offset"] = utc_offset
    return site_data


@app.callback(
    [Output("target-status-match", "options"), Output("target-status-match", "value")],
    [Input("profile-selection", "value")],
    [State("status-match", "value")],
)
def update_target_for_status_callback(profile_list, status_match):
    df_combined_group = df_combined[df_combined["GROUP"].isin(profile_list)]
    targets = df_combined_group.index.values
    return make_options(targets), ""


@app.callback(
    Output("target-status-selector", "value"),
    [Input("target-status-match", "value")],
    [State("store-target-status", "data"), State("profile-selection", "value")],
)
def update_radio_status_for_targets_callback(
    targets, target_status_store, profile_list
):
    global df_target_status
    status_set = set()
    status = df_target_status.set_index(["OBJECT", "GROUP"])
    for target in targets:
        status_values = [status.loc[target].loc[profile_list]["status"]]
        status_set = status_set.union(set(status_values))
    if len(status_set) == 1:
        log.debug(f"Fetching targets: {targets} with status of {status_set}")
        return list(status_set)[0]
    else:
        log.debug(f"Conflict fetching targets: {targets} with status of {status_set}")
        return ""


@app.callback(
    Output("store-target-status", "data"),
    [Input("target-status-selector", "value")],
    [
        State("target-status-match", "value"),
        State("store-target-status", "data"),
        State("profile-selection", "value"),
    ],
)
def update_target_with_status_callback(
    status, targets, target_status_store, profile_list
):
    global df_combined
    global df_target_status
    df_combined, df_target_status = update_targets_with_status(
        targets, status, df_combined, profile_list
    )
    return ""


@app.callback(
    [
        Output("tab-target-div", "style"),
        Output("tab-data-table-div", "style"),
        Output("tab-files-table-div", "style"),
        Output("tab-config-div", "style"),
    ],
    [Input("tabs", "active_tab")],
)
def render_content(tab):

    tab_names = [
        "tab-target",
        "tab-data-table",
        "tab-files-table",
        "tab-config",
    ]

    styles = [{"display": "none"}] * len(tab_names)

    indx = tab_names.index(tab)

    styles[indx] = {}
    return styles


def shutdown():
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()


@app.server.route("/getLogs/<file>")
def download_log(file):
    return flask.send_file(
        f"/logs/{file}", attachment_filename=file, as_attachment=True, cache_timeout=-1
    )


def shutdown_watchdog():
    mqtt_client.publish("watchdog", "restart")


# Callbacks
@app.callback(
    Output("dummy-id", "children"),
    [
        Input("button-clear-tables", "n_clicks"),
        Input("button-clear-star-tables", "n_clicks"),
        Input("button-clear-header-tables", "n_clicks"),
        Input("button-clear-target-tables", "n_clicks"),
        Input("button-restart-app", "n_clicks"),
        Input("button-restart-watchdog", "n_clicks"),
    ],
)
def config_buttons(n1, n2, n3, n4, n5, n6):
    ctx = dash.callback_context

    if ctx.triggered:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "button-clear-tables":
            log.info("Clearing all tables")
            init_tables()
        if button_id == "button-clear-star-tables":
            tables_to_clear = [
                "aggregated_star_metrics",
                "xy_frame_metrics",
                "radial_frame_metrics",
            ]
            log.info(f"Clearing tables: {tables_to_clear}")
            clear_tables(tables_to_clear)
        if button_id == "button-clear-header-tables":
            tables_to_clear = [
                "fits_headers",
                "fits_status",
            ]
            log.info(f"Clearing tables: {tables_to_clear}")
            clear_tables(tables_to_clear)
        if button_id == "button-clear-target-tables":
            tables_to_clear = ["targets"]
            log.info(f"Clearing tables: {tables_to_clear}")
            clear_tables(tables_to_clear)
        if button_id == "button-restart-app":
            log.info(f"Restarting app")
            shutdown()
        if button_id == "button-restart-watchdog":
            log.info(f"Restarting watchdog")
            shutdown_watchdog()

    return ""


def store_target_coordinate_data(date_string, site_data):
    global df_combined, object_data, all_target_coords

    t0 = time.time()
    site = update_site(
        site_data,
        default_lat=DEFAULT_LAT,
        default_lon=DEFAULT_LON,
        default_utc_offset=DEFAULT_UTC_OFFSET,
    )
    targets = []
    target_list = object_data.target_list
    for profile in target_list:
        targets += list(target_list[profile].values())

    all_target_coords = get_coordinates(
        targets, date_string, site, time_resolution_in_sec=DEFAULT_TIME_RESOLUTION
    )

    log.info(f"store_target_coordinate_data: {time.time() - t0:.3f}")

    return all_target_coords, targets


def filter_targets_for_matches_and_filters(
    targets, status_matches, filters, profile_list
):
    global df_combined
    global object_data

    targets = []
    for profile in profile_list:
        if profile in object_data.target_list:
            targets += list(object_data.target_list[profile].values())

    if filters:
        targets = target_filter(targets, filters)

    if status_matches:
        matching_targets = df_combined[df_combined["status"].isin(status_matches)].index
        targets = [target for target in targets if target.name in matching_targets]

    return targets


@app.callback(
    Output("dummy-id-target-data", "children"),
    [Input("date-picker", "date"), Input("store-site-data", "data"),],
)
def get_target_data(
    date_string, site_data,
):
    global all_target_coords, all_targets
    t0 = time.time()

    all_target_coords, all_targets = store_target_coordinate_data(
        date_string, site_data
    )
    all_target_coords = add_moon_distance(all_target_coords)

    log.info(f"store_target_coordinate_data: {time.time() - t0}")
    return ""


@app.callback(
    Output("dummy-id-contrast-data", "children"),
    [
        Input("dummy-id-target-data", "children"),
        Input("local-mpsas", "value"),
        Input("k-ext", "value"),
    ],
)
def update_contrast(
    dummy_input, local_mpsas, k_ext,
):
    global all_target_coords

    if local_mpsas is None:
        local_mpsas = DEFAULT_MPSAS
    filter_bandwidth = DEFAULT_BANDWIDTH
    if k_ext is None:
        k_ext = DEFAULT_K_EXTINCTION

    all_target_coords = add_contrast(
        all_target_coords,
        filter_bandwidth=filter_bandwidth,
        mpsas=local_mpsas,
        include_airmass=True,
        k_ext=k_ext,
    )

    return ""


@app.callback(
    [
        Output("store-target-data", "data"),
        Output("store-target-list", "data"),
        Output("store-target-metadata", "data"),
        Output("dark-sky-duration", "data"),
    ],
    [
        Input("dummy-id-contrast-data", "children"),
        Input("store-target-status", "data"),
        Input("y-axis-type", "value"),
        Input("filter-targets", "checked"),
        Input("status-match", "value"),
        Input("filter-match", "value"),
        Input("min-moon-distance", "value"),
    ],
    [State("profile-selection", "value")],
)
def store_data(
    dummy_input,
    target_status_store,
    value,
    filter_targets,
    status_matches,
    filters,
    min_moon_distance,
    profile_list,
):

    global df_combined, object_data, all_target_coords, all_targets
    t0 = time.time()

    if min_moon_distance is None:
        min_moon_distance = DEFAULT_MIN_MOON_DISTANCE

    targets = filter_targets_for_matches_and_filters(
        all_targets, status_matches, filters, profile_list
    )
    target_names = [t.name for t in targets]
    target_names.append("sun")
    target_names.append("moon")

    log.info(f"filter_targets_for_matches_and_filters: {time.time() - t0}")

    target_coords = dict(
        [[k, v] for k, v in all_target_coords.items() if k in target_names]
    )

    sun_down_range = get_time_limits(target_coords)
    log.info(f"get_time_limits: {time.time() - t0}")

    data, duration_sun_down_hrs = get_data(
        target_coords,
        df_combined,
        value=value,
        filter_targets=filter_targets,
        min_moon_distance=min_moon_distance,
    )
    log.info(f"get_data: {time.time() - t0}")

    dark_sky_duration_text = (
        f"Length of sky darkness: {duration_sun_down_hrs:.1f} hours"
    )

    metadata = dict(date_range=sun_down_range, value=value)
    filtered_targets = [d["name"] for d in data if d["name"]]

    return data, filtered_targets, metadata, dark_sky_duration_text


@app.callback(
    [
        Output("target-graph", "children"),
        Output("progress-graph", "children"),
        Output("data-table", "children"),
    ],
    [
        Input("store-target-data", "data"),
        Input("store-progress-data", "data"),
        Input("store-target-goals", "data"),
    ],
    [
        State("store-target-metadata", "data"),
        State("dark-sky-duration", "data"),
        State("profile-selection", "value"),
        State("status-match", "value"),
        State("date-picker", "date"),
    ],
)
def update_target_graph(
    target_data,
    progress_data,
    target_goals,
    metadata,
    dark_sky_duration,
    profile_list,
    status_list,
    date,
):
    global df_target_status
    global df_combined
    global df_stored_data

    log.debug(f"Calling update_target_graph")
    if not metadata:
        metadata = {}
    try:
        value = metadata["value"]
        date_range = metadata["date_range"]
    except KeyError:
        return None, None, None

    date_string = str(date.split("T")[0])
    title = "Imaging Targets For Night of {date_string} <br> {dark_sky_duration}".format(
        date_string=date_string, dark_sky_duration=dark_sky_duration
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

    target_graph = dcc.Graph(
        config={"displaylogo": False, "modeBarButtonsToRemove": ["pan2d", "lasso2d"]},
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
            ),
        },
    )

    df_combined_group = df_combined[df_combined["GROUP"].isin(profile_list)]

    df_combined_group = set_target_status(df_combined_group, df_target_status)
    targets = get_targets_with_status(df_combined_group, status_list=status_list)
    progress_days_ago = int(CONFIG.get("progress_days_ago", 0))

    progress_graph, df_summary = get_progress_graph(
        df_stored_data,
        date_string=date_string,
        profile_list=profile_list,
        days_ago=progress_days_ago,
        targets=targets,
    )

    df_combined_group = df_combined_group.reset_index()
    columns = []
    for col in df_combined_group.columns:
        entry = {"name": col, "id": col, "deletable": False, "selectable": True}
        columns.append(entry)

    # target table
    data = df_combined.reset_index().to_dict("records")
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
            )
        ]
    )

    return target_graph, progress_graph, target_table


@app.callback(
    [
        Output("files-table", "children"),
        Output("summary-table", "children"),
        Output("header-col-match", "options"),
        Output("target-match", "options"),
        Output("x-axis-field", "options"),
        Output("y-axis-field", "options"),
        Output("scatter-size-field", "options"),
    ],
    [
        Input("store-target-data", "data"),
        Input("header-col-match", "value"),
        Input("target-match", "value"),
    ],
)
def update_files_table(target_data, header_col_match, target_match):
    global df_target_status
    global df_combined
    global df_stars_headers

    targets = sorted(df_stars_headers["OBJECT"].unique())
    target_options = make_options(targets)

    df0 = df_stars_headers.copy()
    if target_match:
        log.info("Selecting target match")
        df0 = df_stars_headers[df_stars_headers["OBJECT"] == target_match]
    log.info("Done with queries")

    columns = []
    default_cols = [
        "OBJECT",
        "DATE-OBS",
        "FILTER",
        "EXPOSURE",
        "XPIXSZ",
        "FOCALLEN",
        "arcsec_per_pixel",
        "CCD-TEMP",
        "fwhm_mean_arcsec",
        "eccentricity_mean",
        "star_trail_strength",
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
    # df0["FILTER"] = df0["FILTER"].replace(FILTER_MAP)
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
            )
        ]
    )
    df0["DATE-OBS"] = pd.to_datetime(df0["DATE-OBS"])
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

    df_agg = df0.groupby(["OBJECT", "FILTER", "XBINNING", "FOCALLEN", "XPIXSZ"]).agg(
        {
            "EXPOSURE": "sum",
            "CCD-TEMP": "std",
            "DATE-OBS": "count",
            "star_orientation_score": "mean",
        }
    )
    df_agg["EXPOSURE"] = df_agg["EXPOSURE"] / 3600
    col_map = {
        "DATE-OBS": "n_subs",
        "CCD-TEMP": "CCD-TEMP Dispersion",
        "EXPOSURE": "EXPOSURE (hour)",
    }
    df_agg = df_agg.reset_index().rename(col_map, axis=1)
    df_agg["FILTER_indx"] = df_agg["FILTER"].map(
        dict(zip(FILTER_LIST, range(len(FILTER_LIST))))
    )
    df_agg = df_agg.sort_values(by=["FILTER_indx"]).drop("FILTER_indx", axis=1)
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
            )
        ]
    )

    return (
        files_table,
        summary_table,
        header_options,
        target_options,
        scatter_field_options,
        scatter_field_options,
        scatter_field_options,
    )


@app.callback(
    [Output("x-axis-field", "value"), Output("y-axis-field", "value")],
    [Input("scatter-radio-selection", "value")],
)
def update_scatter_axes(value):
    x_col = "fwhm_mean_arcsec"
    y_col = "eccentricity_mean"
    if value:
        x_col, y_col = value.split(" vs. ")
    return x_col, y_col


@app.callback(
    Output("target-scatter-graph", "figure"),
    [
        Input("store-target-data", "data"),
        Input("target-match", "value"),
        Input("x-axis-field", "value"),
        Input("y-axis-field", "value"),
        Input("scatter-size-field", "value"),
    ],
)
def update_scatter_plot(target_data, target_match, x_col, y_col, size_col):
    global df_stars_headers

    df0 = df_stars_headers[(df_stars_headers["OBJECT"] == target_match)]
    df0["FILTER"] = df0["FILTER"].replace(FILTER_MAP)
    filters = df0["FILTER"].unique()
    if not x_col:
        x_col = "fwhm_mean_arcsec"
    if not y_col:
        y_col = "eccentricity_mean"
    p = go.Figure()
    sizeref = float(2.0 * df0[size_col].max() / (5 ** 2))
    for filter in FILTER_LIST:
        if filter not in filters:
            continue
        df1 = df0[df0["FILTER"] == filter].reset_index()

        df1["text"] = df1.apply(
            lambda row: "<br>Date: "
            + str(row["DATE-OBS"])
            + f"<br>Star count: {row['n_stars']}"
            + f"<br>FWHM: {row['fwhm_mean']:.2f}"
            + f"<br>Eccentricity: {row['eccentricity_mean']:.2f}"
            + f"<br>{size_col}: {row[size_col]:.2f}",
            axis=1,
        )
        default_size = df1[size_col].median()
        if np.isnan(default_size):
            default_size = 1
        size = df1[size_col].fillna(default_size)

        if filter in [HA_FILTER, OIII_FILTER, SII_FILTER]:
            symbol = "diamond"
        else:
            symbol = "circle"

        p.add_trace(
            go.Scatter(
                x=df1[x_col],
                y=df1[y_col],
                mode="markers",
                name=filter,
                hovertemplate="<b>%{text}</b><br>"
                + f"{x_col}: "
                + "%{x:.2f}<br>"
                + f"{y_col}: "
                + "%{y:.2f}<br>",
                text=df1["text"],
                marker=dict(
                    color=COLORS[filter], size=size, sizeref=sizeref, symbol=symbol
                ),
                customdata=df1["filename"],
            )
        )
    p.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        title=f"Subframe data for {target_match}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return p


@app.callback(
    [
        Output("radial-frame-graph", "figure"),
        Output("xy-frame-graph", "figure"),
        Output("inspector-frame", "figure"),
    ],
    [
        Input("target-scatter-graph", "clickData"),
        Input("aberration-preview", "checked"),
        Input("frame-heatmap-dropdown", "value"),
    ],
)
def inspect_frame_analysis(data, as_aberration_inspector, frame_heatmap_col):
    p0 = go.Figure()
    p1 = p0
    p2 = p0

    if data is None:
        return p0, p1, p2
    base_filename = data["points"][0]["customdata"]
    if not base_filename:
        return p0, p1, p2

    file_full_path = df_stars_headers[df_stars_headers["filename"] == base_filename]
    if file_full_path.shape[0] == 1:
        log.info(file_full_path)
        filename = file_full_path["file_full_path"].values[0]
    else:
        return p0, p1, p2

    t0 = time.time()
    radial_query = (
        f"""select * from radial_frame_metrics where filename = '{base_filename}';"""
    )
    df_radial = pd.read_sql(radial_query, POSTGRES_ENGINE)

    xy_query = f"""select * from xy_frame_metrics where filename = '{base_filename}';"""
    df_xy = pd.read_sql(xy_query, POSTGRES_ENGINE)
    log.info(f"Time for query: {time.time() - t0:.2f} seconds")

    p2, canvas = show_inspector_image(
        filename,
        as_aberration_inspector=as_aberration_inspector,
        with_overlay=False,
        n_cols=3,
        n_rows=3,
        border=5,
    )

    p0 = show_fwhm_ellipticity_vs_r(df_radial, filename)
    p1 = show_frame_analysis(df_xy, filename=filename, feature_col=frame_heatmap_col)

    return p0, p1, p2


@app.callback(
    [
        Output("alert-auto", "children"),
        Output("alert-auto", "is_open"),
        Output("alert-auto", "duration"),
        Output("alert-auto", "color"),
    ],
    [Input("interval-component", "n_intervals")],
)
def toggle_alert(n):
    global df_stars_headers
    df_old = df_stars_headers.copy()
    update_data()
    df_new = df_stars_headers.drop(df_old.index)
    new_row_count = df_new.shape[0]

    new_files_available = new_row_count > 0

    if new_files_available:
        filenames = df_new["filename"].values
        response = [f"Detected {new_row_count} new files available:"]
        for filename in filenames:
            response.append(html.Br())
            response.append(filename)
        is_open = True
        duration = 60000
        color = "primary"
        return response, is_open, duration, color
    return "", False, 0, "primary"


@app.callback(
    [Output("glossary-modal", "is_open"), Output("glossary", "children")],
    [Input("glossary-open", "n_clicks"), Input("glossary-close", "n_clicks")],
    [State("glossary-modal", "is_open")],
)
def toggle_glossary_modal_callback(n1, n2, is_open):

    with open("/app/src/glossary.md", "r") as f:
        text = f.readlines()

    status = is_open
    if n1 or n2:
        status = not is_open

    return status, dcc.Markdown(text)


if __name__ == "__main__":
    localhost_only = strtobool(CONFIG.get("localhost_only", "True"))
    debug = strtobool(CONFIG.get("debug", "False"))
    host = "0.0.0.0"
    if localhost_only:
        host = "localhost"
    app.run_server(debug=True, host=host, dev_tools_serve_dev_bundles=True)
