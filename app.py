import base64
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

from dash.dependencies import Input, Output, State

from scipy.interpolate import interp1d
from astro_planner.weather import NWS_Forecast
from astro_planner.target import object_file_reader, normalize_target_name
from astro_planner.contrast import add_contrast
from astro_planner.site import update_site
from astro_planner.ephemeris import get_coordinates
from astro_planner.data_parser import (
    INSTRUMENT_COL,
    EXPOSURE_COL,
    FOCALLENGTH_COL,
    BINNING_COL,
)

from astro_planner.data_merge import (
    compute_ra_order,
    merge_roboclip_stored_metadata,
    get_targets_with_status,
    dump_target_status_to_file,
    load_target_status_from_file,
    set_target_status,
    update_targets_with_status,
)
from layout import serve_layout, yaxis_map
import seaborn as sns

from astropy.utils.exceptions import AstropyWarning

import flask

from astro_planner.logger import log

warnings.simplefilter("ignore", category=AstropyWarning)


server = flask.Flask(__name__)

BS = "https://stackpath.bootstrapcdn.com/bootswatch/4.4.1/cosmo/bootstrap.min.css"
BS = dbc.themes.FLATLY
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO], server=server)

app.title = "The AstroImaging Planner"


with open("./conf/config.yml", "r") as f:
    CONFIG = yaml.safe_load(f)
HORIZON_DATA = CONFIG.get("horizon_data", {})

with open("./conf/equipment.yml", "r") as f:
    EQUIPMENT = yaml.safe_load(f)

DATA_DIR = os.getenv("DATA_DIR", "/data")
ROBOCLIP_FILE = os.getenv("ROBOCLIP_FILE", "/roboclip/VoyRC.mdb")

DEFAULT_LAT = CONFIG.get("lat", 43.37)
DEFAULT_LON = CONFIG.get("lon", -88.37)
DEFAULT_UTC_OFFSET = CONFIG.get("utc_offset", -5)
DEFAULT_MPSAS = CONFIG.get("mpsas", 20.1)
DEFAULT_BANDWIDTH = CONFIG.get("bandwidth", 120)
DEFAULT_K_EXTINCTION = CONFIG.get("k_extinction", 0.2)
DEFAULT_TIME_RESOLUTION = CONFIG.get("time_resolution", 300)


USE_CONTRAST = os.getenv("USE_CONTRAST", False)
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


FILTER_LIST = [
    L_FILTER,
    R_FILTER,
    G_FILTER,
    B_FILTER,
    HA_FILTER,
    OIII_FILTER,
    SII_FILTER,
    BAYER,
]

COLORS = {
    L_FILTER: "black",
    R_FILTER: "red",
    G_FILTER: "green",
    B_FILTER: "blue",
    HA_FILTER: "crimson",
    SII_FILTER: "maroon",
    OIII_FILTER: "teal",
    BAYER: "gray",
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
df_combined = None
df_target_status = None
df_stored_data = None
df_stars_headers = None


def update_data():
    global object_data
    global df_combined
    global df_target_status
    global df_stored_data
    global df_stars_headers

    log.info("Updating Data")

    query = """
    select file_full_path as filename,
        "OBJECT",
        "DATE-OBS",
        "FILTER",
        cast("CCD-TEMP" as float),
        "OBJCTRA",
        "OBJCTDEC",
        cast("AIRMASS" as float),
        "OBJCTALT",
        "INSTRUME" as "Instrument",
        cast("FOCALLEN" as float) as "Focal Length",
        cast("EXPOSURE" as float) as "Exposure",
        cast("XBINNING" as float) as "Binning",
        date("DATE-OBS") as "date"
        from fits_headers
    """
    conn = sqlalchemy.create_engine(
        "postgresql+psycopg2://astro_user:andromeda@db:5431/fits_files"
    )

    df_stored_data = pd.read_sql(query, conn)
    df_stored_data["date"] = pd.to_datetime(df_stored_data["date"])
    df_stored_data["filename"] = df_stored_data["filename"].apply(
        lambda f: f.replace("/Volumes/Users/gshau/Dropbox/AstroBox", "")
    )

    df_stored_data["OBJECT"] = df_stored_data["OBJECT"].apply(normalize_target_name)

    object_data = object_file_reader(ROBOCLIP_FILE)

    default_status = CONFIG.get("default_target_status", "")

    df_combined = merge_roboclip_stored_metadata(
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

    # files tables
    log.info("Ready for queries")
    engine = sqlalchemy.create_engine(
        "postgresql+psycopg2://astro_user:andromeda@db:5431/fits_files"
    )

    header_query = "select * from fits_headers;"
    log.info("Ready to query headers")
    df_headers = pd.read_sql(header_query, engine)
    star_query = "select * from star_metrics;"

    log.info("Ready to query stars")
    df_stars = pd.read_sql(star_query, engine)

    df_stars_headers = pd.merge(df_headers, df_stars, on="filename", how="left")
    df_stars_headers["fwhm_mean_arcsec"] = (
        df_stars_headers["fwhm_mean"] * df_stars_headers["arcsec_per_pixel"]
    )
    df_stars_headers["fwhm_std_arcsec"] = (
        df_stars_headers["fwhm_std"] * df_stars_headers["arcsec_per_pixel"]
    )

    df_stars_headers["frame_snr"] = (
        10 ** df_stars_headers["log_flux_mean"] * df_stars_headers["n_stars"]
    ) / df_stars_headers["bkg_val"]


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
    targets,
    df_combined,
    value="alt",
    sun_alt_for_twilight=-18,
    local_mpsas=20,
    filter_bandwidth=300,
    k_ext=0.2,
    filter_targets=True,
):
    log.debug("Starting get_data")
    target_names = [
        name for name in (list(target_coords.keys())) if name not in ["sun", "moon"]
    ]
    if local_mpsas is None:
        local_mpsas = DEFAULT_MPSAS
    if filter_bandwidth is None:
        filter_bandwidth = DEFAULT_BANDWIDTH
    if k_ext is None:
        k_ext = DEFAULT_K_EXTINCTION

    if value == "contrast":
        target_coords = add_contrast(
            target_coords,
            filter_bandwidth=filter_bandwidth,
            mpsas=local_mpsas,
            object_brightness=19,
            include_airmass=True,
            k_ext=k_ext,
        )
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
        name="sun_up",
    )
    sun_dn_data = dict(
        x=[sun.index[sun_dn], sun.index[sun_dn], sun.index[0], sun.index[0]],
        y=[0, 90, 90, 0],
        mode="lines",
        line=dict(color="Orange", width=1),
        showlegend=False,
        fill="toself",
        name="sun_dn",
    )
    data = [sun_data, sun_up_data, sun_dn_data, moon_data]
    if value == "contrast":
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
            notes_text = df_combined.loc[target_name, "NOTE"]

            for horizon_status in ["above", "below"]:
                df0 = df.copy()
                show_trace = df["alt"] >= f_horizon(np.clip(df["az"], 0, 360))
                in_legend = True
                opacity = 1
                width = 3
                if horizon_status == "below":
                    show_trace = df["alt"] > -90
                    in_legend = False
                    opacity = 0.15
                    width = 1

                df0.loc[~show_trace, value] = np.nan
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
                        text="Notes: {notes_text}".format(notes_text=notes_text),
                        opacity=opacity,
                    )
                )

    return data, duraion_sun_down_hrs


def parse_loaded_contents(contents, filename, date):
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
            log.debug("Done!")
            object_data = object_file_reader(local_file)
            log.debug(object_data.df_objects.head())
            log.debug(local_file)
        else:
            return html.Div(["Unsupported file!"])
    except Exception as e:
        log.warning(e)
        return html.Div(["There was an error processing this file."])
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


def get_progress_graph(df, date_string, profile, days_ago, targets=[]):

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

    p = go.Figure()
    if df0.shape[0] == 0:
        return dcc.Graph(figure=p), pd.DataFrame()

    df_summary = (
        df0.groupby(["OBJECT", "FILTER", BINNING_COL, FOCALLENGTH_COL, INSTRUMENT_COL])
        .agg({EXPOSURE_COL: "sum"})
        .dropna()
    )
    df_summary = df_summary.unstack(1).fillna(0)[EXPOSURE_COL] / 3600
    cols = ["OBJECT", "L", "R", "G", "B", "Ha", "OIII", "SII", "OSC"]
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
    [State("upload-data", "filename"), State("upload-data", "last_modified")],
)
def update_output_callback(list_of_contents, list_of_names, list_of_dates):
    global object_data
    profile = object_data.profiles[0]

    inactive_profiles = CONFIG.get("inactive_profiles", [])

    options = [
        {"label": profile, "value": profile}
        for profile in object_data.profiles
        if profile not in inactive_profiles
    ]
    default_option = options[0]["value"]

    if list_of_contents is not None:
        options = []
        for (c, n, d) in zip(list_of_contents, list_of_names, list_of_dates):
            object_data = parse_loaded_contents(c, n, d)
            for profile in object_data.profiles:
                if profile in inactive_profiles:
                    options.append({"label": profile, "value": profile})
        default_option = options[0]["value"]
        return options, default_option
    return options, default_option


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
    Output("store-placeholder", "data"), [Input("update-data", "n_clicks")],
)
def update_data_callback(n_click):
    log.info(n_click)
    if n_click != 0:
        update_data()
    return ""


@app.callback(
    [Output("target-status-match", "options"), Output("target-status-match", "value")],
    [Input("profile-selection", "value")],
    [State("status-match", "value")],
)
def update_target_for_status_callback(profile, status_match):
    df_combined_group = df_combined[df_combined["GROUP"] == profile]
    targets = df_combined_group.index.values
    return make_options(targets), ""


@app.callback(
    Output("target-status-selector", "value"),
    [Input("target-status-match", "value")],
    [State("store-target-status", "data"), State("profile-selection", "value")],
)
def update_radio_status_for_targets_callback(targets, target_status_store, profile):
    global df_target_status
    status_set = set()
    status = df_target_status.set_index(["OBJECT", "GROUP"])
    for target in targets:
        status_values = [status.loc[target].loc[profile]["status"]]
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
def update_target_with_status_callback(status, targets, target_status_store, profile):
    global df_combined
    global df_target_status
    df_combined, df_target_status = update_targets_with_status(
        targets, status, df_combined, profile
    )
    return ""


@app.callback(
    [
        Output("tab-target-div", "style"),
        Output("tab-data-table-div", "style"),
        Output("tab-files-table-div", "style"),
    ],
    [Input("tabs", "active_tab")],
)
def render_content(tab):
    print(tab)

    styles = [{"display": "none"}] * 3

    tab_names = [
        "tab-target",
        "tab-data-table",
        "tab-files-table",
    ]

    indx = tab_names.index(tab)

    styles[indx] = {}
    return styles


@app.callback(
    [
        Output("store-target-data", "data"),
        Output("store-target-list", "data"),
        Output("store-target-metadata", "data"),
        Output("dark-sky-duration", "data"),
    ],
    [
        Input("date-picker", "date"),
        Input("store-site-data", "data"),
        Input("store-target-status", "data"),
        Input("y-axis-type", "value"),
        Input("local-mpsas", "value"),
        Input("k-ext", "value"),
        Input("filter-targets", "checked"),
        Input("status-match", "value"),
        Input("filter-match", "value"),
    ],
    [State("profile-selection", "value")],
)
def store_data(
    date_string,
    site_data,
    target_status_store,
    value,
    local_mpsas,
    k_ext,
    filter_targets,
    status_matches,
    filters,
    profile,
):
    global df_combined, object_data
    log.debug(f"Calling store_data")
    targets = list(object_data.target_list[profile].values())
    site = update_site(
        site_data,
        default_lat=DEFAULT_LAT,
        default_lon=DEFAULT_LON,
        default_utc_offset=DEFAULT_UTC_OFFSET,
    )

    if filters:
        targets = target_filter(targets, filters)

    if status_matches:
        matching_targets = df_combined[df_combined["status"].isin(status_matches)].index
        targets = [target for target in targets if target.name in matching_targets]

    coords = get_coordinates(
        targets, date_string, site, time_resolution_in_sec=DEFAULT_TIME_RESOLUTION
    )
    sun_down_range = get_time_limits(coords)

    data, duration_sun_down_hrs = get_data(
        coords,
        targets,
        df_combined,
        value=value,
        local_mpsas=local_mpsas,
        k_ext=k_ext,
        filter_targets=filter_targets,
    )

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
    profile,
    status_list,
    date,
):
    global df_target_status
    global df_combined
    global df_stored_data

    update_data()

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

    if value == "alt":
        y_range = [0, 90]
    elif value == "airmass":
        y_range = [1, 5]
    elif value == "contrast":
        y_range = [0, 1]

    target_graph = dcc.Graph(
        config={"displaylogo": False, "modeBarButtonsToRemove": ["pan2d", "lasso2d"]},
        figure={
            "data": target_data,
            "layout": dict(
                xaxis={"title": "", "range": date_range},
                yaxis={"title": yaxis_map[value], "range": y_range},
                title=title,
                margin={"l": 50, "b": 100, "t": 50, "r": 50},
                legend={"orientation": "v"},
                height=400,
                plot_bgcolor="#ddd",
                paper_bgcolor="#fff",
                hovermode="closest",
                transition={"duration": 50},
            ),
        },
    )

    df_combined_group = df_combined[df_combined["GROUP"] == profile]

    df_combined_group = set_target_status(df_combined_group, df_target_status)
    targets = get_targets_with_status(df_combined_group, status_list=status_list)
    progress_days_ago = int(CONFIG.get("progress_days_ago", 0))

    progress_graph, df_summary = get_progress_graph(
        df_stored_data,
        date_string=date_string,
        profile=profile,
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
        # Output("scatter-graph", "children"),
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
        # "filename",
        "DATE-OBS",
        "FILTER",
        "EXPOSURE",
        "XPIXSZ",
        "FOCALLEN",
        "arcsec_per_pixel",
        "CCD-TEMP",
        "fwhm_mean_arcsec",
        "ecc_mean",
        # "fwhm_ecc",
        # "fwhm_chip_r",
        # "ecc_chip_r",
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
            )
        ]
    )

    df_numeric = df0.select_dtypes(
        include=["int16", "int32", "int64", "float16", "float32", "float64"]
    )

    numeric_cols = [col for col in df_numeric.columns if "corr__" not in col]
    scatter_field_options = make_options(numeric_cols)

    df_agg = df0.groupby(["OBJECT", "FILTER", "XBINNING", "FOCALLEN", "XPIXSZ"]).agg(
        {"EXPOSURE": "sum", "CCD-TEMP": "std", "DATE-OBS": "count"}
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

    columns = []
    for col in df_agg.columns:
        entry = {"name": col, "id": col, "deletable": False, "selectable": True}
        columns.append(entry)
    data = df_agg.round(2).to_dict("records")
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
    [Input("scatter-radio-selection", "value"),],
)
def update_scatter_axes(value):
    x_col = "fwhm_mean_arcsec"
    y_col = "ecc_mean"
    if value:
        x_col, y_col = value.split(" vs. ")
    return x_col, y_col


@app.callback(
    [Output("scatter-graph", "children")],
    [
        Input("store-target-data", "data"),
        Input("target-match", "value"),
        Input("x-axis-field", "value"),
        Input("y-axis-field", "value"),
        Input("scatter-size-field", "value"),
    ],
)
def update_scatter_plot(target_data, target_match, x_col, y_col, size_col):
    global df_target_status
    global df_combined
    global df_stars_headers

    df0 = df_stars_headers[(df_stars_headers["OBJECT"] == target_match)]

    filters = df0["FILTER"].unique()
    if not x_col:
        x_col = "fwhm_mean_arcsec"
    if not y_col:
        y_col = "ecc_mean"
    p = go.Figure()
    for filter in FILTER_LIST:
        if filter not in filters:
            continue
        df1 = df0[df0["FILTER"] == filter].reset_index()

        df1["text"] = df1.apply(
            lambda row: "Filename: "
            + row["filename"]
            + "<br>Date: "
            + str(row["DATE-OBS"])
            + f"<br>Star count: {row['n_stars']}"
            + f"<br>FWHM: {row['fwhm_mean']:.2f}"
            + f"<br>Eccentricity: {row['ecc_mean']:.2f}"
            + f"<br>{size_col}: {row[size_col]:.2f}",
            axis=1,
        )
        size = df1[size_col]

        sizeref = float(2.0 * size.max() / (4 ** 2))

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
                marker=dict(color=COLORS[filter], size=size, sizeref=sizeref),
            )
        )
    p.update_layout(xaxis_title=x_col, yaxis_title=y_col, height=600)

    scatter_graph = dcc.Graph(id="example-graph-2", figure=p)

    return [scatter_graph]


if __name__ == "__main__":
    localhost_only = strtobool(CONFIG.get("localhost_only", "True"))
    debug = strtobool(CONFIG.get("debug", "False"))
    host = "0.0.0.0"
    if localhost_only:
        host = "localhost"
    app.run_server(debug=True, host=host, dev_tools_serve_dev_bundles=True)
