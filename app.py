import base64
import io
import os

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

import pandas as pd
import numpy as np

# import dash_table
import plotly.graph_objects as go
import warnings

# import os
import datetime
import time
import yaml

# from datetime import datetime as dt
from dash.dependencies import Input, Output, State

# from plotly.subplots import make_subplots
# from collections import OrderedDict, defaultdict
# from astro_planner import *

from astro_planner.weather import DarkSky_Forecast, NWS_Forecast
from astro_planner.target import object_file_reader
from astro_planner.contrast import add_contrast
from astro_planner.site import ObservingSite
from astro_planner.ephemeris import get_coords
from astro_planner.data_parser import get_exposure_summary
from layout import layout, yaxis_map
import seaborn as sns

from astropy.utils.exceptions import AstropyWarning

import flask

import logging

# from flask_caching import Cache


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(module)s %(message)s")
log = logging.getLogger("app")
warnings.simplefilter("ignore", category=AstropyWarning)


server = flask.Flask(__name__)  #

BS = "https://stackpath.bootstrapcdn.com/bootswatch/4.4.1/cosmo/bootstrap.min.css"
BS = dbc.themes.FLATLY
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO], server=server)


app.title = "The AstroImaging Planner"


with open("./config.yml", "r") as f:
    config = yaml.load(f)

DSF_FORECAST = DarkSky_Forecast(key="")
DATA_DIR = os.getenv("DATA_DIR", "/Volumes/Users/gshau/Dropbox/AstroBox/data/")
ROBOCLIP_FILE = os.getenv("ROBOCLIP_FILE", "./data/VoyRC_default.mdb")
DEFAULT_LAT = os.getenv("DEFAULT_LAT", 43.37)
DEFAULT_LON = os.getenv("DEFAULT_LON", -88.37)
DEFAULT_UTC_OFFSET = os.getenv("DEFAULT_UTC_OFFSET", -5)
DEFAULT_MPSAS = os.getenv("DEFAULT_MPSAS", 19.5)
DEFAULT_BANDWIDTH = os.getenv("DEFAULT_BANDWIDTH", 120)
DEFAULT_K_EXTINCTION = os.getenv("DEFAULT_K_EXTINCTION", 0.2)
DEFAULT_TIME_RESOLUTION = os.getenv("DEFAULT_TIME_RESOLUTION", 300)
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

colors = {
    "L": "black",
    "R": "red",
    "G": "green",
    "B": "blue",
    "Ha": "crimson",
    "SII": "maroon",
    "OIII": "teal",
    "OSC": "gray",
}


translated_filter = {
    "ha": ["ho", "sho", "hoo", "hos", "halpha", "h-alpha"],
    "oiii": ["ho", "sho", "hoo", "hos"],
    "nb": ["ha", "oiii", "sii", "sho", "ho", "hoo", "hos", "halpha", "h-alpha"],
    "rgb": ["osc", "bayer", "dslr", "slr", "r ", " g ", " b "],
    "lum": ["luminance", "lrgb"],
}


date_string = datetime.datetime.now().strftime("%Y-%m-%d")
log.info(date_string)


object_data = object_file_reader(ROBOCLIP_FILE)
deploy = False

show_todos = not deploy
debug_status = not deploy

date_range = []


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
    value="alt",
    sun_alt_for_twilight=-18,
    local_mpsas=20,
    filter_bandwidth=300,
    k_ext=0.2,
    filter_targets=True,
):
    log.info("Starting get_data")
    t0 = time.time()
    target_names = [
        name for name in (list(target_coords.keys())) if name not in ["sun", "moon"]
    ]
    if local_mpsas is None:
        local_mpsas = DEFAULT_MPSAS
    if filter_bandwidth is None:
        filter_bandwidth = DEFAULT_BANDWIDTH
    if k_ext is None:
        k_ext = DEFAULT_K_EXTINCTION

    # this is where we sort by transit time
    # log.info(sorted(target_coords.values, key=lambda x: x["alt"].argmax()))
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

    # need better way to line up notes with target - this is messy, and prone to mismatch
    for i_target, (color, target_name) in enumerate(zip(colors, target_names)):
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
        # df.index[])
        notes_text = targets[i_target].info["notes"]
        data.append(
            dict(
                x=df.index,
                y=df[value],
                mode="lines",
                line=dict(color=color, width=3),
                name=target_name,
                text="Notes: {notes_text}".format(notes_text=notes_text),
                opacity=1,
            )
        )
    log.info(f"Done get_data {time.time() - t0}")

    return data


app.layout = layout


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if ".mdb" in filename:
            file_root = filename.replace(".mdb", "")
            local_file = f"./data/uploads/{file_root}.mdb"
            with open(local_file, "wb") as f:
                f.write(decoded)
            object_data = object_file_reader(local_file)
            log.info(object_data.df_objects.head())
            log.info(local_file)
        elif ".sgf" in filename:
            out_data = io.StringIO(decoded.decode("utf-8"))
            file_root = filename.replace(".sgf", "")
            local_file = f"./data/uploads/{file_root}.sgf"
            with open(local_file, "w") as f:
                f.write(out_data.read())
            log.info("Done!")
            object_data = object_file_reader(local_file)
            log.info(object_data.df_objects.head())
            log.info(local_file)
        else:
            return html.Div(["Unsupported file!"])
    except Exception as e:
        log.info(e)
        return html.Div(["There was an error processing this file."])
    return object_data


@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    [Output("profile-selection", "options"), Output("profile-selection", "value")],
    [Input("upload-data", "contents")],
    [State("upload-data", "filename"), State("upload-data", "last_modified")],
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    global object_data
    profile = object_data.profiles[0]
    options = [{"label": profile, "value": profile} for profile in object_data.profiles]
    default_option = options[0]["value"]

    if list_of_contents is not None:
        options = []
        for (c, n, d) in zip(list_of_contents, list_of_names, list_of_dates):
            object_data = parse_contents(c, n, d)
            object_data.df_objects.to_json(orient="table")
            for profile in object_data.profiles:
                options.append({"label": profile, "value": profile})
        default_option = options[0]["value"]
        return options, default_option
    return options, default_option


def update_site(site_data):
    lat = site_data.get("lat", DEFAULT_LAT)
    lon = site_data.get("lon", DEFAULT_LON)
    utc_offset = site_data.get("utc_offset", DEFAULT_UTC_OFFSET)
    log.info("site_data")
    site = ObservingSite(lat, lon, 0, utc_offset=utc_offset)
    return site


def update_weather(site):
    log.info(f"{site.lat} {site.lon}")
    log.info("Trying NWS")
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
                    xaxis={"range": date_range},
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

    navbar_children = [
        dbc.NavItem(
            dbc.NavLink(
                "Clear Outside Report",
                id="clear-outside",
                href=f"http://clearoutside.com/forecast/{site.lat}/{site.lon}?view=current",
                target="_blank",
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "Weather",
                id="nws-weather",
                href=f"http://forecast.weather.gov/MapClick.php?lon={site.lon}&lat={site.lat}#.U1xl5F7N7wI",
                target="_blank",
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "Satellite",
                href="https://www.star.nesdis.noaa.gov/GOES/sector_band.php?sat=G16&sector=umv&band=11&length=12",
                target="_blank",
            )
        ),
    ]
    return graph_data, navbar_children


@app.callback(
    [Output("weather-graph", "children"), Output("navbar", "children")],
    [Input("store-site-data", "data")],
)
def update_weather_data(site_data):
    site = update_site((site_data))
    weather_graph, navbar = update_weather(site)
    return weather_graph, navbar


@app.callback(
    Output("store-site-data", "data"),
    [
        Input("input-lat", "value"),
        Input("input-lon", "value"),
        Input("input-utc-offset", "value"),
    ],
)
def update_time_location_data(lat=None, lon=None, utc_offset=None):

    site_data = dict(lat=DEFAULT_LAT, lon=DEFAULT_LON, utc_offset=DEFAULT_UTC_OFFSET)
    if lat:
        site_data["lat"] = lat
    if lon:
        site_data["lon"] = lon
    if utc_offset:
        site_data["utc_offset"] = utc_offset
    return site_data


def target_filter(targets, filters):
    log.info(filters)
    targets_with_filter = []
    for filter in filters:
        for target in targets:
            if target.info["notes"]:
                if filter in target.info["notes"].lower():
                    targets_with_filter.append(target)
        if filter.lower() in translated_filter:
            for t_filter in translated_filter[filter.lower()]:
                targets_with_filter += [
                    target
                    for target in targets
                    if t_filter.lower() in target.info["notes"].lower()
                ]
    return list(set(targets_with_filter))


def format_name(name):
    name = name.lower()
    name = name.replace(" ", "_")
    if "sh2" not in name:
        name = name.replace("-", "_")
    catalogs = ["m", "ngc", "abell", "ic", "vdb", "ldn"]

    for catalog in catalogs:
        if catalog in name[: len(catalog)]:
            if f"{catalog}_" in name:
                name = name.replace(f"{catalog}_", catalog)
    return name


@app.callback(
    Output("store-progress-data", "data"), [Input("profile-selection", "value")],
)
def get_progress(profile):
    df_exposure_summary = get_exposure_summary(
        data_dir=DATA_DIR, filter_list=FILTER_LIST
    )
    targets_saved = [format_name(target) for target in df_exposure_summary.index]
    matches = [
        target
        for target in df_exposure_summary.index
        if format_name(target) in targets_saved
    ]
    return df_exposure_summary.loc[matches].to_json()


@app.callback(
    [
        Output("store-target-data", "data"),
        Output("store-target-list", "data"),
        Output("store-target-metadata", "data"),
    ],
    [
        Input("date-picker", "date"),
        Input("profile-selection", "value"),
        Input("store-site-data", "data"),
        Input("y-axis-type", "value"),
        Input("local-mpsas", "value"),
        Input("k-ext", "value"),
        Input("filter-targets", "checked"),
        Input("filter-match", "value"),
    ],
)
def store_data(
    date_string,
    profile,
    site_data,
    value,
    local_mpsas,
    k_ext,
    filter_targets,
    filters=[],
):
    log.info(f"Calling store_data")
    targets = list(object_data.target_list[profile].values())
    site = update_site((site_data))

    if filters:
        targets = target_filter(targets, filters)

    coords = get_coords(
        targets, date_string, site, time_resolution_in_sec=DEFAULT_TIME_RESOLUTION
    )
    date_range = get_time_limits(coords)
    log.info(coords.keys())
    log.info(np.sum([df.shape[0] for df in coords.values()]))

    data = get_data(
        coords,
        targets,
        value=value,
        local_mpsas=local_mpsas,
        k_ext=k_ext,
        filter_targets=filter_targets,
    )

    metadata = dict(date_range=date_range, value=value)
    filtered_targets = [d["name"] for d in data if d["name"]]
    return data, filtered_targets, metadata


@app.callback(
    Output("target-graph", "children"),
    [
        Input("store-target-data", "data"),
        Input("store-target-metadata", "data"),
        Input("store-progress-data", "data"),
        Input("store-target-goals", "data"),
    ],
)
def update_target_graph(
    data, metadata, progress_data, target_goals,
):
    log.info(f"Calling update_target_graph")

    try:
        value = metadata["value"]
        date_range = metadata["date_range"]
    except KeyError:
        return []

    date = str(date_string.split("T")[0])
    title = "Imaging Targets on {date_string}".format(date_string=date)

    if value == "alt":
        y_range = [0, 90]
    elif value == "airmass":
        y_range = [1, 5]
    elif value == "contrast":
        y_range = [0, 1]
    target_graph = dcc.Graph(
        config={"displaylogo": False, "modeBarButtonsToRemove": ["pan2d", "lasso2d"]},
        figure={
            "data": data,
            "layout": dict(
                xaxis={"title": "", "range": date_range},
                yaxis={"title": yaxis_map[value], "range": y_range},
                title=title,
                margin={"l": 50, "b": 100, "t": 50, "r": 50},
                legend={"orientation": "v"},
                height=600,
                plot_bgcolor="#ddd",
                paper_bgcolor="#fff",
                hovermode="closest",
                transition={"duration": 150},
            ),
        },
    )

    colors = {
        "L": "black",
        "R": "red",
        "G": "green",
        "B": "blue",
        "Ha": "crimson",
        "SII": "maroon",
        "OIII": "teal",
        "OSC": "gray",
    }

    df_progress = pd.read_json(progress_data)
    df_progress.index = [format_name(t) for t in df_progress.index]
    t = [
        format_name(d["name"])
        for d in data
        if format_name(d["name"]) in df_progress.index
    ]

    df_progress = df_progress.loc[t]

    p = go.Figure()
    for i, filter in enumerate(list(df_progress.columns)):
        p.add_trace(
            go.Bar(
                name=f"{filter}",
                x=df_progress.index,
                y=df_progress[filter] / 60,
                marker_color=colors[filter],
            )
        )

    p.update_layout(barmode="group", height=400, legend_orientation="v")
    progress_graph = dcc.Graph(figure=p)
    return [target_graph, progress_graph]


def show_progress(df, days_ago=90, targets=[], date_string="2020-09-22"):

    selection = df["date"] < "1970-01-01"
    if days_ago > 0:
        selection |= df["date"] > str(
            datetime.datetime.today() - datetime.timedelta(days=days_ago)
        )
    targets_in_last_n_days = list(df[selection]["OBJECT"].unique())
    targets += targets_in_last_n_days
    if targets:
        selection |= df["OBJECT"].isin(targets)
    df0 = df[selection]

    df_summary = (
        df0.groupby(["OBJECT", "FILTER", "XBINNING", "FOCALLEN", "INSTRUME"])
        .agg({"EXPOSURE": "sum"})
        .dropna()
    )
    df_summary = df_summary.unstack(1).fillna(0)["EXPOSURE"] / 3600
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

    p = go.Figure()
    colors = {
        "L": "black",
        "R": "red",
        "G": "green",
        "B": "blue",
        "Ha": "crimson",
        "SII": "maroon",
        "OIII": "teal",
        "OSC": "gray",
    }

    df0 = df_summary.reset_index()
    bin = df0["XBINNING"].astype(int).astype(str)
    fl = df0["FOCALLEN"].astype(int).astype(str)
    df0["text"] = df0["INSTRUME"] + " @ " + bin + "x" + bin + " FL = " + fl + "mm"
    df_summary = df0.set_index("OBJECT").loc[objects_sorted]
    for filter in [col for col in colors if col in df_summary.columns]:
        p.add_trace(
            go.Bar(
                name=f"{filter}",
                x=df_summary.index,
                y=df_summary[filter],
                hovertext=df_summary["text"],
                marker_color=colors[filter],
            )
        )
    p.update_layout(barmode="group")
    p.show()
    return df_summary


if __name__ == "__main__":
    if deploy:
        app.run_server(host="0.0.0.0")
    else:
        app.run_server(debug=True, host="0.0.0.0")
