import base64
import io

# import ntpath

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

import pandas as pd
import numpy as np
import dash_table
import plotly.graph_objects as go
import warnings
import uuid

# import os
import datetime
import time
import yaml

from datetime import datetime as dt
from dash.dependencies import Input, Output, State

# from plotly.subplots import make_subplots
# from collections import OrderedDict, defaultdict
from astro_planner import *
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
# app = dash.Dash(__name__, external_stylesheets=[BS], server=server)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO], server=server)

# cache = Cache(
#     app.server, config={"CACHE_TYPE": "filesystem", "CACHE_DIR": "cache-directory"}
# )


app.title = "The AstroImaging Planner"


with open("./config.yml", "r") as f:
    config = yaml.load(f)

DSF_FORECAST = DarkSky_Forecast(key="")


DEFAULT_LAT = 43.37
DEFAULT_LON = -88.37
DEFAULT_UTC_OFFSET = -5
DEFAULT_MPSAS = 19.5
DEFAULT_BANDWIDTH = 120
DEFAULT_K_EXTINCTION = 0.2
DEFAULT_TIME_RESOLUTION = 300


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


date_string = datetime.datetime.now().strftime("%Y-%m-%d")
log.info(date_string)

data_dir = "/Volumes/Users/gshau/Dropbox/AstroBox/data/"

object_data = object_file_reader("./data/VoyRC_default.mdb")
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
    colors = sns.color_palette("colorblind", n_colors=n_targets).as_hex()

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


settings_dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("", header=True),
        dbc.DropdownMenuItem("Profiles", href="#"),
        dbc.DropdownMenuItem("UI Theme", href="#"),
        dbc.DropdownMenuItem("Config", href="#"),
        dbc.DropdownMenuItem("Logout", href="#"),
    ],
    nav=True,
    in_navbar=True,
    label="Settings",
)


navbar = dbc.NavbarSimple(
    id="navbar",
    children=[
        dbc.NavItem(
            dbc.NavLink(
                "Clear Outside Report",
                id="clear-outside",
                href=f"http://clearoutside.com/forecast/{DEFAULT_LAT}/{DEFAULT_LON}?view=current",
                target="_blank",
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "Weather",
                id="nws-weather",
                href=f"http://forecast.weather.gov/MapClick.php?lon={DEFAULT_LON}&lat={DEFAULT_LAT}#.U1xl5F7N7wI",
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
    ],
    brand="The AstroImaging Planner",
    brand_href="#",
    color="primary",
    dark=True,
)


date_picker = dbc.Row(
    [
        dbc.Col(html.Label("DATE: ")),
        dbc.Col(
            html.Div(
                [dcc.DatePickerSingle(id="date-picker", date=dt.now()),],
                style={"textAlign": "center"},
                className="dash-bootstrap",
            )
        ),
    ]
)

yaxis_map = {
    "alt": "Altitude",
    "airmass": "Airmass",
    "contrast": "Relative Contrast",
}


yaxis_picker = dbc.Col(
    html.Div(
        [
            html.Label("Quantity to plot:"),
            dcc.Dropdown(
                id="y-axis-type",
                options=[{"label": v, "value": k} for k, v in yaxis_map.items()],
                value="alt",
            ),
        ],
        style={"textAlign": "center"},
        className="dash-bootstrap",
    ),
    style={"border": "0px solid"},
)

profile_picker = dbc.Col(
    html.Div(
        [
            html.Label("Group (Equipment Profiles)", style={"textAlign": "center"},),
            dcc.Dropdown(
                id="profile-selection",
                options=[
                    {"label": profile, "value": profile}
                    for profile in object_data.profiles
                ],
                value=object_data.profiles[0],
            ),
        ],
        style={"textAlign": "center"},
        className="dash-bootstrap",
    ),
    style={"border": "0px solid"},
)


filter_targets_check = dbc.FormGroup(
    [
        dbc.Checkbox(id="filter-targets", className="form-check-input", checked=True),
        dbc.Label(
            "Seasonal Targets",
            html_for="standalone-checkbox",
            className="form-check-label",
        ),
    ]
)

filter_picker = dbc.Col(
    [
        html.Div(
            [
                html.Label("Matching Filters in Notes", style={"textAlign": "center"},),
                dcc.Dropdown(
                    id="filter-match",
                    options=[
                        {"label": "Luminance", "value": "lum",},
                        {"label": "RGB", "value": "rgb"},
                        {"label": "Narrowband", "value": "nb",},
                        {"label": "Ha", "value": "ha"},
                        {"label": "OIII", "value": "oiii"},
                        {"label": "SII", "value": "sii"},
                    ],
                    value=["ha"],
                    multi=True,
                ),
            ],
            className="dash-bootstrap",
        )
    ]
)
search_notes = dbc.Col(
    html.Div(
        [
            html.Label("Search Notes:  ", style={"textAlign": "center"},),
            dcc.Input(
                placeholder="NOT ACTIVE: Enter a value...",
                type="text",
                value="",
                debounce=True,
            ),
        ],
        className="dash-bootstrap",
    )
)
location_selection = dbc.Col(
    html.Div(
        children=[
            dbc.Col(
                children=[
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Label("LATITUDE:  ", style={"textAlign": "right"},)
                            ),
                            dbc.Col(
                                html.Div(
                                    dcc.Input(
                                        id="input-lat",
                                        debounce=True,
                                        placeholder=DEFAULT_LAT,
                                        type="number",
                                        className="dash-bootstrap",
                                    ),
                                    className="dash-bootstrap",
                                )
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Label(
                                    "LONGITUDE:  ", style={"textAlign": "left"},
                                ),
                            ),
                            dbc.Col(
                                dcc.Input(
                                    id="input-lon",
                                    debounce=True,
                                    placeholder=DEFAULT_LON,
                                    type="number",
                                )
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Label(
                                    "UTC OFFSET:  ", style={"textAlign": "left"},
                                ),
                            ),
                            dbc.Col(
                                dcc.Input(
                                    id="input-utc-offset",
                                    debounce=True,
                                    placeholder=DEFAULT_UTC_OFFSET,
                                    type="number",
                                ),
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Label(
                                    "Local SQM (mpsas):  ", style={"textAlign": "left"},
                                ),
                            ),
                            dbc.Col(
                                dcc.Input(
                                    id="local-mpsas",
                                    debounce=True,
                                    placeholder=DEFAULT_MPSAS,
                                    type="number",
                                ),
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Label(
                                    "Extinction Coeff:  ", style={"textAlign": "left"},
                                ),
                            ),
                            dbc.Col(
                                dcc.Input(
                                    id="k-ext",
                                    debounce=True,
                                    placeholder=DEFAULT_K_EXTINCTION,
                                    type="number",
                                ),
                            ),
                        ]
                    ),
                ],
                className="dash-bootstrap",
            )
        ]
    ),
)

weather_graph = html.Div(id="weather-graph", children=[dbc.Spinner(color="warning")])

weather_modal = html.Div(
    [
        dbc.Button(
            "Show Weather Forecast",
            id="open",
            color="primary",
            block=True,
            className="mr-1",
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Weather Forecast"),
                dbc.ModalBody(weather_graph),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="close",
                        color="danger",
                        block=True,
                        className="mr-1",
                    ),
                ),
            ],
            id="modal",
            size="xl",
        ),
    ]
)

upload = dcc.Upload(
    id="upload-data",
    children=html.Div(
        [
            dbc.Button(
                "Drag and drop Voyager RoboClip (.mdb) or Sequence Generator Pro (.sgf) file or click here ",
                color="dark",
                className="mr-1",
            )
        ]
    ),
    multiple=True,
)

profile_container = dbc.Container(
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Container(
                        fluid=True,
                        style={"width": "95%"},
                        children=[dbc.Row(profile_picker, justify="around"), html.Br()],
                    )
                ],
                width=3,
                style={"border": "0px solid"},
            )
        ],
    ),
    fluid=True,
    style={},
)


rows = []
for filter in FILTER_LIST:

    row = dbc.InputGroup(
        [
            dbc.InputGroupAddon(
                filter, addon_type="prepend", style={"background-color": colors[filter]}
            ),
            dbc.Input(placeholder="Exposure", id=f"{filter}-exposure", value=0),
            dbc.Input(placeholder="Subs", id=f"{filter}-sub", value=0),
        ],
        className="mb-3",
    )

    rows.append(row)
target_goal_children = [
    dcc.Dropdown(id="target-dropdown", multi=True),
    dbc.Container(children=rows, id="image-goal", fluid=True, style={},),
]


target_container = dbc.Container(
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Container(
                        fluid=True,
                        style={"width": "95%"},
                        children=[
                            dbc.Row(yaxis_picker, justify="around"),
                            html.Br(),
                            dbc.Row(filter_picker, justify="around"),
                            html.Br(),
                            dbc.Row(date_picker, justify="around"),
                            html.Br(),
                            dbc.Row(filter_targets_check, justify="around"),
                            html.Br(),
                            dbc.Row(location_selection, justify="around"),
                            html.Br(),
                            dbc.Row(weather_modal, justify="around"),
                            html.Br(),
                            dbc.Container(target_goal_children, id="target-goals"),
                            html.Br(),
                        ],
                    )
                ],
                width=3,
                style={"border": "0px solid"},
            ),
            dbc.Col(
                children=[
                    dbc.Row(
                        html.Div(id="upload-button", children=[upload]),
                        justify="center",
                    ),
                    html.Div(
                        id="target-graph", children=[dbc.Spinner(color="primary")],
                    ),
                    html.Br(),
                    html.Div(id="progress-chart"),
                ],
                width=9,
            ),
        ]
    ),
    id="tab-target-div",
    fluid=True,
    style={},
)

tabs = dbc.Tabs(
    id="tabs",
    active_tab="tab-target",
    children=[
        dbc.Tab(
            label="Target Review",
            tab_id="tab-target",
            labelClassName="text-primary",
        ),
        dbc.Tab(
            label="Sequence Constructor",
            tab_id="tab-sequence",
            labelClassName="text-warning",
        ),
        dbc.Tab(
            label="Sequence Writer",
            tab_id="tab-sequence-writer",
            labelClassName="text-danger",
        ),
    ],
)

target_picker = dbc.Col(
    [
        html.Div(
            [
                html.Label("Target", style={"textAlign": "center"},),
                dcc.Dropdown(
                    id="targets-available", options=[], value=None, multi=False,
                ),
            ],
            className="dash-bootstrap",
        )
    ]
)

goal_table = html.Div(
    [dash_table.DataTable(id="goal-table", columns=[], data=[], editable=True)]
)




body = dbc.Container(
    fluid=True,
    style={"width": "80%"},
    children=[
        navbar,
        # banner_jumbotron,
        tabs,
        profile_container,
        target_container,
        dbc.Container(children=["seq-div"], id="tab-seq-div", fluid=True, style={},),
        dbc.Container(
            children=["writer-div"], id="tab-writer-div", fluid=True, style={},
        ),
        html.Div(id="date-range", style={"display": "none"}),
    ],
)


app.layout = html.Div(
    [
        body,
        dcc.Store(id="store-target-data"),
        dcc.Store(id="store-target-list"),
        dcc.Store(id="store-site-data", data={}),
        dcc.Store(id="store-goal-data", data="{}"),
        dcc.Store(id="store-progress-data", data="{}"),
        dcc.Store(id="store-target-goals", data={}),
        dcc.Store(id="store-target-metadata"),
    ]
)



@app.callback(
    [
        Output("tab-target-div", "style"),
        Output("tab-seq-div", "style"),
        Output("tab-writer-div", "style"),
    ],
    [Input("tabs", "active_tab")],
)
def render_content(tab):

    styles = [{"display": "none"}] * 3

    tab_names = [
        "tab-target",
        "tab-sequence",
        "tab-sequence-writer",
    ]

    indx = tab_names.index(tab)

    styles[indx] = {}
    return styles


@app.callback(
    Output("target-dropdown", "options"),
    [Input("profile-selection", "value"), Input("store-target-list", "data"), Input("target-graph", "children"),],
)
def target_dropdown_setter(profile, target_list, graph):

    target_names = [t.name for t in object_data.target_list[profile].values()]

    if target_list:
        target_names = [name for name in target_names if name in target_list]

    default_target = ""
    if target_names:
        default_target = target_names[0]

    return make_options(target_names)#, default_target


@app.callback(
    [
        Output("L-exposure", "value"),
        Output("R-exposure", "value"),
        Output("G-exposure", "value"),
        Output("B-exposure", "value"),
        Output("Ha-exposure", "value"),
        Output("OIII-exposure", "value"),
        Output("SII-exposure", "value"),
        Output("OSC-exposure", "value"),
        Output("L-sub", "value"),
        Output("R-sub", "value"),
        Output("G-sub", "value"),
        Output("B-sub", "value"),
        Output("Ha-sub", "value"),
        Output("OIII-sub", "value"),
        Output("SII-sub", "value"),
        Output("OSC-sub", "value"),
    ],
    [Input("target-dropdown", "value"),],
    [State("store-target-goals", "data")],
)
def target_exposure_setter(target, target_goals):

    if isinstance(target, list):
        if target:
            target = target[0]

    if target in target_goals:
        goals = target_goals[target]
        subs = []
        exposures = []
        for filter in FILTER_LIST:
            exposure = 0
            n_sub = 0
            if filter in goals:
                exposure = goals[filter]["sub_exposure"]
                n_sub = goals[filter]["n_subs"]
            exposures.append(exposure)
            subs.append(n_sub)

        return exposures + subs
    return [0] * len(FILTER_LIST) * 2


@app.callback(
    Output("store-target-goals", "data"),
    [
        Input("L-exposure", "value"),
        Input("R-exposure", "value"),
        Input("G-exposure", "value"),
        Input("B-exposure", "value"),
        Input("Ha-exposure", "value"),
        Input("OIII-exposure", "value"),
        Input("SII-exposure", "value"),
        Input("OSC-exposure", "value"),
        Input("L-sub", "value"),
        Input("R-sub", "value"),
        Input("G-sub", "value"),
        Input("B-sub", "value"),
        Input("Ha-sub", "value"),
        Input("OIII-sub", "value"),
        Input("SII-sub", "value"),
        Input("OSC-sub", "value"),
    ],
    [State("target-dropdown", "value"), State("store-target-goals", "data")],
)
def update_target_goals(
    l_exp,
    r_exp,
    g_exp,
    b_exp,
    ha_exp,
    oiii_exp,
    sii_exp,
    osc_exp,
    l_sub,
    r_sub,
    g_sub,
    b_sub,
    ha_sub,
    oiii_sub,
    sii_sub,
    osc_sub,
    targets,
    target_goals,
):

    exposures = dict(
        zip(
            FILTER_LIST,
            [l_exp, r_exp, g_exp, b_exp, ha_exp, oiii_exp, sii_exp, osc_exp],
        )
    )
    subs = dict(
        zip(
            FILTER_LIST,
            [l_sub, r_sub, g_sub, b_sub, ha_sub, oiii_sub, sii_sub, osc_sub],
        )
    )
    for filter in FILTER_LIST:
        d = {
            filter: {
                "sub_exposure": int(exposures[filter]),
                "n_subs": int(subs[filter]),
            }
        }
        if targets:
            for target in targets:
                if target in target_goals:
                    target_goals[target].update(d)
                else:
                    target_goals[target] = d
    return target_goals





def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if ".mdb" in filename:
            file_id = uuid.uuid1()
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
            file_id = uuid.uuid1()
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
    [
        Output("profile-selection", "options"),
        Output("profile-selection", "value"),
    ],
    [Input("upload-data", "contents")],
    [State("upload-data", "filename"), State("upload-data", "last_modified")],
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    global object_data
    children = [None]

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


translated_filter = {
    "ha": ["ho", "sho", "hoo", "hos", "halpha", "h-alpha"],
    "oiii": ["ho", "sho", "hoo", "hos"],
    "nb": ["ha", "oiii", "sii", "sho", "ho", "hoo", "hos", "halpha", "h-alpha"],
    "rgb": ["osc", "bayer", "dslr", "slr", "r ", " g ", " b "],
    "lum": ["luminance", "lrgb"],
}


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
    [Output("weather-graph", "children"), Output("navbar", "children"),],
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
    Output("store-progress-data", "data"), [Input("profile-selection", "value"),],
)
def get_progress(profile):
    df_exposure_summary = get_exposure_summary(
        data_dir=data_dir, filter_list=FILTER_LIST
    )
    df_files = get_data_info(data_dir=data_dir, skip_header=True)

    optic, sensor = profile.split()
    df0 = df_files
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
    except:
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
        config={"displaylogo": False, "modeBarButtonsToRemove": ["pan2d", "lasso2d"],},
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
        "complete": {
            "L": "black",
            "R": "red",
            "G": "green",
            "B": "blue",
            "Ha": "crimson",
            "SII": "maroon",
            "OIII": "teal",
            "OSC": "gray",
        },
        "pending": {
            "L": "rgba(0, 0, 0, 0.5)",
            "R": "rgba(255, 0, 0, 0.5)",
            "G": "rgba(0, 255, 0, 0.5)",
            "B": "rgba(0, 0, 255, 0.5)",
            "Ha": "rgba(214, 25, 55, 0.5)",
            "SII": "rgba(116, 4, 7, 0.5)",
            "OIII": "rgba(10, 116, 116, 0.5)",
            "OSC": "rgba(128, 128, 128, 0.5)",
        },
    }

    df_progress = pd.read_json(progress_data)
    df_progress.index = [format_name(t) for t in df_progress.index]
    t = [
        format_name(d["name"])
        for d in data
        if format_name(d["name"]) in df_progress.index
    ]

    df_progress = df_progress.loc[t]
    df_progress["status"] = "complete"

    p = go.Figure()
    r = []
    for target, d in target_goals.items():
        if target == "null":
            continue
        x = pd.DataFrame(d).T
        y = (x["sub_exposure"] * x["n_subs"]) / 60.0
        y.name = target
        r.append(y)
    df_requested = pd.DataFrame(r)
    if df_requested.shape[0] == 0:
        df_requested = pd.DataFrame(columns=FILTER_LIST)
    df_requested["status"] = "requested"
    df_pending = np.clip(
        df_requested[FILTER_LIST].sub(df_progress[FILTER_LIST], fill_value=0), 0, None
    ).fillna(0)
    df_pending["status"] = "pending"
    df_progress = pd.concat([df_progress, df_requested, df_pending])
    for status in ["complete", "pending"]:
        for i, filter in enumerate(list(df_progress.columns)):
            if filter == "status":
                continue
            selection = df_progress["status"] == status
            p.add_trace(
                go.Bar(
                    name=f"{filter} {status}",
                    x=df_progress[selection].index,
                    y=df_progress[selection][filter] / 60,
                    marker_color=colors[status][filter],
                )
            )

    p.update_layout(barmode="group", height=400, legend_orientation="h")
    progress_graph = dcc.Graph(figure=p)
    return [target_graph, progress_graph]


if __name__ == "__main__":
    if deploy:
        app.run_server(host="0.0.0.0")
    else:
        app.run_server(debug=True, host="0.0.0.0")
