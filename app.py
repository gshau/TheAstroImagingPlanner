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
from fuzzywuzzy import fuzz
# import os
import datetime
import time
import yaml

from datetime import datetime as dt
from dash.dependencies import Input, Output, State

# from plotly.subplots import make_subplots
# from collections import OrderedDict, defaultdict
from astro_planner import *

from astropy.utils.exceptions import AstropyWarning

import flask

import json
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(module)s %(message)s")
log = logging.getLogger("app")
warnings.simplefilter("ignore", category=AstropyWarning)


server = flask.Flask(__name__)  #

BS = "https://stackpath.bootstrapcdn.com/bootswatch/4.4.1/cosmo/bootstrap.min.css"
BS = dbc.themes.FLATLY
# app = dash.Dash(__name__, external_stylesheets=[BS], server=server)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO], server=server)

# from flask import request

app.title = "The AstroImaging Planner"

markdown_roadmap = """
## Future work:
- [x] Profile Selection - select which equipment profile you want to view
- [x] Airmass/Altitude selection
- [x] RGB vs. Narrowband selection
- [x] Spinner for loading graph
- [x] Bootstrap themes
- [x] Site selection (lat/lon)
- [x] Weather graph for current date - cloud cover, temp, humidity, wind
  - [x] Select only humidity, temperature, wind and cloudcover
- [x] Target progress table
- [ ] Profile details
- [x] Contrast calculations
- [ ] Aladin target view
- [ ] user-controlled
  - [ ] priority
  - [ ] filter goals
  - [ ] exposure totals and progress
- [ ] mpsas level at current location
  - [ ] read from mpsas map
  - [ ] snowpack inflation?
- [x] docker deploymeent
  - [ ] darksky api key
- [ ] add openweathermap datasource
- [x] Organization of divs - use bootstrap columns
- [ ] Notes search
- [x] Allow upload of file
  - [x] Roboclip
  - [x] SGP
  - [x] Allow upload of targets file
- [x] Fix double-click zoom reset

- show all targets vs. show active targets
- alpha reduced on inactive targets
- show color-coded blocks of when filters are used


"""


markdown_info = """
This tool attempts to...

"""

with open('./config.yml', 'r') as f:
    config = yaml.load(f)

DSF_FORECAST = DarkSky_Forecast(key="")


DEFAULT_LAT = 43.37
DEFAULT_LON = -88.37
DEFAULT_UTC_OFFSET = -5
DEFAULT_MPSAS = 19.5
DEFAULT_BANDWIDTH = 120
DEFAULT_K_EXTINCTION = 0.2
DEFAULT_TIME_RESOLUTION = 300


date_string = datetime.datetime.now().strftime("%Y-%m-%d")
log.info(date_string)

data_dir = "/Volumes/Users/gshau/Dropbox/AstroBox/data/"

object_data = object_file_reader("./data/VoyRC_default.mdb")
deploy = False

show_todos = not deploy
debug_status = not deploy

date_range = []


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

    ### need better way to line up notes with target - this is messy, and prone to mismatch
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

roadmap_modal = html.Div(
    [
        dbc.Button(
            "Roadmap",
            id="open_roadmap_modal",
            color="primary",
            block=True,
            className="mr-1",
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Roadmap"),
                dbc.ModalBody(dcc.Markdown(children=markdown_roadmap)),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="close_roadmap_modal",
                        color="danger",
                        block=True,
                        className="mr-1",
                    ),
                ),
            ],
            id="roadmap_modal",
            size="xl",
        ),
    ],
    style={"display": "none" if deploy else "block"},
)


navbar = dbc.NavbarSimple(
    id="navbar",
    children=[
        dbc.NavItem(
            dbc.NavLink(
                "Clear Outside Report",
                id="clear_outside",
                href=f"http://clearoutside.com/forecast/{DEFAULT_LAT}/{DEFAULT_LON}?view=current",
                target="_blank",
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "Weather",
                id="nws_weather",
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
        roadmap_modal,
        # weather_dropdown,
        # settings_dropdown,
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
                [dcc.DatePickerSingle(id="date_picker", date=dt.now()),],
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
                id="y_axis_type",
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
                id="profile_selection",
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
        dbc.Checkbox(id="filter_targets", className="form-check-input", checked=True),
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
                    id="filter_match",
                    options=[
                        {"label": "Luminance", "value": "lum",},
                        {"label": "RGB", "value": "rgb"},
                        {"label": "Narrowband", "value": "nb",},
                        {"label": "Ha", "value": "ha"},
                        {"label": "OIII", "value": "oiii"},
                        {"label": "SII", "value": "sii"},
                    ],
                    value=["lum", "rgb", "ha"],
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
                                        id="input_lat",
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
                                    id="input_lon",
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
                                    id="input_utc_offset",
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
                                    id="local_mpsas",
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
                                    id="k_ext",
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

weather_graph = html.Div(id="weather_graph", children=[dbc.Spinner(color="warning")])

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
    id="upload_data",
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
                        ],
                    )
                ],
                width=3,
                style={"border": "0px solid"},
            ),
            dbc.Col(
                children=[
                    dbc.Row(
                        html.Div(id="upload_button", children=[upload]),
                        justify="center",
                    ),
                    html.Div(
                        id="target_graph", children=[dbc.Spinner(color="primary")],
                    ),
                    html.Br(),
                    html.Div(id="progress_chart"),
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
            label="Target Goals",
            tab_id="tab-goals",
            # tabClassName="ml-auto",
            labelClassName="text-info",
        ),
        dbc.Tab(
            label="Target Review",
            tab_id="tab-target",
            # tabClassName="ml-auto",
            labelClassName="text-primary",
        ),
        dbc.Tab(
            label="Data Review",
            tab_id="tab-data-review",
            # tabClassName="ml-auto",
            labelClassName="text-success",
        ),
        dbc.Tab(
            label="Sequence Constructor",
            tab_id="tab-sequence",
            # tabClassName="ml-auto",
            labelClassName="text-warning",
        ),
        dbc.Tab(
            label="Sequence Writer",
            tab_id="tab-sequence-writer",
            # tabClassName="ml-auto",
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


progress_container = dbc.Container(
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Container(
                        fluid=True,
                        style={"width": "95%"},
                        children=[
                            # dbc.Row(yaxis_picker, justify="around"),
                            # html.Br(),
                            # dbc.Row(filter_picker, justify="around"),
                            # html.Br(),
                            # dbc.Row(date_picker, justify="around"),
                            # html.Br(),
                            # dbc.Row(filter_targets_check, justify="around"),
                            # html.Br(),
                            # dbc.Row(location_selection, justify="around"),
                            # html.Br(),
                            # dbc.Row(weather_modal, justify="around"),
                        ],
                    )
                ],
                width=3,
                style={"border": "0px solid"},
            ),
            dbc.Col(
                # children=html.Div(id="progress_chart"),
                width=9,
                style={"height": "100vh"},
            ),
        ]
    ),
    id="tab-data-div",
    fluid=True,
    style={"height": "100vh"},
)



goal_container = dbc.Container(
    children=[
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Container(
                            fluid=True,
                            style={"width": "95%"},
                            children=[
                                dbc.Row(target_picker, justify="around"),
                                html.Br(),
                                goal_table,
                                # dbc.Row(filter_selector, justify="around"),
                                # html.Br(),
                                # dbc.Row(subexposure_selector, justify="around"),
                                # html.Br(),
                                # dbc.Row(subnumber_selector, justify="around"),
                                # html.Br(),
                            ],
                        )
                    ],
                    width=3,
                    style={"border": "0px solid"},
                ),
                dbc.Col(
                    children=[
                        # html.Div(
                        #     id="target_graph",
                        #     children=[dbc.Spinner(color="primary")],
                        # ),
                        # html.Br(),
                        # dbc.Row(
                        #     html.Div(id="upload_button", children=[upload]),
                        #     justify="center",
                        # ),
                    ],
                    width=9,
                ),
            ]
        ),
    ],
    id="tab-goal-div",
    fluid=True,
    style={},
)

body = dbc.Container(
    fluid=True,
    style={"width": "80%"},
    children=[
        navbar,
        # banner_jumbotron,
        tabs,
        profile_container,
        goal_container,
        target_container,
        progress_container,
        dbc.Container(children=["seq-div"], id="tab-seq-div", fluid=True, style={},),
        dbc.Container(
            children=["writer-div"], id="tab-writer-div", fluid=True, style={},
        ),
        html.Div(id="date_range", style={"display": "none"}),
    ],
)


app.layout = html.Div(
    [
        body,
        dcc.Store(id="target_data"),
        dcc.Store(id="site_data", data={}),
        dcc.Store(id="progress_data", data='{}'),
    ]
)



def match_profile_to_data(object_data, profile, df_files, weight_thr=80):
    matchup = {}
    df0 = object_data.df_objects
    df1 = df0[df0['GROUP'] == profile]
    for db_target_name in df1['TARGET'].values:
        matchup[db_target_name] = dict([[file_target_name, fuzz.ratio(db_target_name, file_target_name)] for file_target_name in df_files['target'].unique()])
    df = pd.DataFrame(matchup).T
    df = df.apply(lambda row: [row.max(), row.idxmax()], result_type='expand')
    df.index = ['weight', 'name']
    df = df.T
    return df[df['weight'] > weight_thr]

def make_options(elements):
    return [{"label": element, "value": element} for element in elements]


@app.callback(
    [Output("targets-available", "options"), Output("targets-available", "value")],
    [Input("profile_selection", "value")],
)
def get_available_targets(profile):
    target_names = list(object_data.target_list[profile].keys())
    print(target_names)
    return make_options(target_names), target_names[0]


@app.callback(
    [Output("goal-table", "columns"), Output("goal-table", "data")],
    [Input("profile_selection", "value"), Input("targets-available", "value")],
)
def get_target_goals(profile, target):

    columns = ["Target"]
    for filter in ["L", "R", "G", "B", "HA", "OIII", "SII"]:
        columns += [f"{filter} exp (min)", f"{filter} subs"]

    target_names = list(object_data.target_list[profile].keys())
    # data = {'Target': target_names}
    data = []
    n_targets = len(target_names)
    for target in target_names:
        entry = {}
        for column in columns:
            if column == "Target":
                entry[column] = target
                continue
            entry[column] = 0
        data.append(entry)
    columns_entry = [{"name": col, "id": col} for col in columns]
    return columns_entry, data


@app.callback(
    [
        Output("tab-goal-div", "style"),
        Output("tab-target-div", "style"),
        Output("tab-data-div", "style"),
        Output("tab-seq-div", "style"),
        Output("tab-writer-div", "style"),
    ],
    [Input("tabs", "active_tab")],
)
def render_content(tab):

    styles = [{"display": "none"}] * 5

    tab_names = [
        "tab-goals",
        "tab-target",
        "tab-data-review",
        "tab-sequence",
        "tab-sequence-writer",
    ]

    indx = tab_names.index(tab)

    styles[indx] = {}
    return styles


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
    Output("roadmap_modal", "is_open"),
    [Input("open_roadmap_modal", "n_clicks"), Input("close_roadmap_modal", "n_clicks")],
    [State("roadmap_modal", "is_open")],
)
def toggle_roadmap_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    [
        Output("target_data", "data"),
        Output("profile_selection", "options"),
        Output("profile_selection", "value"),
    ],
    [Input("upload_data", "contents")],
    [State("upload_data", "filename"), State("upload_data", "last_modified")],
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    global object_data
    # children = [object_data.df_objects.to_json(orient="table")]
    children = [None]
    profile = object_data.profiles[0]
    options = [{"label": profile, "value": profile} for profile in object_data.profiles]
    default_option = options[0]["value"]

    if list_of_contents is not None:
        children = []
        options = []
        for (c, n, d) in zip(list_of_contents, list_of_names, list_of_dates):
            object_data = parse_contents(c, n, d)
            object_data.df_objects.to_json(orient="table")
            children.append(object_data.df_objects.to_json(orient="table"))
            for profile in object_data.profiles:
                options.append({"label": profile, "value": profile})
        default_option = options[0]["value"]
        return children[0], options, default_option
    return children[0], options, default_option


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
    try:
        log.info("Trying NWS")
        nws_forecast = NWS_Forecast(site.lat, site.lon)
        df_weather = nws_forecast.parse_data()
    except:
        log.info("Trying Dark Sky")
        DSF_FORECAST.get_forecast_data(site.lat, site.lon)
        df_weather = DSF_FORECAST.forecast_data_to_df()["hourly"]
        df_weather = df_weather[
            df_weather.columns[df_weather.dtypes != "object"]
        ].fillna(0)

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
                id="clear_outside",
                href=f"http://clearoutside.com/forecast/{site.lat}/{site.lon}?view=current",
                target="_blank",
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "Weather",
                id="nws_weather",
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
        roadmap_modal,
    ]
    return graph_data, navbar_children


@app.callback(
    [Output("weather_graph", "children"), Output("navbar", "children"),],
    [Input("site_data", "data")],
)
def update_weather_data(site_data):
    site = update_site((site_data))
    weather_graph, navbar = update_weather(site)
    return weather_graph, navbar


@app.callback(
    Output("site_data", "data"),
    [
        Input("input_lat", "value"),
        Input("input_lon", "value"),
        Input("input_utc_offset", "value"),
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
    name = name.replace(' ', '_')
    if 'sh2' not in name:
        name = name.replace('-', '_')
    catalogs = ['m', 'ngc', 'abell', 'ic', 'vdb', 'ldn']
    
    for catalog in catalogs:
        if catalog in name[:len(catalog)]:
            if f'{catalog}_' in name:
                continue
            number = name.replace(catalog, '')
            name = f'{catalog}_{number}'
    return name

@app.callback(
    Output("progress_data", "data"), [Input("profile_selection", "value"),],
)
def get_progress(profile):
    df_exposure_summary = get_exposure_summary(data_dir=data_dir)
    df_files = get_data_info(data_dir=data_dir, skip_header=False)

    optic, sensor = profile.split()

    selection = df_files['XPIXSZ'] == config['equipment']['sensor'][sensor]['pixel_size'] 
    selection &= df_files['FOCALLEN'] == config['equipment']['optic'][optic]['focal_length'] 

    df0 = df_files[selection]

    targets_saved = [format_name(target) for target in df_exposure_summary.index]

    targets_saved = [format_name(obj) for obj in df0['OBJECT'].unique()]
    matches = [target for target in df_exposure_summary.index if format_name(target) in targets_saved]
    print('matches', matches)
    return df_exposure_summary.loc[matches].to_json()



@app.callback(
    Output("target_graph", "children"),
    [
        Input("date_picker", "date"),
        Input("profile_selection", "value"),
        Input("site_data", "data"),
        Input("y_axis_type", "value"),
        Input("local_mpsas", "value"),
        Input("k_ext", "value"),
        Input("filter_targets", "checked"),
        Input("progress_data", "data"),
        Input("filter_match", "value"),
    ],
)
def update_target_graph(
    date_string,
    profile,
    site_data,
    value,
    local_mpsas,
    k_ext,
    filter_targets,
    progress_data,
    filters=[],
):
    log.info(f"Calling update_target_graph")
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

    date = str(date_string.split("T")[0])
    title = "Imaging Targets on {date_string}".format(date_string=date)

    if value == "alt":
        y_range = [0, 90]
    elif value == "airmass":
        y_range = [1, 5]
    elif value == "contrast":
        y_range = [0, 1]
    target_graph = dcc.Graph(
            config={
                "displaylogo": False,
                "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
            },
            figure={
                "data": data,
                "layout": dict(
                    xaxis={"title": "", "range": date_range},
                    yaxis={"title": yaxis_map[value], "range": y_range},
                    title=title,
                    margin={"l": 50, "b": 100, "t": 50, "r": 50},
                    legend={"x": 1, "y": 0.5},
                    height=600,
                    plot_bgcolor="#ddd",
                    paper_bgcolor="#fff",
                    hovermode="closest",
                    transition={"duration": 150},
                ),
            },
        )

    colors = {'complete': {'L': 'black', 'R': 'red', 'G': 'green', 'B': 'blue', 'Ha': 'crimson', 'SII': 'maroon', 'OIII': 'teal', 'OSC': 'gray'},
          'pending': {'L': 'rgba(0, 0, 0, 0.5)', 
                      'R': 'rgba(255, 0, 0, 0.5)', 
                      'G': 'rgba(0, 255, 0, 0.5)', 
                      'B': 'rgba(0, 0, 255, 0.5)', 
                      'Ha': 'rgba(214, 25, 55, 0.5)', 
                      'SII': 'rgba(116, 4, 7, 0.5)', 
                      'OIII': 'rgba(10, 116, 116, 0.5)', 
                      'OSC': 'rgba(128, 128, 128, 0.5)', 
                     }
         }

    df_progress = pd.read_json(progress_data)
    df_progress.index = [format_name(t) for t in df_progress.index]
    for d in data:
        print('data', d['name'])
    t = [format_name(d['name']) for d in data if format_name(d['name']) in df_progress.index]

    df_progress = df_progress.loc[t]


    p = go.Figure()
    dfp = df_progress.copy() / 3
    df_progress['status'] = 'complete'
    dfp['status'] = 'pending'
    df_progress = pd.concat([df_progress, dfp])
    for status in ['complete', 'pending']:
        for i, filter in enumerate(list(df_progress.columns)):
            if filter == 'status':
                continue
            selection = df_progress['status'] == status
            p.add_trace(go.Bar(name=f'{filter} {status}', 
                            x=df_progress[selection].index, 
                            y=df_progress[selection][filter] / 60,
                            marker_color=colors[status][filter]))

    p.update_layout(barmode="group", height=400)
    progress_graph = dcc.Graph(figure=p)
    return [target_graph, progress_graph]

   
if __name__ == "__main__":
    if deploy:
        app.run_server(host="0.0.0.0")
    else:
        app.run_server(debug=True, host="0.0.0.0")
