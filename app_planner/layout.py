import yaml
import os

from datetime import datetime as dt
from distutils.util import strtobool

import glob

import dash_daq as daq
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_leaflet as dl
from astro_planner.logger import log
from pathlib import Path

base_dir = Path(__file__).parents[1]
with open(f"{base_dir}/conf/config.yml", "r") as f:
    CONFIG = yaml.safe_load(f)

DEFAULT_LAT = CONFIG.get("lat", 43.37)
DEFAULT_LON = CONFIG.get("lon", -88.37)
DEFAULT_MPSAS = CONFIG.get("mpsas", None)
DEFAULT_BANDWIDTH = CONFIG.get("bandwidth", 120)
DEFAULT_K_EXTINCTION = CONFIG.get("k_extinction", 0.2)
MIN_MOON_DISTANCE = CONFIG.get("min_moon_distance", 30)
MIN_FRAME_OVERLAP_FRACTION = CONFIG.get("min_frame_overlap_fraction", 0.95)

USE_CONTRAST = strtobool(os.getenv("USE_CONTRAST", "True")) == 1
styles = {}
if not USE_CONTRAST:
    styles["k-ext"] = {"display": "none"}
    styles["local-mpsas"] = {"display": "none"}

yaxis_map = {
    "alt": "Altitude",
    "airmass": "Airmass",
}
if USE_CONTRAST:
    yaxis_map = {
        "alt": "Altitude",
        "airmass": "Airmass",
        "sky_mpsas": "Sky Brightness (Experimental)",
        "contrast": "Relative Contrast (Experimental)",
    }

switch_color = "#427EDC"


def serve_layout(app):
    def foo():
        navbar = dbc.NavbarSimple(
            id="navbar",
            children=[
                dbc.NavItem(
                    dbc.NavLink(
                        "Project Repository",
                        id="github-link",
                        href="https://github.com/gshau/AstroPlanner/",
                        className="fa-github",
                        target="_blank",
                    )
                ),
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
                        id="goes-satellite",
                        href="https://www.star.nesdis.noaa.gov/GOES/sector_band.php?sat=G16&sector=umv&band=11&length=12",
                        target="_blank",
                    )
                ),
                dbc.NavItem(
                    dbc.NavLink(
                        "Smoke Forecast",
                        id="smoke-forecast",
                        href="https://rapidrefresh.noaa.gov/hrrr/HRRRsmoke/",
                        target="_blank",
                    )
                ),
            ],
            brand="The AstroImaging Planner",
            brand_href="https://github.com/gshau/AstroPlanner/",
            color="primary",
            dark=True,
        )

        target_status_picker = dbc.Col(
            [
                html.Div(
                    [
                        html.Label(
                            "Change Target Status", style={"textAlign": "center"},
                        ),
                        dcc.Dropdown(
                            id="target-status-match", options=[], value=[], multi=True,
                        ),
                    ],
                    style={"textAlign": "center"},
                    className="dash-bootstrap",
                )
            ]
        )

        target_status_selector = dbc.FormGroup(
            [
                dbc.Col(
                    dbc.RadioItems(
                        options=[
                            {"label": "Pending", "value": "pending"},
                            {"label": "Active", "value": "active"},
                            {"label": "Acquired", "value": "acquired"},
                            {"label": "Closed", "value": "closed"},
                        ],
                        labelStyle={"display": "block"},
                        inputStyle={"margin-right": "10px"},
                        id="target-status-selector",
                    ),
                    width=10,
                ),
            ],
            row=True,
        )

        target_priority_selector = dbc.FormGroup(
            [
                dbc.Col(
                    dbc.RadioItems(
                        options=[
                            {"label": "Very High", "value": 5},
                            {"label": "High", "value": 4},
                            {"label": "Medium", "value": 3},
                            {"label": "Low", "value": 2},
                            {"label": "Very Low", "value": 1},
                        ],
                        labelStyle={"display": "block"},
                        inputStyle={"margin-right": "10px"},
                        id="target-priority-selector",
                    ),
                    width=10,
                ),
            ],
            row=True,
        )

        date_picker = dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.Label("DATE:   "),
                            html.Br(),
                            dcc.DatePickerSingle(id="date-picker", date=dt.now()),
                        ],
                        style={"textAlign": "center"},
                        className="dash-bootstrap",
                    ),
                ),
            ]
        )

        yaxis_picker = dbc.Col(
            html.Div(
                [
                    html.Label("Quantity to plot:"),
                    dcc.Dropdown(
                        id="y-axis-type",
                        options=[
                            {"label": v, "value": k} for k, v in yaxis_map.items()
                        ],
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
                    html.Label(
                        "Group or Sequence file", style={"textAlign": "center"},
                    ),
                    dcc.Dropdown(id="profile-selection", multi=True),
                ],
                style={"textAlign": "center"},
                className="dash-bootstrap",
            ),
            style={"border": "0px solid"},
        )

        filter_targets_check = dbc.FormGroup(
            [
                daq.BooleanSwitch(
                    id="filter-seasonal-targets",
                    on=True,
                    label="Display Only Seasonal Targets",
                    labelPosition="bottom",
                    color=switch_color,
                ),
            ]
        )

        filter_picker = dbc.Col(
            [
                html.Div(
                    [
                        html.Label("Matching Filters in Notes",),
                        dcc.Dropdown(
                            id="filter-match",
                            options=[
                                {"label": "Narrowband", "value": "nb"},
                                {"label": "Broadband", "value": "bb"},
                                {"label": "Luminance", "value": "lum"},
                                {"label": "RGB", "value": "rgb"},
                                {"label": "Ha", "value": "ha"},
                                {"label": "OIII", "value": "oiii"},
                                {"label": "SII", "value": "sii"},
                            ],
                            value=[],
                            multi=True,
                        ),
                    ],
                    style={"textAlign": "center"},
                    className="dash-bootstrap",
                )
            ]
        )

        status_picker = dbc.Col(
            [
                html.Div(
                    [
                        html.Label(
                            "Selected Target Status", style={"textAlign": "center"},
                        ),
                        dcc.Dropdown(
                            id="status-match",
                            options=[
                                {"label": "Pending", "value": "pending"},
                                {"label": "Active", "value": "active"},
                                {"label": "Acquired", "value": "acquired"},
                                {"label": "Closed", "value": "closed"},
                            ],
                            value=[],
                            multi=True,
                        ),
                    ],
                    style={"textAlign": "center"},
                    className="dash-bootstrap",
                )
            ]
        )

        priority_picker = dbc.Col(
            [
                html.Div(
                    [
                        html.Label(
                            "Selected Target Priority", style={"textAlign": "center"},
                        ),
                        dcc.Dropdown(
                            id="priority-match",
                            options=[
                                {"label": "Very High", "value": 5},
                                {"label": "High", "value": 4},
                                {"label": "Medium", "value": 3},
                                {"label": "Low", "value": 2},
                                {"label": "Very Low", "value": 1},
                            ],
                            value=[],
                            multi=True,
                        ),
                    ],
                    style={"textAlign": "center"},
                    className="dash-bootstrap",
                )
            ]
        )

        header_col_picker = dbc.Col(
            [
                html.Div(
                    [
                        html.Label(
                            "Show FITs Header Cols", style={"textAlign": "center"}
                        ),
                        dcc.Dropdown(
                            id="header-col-match", options=[], value=[], multi=True
                        ),
                    ],
                    style={"textAlign": "center"},
                    className="dash-bootstrap",
                )
            ]
        )

        scatter_col_picker = dbc.Col(
            [
                html.Div(
                    [
                        html.Label("X-axis", style={"textAlign": "center"}),
                        dcc.Dropdown(id="x-axis-field", options=[], value=[]),
                    ],
                    style={"textAlign": "center"},
                    className="dash-bootstrap",
                ),
                html.Div(
                    [
                        html.Label("Y-axis", style={"textAlign": "center"}),
                        dcc.Dropdown(id="y-axis-field", options=[], value=[]),
                    ],
                    style={"textAlign": "center"},
                    className="dash-bootstrap",
                ),
                html.Div(
                    [
                        html.Label("Marker Size", style={"textAlign": "center"}),
                        dcc.Dropdown(id="scatter-size-field", options=[], value=""),
                    ],
                    style={"textAlign": "center"},
                    className="dash-bootstrap",
                ),
            ]
        )

        location_selection = html.Div(
            [
                dbc.InputGroup(
                    [
                        dbc.InputGroupAddon(
                            "Min Frame Overlap Fraction".ljust(20), addon_type="append",
                        ),
                        dbc.Input(
                            id="min-frame-overlap-fraction",
                            value=MIN_FRAME_OVERLAP_FRACTION,
                            placeholder=MIN_FRAME_OVERLAP_FRACTION,
                            type="number",
                            debounce=True,
                        ),
                    ]
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupAddon(
                            "Min Moon Distance".ljust(20), addon_type="append"
                        ),
                        dbc.Input(
                            id="min-moon-distance",
                            value=MIN_MOON_DISTANCE,
                            placeholder=MIN_MOON_DISTANCE,
                            type="number",
                            debounce=True,
                        ),
                    ]
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupAddon(
                            "Extinction Coefficient".ljust(20), addon_type="append"
                        ),
                        dbc.Input(
                            id="k-ext",
                            value=DEFAULT_K_EXTINCTION,
                            placeholder=DEFAULT_K_EXTINCTION,
                            type="number",
                            debounce=True,
                        ),
                    ]
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupAddon(
                            "Sky Brightness (mpsas)".ljust(20), addon_type="append",
                        ),
                        dbc.Input(
                            id="local-mpsas",
                            value=DEFAULT_MPSAS,
                            placeholder=DEFAULT_MPSAS,
                            type="number",
                            debounce=True,
                        ),
                    ]
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupAddon("Latitude".ljust(20), addon_type="append"),
                        dbc.Input(
                            id="input-lat",
                            value=DEFAULT_LAT,
                            placeholder=DEFAULT_LAT,
                            type="number",
                            debounce=True,
                            disabled=True,
                        ),
                    ],
                    style={"display": "none"},
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupAddon("Longitude".ljust(20), addon_type="append"),
                        dbc.Input(
                            id="input-lon",
                            value=DEFAULT_LON,
                            placeholder=DEFAULT_LON,
                            type="number",
                            debounce=True,
                            disabled=True,
                        ),
                    ],
                    style={"display": "none"},
                ),
            ]
        )

        weather_graph = html.Div(
            id="weather-graph", children=[dbc.Spinner(color="warning")]
        )

        clear_outside_forecast_img = html.Img(id="clear-outside-img")

        location_picker_modal = html.Div(
            [
                dbc.Button(
                    "Change Location",
                    id="open-location",
                    color="info",
                    block=True,
                    className="mr-1",
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Select Location"),
                        dbc.ModalBody(
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            dl.Map(
                                                style={
                                                    "width": "700px",
                                                    "height": "500px",
                                                },
                                                center=[DEFAULT_LAT, DEFAULT_LON],
                                                zoom=6,
                                                children=[
                                                    dl.TileLayer(
                                                        url="http://www.google.com/maps/vt?x={x}&y={y}&z={z}",
                                                        attribution='<a href="https://maps.google.com/">Google Maps</a>',
                                                    ),
                                                    dl.LayerGroup(
                                                        id="location-marker",
                                                    ),
                                                ],
                                                id="map-id",
                                            ),
                                            html.Div(id="location-text"),
                                        ],
                                        style={"textAlign": "center"},
                                    )
                                ]
                            )
                        ),
                        dbc.ModalFooter(
                            dbc.Button(
                                "Close",
                                id="close-location",
                                color="danger",
                                block=True,
                                className="mr-1",
                            ),
                        ),
                    ],
                    id="modal-location",
                    size="lg",
                ),
            ]
        )

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
                        dbc.ModalBody(
                            html.Div(
                                [clear_outside_forecast_img, weather_graph],
                                style={"textAlign": "center"},
                            )
                        ),
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
                    size="lg",
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

        quick_options_col_picker = dbc.Col(
            [
                html.Div(
                    [
                        html.Label("Quick Options:", style={"textAlign": "center"}),
                        dcc.RadioItems(
                            id="scatter-radio-selection",
                            options=[
                                {
                                    "label": "FWHM vs. Date",
                                    "value": "DATE-OBS vs. fwhm_median",
                                },
                                {
                                    "label": "Eccentricity vs. FWHM",
                                    "value": "fwhm_median vs. eccentricity_median",
                                },
                                {
                                    "label": "Altitude vs. Azimuth",
                                    "value": "OBJCTAZ vs. OBJCTALT",
                                },
                                {
                                    "label": "Star Count vs. Sky Background (ADU)",
                                    "value": "bkg_val vs. n_stars",
                                },
                                {
                                    "label": "Focus Position vs. Temperature",
                                    "value": "FOCUSTEM vs. FOCUSPOS",
                                },
                                {
                                    "label": "Sky Background (ADU) vs. Altitude",
                                    "value": "OBJCTALT vs. bkg_val",
                                },
                                {
                                    "label": "Spacing Metric vs. Star Trailing Strength",
                                    "value": "star_trail_strength vs. star_orientation_score",
                                },
                                {
                                    "label": "Eccentricity vs. Star Trailing Strength",
                                    "value": "star_trail_strength vs. eccentricity_mean",
                                },
                            ],
                            labelStyle={"display": "block"},
                            inputStyle={"margin-right": "10px"},
                        ),
                    ],
                    className="dash-bootstrap",
                ),
            ]
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
                                    dbc.Row(date_picker, justify="around"),
                                    html.Br(),
                                    dbc.Row(location_picker_modal, justify="around"),
                                    html.Br(),
                                    dbc.Row(profile_picker, justify="around"),
                                    html.Br(),
                                    dbc.Row(yaxis_picker, justify="around"),
                                    html.Br(),
                                    dbc.Row(filter_picker, justify="around"),
                                    html.Br(),
                                    dbc.Row(status_picker, justify="around"),
                                    html.Br(),
                                    dbc.Row(priority_picker, justify="around"),
                                    html.Br(),
                                    dbc.Row(filter_targets_check, justify="around"),
                                    html.Br(),
                                    dbc.Row(target_status_picker, justify="around"),
                                    html.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label("Status"),
                                                    target_status_selector,
                                                ],
                                                width=6,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Priority"),
                                                    target_priority_selector,
                                                ],
                                                width=6,
                                            ),
                                        ],
                                        justify="around",
                                    ),
                                    html.Br(),
                                    dbc.Row(location_selection, justify="around"),
                                    html.Br(),
                                    dbc.Row(weather_modal, justify="around"),
                                    html.Br(),
                                    dbc.Row(
                                        html.Div(
                                            id="upload-button",
                                            children=[upload],
                                            style={"display": "none"},
                                        ),
                                        justify="around",
                                    ),
                                ],
                            ),
                        ],
                        width=3,
                        style={"border": "0px solid"},
                    ),
                    dbc.Col(
                        children=[
                            html.Div(
                                id="target-graph",
                                children=[dbc.Spinner(color="primary")],
                            ),
                            html.Br(),
                            html.Div(id="progress-graph"),
                        ],
                        width=9,
                    ),
                ]
            ),
            id="tab-target-div",
            fluid=True,
            style={},
        )

        config_container = dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Button(
                                "Clear All Tables",
                                id="button-clear-tables",
                                color="warning",
                                block=True,
                                className="mr-1",
                            ),
                            dbc.Button(
                                "Clear Targets Table",
                                id="button-clear-target-tables",
                                color="warning",
                                block=True,
                                className="mr-1",
                            ),
                            dbc.Button(
                                "Clear Header Tables",
                                id="button-clear-header-tables",
                                color="warning",
                                block=True,
                                className="mr-1",
                            ),
                            dbc.Button(
                                "Clear Star Tables",
                                id="button-clear-star-tables",
                                color="warning",
                                block=True,
                                className="mr-1",
                            ),
                        ],
                        width=2,
                        style={"border": "0px solid"},
                    ),
                    dbc.Col(
                        [
                            dbc.Button(
                                html.A(
                                    "Download Planner Log",
                                    href="getLogs/planner.log",
                                    style={"color": "white"},
                                ),
                                color="info",
                                block=True,
                                className="mr-1",
                            ),
                            dbc.Button(
                                html.A(
                                    "Download Watchdog Log",
                                    href="getLogs/watchdog.log",
                                    style={"color": "white"},
                                ),
                                color="info",
                                block=True,
                                className="mr-1",
                            ),
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Button(
                                "Restart App",
                                id="button-restart-app",
                                color="danger",
                                block=True,
                                className="mr-1",
                            ),
                            dbc.Button(
                                "Restart Watchdog",
                                id="button-restart-watchdog",
                                color="danger",
                                block=True,
                                className="mr-1",
                            ),
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Button(
                                "Show File Skiplist",
                                id="button-show-file-skiplist",
                                color="primary",
                                block=True,
                                className="mr-1",
                            ),
                            dbc.Button(
                                "Clear File Skiplist",
                                id="button-clear-file-skiplist",
                                color="primary",
                                block=True,
                                className="mr-1",
                            ),
                        ],
                        width=2,
                    ),
                ]
            ),
            id="tab-config-div",
            fluid=True,
            style={},
        )
        md_files = glob.glob("docs/*.md")
        markdown_from_files = {}
        for md_file in md_files:
            root_name = os.path.basename(md_file).replace(".md", "")
            with open(md_file, "r") as file:
                markdown_from_files[root_name] = "".join(file.readlines())
        help_container = dbc.Container(
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Target Planning"),
                                        dbc.CardImg(
                                            src=app.get_asset_url("planner_tab.png"),
                                            top=True,
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Markdown(
                                                    markdown_from_files.get(
                                                        "01_target_planning"
                                                    )
                                                )
                                            ],
                                        ),
                                        dbc.CardImg(
                                            src=app.get_asset_url(
                                                "weather_forecast_modal.png"
                                            ),
                                            className="align-self-center w-50",
                                        ),
                                    ],
                                    style={"width": "18rem"},
                                    color="primary",
                                    outline=True,
                                    className="w-100",
                                ),
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Utilities"),
                                        dbc.CardImg(
                                            src=app.get_asset_url("utilities_tab.png"),
                                            top=True,
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Markdown(
                                                    markdown_from_files.get(
                                                        "05_utilities"
                                                    )
                                                )
                                            ]
                                        ),
                                    ],
                                    style={"width": "18rem"},
                                    color="danger",
                                    outline=True,
                                    className="w-100",
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Frame Inspector"),
                                        dbc.CardImg(
                                            src=app.get_asset_url("inspector_tab.png"),
                                            top=True,
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Markdown(
                                                    markdown_from_files.get(
                                                        "02_frame_inspector"
                                                    )
                                                )
                                            ]
                                        ),
                                    ],
                                    style={"width": "18rem"},
                                    color="info",
                                    outline=True,
                                    className="w-100",
                                ),
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Subframe Analysis"),
                                        dbc.CardImg(
                                            src=app.get_asset_url("frame_analysis.png"),
                                            top=True,
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Markdown(
                                                    markdown_from_files.get(
                                                        "03_frame_analysis"
                                                    )
                                                )
                                            ]
                                        ),
                                    ],
                                    style={"width": "18rem"},
                                    color="info",
                                    outline=True,
                                    className="w-100",
                                ),
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Targets Table"),
                                        dbc.CardImg(
                                            src=app.get_asset_url("table.png"),
                                            top=True,
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Markdown(
                                                    markdown_from_files.get(
                                                        "04_targets_table"
                                                    )
                                                )
                                            ]
                                        ),
                                    ],
                                    style={"width": "18rem"},
                                    color="success",
                                    outline=True,
                                    className="w-100",
                                ),
                            ],
                            width=6,
                        ),
                    ]
                )
            ],
            id="tab-help-div",
            fluid=True,
            style={},
        )

        data_table_container = dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        children=[html.Div(id="data-table")],
                        width=12,
                        style={"border": "20px solid white"},
                    ),
                ]
            ),
            id="tab-data-table-div",
            fluid=True,
            style={},
        )

        target_picker = dbc.Col(
            [
                html.Div(
                    [
                        dcc.Dropdown(
                            id="inspector-dates",
                            placeholder="Select Date",
                            options=[],
                            multi=True,
                        ),
                    ],
                    style={"textAlign": "center"},
                    className="dash-bootstrap",
                ),
                html.Div(
                    [
                        dcc.Dropdown(
                            id="target-matches",
                            placeholder="Select Target",
                            options=[],
                            multi=True,
                        ),
                    ],
                    style={"textAlign": "center"},
                    className="dash-bootstrap",
                ),
                html.Div(
                    [
                        dcc.Dropdown(
                            id="fl-matches",
                            placeholder="Select Focal Length",
                            options=[],
                            multi=True,
                        ),
                    ],
                    style={"textAlign": "center"},
                    className="dash-bootstrap",
                ),
                html.Div(
                    [
                        dcc.Dropdown(
                            id="px-size-matches",
                            placeholder="Select Pixel Size",
                            options=[],
                            multi=True,
                        ),
                    ],
                    style={"textAlign": "center"},
                    className="dash-bootstrap",
                ),
            ]
        )

        frame_acceptance_criteria = dbc.Col(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Label(
                                "Frame Acceptance Criteria:  ",
                                id="frame-acceptance-label",
                                style={"textAlign": "left"},
                            ),
                            width=12,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.InputGroup(
                            [
                                dbc.InputGroupAddon(
                                    "Max Eccentricity".ljust(20),
                                    addon_type="append",
                                    id="ecc-label",
                                ),
                                dbc.Input(
                                    id="ecc-thr-field",
                                    value=0.6,
                                    placeholder=0.6,
                                    type="number",
                                    debounce=True,
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.InputGroup(
                            [
                                dbc.InputGroupAddon(
                                    "Min Star Fraction:  ",
                                    addon_type="append",
                                    id="min-star-label",
                                ),
                                dbc.Input(
                                    id="star-frac-thr-field",
                                    debounce=True,
                                    placeholder=0.5,
                                    value=0.5,
                                    type="number",
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.InputGroup(
                            [
                                dbc.InputGroupAddon(
                                    "z score:  ",
                                    id="z-score-label",
                                    addon_type="append",
                                ),
                                dbc.Input(
                                    id="z-score-field",
                                    value=5,
                                    placeholder=5,
                                    type="number",
                                    debounce=True,
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.InputGroup(
                            [
                                dbc.InputGroupAddon(
                                    "IQR Scale".ljust(20),
                                    addon_type="append",
                                    id="iqr-scale-label",
                                ),
                                dbc.Input(
                                    id="iqr-scale-field",
                                    value=1.5,
                                    placeholder=1.5,
                                    type="number",
                                    debounce=True,
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.InputGroup(
                            [
                                dbc.InputGroupAddon(
                                    "Trail Threshold".ljust(20),
                                    addon_type="append",
                                    id="trail-label",
                                ),
                                dbc.Input(
                                    id="trail-thr-field",
                                    value=8,
                                    placeholder=8,
                                    type="number",
                                    debounce=True,
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )

        tooltips = html.Div(
            [
                dbc.Tooltip(
                    """Change to most recent date, and live-update data.""",
                    target="monitor-mode",
                    placement="left",
                ),
                dbc.Tooltip(
                    """Criteria for acceptance of subframes within a group.
                    Each group is defined by unique target, sensor, optic, filter and binning.""",
                    target="frame-acceptance-label",
                    placement="left",
                ),
                dbc.Tooltip(
                    """Threshold value of the z score `(x - mean(x)) / std(x)` for fwhm and star count in comparison to the group.
                For fwhm, only frames with a z score below the positive threshold value are used.
                For star count, only frames with a z score above the negative threshold value are used.""",
                    target="z-score-label",
                    placement="left",
                ),
                dbc.Tooltip("IQR scale", target="iqr-scale-label", placement="left"),
                dbc.Tooltip(
                    "Maximum average eccentricity allowed",
                    target="ecc-label",
                    placement="left",
                ),
                dbc.Tooltip(
                    "Maximum star traling metric allowed",
                    target="trail-label",
                    placement="left",
                ),
                dbc.Tooltip(
                    "Minimum star count allowed, as a fraction compared to the maximum star count of a frame in the group",
                    target="min-star-label",
                    placement="left",
                ),
            ]
        )

        single_target_progress = html.Div(id="single-target-progress-graph")

        filter_targets_check = dbc.FormGroup(
            [
                daq.BooleanSwitch(
                    id="aberration-preview",
                    on=True,
                    label="As Abberation Inspector View",
                    labelPosition="bottom",
                    color=switch_color,
                ),
            ]
        )

        monitor_mode_check = dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            daq.BooleanSwitch(
                                id="monitor-mode",
                                on=False,
                                label="Monitor Mode",
                                labelPosition="bottom",
                                color=switch_color,
                            ),
                        ],
                        className="dash-bootstrap",
                    ),
                    width=6,
                ),
                dbc.Col(
                    html.Div(
                        [
                            daq.BooleanSwitch(
                                id="show-text-in-scatter",
                                on=False,
                                label="Label Points",
                                labelPosition="bottom",
                                color=switch_color,
                            ),
                        ],
                        className="dash-bootstrap",
                    ),
                    width=6,
                ),
            ]
        )

        data_files_table_container = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Container(
                                    fluid=True,
                                    style={"width": "100%"},
                                    children=[
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                target_picker,
                                                                html.Br(),
                                                                monitor_mode_check,
                                                                # html.Br(),
                                                                # auto_reject_file_move,
                                                            ]
                                                        )
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [frame_acceptance_criteria],
                                                    width=6,
                                                ),
                                            ]
                                        )
                                    ],
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Container(
                                    fluid=True,
                                    style={"width": "100%"},
                                    children=[
                                        dbc.Row(
                                            [
                                                dbc.Col(scatter_col_picker, width=6,),
                                                dbc.Col(
                                                    quick_options_col_picker, width=6,
                                                ),
                                            ]
                                        )
                                    ],
                                )
                            ],
                            width=6,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col([single_target_progress], width=6,),
                        dbc.Col(
                            html.Div(
                                id="scatter-graph",
                                children=[
                                    dcc.Graph(
                                        id="target-scatter-graph",
                                        style={"width": "100%", "height": "600px"},
                                        config={
                                            "displaylogo": False,
                                            "modeBarButtonsToRemove": ["lasso2d"],
                                        },
                                    )
                                ],
                            ),
                            width=6,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    [filter_targets_check], className="dash-bootstrap"
                                ),
                                dcc.Graph(
                                    id="inspector-frame",
                                    style={"width": "100%", "height": "800px"},
                                    config={
                                        "displaylogo": False,
                                        "modeBarButtonsToRemove": ["lasso2d"],
                                    },
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "Select Heatmap Data",
                                            style={"textAlign": "center"},
                                        ),
                                        dcc.Dropdown(
                                            id="frame-heatmap-dropdown",
                                            value="ellipticity",
                                            options=[
                                                {
                                                    "label": "Ellipticity",
                                                    "value": "ellipticity",
                                                },
                                                {
                                                    "label": "Eccentricity",
                                                    "value": "eccentricity",
                                                },
                                                {"label": "FWHM", "value": "fwhm"},
                                            ],
                                        ),
                                    ],
                                    className="dash-bootstrap",
                                ),
                                dcc.Graph(
                                    id="xy-frame-graph",
                                    style={"width": "100%", "height": "800px"},
                                    config={
                                        "displaylogo": False,
                                        "modeBarButtonsToRemove": ["lasso2d"],
                                    },
                                ),
                            ],
                            width=6,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(),
                        dbc.Col(
                            dcc.Graph(
                                id="radial-frame-graph",
                                style={"width": "100%", "height": "600px"},
                                config={
                                    "displaylogo": False,
                                    "modeBarButtonsToRemove": ["lasso2d"],
                                },
                            ),
                            width=10,
                        ),
                        dbc.Col(),
                    ]
                ),
                dcc.Markdown(
                    """
                ## Summary Table"""
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            children=[html.Div(id="summary-table")],
                            width=12,
                            style={"border": "20px solid white"},
                        ),
                    ]
                ),
                dcc.Markdown(
                    """
                ## Subexposure data - star measurements and FITs header"""
                ),
                dbc.Row([dbc.Col(header_col_picker, width=3,)]),
                dbc.Row(
                    [
                        dbc.Col(
                            children=[html.Div(id="files-table")],
                            width=12,
                            style={"border": "20px solid white"},
                        ),
                    ]
                ),
            ],
            id="tab-files-table-div",
            fluid=True,
            style={},
        )

        loading = dcc.Loading(
            id="loading-2",
            children=[
                html.Div(
                    [html.Div(id="loading-output"), html.Div(id="loading-output-click")]
                )
            ],
            type="default",
        )

        tabs = dbc.Tabs(
            id="tabs",
            active_tab="tab-target",
            children=[
                dbc.Tab(
                    id="tab-target-review",
                    label="Target Planning",
                    tab_id="tab-target",
                    labelClassName="text-primary",
                    label_style={"font-size": 16},
                ),
                dbc.Tab(
                    label="Frame Inspector",
                    tab_id="tab-files-table",
                    labelClassName="text-info",
                    label_style={"font-size": 16},
                ),
                dbc.Tab(
                    id="tab-targets-table",
                    label="Targets Table",
                    tab_id="tab-data-table",
                    labelClassName="text-success",
                    label_style={"font-size": 16},
                ),
                dbc.Tab(
                    label="Utilities",
                    tab_id="tab-config",
                    labelClassName="text-danger",
                    label_style={"font-size": 16},
                ),
                dbc.Tab(
                    label="Help",
                    tab_id="tab-help",
                    labelClassName="text-dark",
                    label_style={"font-size": 16},
                ),
            ],
        )

        alerts = html.Div(
            [
                dbc.Alert(
                    "", id="alert-auto", is_open=False, duration=1, dismissable=True,
                ),
                dbc.Alert(
                    "",
                    id="alert-file-skiplist",
                    is_open=False,
                    duration=1,
                    dismissable=True,
                ),
                dcc.Interval(
                    id="monitor-mode-interval",
                    interval=5 * 1000,
                    n_intervals=0,
                    disabled=True,
                ),
            ]
        )

        body = dbc.Container(
            fluid=True,
            style={"width": "90%"},
            children=[
                navbar,
                dbc.Row(
                    [
                        dbc.Col(tabs, width=6),
                        dbc.Col([]),
                        dbc.Col([], width=3, id="location-tab-text"),
                        dbc.Col([], width=1, id="bortle-tab-badge"),
                        dbc.Col(
                            daq.Indicator(
                                value=False,
                                label="Monitor Mode",
                                labelPosition="right",
                                color="#888888",
                                id="monitor-mode-indicator",
                            ),
                            width=1,
                        ),
                    ],
                    justify="center",
                    align="center",
                    className="h-50",
                ),
                html.Br(),
                alerts,
                loading,
                html.Br(),
                data_table_container,
                target_container,
                config_container,
                help_container,
                data_files_table_container,
                html.Div(id="dummy-id", style={"display": "none"}),
                html.Div(id="dummy-id-target-data", style={"display": "none"}),
                html.Div(id="dummy-id-contrast-data", style={"display": "none"}),
                html.Div(id="dummy-rejection-criteria-id", style={"display": "none"}),
                html.Div(id="dummy-interval-update", style={"display": "none"}),
            ],
        )

        layout = html.Div(
            [
                body,
                tooltips,
                dcc.Store(id="store-site-data", data={}),
                dcc.Store(id="store-target-data"),
                dcc.Store(id="store-target-status"),
                dcc.Store(id="store-target-list"),
                dcc.Store(id="store-target-metadata"),
                dcc.Store(id="store-dark-sky-duration"),
            ]
        )
        return layout

    return foo
