import yaml
import os

from datetime import datetime as dt
from distutils.util import strtobool

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

from pathlib import Path

base_dir = Path(__file__).parents[1]
with open(f"{base_dir}/conf/config.yml", "r") as f:
    CONFIG = yaml.safe_load(f)

DEFAULT_LAT = CONFIG.get("lat", 43.37)
DEFAULT_LON = CONFIG.get("lon", -88.37)
DEFAULT_UTC_OFFSET = CONFIG.get("utc_offset", -5)
DEFAULT_MPSAS = CONFIG.get("mpsas", 19.5)
DEFAULT_BANDWIDTH = CONFIG.get("bandwidth", 120)
DEFAULT_K_EXTINCTION = CONFIG.get("k_extinction", 0.2)
MIN_MOON_DISTANCE = CONFIG.get("min_moon_distance", 30)

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
        "contrast": "Relative Contrast",
        "sky_mpsas": "Local Sky Brightness",
    }


def serve_layout():

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
                    html.Label("Change Target Status", style={"textAlign": "center"},),
                    dcc.Dropdown(
                        id="target-status-match", options=[], value=[], multi=True,
                    ),
                ],
                className="dash-bootstrap",
            )
        ]
    )

    target_status_selector = dbc.Col(
        [
            html.Div(
                [
                    dcc.RadioItems(
                        options=[
                            {"label": "Active", "value": "active"},
                            {"label": "Acquired", "value": "acquired"},
                            {"label": "Pending", "value": "pending"},
                            {"label": "Closed", "value": "closed"},
                        ],
                        labelStyle={"display": "block"},
                        id="target-status-selector",
                    )
                ],
                className="dash-bootstrap",
            )
        ]
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
                html.Label(
                    "Group (Equipment Profiles)", style={"textAlign": "center"},
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
            dbc.Checkbox(
                id="filter-targets", className="form-check-input", checked=True
            ),
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
                    html.Label(
                        "Matching Filters in Notes", style={"textAlign": "center"},
                    ),
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
                className="dash-bootstrap",
            )
        ]
    )

    header_col_picker = dbc.Col(
        [
            html.Div(
                [
                    html.Label("Show FITs HEADER Cols", style={"textAlign": "center"}),
                    dcc.Dropdown(
                        id="header-col-match", options=[], value=[], multi=True
                    ),
                ],
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
                className="dash-bootstrap",
            ),
            html.Div(
                [
                    html.Label("Y-axis", style={"textAlign": "center"}),
                    dcc.Dropdown(id="y-axis-field", options=[], value=[]),
                ],
                className="dash-bootstrap",
            ),
            html.Div(
                [
                    html.Label("Marker Size", style={"textAlign": "center"}),
                    dcc.Dropdown(id="scatter-size-field", options=[], value=""),
                ],
                className="dash-bootstrap",
            ),
        ]
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
                                "label": "FWHM vs. Eccentricity",
                                "value": "fwhm_mean_arcsec vs. eccentricity_mean",
                            },
                            {"label": "Az. vs Alt", "value": "OBJCTAZ vs. OBJCTALT"},
                            {
                                "label": "Background vs. Star count",
                                "value": "bkg_val vs. n_stars",
                            },
                            {
                                "label": "Focus position vs. temperature",
                                "value": "FOCUSTEM vs. FOCUSPOS",
                            },
                            {
                                "label": "Alt. vs. Background",
                                "value": "OBJCTALT vs. bkg_val",
                            },
                            {
                                "label": "Star Trailing vs. Spacing Metric",
                                "value": "star_trail_strength vs. star_orientation_score",
                            },
                            {
                                "label": "Star Trailing vs. Eccentricity",
                                "value": "star_trail_strength vs. eccentricity_mean",
                            },
                        ],
                        labelStyle={"display": "block"},
                    ),
                ],
                className="dash-bootstrap",
            ),
        ]
    )

    location_selection = dbc.Col(
        html.Div(
            children=[
                dbc.Col(
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Label(
                                        "Min Moon Distance:  ",
                                        style={"textAlign": "left"},
                                    ),
                                ),
                                dbc.Col(
                                    dcc.Input(
                                        id="min-moon-distance",
                                        debounce=True,
                                        placeholder=MIN_MOON_DISTANCE,
                                        type="number",
                                    )
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Label(
                                        "LATITUDE:  ", style={"textAlign": "right"},
                                    )
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
                                        "Local SQM (mpsas):  ",
                                        style={"textAlign": "left"},
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
                            ],
                            style=styles.get("local-mpsas", {}),
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Label(
                                        "Extinction Coeff:  ",
                                        style={"textAlign": "left"},
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
                            ],
                            style=styles.get("k-ext", {}),
                        ),
                    ],
                    className="dash-bootstrap",
                )
            ]
        ),
    )

    weather_graph = html.Div(
        id="weather-graph", children=[dbc.Spinner(color="warning")]
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

    glossary_md = html.Div(id="glossary", children=[])

    glossary_modal = html.Div(
        [
            dbc.Button(
                "Quantity Glossary",
                id="glossary-open",
                color="primary",
                block=True,
                className="mr-1",
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader("Quantity Glossary"),
                    dbc.ModalBody(glossary_md),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Close",
                            id="glossary-close",
                            color="danger",
                            block=True,
                            className="mr-1",
                        ),
                    ),
                ],
                id="glossary-modal",
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
                                dbc.Row(profile_picker, justify="around"),
                                html.Br(),
                                dbc.Row(yaxis_picker, justify="around"),
                                html.Br(),
                                dbc.Row(filter_picker, justify="around"),
                                html.Br(),
                                dbc.Row(status_picker, justify="around"),
                                html.Br(),
                                dbc.Row(filter_targets_check, justify="around"),
                                html.Br(),
                                dbc.Row(target_status_picker, justify="around"),
                                html.Br(),
                                dbc.Row(target_status_selector, justify="around"),
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
                            id="target-graph", children=[dbc.Spinner(color="primary")],
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
                                "Download Planner log",
                                href="getLogs/planner.log",
                                style={"color": "white"},
                            ),
                            color="info",
                            block=True,
                            className="mr-1",
                        ),
                        dbc.Button(
                            html.A(
                                "Download Watchdog log",
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
                            "Show File blacklist",
                            id="button-show-file-blacklist",
                            color="primary",
                            block=True,
                            className="mr-1",
                        ),
                        dbc.Button(
                            "Clear File blacklist",
                            id="button-clear-file-blacklist",
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
                    html.Label("Select Target", style={"textAlign": "center"},),
                    dcc.Dropdown(id="target-matches", options=[], multi=True),
                ],
                className="dash-bootstrap",
            ),
            html.Br(),
            html.Div(
                [
                    html.Label("Select Date", style={"textAlign": "center"},),
                    dcc.Dropdown(id="inspector-dates", options=[], multi=True),
                ],
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
                            "Frame Acceptance Criteria:  ", style={"textAlign": "left"},
                        ),
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(html.Label("Z score:  ", style={"textAlign": "left"},),),
                    dbc.Col(
                        dcc.Input(
                            id="z-score-field",
                            debounce=True,
                            placeholder=5,
                            value=5,
                            type="number",
                        )
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(html.Label("IQR scale:  ", style={"textAlign": "left"},),),
                    dbc.Col(
                        dcc.Input(
                            id="iqr-scale-field",
                            debounce=True,
                            placeholder=1.5,
                            value=1.5,
                            type="number",
                        )
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Label(
                            "Eccentricity threshold:  ", style={"textAlign": "left"},
                        ),
                    ),
                    dbc.Col(
                        dcc.Input(
                            id="ecc-thr-field",
                            debounce=True,
                            placeholder=0.6,
                            value=0.6,
                            type="number",
                        )
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Label("Trail threshold:  ", style={"textAlign": "left"},),
                    ),
                    dbc.Col(
                        dcc.Input(
                            id="trail-thr-field",
                            debounce=True,
                            placeholder=5,
                            value=5,
                            type="number",
                        )
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Label(
                            "Min Star Fraction:  ", style={"textAlign": "left"},
                        ),
                    ),
                    dbc.Col(
                        dcc.Input(
                            id="star-frac-thr-field",
                            debounce=True,
                            placeholder=0.5,
                            value=0.5,
                            type="number",
                        )
                    ),
                ]
            ),
        ]
    )

    single_target_progress = html.Div(id="single-target-progress-graph")

    filter_targets_check = dbc.FormGroup(
        [
            dbc.Checkbox(
                id="aberration-preview", className="form-check-input", checked=True
            ),
            dbc.Label(
                "As Aberration Inspector View",
                html_for="standalone-checkbox",
                className="form-check-label",
            ),
        ]
    )

    data_files_table_container = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Container(
                                        fluid=True,
                                        style={"width": "100%"},
                                        children=[
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            target_picker,
                                                            html.Br(),
                                                            frame_acceptance_criteria,
                                                        ],
                                                        width=4,
                                                    ),
                                                    dbc.Col(
                                                        [single_target_progress],
                                                        width=8,
                                                    ),
                                                ]
                                            )
                                        ],
                                    )
                                ],
                            ),
                            dbc.Row(
                                [
                                    dbc.Container(
                                        fluid=True,
                                        style={"width": "100%"},
                                        children=[
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        scatter_col_picker, width=3,
                                                    ),
                                                    dbc.Col(
                                                        quick_options_col_picker,
                                                        width=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            filter_targets_check,
                                                            glossary_modal,
                                                        ],
                                                        width=2,
                                                    ),
                                                ]
                                            )
                                        ],
                                    )
                                ],
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            id="scatter-graph",
                            children=[
                                dcc.Graph(
                                    id="target-scatter-graph",
                                    style={"width": "100%", "height": "800px"},
                                )
                            ],
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="inspector-frame",
                                style={"width": "100%", "height": "800px"},
                            ),
                        ],
                        width=6,
                    ),
                ]
            ),
            dbc.Row(
                [
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
                                style={"width": "100%", "height": "600px"},
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id="radial-frame-graph",
                            style={"width": "100%", "height": "600px"},
                        ),
                        width=6,
                    ),
                ],
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
                label="Target Review",
                tab_id="tab-target",
                labelClassName="text-primary",
            ),
            dbc.Tab(
                label="Frame Inspector",
                tab_id="tab-files-table",
                labelClassName="text-info",
            ),
            dbc.Tab(
                label="Review Stored Targets",
                tab_id="tab-data-table",
                labelClassName="text-success",
            ),
            dbc.Tab(label="Utils", tab_id="tab-config", labelClassName="text-danger",),
        ],
    )

    alerts = html.Div(
        [
            dbc.Alert(
                "", id="alert-auto", is_open=False, duration=1, dismissable=True,
            ),
            dbc.Alert(
                "",
                id="alert-file-blacklist",
                is_open=False,
                duration=1,
                dismissable=True,
            ),
            dcc.Interval(
                id="interval-component",
                interval=15 * 1000,  # in milliseconds
                n_intervals=0,
            ),
        ]
    )

    body = dbc.Container(
        fluid=True,
        style={"width": "80%"},
        children=[
            navbar,
            tabs,
            html.Br(),
            alerts,
            loading,
            html.Br(),
            data_table_container,
            target_container,
            config_container,
            data_files_table_container,
            html.Div(id="dummy-id", style={"display": "none"}),
            html.Div(id="dummy-id-target-data", style={"display": "none"}),
            html.Div(id="dummy-id-contrast-data", style={"display": "none"}),
            html.Div(id="dummy-rejection-criteria-id", style={"display": "none"}),
        ],
    )

    layout = html.Div(
        [
            body,
            dcc.Store(id="store-site-data", data={}),
            dcc.Store(id="store-target-data"),
            dcc.Store(id="store-target-status"),
            dcc.Store(id="store-target-list"),
            dcc.Store(id="store-target-metadata"),
            dcc.Store(id="store-dark-sky-duration"),
        ]
    )
    return layout
