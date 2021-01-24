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
        "contrast": "Relative Contrast",
        "airmass": "Airmass",
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
                    dcc.Dropdown(
                        id="scatter-size-field", options=[], value="fwhm_mean"
                    ),
                ],
                className="dash-bootstrap",
            ),
        ]
    )

    quick_options_col_picker = dbc.Col(
        [
            html.Div(
                [
                    html.Label("Quick Options", style={"textAlign": "center"}),
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
                                "label": "FWHM std. vs. FWHM mean",
                                "value": "fwhm_std_arcsec vs. fwhm_mean_arcsec",
                            },
                            {
                                "label": "Alt. vs. Background",
                                "value": "OBJCTALT vs. bkg_val",
                            },
                            {
                                "label": "Star Trailing vs. Spacing Metric",
                                "value": "star_trail_strength vs. star_orientation_score",
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
                                    html.Div(id="upload-button", children=[upload]),
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
                    children=[
                        html.Div(
                            id="log-div", style=dict(height="700px", overflow="auto")
                        )
                    ],
                    width=9,
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
                    dcc.Dropdown(id="target-match", options=[]),
                ],
                className="dash-bootstrap",
            )
        ]
    )

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
                                        style={"width": "95%"},
                                        children=[
                                            dbc.Row(
                                                [
                                                    dbc.Col(target_picker, width=3),
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
                                                        width=3,
                                                    ),
                                                ]
                                            )
                                        ],
                                    )
                                ],
                            ),
                        ],
                        width=10,
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
                                        value="fwhm",
                                        options=[
                                            {"label": "FWHM", "value": "fwhm"},
                                            {
                                                "label": "Ellipticity",
                                                "value": "ellipticity",
                                            },
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
            dbc.Tab(
                label="Config",
                tab_id="tab-config",
                labelClassName="text-danger",
                style={"justify": "right"},
            ),
        ],
    )

    alerts = html.Div(
        [
            dbc.Alert("", id="alert-auto", is_open=False, duration=1,),
            dcc.Interval(
                id="interval-component",
                interval=60 * 1000,  # in milliseconds
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
            data_table_container,
            target_container,
            config_container,
            data_files_table_container,
            html.Div(id="dummy-id", style={"display": "none"}),
        ],
    )

    layout = html.Div(
        [
            body,
            dcc.Store(id="store-target-data"),
            dcc.Store(id="store-target-list"),
            dcc.Store(id="store-site-data", data={}),
            dcc.Store(id="store-goal-data", data="{}"),
            dcc.Store(id="store-progress-data", data="{}"),
            dcc.Store(id="store-target-goals", data={}),
            dcc.Store(id="store-target-metadata"),
            dcc.Store(id="store-target-status"),
            dcc.Store(id="store-placeholder"),
            dcc.Store(id="dark-sky-duration"),
        ]
    )
    return layout
