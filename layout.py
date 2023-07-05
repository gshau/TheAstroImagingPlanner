import yaml
import os
import glob
import multiprocessing

from datetime import datetime as dt

import pandas as pd

import dash_daq as daq
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import dash_leaflet as dl
import dash_mantine_components as dmc
from dash_iconify import DashIconify

from astro_planner.globals import BASE_DIR, BINNING_COL, FOCALLENGTH_COL, IS_PROD
from astro_planner.utils import get_config
from astro_planner.logger import log
from config import PlannerConfig, VoyagerConfig, InspectorThresholds

yaxis_map = {
    "alt": "Altitude",
    "airmass": "Airmass",
    "sky_mpsas": "Sky Brightness (Experimental)",
    "contrast": "Relative Contrast (Experimental)",
}


DEBUG_STYLE = {}
if IS_PROD:
    DEBUG_STYLE = {"display": "none"}


SIDEBAR_STYLE = {
    "position": "relative",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "90%",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}
with open(f"{BASE_DIR}/git.hash", "r") as f:
    git_hash = f.readline()[:6]
with open(f"{BASE_DIR}/metadata.yml", "r") as f:
    metadata = yaml.safe_load(f)
    version = metadata.get("Version")


def make_options(elements):
    return [{"label": element, "value": element} for element in elements]


def disable_options(options, valid_options):
    for option in options:
        if option["value"] not in valid_options:
            option["disabled"] = True
    return options


def serve_layout(app, monitor_mode_on=True, update_data_fn=None):
    def foo():
        env = app.env
        available_themes = [t for t in dbc.themes.__dict__ if "_" not in t]
        config = get_config(env=env)
        log.info(f"env = {app.env}")
        assert env == config.get("env")
        planner_config = PlannerConfig()
        planner_config.set_var(config.get("planner_config", {}))
        inspector_thresholds = InspectorThresholds()
        inspector_thresholds.set_var(config.get("inspector_thresholds", {}))

        voyager_config = VoyagerConfig()
        voyager_config.set_var(config.get("voyager_config", {}))

        yaxis_map = {
            "alt": "Altitude",
            "airmass": "Airmass",
            "sky_mpsas": "Sky Brightness (Experimental)",
            "contrast": "Relative Contrast (Experimental)",
        }

        switch_color = "#427EDC"

        df = pd.DataFrame(
            columns=[
                "TARGET",
                "GROUP",
                "status",
                "exposure_goal",
                "exposure_acquired",
                "priority",
                "metadata",
            ]
        )

        dropdown_cols = ["status", "priority"]
        dropdowns = {}
        column_properties = []
        for col in df.columns:
            column_property = {"name": col, "id": col, "editable": False}
            if col in dropdown_cols:
                column_property["presentation"] = "dropdown"
                column_property["editable"] = True
                dropdowns[col] = {
                    "options": [
                        {"label": str(i), "value": str(i)} for i in df[col].unique()
                    ]
                }
            column_properties.append(column_property)

        process_elements = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button(
                                            "Select All",
                                            color="primary",
                                            id="select-all",
                                            className="mr-1",
                                        ),
                                        dbc.Button(
                                            "Select None",
                                            color="primary",
                                            id="select-none",
                                            className="mr-1",
                                        ),
                                        dbc.Button(
                                            "Save Changes",
                                            color="warning",
                                            id="save-changes",
                                            className="mr-1",
                                        ),
                                    ],
                                    className="mr-2",
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.Label(
                                    "Change Target Status",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Dropdown(
                                    id="bulk-status",
                                    options=[],
                                    value=[],
                                ),
                            ],
                            width=2,
                        ),
                        dbc.Col(
                            [
                                html.Label(
                                    "Change Target Priority",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Dropdown(
                                    id="bulk-priority",
                                    options=[],
                                    value=[],
                                ),
                            ],
                            width=2,
                        ),
                        dbc.Col(
                            [
                                html.Label(
                                    "Change Target Goals",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Dropdown(
                                    id="goal-dropdown",
                                    options=[],
                                    value=[],
                                ),
                            ],
                            width=2,
                            style={"display": "none"},
                        ),
                        dbc.Col(html.Div(id="goal-table"), width=4),
                    ]
                ),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dash_table.DataTable(
                                    id="table-progress",
                                    columns=column_properties,
                                    data=[],
                                    dropdown=dropdowns,
                                    filter_action="native",
                                    sort_action="native",
                                    sort_mode="multi",
                                    column_selectable="single",
                                    row_selectable="multi",
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
                                    export_format="csv",
                                    style_cell_conditional=[
                                        {
                                            "if": {"column_id": "exposure_goal"},
                                            "display": "None",
                                        },
                                        {
                                            "if": {"column_id": "exposure_acquired"},
                                            "display": "None",
                                        },
                                        {
                                            "if": {"column_id": "metadata"},
                                            "display": "None",
                                        },
                                    ],
                                )
                            ],
                            width=12,
                        ),
                    ]
                ),
            ],
            fluid=True,
        )

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
                        href=f"http://clearoutside.com/forecast/{planner_config.lat}/{planner_config.lon}?view=current",
                        target="_blank",
                    )
                ),
                dbc.NavItem(
                    dbc.NavLink(
                        "Weather",
                        id="nws-weather",
                        href=f"http://forecast.weather.gov/MapClick.php?lon={planner_config.lon}&lat={planner_config.lat}#.U1xl5F7N7wI",
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
            style={"display": "none"},
        )

        target_status_picker = dbc.Col(
            [
                html.Div(
                    [
                        html.Label(
                            "Change Target Status and Priority",
                            style={"textAlign": "center"},
                        ),
                        dcc.Dropdown(
                            id="target-status-match",
                            options=[],
                            value=[],
                            multi=True,
                            placeholder="Select Target...",
                        ),
                    ],
                    style={"textAlign": "center"},
                    className="dash-bootstrap",
                )
            ]
        )

        target_status_selector = dbc.Row(
            [
                dbc.Col(
                    dbc.RadioItems(
                        options=[],
                        labelStyle={"display": "block"},
                        inputStyle={"margin-right": "10px"},
                        id="target-status-selector",
                    ),
                    width=10,
                ),
            ],
        )

        target_priority_selector = dbc.Row(
            [
                dbc.Col(
                    dbc.RadioItems(
                        options=[],
                        labelStyle={"display": "block"},
                        inputStyle={"margin-right": "10px"},
                        id="target-priority-selector",
                    ),
                    width=10,
                ),
            ],
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
                        "Group or Sequence file",
                        style={"textAlign": "center"},
                    ),
                    dcc.Dropdown(id=PlannerConfig.PROFILES, multi=True),
                ],
                style={"textAlign": "center"},
                className="dash-bootstrap",
            ),
            style={"border": "0px solid"},
        )

        palette_picker = dbc.Col(
            html.Div(
                [
                    html.Label(
                        "Color Palette",
                        style={"textAlign": "center"},
                    ),
                    dcc.Dropdown(
                        id=PlannerConfig.COLOR_PALETTE,
                        options=[
                            {"label": element, "value": element}
                            for element in [
                                "base",
                                "deep",
                                "muted",
                                "bright",
                                "pastel",
                                "dark",
                                "colorblind",
                                "husl",
                                "hls",
                                "ch:s=.25,rot=-.25",
                            ]
                        ],
                        value=planner_config.color_palette,
                    ),
                ],
                style={"textAlign": "center"},
                className="dash-bootstrap",
            ),
            style={"border": "0px solid"},
        )

        filter_targets_check = dbc.Row(
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
                        html.Label(
                            "Display Targets With Filters",
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
                            placeholder="Select Filters...",
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
                            "Display Targets With Status",
                            style={"textAlign": "center"},
                        ),
                        dcc.Dropdown(
                            id=PlannerConfig.TARGET_STATUS,
                            options=[
                                {"label": "Pending", "value": "Pending"},
                                {"label": "Active", "value": "Active"},
                                {"label": "Acquired", "value": "Acquired"},
                                {"label": "Closed", "value": "Closed"},
                            ],
                            placeholder="Select Target Status...",
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
                            "Display Targets With Priority",
                            style={"textAlign": "center"},
                        ),
                        dcc.Dropdown(
                            id=PlannerConfig.TARGET_PRIORITIES,
                            options=[
                                {"label": "Very High", "value": "Very High"},
                                {"label": "High", "value": "High"},
                                {"label": "Medium", "value": "Medium"},
                                {"label": "Low", "value": "Low"},
                                {"label": "Very Low", "value": "Very Low"},
                            ],
                            value=[],
                            multi=True,
                            placeholder="Select Target Priority...",
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
                        dbc.InputGroupText(
                            "Min Frame Overlap Fraction".ljust(20),
                        ),
                        dbc.Input(
                            id=PlannerConfig.MIN_FRAME_OVERLAP_FRACTION,
                            value=planner_config.min_frame_overlap_fraction,
                            placeholder=planner_config.min_frame_overlap_fraction,
                            type="number",
                            step=0.05,
                            debounce=True,
                        ),
                    ]
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Min Moon Distance".ljust(20)),
                        dbc.Input(
                            id=PlannerConfig.MIN_MOON_DISTANCE,
                            value=planner_config.min_moon_distance,
                            placeholder=planner_config.min_moon_distance,
                            type="number",
                            step=1,
                            min=0,
                            max=180,
                            debounce=True,
                        ),
                    ]
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Extinction Coefficient".ljust(20)),
                        dbc.Input(
                            id=PlannerConfig.K_EXTINCTION,
                            value=planner_config.k_extinction,
                            placeholder=planner_config.k_extinction,
                            type="number",
                            step=0.01,
                            min=0,
                            max=1,
                            debounce=True,
                        ),
                    ]
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(
                            "Solar Altitude for Nightfall".ljust(20),
                        ),
                        dbc.Input(
                            id=PlannerConfig.SOLAR_ALTITUDE_FOR_NIGHT,
                            value=planner_config.solar_altitude_for_night,
                            placeholder=planner_config.solar_altitude_for_night,
                            type="number",
                            step=1,
                            min=-18,
                            max=0,
                            debounce=True,
                        ),
                    ]
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Latitude".ljust(20)),
                        dbc.Input(
                            id=PlannerConfig.LAT,
                            value=planner_config.lat,
                            placeholder=planner_config.lat,
                            type="number",
                            debounce=True,
                            disabled=False,
                            min=-90,
                            max=90,
                        ),
                    ],
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Longitude".ljust(20)),
                        dbc.Input(
                            id=PlannerConfig.LON,
                            value=planner_config.lon,
                            placeholder=planner_config.lon,
                            type="number",
                            debounce=True,
                            disabled=False,
                            min=-180,
                            max=180,
                        ),
                    ],
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("UTC Offset".ljust(20)),
                        dbc.Input(
                            id=PlannerConfig.UTC_OFFSET,
                            value=planner_config.utc_offset,
                            placeholder=planner_config.utc_offset,
                            type="number",
                            debounce=True,
                            disabled=False,
                            min=-12,
                            max=12,
                        ),
                    ],
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(
                            "Sky Brightness (mpsas)".ljust(20),
                        ),
                        dbc.Input(
                            id=PlannerConfig.MPSAS,
                            value=planner_config.mpsas,
                            placeholder=planner_config.mpsas,
                            type="number",
                            debounce=True,
                            min=0,
                            max=22,
                        ),
                    ]
                ),
            ]
        )

        weather_graph = html.Div(
            id="weather-graph",
            children=[],
        )

        clear_outside_forecast_img = html.Img(id="clear-outside-img")

        leaflet_map = dl.Map(
            [
                dl.TileLayer(
                    url="https://www.google.com/maps/vt?lyrs=m&x={x}&y={y}&z={z}&",
                ),
                dl.LayerGroup(
                    id="location-marker",
                ),
                # dl.GestureHandling(),
                # dl.LocateControl(
                #     options={"locateOptions": {"enableHighAccuracy": True}}
                # ),  # requires https
            ],
            style={
                "height": "50vh",
                "margin": "auto",
                "display": "block",
                "position": "relative",
            },
            center=[planner_config.lat, planner_config.lon],
            zoom=12,
            id="map",
            # boundsOptions={"padding": [15, 15]},  ## fails
        )

        location_picker_modal = html.Div(
            [
                dbc.Button(
                    "Change Location",
                    id="open-location",
                    color="info",
                    className="mr-1",
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Select Location"),
                        dbc.ModalBody(
                            dbc.Col(
                                [
                                    html.Div(
                                        [leaflet_map, html.Div(id="location-text")],
                                    ),
                                    html.Br(),
                                    # dcc.Markdown(
                                    #     """Note: if the map does not render, please adjust the size of the main window.
                                    #     This is a known bug, and we are awaiting a fix."""
                                    # ),
                                ]
                            )
                        ),
                        dbc.ModalFooter(
                            [
                                dbc.Button(
                                    "Close",
                                    id="close-location",
                                    color="danger",
                                    className="mr-1",
                                ),
                            ]
                        ),
                    ],
                    id="modal-location",
                    size="lg",
                ),
            ],
            className="d-grid gap-2",
        )

        weather_modal = html.Div(
            [
                dbc.Button(
                    "Show Weather Forecast",
                    id="open",
                    color="primary",
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
                                className="mr-1",
                            ),
                        ),
                    ],
                    id="modal",
                    size="lg",
                ),
            ],
            className="d-grid gap-2",
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
                        html.Br(),
                        html.Label("Quick Options:", style={"textAlign": "center"}),
                        html.Br(),
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
                                {
                                    "label": "Relative Gradient Strength vs. Sky Background (ADU)",
                                    "value": "bkg_val vs. relative_gradient_strength",
                                },
                                {
                                    "label": "FWHM Angle vs. Slope",
                                    "value": "fwhm_slope vs. fwhm_theta",
                                },
                                {
                                    "label": "Elongation Angle vs. Star Trailing Strength",
                                    "value": "star_trail_strength vs. theta_median",
                                },
                            ],
                            value="fwhm_median vs. eccentricity_median",
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
                            html.Div(
                                children=[
                                    dbc.Row(date_picker, justify="around"),
                                    html.Br(),
                                    dbc.Row(profile_picker, justify="around"),
                                    html.Br(),
                                    dbc.Row(palette_picker, justify="around"),
                                    html.Br(),
                                    dbc.Row(yaxis_picker, justify="around"),
                                    html.Br(),
                                    html.Br(),
                                    dbc.Row(filter_targets_check, justify="around"),
                                    html.Br(),
                                    dbc.Row(filter_picker, justify="around"),
                                    html.Br(),
                                    dbc.Row(status_picker, justify="around"),
                                    html.Br(),
                                    dbc.Row(priority_picker, justify="around"),
                                    html.Br(),
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
                                    dbc.Row(location_picker_modal, justify="around"),
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
                                style=SIDEBAR_STYLE,
                            ),
                        ],
                        width=3,
                        style={"border": "0px solid"},
                    ),
                    dbc.Col(
                        children=[
                            html.Div(
                                id="target-graph-div",
                                children=[
                                    dcc.Graph(
                                        id="target-graph",
                                    )
                                ],
                                style={"display": "none"},
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
            style={"display": "none"},
        )

        running_mode_checklist = html.Div(
            [
                dcc.Markdown("""## UI Visibility"""),
                dbc.Checklist(
                    options=[
                        {"label": "Use Planner", "value": "planner"},
                        {"label": "Use Inspector", "value": "inspector"},
                    ],
                    value=config.get("running_mode"),
                    id="running-mode-switch",
                    switch=True,
                ),
            ],
            style={"display": "none"},
        )

        thread_slider = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label(
                                    "Thread Count",
                                    html_for="thread-count-slider",
                                ),
                                dbc.Input(
                                    id="thread-count-slider",
                                    value=config.get("n_threads", 1),
                                    min=1,
                                    max=multiprocessing.cpu_count(),
                                    step=1,
                                    type="number",
                                ),
                            ],
                            width=2,
                        ),
                        dbc.Col(
                            [
                                dbc.Label(
                                    "Update Interval",
                                    html_for="monitor-mode-update-frequency",
                                ),
                                dbc.Input(
                                    id="monitor-mode-update-frequency",
                                    value=config.get(
                                        "monitor_mode_update_frequency", 1
                                    ),
                                    min=5,
                                    step=5,
                                    type="number",
                                ),
                            ],
                            width=2,
                        ),
                        dbc.Col(
                            [
                                dbc.Label(
                                    "Planner Time Resolution",
                                    html_for=PlannerConfig.TIME_RESOLUTION,
                                ),
                                dbc.Input(
                                    id=PlannerConfig.TIME_RESOLUTION,
                                    value=planner_config.time_resolution,
                                    min=60,
                                    max=3600,
                                    step=60,
                                    type="number",
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Label(
                                    "UI Theme",
                                    html_for="themes",
                                ),
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="themes",
                                            placeholder="Select UI Theme",
                                            value=config.get("theme", "LITERA"),
                                            options=make_options(available_themes),
                                            multi=False,
                                        ),
                                    ],
                                    style={"textAlign": "center"},
                                    className="dash-bootstrap",
                                ),
                            ],
                            width=3,
                        ),
                    ],
                    align="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                # dbc.Label(
                                #     "Silence Alerts",
                                #     html_for="silence-alerts",
                                # ),
                                daq.BooleanSwitch(
                                    id="silence-alerts",
                                    on=False,
                                    label="Silence Alerts",
                                    labelPosition="top",
                                    color=switch_color,
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                # dbc.Label(
                                #     "Silence Alerts",
                                #     html_for="silence-alerts",
                                # ),
                                daq.BooleanSwitch(
                                    id="planner-toggle",
                                    on=False,
                                    label="Use Planner",
                                    labelPosition="top",
                                    color=switch_color,
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                # dbc.Label(
                                #     "Silence Alerts",
                                #     html_for="silence-alerts",
                                # ),
                                daq.BooleanSwitch(
                                    id="inspector-toggle",
                                    on=False,
                                    label="Use Data Review",
                                    labelPosition="top",
                                    color=switch_color,
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                daq.BooleanSwitch(
                                    id="remove-files-when-deleted",
                                    on=False,
                                    label="Remove Frame Data When Deleted (Inactive)",
                                    labelPosition="top",
                                    color=switch_color,
                                ),
                            ],
                            width=3,
                        ),
                    ],
                    align="center",
                ),
            ],
            className="pad-row",
        )

        profile_list = html.Div(
            [
                # dmc.Group(
                #     [
                #         DashIconify(
                #             icon="ion:logo-github",
                #             width=30,
                #             rotate=1,
                #             flip="horizontal",
                #         ),
                #         DashIconify(icon="flat-ui:settings", width=30),
                #         DashIconify(icon="feather:info", color="pink", width=30),
                #     ]
                # ),
                # change to modal for changing profile
                # need button to add new profile
                dcc.Markdown(f"""Current AIP Profile: {config.get('env')}"""),
                dcc.Dropdown(
                    id="aip-profile",
                    value=env,
                    placeholder="Profile",
                    options=make_options(["primary", "testing", "demo"]),
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    dbc.Button(
                                        "Change Profile",
                                        id="aip-profile-change-button",
                                        n_clicks=0,
                                        color="info",
                                    )
                                ],
                                className="d-grid gap-2",
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    dbc.Button(
                                        "Add New Profile (Inactive)",
                                        id="aip-profile-new-button",
                                        n_clicks=0,
                                        color="success",
                                    )
                                ],
                                className="d-grid gap-2",
                            ),
                            width=6,
                        ),
                    ]
                ),
            ]
        )

        # with open(f"{BASE_DIR}/data/license/license.txt", "r") as f:
        #     license_text = "".join(f.readlines())

        # license_div = html.Div(
        #     [
        #         dbc.Button("Open License Info", id="license-modal-open", n_clicks=0),
        #         dbc.Modal(
        #             [
        #                 dbc.ModalHeader(dbc.ModalTitle("License Info")),
        #                 dbc.ModalBody(
        #                     [
        #                         dbc.Textarea(
        #                             value=license_text,
        #                             size="sm",
        #                             className="mb-3",
        #                             placeholder="License data",
        #                             id="license-text",
        #                             style={
        #                                 "width": "100%",
        #                                 "height": 400,
        #                             },
        #                             debounce=True,
        #                         ),
        #                         dbc.Alert(
        #                             "",
        #                             id="license-status-text",
        #                             color="primary",
        #                             is_open=False,
        #                         ),
        #                     ]
        #                 ),
        #                 dbc.ModalFooter(
        #                     [
        #                         dbc.Button(
        #                             "Save",
        #                             id="license-save",
        #                             n_clicks=0,
        #                             color="success",
        #                         ),
        #                         dbc.Button(
        #                             "Close",
        #                             id="license-modal-close",
        #                             className="ms-auto",
        #                             n_clicks=0,
        #                         ),
        #                     ]
        #                 ),
        #             ],
        #             id="license-modal",
        #             is_open=False,
        #         ),
        #     ]
        # )

        input_dir_list = html.Div(
            [
                # dcc.Markdown("""## Directory Settings"""),
                dbc.FormFloating(
                    [
                        dbc.Input(
                            id={"type": "dir", "sub-type": "target", "index": 0},
                            type="text",
                            placeholder="foo",
                        ),
                        dbc.Label(
                            "Target Directory (Directory with Voyager Roboclip, or SGP/NINA sequences)"
                        ),
                    ],
                ),
                dbc.FormFloating(
                    [
                        dbc.Input(
                            id={"type": "dir", "sub-type": "data", "index": 0},
                            type="text",
                            placeholder="foo",
                        ),
                        dbc.Label("Raw FITs Data Directory"),
                    ],
                    id="fit-data-dir-inputs",
                ),
                dbc.FormFloating(
                    [
                        dbc.Input(
                            id={"type": "dir", "sub-type": "calibration", "index": 0},
                            type="text",
                            placeholder="foo",
                        ),
                        dbc.Label("Calibration FITs Data Directory"),
                    ],
                    id="fit-calibration-dir-inputs",
                ),
                dbc.FormFloating(
                    [
                        dbc.Input(
                            id={"type": "dir", "sub-type": "preproc-out", "index": 0},
                            type="text",
                            placeholder="foo",
                        ),
                        dbc.Label("Preprocessed Output Directory"),
                    ],
                    id="fit-preproc-output-dir-inputs",
                ),
            ]
        )

        siril_inputs = html.Div(
            [
                dbc.Label("Auto-preprocessing with Siril", html_for="siril-switch"),
                daq.BooleanSwitch(
                    id="siril-switch",
                    on=voyager_config.voyager_switch,
                    labelPosition="bottom",
                    color=switch_color,
                ),
            ]
        )

        voyager_inputs = html.Div(
            [
                dbc.Label(
                    "Connect With Voyager Advanced",
                    html_for=VoyagerConfig.SWITCH,
                ),
                daq.BooleanSwitch(
                    id=VoyagerConfig.SWITCH,
                    on=voyager_config.voyager_switch,
                    labelPosition="bottom",
                    color=switch_color,
                ),
                html.Br(),
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText(
                                            "Voyager Hostname",
                                            id="voyager-hostname-label",
                                        ),
                                        dbc.Input(
                                            id=VoyagerConfig.HOSTNAME,
                                            type="text",
                                            value=voyager_config.hostname,
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
                                        dbc.InputGroupText(
                                            "Voyager Port",
                                            id="voyager-port-label",
                                        ),
                                        dbc.Input(
                                            id=VoyagerConfig.PORT,
                                            value=voyager_config.port,
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
                                        dbc.InputGroupText(
                                            "Voyager User",
                                            id="voyager-user-label",
                                        ),
                                        dbc.Input(
                                            id=VoyagerConfig.USER,
                                            type="text",
                                            value=voyager_config.user,
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
                                        dbc.InputGroupText(
                                            "Voyager Password",
                                            id="voyager-password-label",
                                        ),
                                        dbc.Input(
                                            id=VoyagerConfig.PASSWORD,
                                            type="password",
                                            value=voyager_config.password,
                                            debounce=True,
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    id="voyager-inputs",
                ),
            ],
        )

        downloads = dbc.Col(
            [
                dbc.ButtonGroup(
                    [
                        dbc.Button(
                            "Download Target Status",
                            id="button-download-target-status-table",
                            color="success",
                        ),
                        dbc.Button(
                            "Download Target Data",
                            id="button-download-target-data-table",
                            color="success",
                        ),
                        dbc.Button(
                            "Download FITs Data Tables",
                            id="button-download-data-table",
                            color="success",
                        ),
                        dbc.Button(
                            "Download App Log",
                            id="button-download-log",
                            color="info",
                        ),
                        dbc.Button(
                            "Download App Database",
                            id="button-download-database",
                            color="info",
                            style={"display": "none"},
                        ),
                        dbc.Button(
                            "Download Database and Config",
                            id="button-download-zip",
                            color="info",
                        ),
                    ],
                    vertical=True,
                ),
            ],
            className="col-2 mx-auto",
            width=12,
        )

        settings_accordian = dmc.AccordionMultiple(
            children=[
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl(
                            dcc.Markdown("### AIP Profiles"),
                            icon=DashIconify(
                                icon="tabler:user",
                                color=dmc.theme.DEFAULT_COLORS["blue"][6],
                                width=20,
                            ),
                        ),
                        dmc.AccordionPanel(profile_list),
                    ],
                    value="profile",
                ),
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl(
                            dcc.Markdown("### Directories"),
                            icon=DashIconify(
                                icon="clarity:directory-line",
                                color=dmc.theme.DEFAULT_COLORS["violet"][6],
                                width=20,
                            ),
                        ),
                        dmc.AccordionPanel([input_dir_list, running_mode_checklist]),
                    ],
                    value="directories",
                ),
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl(
                            dcc.Markdown("### Target Groups/Sequences"),
                            icon=DashIconify(
                                icon="solar:telescope-outline",
                                color=dmc.theme.DEFAULT_COLORS["cyan"][6],
                                width=20,
                            ),
                        ),
                        dmc.AccordionPanel(
                            [
                                html.Div(
                                    [
                                        dbc.Checklist(id="profile-list", switch=True),
                                    ],
                                    id="profile-list-div",
                                )
                            ]
                        ),
                    ],
                    value="target-profiles",
                ),
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl(
                            dcc.Markdown("### Settings"),
                            icon=DashIconify(
                                icon="gala:settings",
                                color=dmc.theme.DEFAULT_COLORS["red"][6],
                                width=20,
                            ),
                        ),
                        dmc.AccordionPanel(
                            [
                                thread_slider,
                            ]
                        ),
                    ],
                    value="basic-settings",
                ),
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl(
                            dcc.Markdown("### Advanced Settings"),
                            icon=DashIconify(
                                icon="gala:settings",
                                color=dmc.theme.DEFAULT_COLORS["orange"][6],
                                width=20,
                            ),
                        ),
                        dmc.AccordionPanel(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col([voyager_inputs], width=6),
                                        dbc.Col([siril_inputs], width=6),
                                    ]
                                )
                            ]
                        ),
                    ],
                    value="special-settings",
                ),
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl(
                            dcc.Markdown("### Downloads"),
                            icon=DashIconify(
                                icon="material-symbols:download",
                                color=dmc.theme.DEFAULT_COLORS["blue"][6],
                                width=20,
                            ),
                        ),
                        dmc.AccordionPanel([downloads]),
                    ],
                    value="downloads",
                ),
                # dmc.AccordionItem(
                #     [
                #         dmc.AccordionControl(
                #             dcc.Markdown("### License"),
                #             icon=DashIconify(
                #                 icon="clarity:license-line",
                #                 color=dmc.theme.DEFAULT_COLORS["red"][6],
                #                 width=20,
                #             ),
                #         ),
                #         dmc.AccordionPanel(license_div),
                #     ],
                #     value="license",
                # ),
            ],
            value=[
                # "profile",
                # "directories",
                # "target-profiles",
                "basic-settings",
                # "special-settings",
                # "downloads",
                # "license",
            ],
        )
        settings_container = dbc.Container(
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(settings_accordian, width=8),
                            dbc.Col(
                                [
                                    dbc.Button(
                                        "Save All Settings",
                                        id="config-save",
                                        n_clicks=0,
                                        color="success",
                                    ),
                                    dbc.Button(
                                        "Show config",
                                        id="config-show",
                                        n_clicks=0,
                                        color="success",
                                        style=DEBUG_STYLE,
                                    ),
                                    html.Div(f"Version: {version} - Build {git_hash}"),
                                    dcc.Textarea(
                                        id="config-text",
                                        value=yaml.dump(config),
                                        style={
                                            "width": "80%",
                                            "height": 600,
                                        },
                                    ),
                                ],
                                width=4,
                            ),
                        ]
                    ),
                ],
                id="config-div",
            ),
            id="tab-settings-div",
            fluid=True,
            style={"display": "none"},
        )
        markdown_from_files = {}
        md_files = glob.glob("docs/*.md")
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
            style={"display": "none"},
        )

        data_table_container = dbc.Container(
            dbc.Row(
                [
                    process_elements,
                    dbc.Col(
                        children=[html.Div(id="data-table")],
                        width=12,
                        style={"border": "20px solid white"},
                    ),
                ]
            ),
            id="tab-data-table-div",
            fluid=True,
            style={"display": "none"},
        )

        target_picker = html.Div(
            [
                html.Div(
                    [
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText(
                                    "Recent Sessions (0 total)",
                                    id="session-avail-count",
                                ),
                                dbc.Input(
                                    id="inspector-last-n-days",
                                    min=-365 * 10,
                                    max=365 * 10,
                                    value=1 if IS_PROD else 0,
                                    step=1,
                                    debounce=True,
                                    type="number",
                                ),
                            ],
                            className="mb-3",
                        ),
                    ],
                    style={"textAlign": "center"},
                    className="dash-bootstrap",
                ),
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
                            id="filter-matches",
                            placeholder="Select Filter",
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
                            id="sensor-matches",
                            placeholder="Select Sensor",
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
                html.Div(
                    [
                        dcc.Dropdown(
                            id="img-type-matches",
                            placeholder="Image Type",
                            options=[],
                            multi=True,
                        ),
                    ],
                    style={"textAlign": "center"},
                    className="dash-bootstrap",
                ),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                "Force Update",
                                id="force-update-button",
                                color="warning",
                                # style={"display": "none"},
                                style={"textAlign": "center"},
                                className="d-grid col-10 mx-auto",
                                n_clicks=0,
                            ),
                            width=12,
                        ),
                    ]
                ),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                "Show All Data From Current Targets",
                                color="info",
                                id="all-data-curr-targets",
                                style={"textAlign": "center"},
                                className="d-grid col-10 mx-auto",
                            ),
                            width=12,
                        ),
                    ]
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
                                dbc.InputGroupText(
                                    "Max Eccentricity".ljust(20),
                                    id="ecc-label",
                                ),
                                dbc.Input(
                                    id=InspectorThresholds.ECC_THR,
                                    value=config.get("inspector_thresholds", {}).get(
                                        InspectorThresholds.ECC_THR, 0.6
                                    ),
                                    placeholder=config.get(
                                        "inspector_thresholds", {}
                                    ).get(InspectorThresholds.ECC_THR, 0.6),
                                    step=0.05,
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
                                dbc.InputGroupText(
                                    "Min Star Fraction:  ",
                                    id="min-star-label",
                                ),
                                dbc.Input(
                                    id=InspectorThresholds.STAR_FRAC_THR,
                                    debounce=True,
                                    placeholder=config.get(
                                        "inspector_thresholds", {}
                                    ).get(InspectorThresholds.STAR_FRAC_THR, 0.5),
                                    value=config.get("inspector_thresholds", {}).get(
                                        InspectorThresholds.STAR_FRAC_THR, 0.5
                                    ),
                                    step=0.05,
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
                                dbc.InputGroupText(
                                    "z score:  ",
                                    id="z-score-label",
                                ),
                                dbc.Input(
                                    id=InspectorThresholds.Z_SCORE,
                                    value=config.get("inspector_thresholds", {}).get(
                                        InspectorThresholds.Z_SCORE, 5
                                    ),
                                    placeholder=config.get(
                                        "inspector_thresholds", {}
                                    ).get(InspectorThresholds.Z_SCORE, 5),
                                    step=0.5,
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
                                dbc.InputGroupText(
                                    "IQR Scale".ljust(20),
                                    id="iqr-scale-label",
                                ),
                                dbc.Input(
                                    id=InspectorThresholds.IQR_SCALE,
                                    value=config.get("inspector_thresholds", {}).get(
                                        InspectorThresholds.IQR_SCALE, 1.5
                                    ),
                                    placeholder=config.get(
                                        "inspector_thresholds", {}
                                    ).get(InspectorThresholds.IQR_SCALE, 1.5),
                                    step=0.5,
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
                                dbc.InputGroupText(
                                    "Trail Threshold".ljust(20),
                                    id="trail-label",
                                ),
                                dbc.Input(
                                    id=InspectorThresholds.TRAIL_THR,
                                    value=config.get("inspector_thresholds", {}).get(
                                        InspectorThresholds.TRAIL_THR, 8
                                    ),
                                    placeholder=config.get(
                                        "inspector_thresholds", {}
                                    ).get(InspectorThresholds.TRAIL_THR, 8),
                                    step=1,
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
                                dbc.InputGroupText(
                                    "Gradient Threshold".ljust(20),
                                    id="gradient-label",
                                ),
                                dbc.Input(
                                    id=InspectorThresholds.GRADIENT_THR,
                                    value=config.get("inspector_thresholds", {}).get(
                                        InspectorThresholds.GRADIENT_THR, 0.1
                                    ),
                                    placeholder=config.get(
                                        "inspector_thresholds", {}
                                    ).get(InspectorThresholds.GRADIENT_THR, 0.1),
                                    step=0.02,
                                    type="number",
                                    debounce=True,
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )

        single_target_progress = html.Div(id="single-target-progress-graph")

        monitor_mode_check = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                "Preprocess accepted files",
                                color="primary",
                                id="preprocess",
                                className="d-grid col-10 mx-auto",
                                style={"textAlign": "center"},
                            ),
                            id="preprocess-col",
                            width=12,
                        ),
                    ]
                ),
                dbc.Row(
                    html.Div(
                        [
                            dbc.Col(
                                dbc.Button(
                                    "Sync Ratings To Voyager RoboTarget",
                                    color="warning",
                                    id="sync-ratings",
                                    className="d-grid col-10 mx-auto",
                                    style={"textAlign": "center"},
                                ),
                                width=12,
                            ),
                        ],
                        id="sync-ratings-col",
                        style={},
                    ),
                ),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    daq.BooleanSwitch(
                                        id="monitor-mode",
                                        on=monitor_mode_on,
                                        label="Monitor Mode",
                                        labelPosition="bottom",
                                        color=switch_color,
                                    ),
                                ],
                                className="dash-bootstrap",
                            ),
                            width=4,
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
                            width=4,
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    daq.BooleanSwitch(
                                        id="disable-scatter-graph",
                                        on=False,
                                        label="Disable Scatter Graph",
                                        labelPosition="bottom",
                                        color=switch_color,
                                    ),
                                ],
                                className="dash-bootstrap",
                            ),
                            width=4,
                        ),
                    ],
                    className="col-8 mx-auto",
                ),
            ],
            className="d-grid gap-2 col-12 mx-auto",
        )

        # the styles for the main content position it to the right of the sidebar and
        # add some padding.
        CONTENT_STYLE = {
            # "position": "relative",
            # "margin-left": "22%",
            # "margin-right": "2rem",
            # "padding": "2rem 1rem",
        }

        sidebar = html.Div(
            id="sidebar",
            children=dbc.Col(
                [
                    target_picker,
                    html.Br(),
                    html.Div(
                        monitor_mode_check, className="d-grid gap-2 col-12 mx-auto"
                    ),
                    html.Br(),
                    scatter_col_picker,
                    html.Br(),
                    quick_options_col_picker,
                    html.Br(),
                    frame_acceptance_criteria,
                ],
                width=12,
            ),
            style=SIDEBAR_STYLE,
        )

        content = html.Div(
            id="content",
            children=dbc.Col(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [single_target_progress],
                                width=6,
                            ),
                            dbc.Col(
                                html.Div(
                                    id="scatter-graph-div",
                                    children=[
                                        dcc.Graph(
                                            id="subframe-scatter-graph",
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
                                        [
                                            dbc.Row(
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
                                        ],
                                        className="dash-bootstrap",
                                    ),
                                    dcc.Graph(
                                        id="inspector-frame",
                                        style={"width": "35vw", "height": "35vw"},
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
                                        style={"width": "35vw", "height": "35vw"},
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
                                    id="frame-gradient-graph",
                                    style={"display": "none"},
                                    config={
                                        "displaylogo": False,
                                        "modeBarButtonsToRemove": ["lasso2d"],
                                    },
                                ),
                                width=12,
                            ),
                            dbc.Col(),
                        ]
                    ),
                    dbc.Button(
                        "Show Tables",
                        id="collapse-button",
                        className="mb-3",
                        color="primary",
                        n_clicks=0,
                    ),
                    dbc.Collapse(
                        id="table-collapse",
                        is_open=False,
                        children=[
                            dcc.Markdown(
                                """
                ## Summary Table
                Table Group Keys:"""
                            ),
                            dbc.Checklist(
                                options=[
                                    {"label": "Object", "value": "OBJECT"},
                                    {"label": "Filter", "value": "FILTER"},
                                    {"label": "Binning", "value": BINNING_COL},
                                    {"label": "Focal Length", "value": FOCALLENGTH_COL},
                                    {"label": "Pixel Size", "value": "XPIXSZ"},
                                    {"label": "Date", "value": "date_night_of"},
                                ],
                                value=[
                                    "OBJECT",
                                    "FILTER",
                                    BINNING_COL,
                                    "XPIXSZ",
                                    FOCALLENGTH_COL,
                                ],
                                id="summary-table-group-keys",
                                switch=True,
                                labelStyle={"display": "inline-block"},
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
                            dbc.Row(
                                [
                                    dbc.Col(
                                        header_col_picker,
                                        width=3,
                                    )
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        children=[html.Div(id="files-table")],
                                        width=12,
                                    ),
                                ]
                            ),
                        ],
                    ),
                ]
            ),
            style=CONTENT_STYLE,
        )

        inspector_container = dbc.Container(
            dbc.Row([dbc.Col(sidebar, width=3), dbc.Col(content, width=9)]),
            id="tab-files-table-div",
            fluid=True,
            style={"display": "none"},
        )

        loading = dcc.Loading(
            id="loading-2",
            children=[
                html.Div(
                    [
                        html.Div(id="loading-output"),
                        html.Div(id="loading-output-preproc"),
                        html.Div(id="loading-output-rating-sync"),
                        html.Div(id="loading-output-click"),
                    ]
                )
            ],
            type="default",
        )

        # tabs = dbc.Tabs(
        #     id="tabs",
        #     active_tab="tab-settings",
        #     children=[
        #         dbc.Tab(
        #             id="tab-target-review",
        #             label="Target Planning",
        #             # label=DashIconify(icon="flat-ui:settings", width=30),
        #             # label=[html.I(className="bi bi-check-circle-fill me-2")],
        #             tab_id="tab-target",
        #             labelClassName="text-primary",
        #             label_style={"font-size": 16},
        #             tab_style={"display": "none"},
        #         ),
        #         dbc.Tab(
        #             id="tab-inspector",
        #             label="Frame Inspector",
        #             tab_id="tab-inspector",
        #             labelClassName="text-info",
        #             label_style={"font-size": 16},
        #             tab_style={"display": "none"},
        #         ),
        #         dbc.Tab(
        #             id="tab-targets-table",
        #             label="Target Tables",
        #             tab_id="tab-targets-table",
        #             labelClassName="text-success",
        #             label_style={"font-size": 16},
        #             tab_style={"display": "none"},
        #         ),
        #         dbc.Tab(
        #             label="Settings & Utilities",
        #             tab_id="tab-settings",
        #             labelClassName="text-danger",
        #             label_style={"font-size": 16},
        #         ),
        #         dbc.Tab(
        #             label="Help",
        #             tab_id="tab-help",
        #             labelClassName="text-warning",
        #             label_style={"font-size": 16},
        #         ),
        #     ],
        # )

        i_color = 9
        tabs = dmc.Tabs(
            [
                dmc.TabsList(
                    [
                        dmc.Tab(
                            "Planning",
                            icon=DashIconify(
                                icon="mdi:planner-outline",
                                color=dmc.theme.DEFAULT_COLORS["blue"][i_color],
                            ),
                            value="tab-target",
                            id="tab-target-review",
                            style={"display": "none"},
                            color="blue",
                        ),
                        dmc.Tab(
                            "Data Review",
                            icon=DashIconify(
                                icon="icon-park:checklist",
                                color=dmc.theme.DEFAULT_COLORS["green"][i_color],
                            ),
                            value="tab-inspector",
                            id="tab-inspector",
                            style={"display": "none"},
                            color="green",
                        ),
                        dmc.Tab(
                            "Target Data",
                            icon=DashIconify(
                                icon="ph:table-fill",
                                color=dmc.theme.DEFAULT_COLORS["blue"][i_color],
                            ),
                            value="tab-targets-table",
                            id="tab-targets-table",
                            style={"display": "none"},
                            color="blue",
                        ),
                        dmc.Tab(
                            "Settings",
                            icon=DashIconify(
                                icon="tabler:settings",
                                color=dmc.theme.DEFAULT_COLORS["yellow"][i_color],
                            ),
                            value="tab-settings",
                            id="tab-settings",
                            color="yellow",
                        ),
                        dmc.Tab(
                            "Help",
                            icon=DashIconify(
                                icon="tabler:help",
                                color=dmc.theme.DEFAULT_COLORS["orange"][i_color],
                            ),
                            value="tab-help",
                            id="tab-help",
                            color="orange",
                        ),
                    ]
                ),
            ],
            variant="pills",
            value="tab-settings",
            id="tabs",
        )

        alerts = html.Div(
            [
                dbc.Alert(
                    "",
                    id="alert-preproc",
                    is_open=False,
                    duration=2,
                    dismissable=True,
                ),
                dbc.Toast(
                    "",
                    id="alert-auto",
                    is_open=True,
                    duration=10,
                    dismissable=True,
                    style={
                        "position": "fixed",
                        "top": 66,
                        "right": 10,
                        "width": 800,
                        "zIndex": 999,
                        "opacity": 0.95,
                    },
                ),
                dbc.Alert(
                    "",
                    id="alert-file-download",
                    is_open=False,
                    duration=1,
                    dismissable=True,
                ),
                dcc.Interval(
                    id="monitor-mode-interval",
                    interval=5000,
                    n_intervals=0,
                    disabled=False,
                ),
                dcc.Interval(
                    id="target-graph-interval",
                    interval=30000,
                    n_intervals=0,
                    disabled=False,
                ),
            ]
        )

        body = dbc.Container(
            fluid=True,
            style={"width": "98%"},
            children=[
                navbar,
                dcc.Interval(
                    id="processing-interval",
                    interval=1000,
                    n_intervals=0,
                    disabled=False,
                ),
                html.Div(
                    [
                        dbc.Progress(
                            id="preproc-progress",
                            style={"display": "none"},
                            color="warning",
                            value=0,
                            animated=True,
                        ),
                        html.Div(
                            html.H4(
                                "",
                                id="preprocessing-label",
                                className="text-center w-100 mb-0",
                            ),
                            className="position-absolute h-100 w-100 d-flex align-items-center",
                            style={"top": "0"},
                        ),
                    ],
                    className="position-relative mb-1",
                ),
                html.Div(
                    [
                        dbc.Progress(
                            id="processing-progress",
                            style={"height": "40px"},
                            value=0,
                        ),
                        html.Div(
                            html.H4(
                                "",
                                id="processing-label",
                                className="text-center w-100 mb-0",
                            ),
                            className="position-absolute h-100 w-100 d-flex align-items-center",
                            style={"top": "0"},
                        ),
                    ],
                    className="position-relative mb-1",
                ),
                dbc.Row(
                    [
                        dbc.Col(tabs, width=5),
                        # dbc.Col(new_tabs, width=5),
                        dbc.Col([]),
                        dbc.Col([], width=3, id="location-tab-text"),
                        dbc.Col([], width=1, id="bortle-tab-badge"),
                        dbc.Col(
                            f"AIP Profile: {config.get('env')}",
                            width=1,
                            id="profile-tab-text",
                        ),
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
                settings_container,
                help_container,
                inspector_container,
                html.Div(id="dummy-radio-change", style={"display": "none"}),
                html.Div(id="dummy-id-restore", style={"display": "none"}),
                html.Div(id="dummy-id-2", style={"display": "none"}),
                html.Div(id="dummy-id-3", style={"display": "none"}),
                html.Div(id="dummy-id-4", style={"display": "none"}),
                html.Div(id="dummy-id", style={"display": "none"}),
                html.Div(id="dummy-id-target-data", style={"display": "none"}),
                html.Div(id="dummy-id-contrast-data", style={"display": "none"}),
                html.Div(id="dummy-rejection-criteria-id", style={"display": "none"}),
                html.Div(id="new-data-available-trigger", style={"display": "none"}),
                html.Div(id="progress-graph-files", style={"display": "none"}),
                html.Div(id="dummy-use-planner-check", style={"display": "none"}),
                html.Div(
                    id="dummy-profile",
                    #  style={"display": "none"},
                ),
                dcc.Download(id="all-download-data"),
            ],
        )

        layout = html.Div(
            [
                body,
                dcc.Store(id="store-site-data", data={}),
                dcc.Store(id="store-target-planner-data"),
                dcc.Store(id="store-target-status"),
                dcc.Store(id="store-target-metadata"),
                dcc.Store(id="store-dark-sky-duration"),
                dcc.Store(id="store-config", storage_type="memory", data=config),
                dcc.Location(id="url", refresh=True),
            ],
            id="page-content",
        )
        return layout

    return foo
