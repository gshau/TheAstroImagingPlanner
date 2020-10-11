import base64
import io
import os

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
import time
import yaml

from dash.dependencies import Input, Output, State

from astro_planner.weather import DarkSky_Forecast, NWS_Forecast
from astro_planner.target import object_file_reader
from astro_planner.contrast import add_contrast
from astro_planner.site import ObservingSite
from astro_planner.ephemeris import get_coords
from astro_planner.data_parser import get_data_info
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

import logging


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(module)s %(message)s")
log = logging.getLogger("app")
warnings.simplefilter("ignore", category=AstropyWarning)


server = flask.Flask(__name__)  #

BS = "https://stackpath.bootstrapcdn.com/bootswatch/4.4.1/cosmo/bootstrap.min.css"
BS = dbc.themes.FLATLY
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO], server=server)

app.title = "The AstroImaging Planner"


with open("./conf/config.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.BaseLoader)

DSF_FORECAST = DarkSky_Forecast(key="")
DATA_DIR = os.getenv("DATA_DIR", "/Volumes/Users/gshau/Dropbox/AstroBox/data/")
ROBOCLIP_FILE = os.getenv(
    "ROBOCLIP_FILE", "/Volumes/Users/gshau/Dropbox/AstroBox/roboclip/VoyRC.mdb"
)
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
    "nb": ["ha", "oiii", "sii", "sho", "ho", "hoo", "hos", "halpha", "h-alpha"],
    "bb": ["luminance", "lrgb"],
    "rgb": ["osc", "bayer", "dslr", "slr", "r ", " g ", " b "],
    "lum": ["luminance"],
}


## load main data
# df_stored_data = get_data_info(data_dir=DATA_DIR)
# object_data = object_file_reader(ROBOCLIP_FILE)


# df_combined = merge_roboclip_stored_metadata(
#     df_stored_data, object_data.df_objects, config
# )

# df_target_status = load_target_status_from_file()
# df_combined = set_target_status(df_combined, df_target_status)
# if df_combined["status"].isnull().sum() > 0:
#     df_combined["status"] = df_combined["status"].fillna("pending")
#     dump_target_status_to_file(df_combined)

object_data = None
df_combined = None
df_target_status = None
df_stored_data = None


def update_data():
    global object_data
    global df_combined
    global df_target_status
    global df_stored_data
    df_stored_data = get_data_info(data_dir=DATA_DIR)
    object_data = object_file_reader(ROBOCLIP_FILE)

    df_combined = merge_roboclip_stored_metadata(
        df_stored_data, object_data.df_objects, config
    )

    df_target_status = load_target_status_from_file()
    df_combined = set_target_status(df_combined, df_target_status)
    if df_combined["status"].isnull().sum() > 0:
        df_combined["status"] = df_combined["status"].fillna("pending")
        dump_target_status_to_file(df_combined)


update_data()


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


app.layout = serve_layout


@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal_callback(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


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
    [Output("profile-selection", "options"), Output("profile-selection", "value")],
    [Input("upload-data", "contents")],
    [State("upload-data", "filename"), State("upload-data", "last_modified")],
)
def update_output_callback(list_of_contents, list_of_names, list_of_dates):
    global object_data
    profile = object_data.profiles[0]
    options = [{"label": profile, "value": profile} for profile in object_data.profiles]
    default_option = options[0]["value"]

    if list_of_contents is not None:
        options = []
        for (c, n, d) in zip(list_of_contents, list_of_names, list_of_dates):
            object_data = parse_contents(c, n, d)
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
                    # xaxis={"range": date_range},
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
        dbc.NavItem(
            dbc.NavLink(
                "Smoke Forecast",
                href="https://rapidrefresh.noaa.gov/hrrr/HRRRsmoke/",
                target="_blank",
            )
        ),
    ]
    return graph_data, navbar_children


@app.callback(
    [Output("weather-graph", "children"), Output("navbar", "children")],
    [Input("store-site-data", "data")],
)
def update_weather_data_callback(site_data):
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
    if n_click != 0:
        update_data()
    return ""


@app.callback(
    [Output("target-match", "options"), Output("target-match", "value")],
    [Input("profile-selection", "value")],
    [State("status-match", "value")],
)
def update_target_for_status_callback(profile, status_match):
    df_combined_group = df_combined[df_combined["GROUP"] == profile]
    targets = df_combined_group.index.values
    return make_options(targets), ""


@app.callback(
    Output("target-status-selector", "value"),
    [Input("target-match", "value")],
    [State("store-target-status", "data")],
)
def update_radio_status_for_targets_callback(targets, target_status_store):
    global df_target_status
    status_set = set()
    print(df_target_status)
    status = df_target_status.set_index("OBJECT")
    log.info(df_target_status)
    for target in targets:
        log.info(f"Target: {target}")
        status_values = [status.loc[target, "status"]]
        log.info(f"Target: {status_values}")
        status_set = status_set.union(set(status_values))
    log.info(f"Target: {status_set}")
    if len(status_set) == 1:
        return list(status_set)[0]
    else:
        return ""


@app.callback(
    Output("store-target-status", "data"),
    [Input("target-status-selector", "value")],
    [State("target-match", "value"), State("store-target-status", "data")],
)
def update_target_with_status_callback(status, targets, target_status_store):
    global df_combined
    df_combined, df_target_status = update_targets_with_status(
        targets, status, df_combined
    )
    return ""  # df_target_status  # .to_dict()


def target_filter(targets, filters):
    log.info(filters)
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
    [Output("tab-target-div", "style"), Output("tab-data-table-div", "style"),],
    [Input("tabs", "active_tab")],
)
def render_content(tab):

    styles = [{"display": "none"}] * 2

    tab_names = [
        "tab-target",
        "tab-data-table",
    ]

    indx = tab_names.index(tab)

    styles[indx] = {}
    return styles


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
        Input("store-target-status", "data"),
        Input("y-axis-type", "value"),
        Input("local-mpsas", "value"),
        Input("k-ext", "value"),
        Input("filter-targets", "checked"),
        Input("status-match", "value"),
        Input("filter-match", "value"),
    ],
)
def store_data(
    date_string,
    profile,
    site_data,
    target_status_store,
    value,
    local_mpsas,
    k_ext,
    filter_targets,
    status_matches,
    filters=[],
):
    global df_combined, object_data
    log.info(f"Calling store_data")
    targets = list(object_data.target_list[profile].values())
    site = update_site((site_data))

    if filters:
        targets = target_filter(targets, filters)

    if status_matches:
        matching_targets = df_combined[df_combined["status"].isin(status_matches)].index
        targets = [target for target in targets if target.name in matching_targets]

    coords = get_coords(
        targets, date_string, site, time_resolution_in_sec=DEFAULT_TIME_RESOLUTION
    )
    date_range = get_time_limits(coords)

    data = get_data(
        coords,
        targets,
        df_combined,
        value=value,
        local_mpsas=local_mpsas,
        k_ext=k_ext,
        filter_targets=filter_targets,
    )

    metadata = dict(date_range=date_range, value=value)
    filtered_targets = [d["name"] for d in data if d["name"]]
    return data, filtered_targets, metadata


@app.callback(
    [
        Output("target-graph", "children"),
        Output("progress-graph", "children"),
        Output("data-table", "children"),
    ],
    [
        Input("date-picker", "date"),
        Input("store-target-data", "data"),
        Input("store-target-metadata", "data"),
        Input("store-progress-data", "data"),
        Input("store-target-goals", "data"),
        Input("profile-selection", "value"),
        Input("status-match", "value"),
    ],
)
def update_target_graph(
    date, target_data, metadata, progress_data, target_goals, profile, status_list,
):
    log.info(f"Calling update_target_graph")
    if not metadata:
        metadata = {}
    try:
        value = metadata["value"]
        date_range = metadata["date_range"]
    except KeyError:
        return None, None, None

    date_string = str(date.split("T")[0])
    title = "Imaging Targets <br> For Night of {date_string}".format(
        date_string=date_string
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
                transition={"duration": 150},
            ),
        },
    )

    df_combined_group = df_combined[df_combined["GROUP"] == profile]

    df_target_status = load_target_status_from_file()
    df_combined_group = set_target_status(df_combined_group, df_target_status)
    targets = get_targets_with_status(df_combined_group, status_list=status_list)
    progress_graph, df_summary = get_progress_graph(
        df_stored_data, date_string=date_string, days_ago=0, targets=targets
    )

    df_combined_group = df_combined_group.reset_index()
    columns = []
    for col in df_combined_group.columns:
        entry = {"name": col, "id": col, "deletable": False, "selectable": True}
        columns.append(entry)

    data = df_combined.reset_index().to_dict("records")

    table = html.Div(
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

    return target_graph, progress_graph, table


def get_progress_graph(df, date_string, days_ago=90, targets=[]):

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

    df0 = df_summary.reset_index()
    bin = df0["XBINNING"].astype(int).astype(str)
    fl = df0["FOCALLEN"].astype(int).astype(str)
    df0["text"] = df0["INSTRUME"] + " @ " + bin + "x" + bin + " FL = " + fl + "mm"
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
        barmode="group", yaxis_title="Total Exposure (hr)", title="Acquired Data"
    )
    return dcc.Graph(figure=p), df_summary


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0")
