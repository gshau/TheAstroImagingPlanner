import base64
import io
import ntpath

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import pandas as pd
import dash_table
import plotly.graph_objects as go
import warnings
import uuid
import os
import datetime

from datetime import datetime as dt
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
from collections import OrderedDict, defaultdict
from astro_planner import *
from astropy.utils.exceptions import AstropyWarning


warnings.simplefilter("ignore", category=AstropyWarning)

# app = dash.Dash(external_stylesheets=[dbc.themes.SLATE])
app = dash.Dash(external_stylesheets=[dbc.themes.COSMO])
# app = dash.Dash(external_stylesheets=[dbc.themes.DARKLY])
# app = dash.Dash(external_stylesheets=[dbc.themes.LUX])
# app = dash.Dash(external_stylesheets=[dbc.themes.GRID])


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
- [ ] docker deploymeent
  - [ ] darksky api key
- [ ] add openweathermap datasource
- [x] Organization of divs - use bootstrap columns
- [ ] Notes search
- [ ] Allow upload of file
  - [x] Roboclip
  - [ ] json
  - [ ] astroplanner
  - [ ] Allow upload of targets file
- [x] Fix double-click zoom reset
"""

markdown_info = """
This tool attempts to...

"""


DSF_FORECAST = DarkSky_Forecast(key="")


DEFAULT_LAT = 43.37
DEFAULT_LON = -88.37
DEFAULT_UTC_OFFSET = -6
DEFAULT_MPSAS = 19.5
DEFAULT_BANDWIDTH = 120
DEFAULT_K_EXTINCTION = 0.2

DEFAULT_TIME_RESOLUTION = 300

date_string = datetime.datetime.now().strftime("%Y-%m-%d")



class Objects:
    def __init__(self):
        self.target_list = defaultdict(dict)
        self.profiles = []

    def process_objects(self, df_input):
        self.target_list = defaultdict(dict)
        for row in df_input.itertuples():
            profile = row.GROUP
            target = Target(
                row.TARGET,
                ra=row.RAJ2000 * u.hourangle,
                dec=row.DECJ2000 * u.deg,
                notes=row.NOTE,
            )
            self.target_list[profile][row.TARGET] = target

    def load_from_df(self, df_input):
        self.df_objects = df_input
        self.process_objects(df_input)
        self.profiles = list(self.target_list.keys())


class RoboClipObjects(Objects):
    def __init__(self, filename):
        super().__init__()
        self.df_objects = mdb.read_table(
            filename, "RoboClip", converters_from_schema=False
        )
        self.df_objects.rename({"GRUPPO": "GROUP"}, axis=1, inplace=True)
        self.load_from_df(self.df_objects)


class SGPSequenceObjects(Objects):
    def __init__(self, filename):
        super().__init__()
        for self.filename in [filename]:
            with open(self.filename, "r") as f:
                self.data = json.load(f)
                self.df_objects = self.parse_data()
                self.process_objects(self.df_objects)

    def parse_data(self):
        self.sequence = {}
        root_name = ntpath.basename(self.filename)
        self.profiles.append(root_name)
        for sequence in self.data["arEventGroups"]:
            name = sequence["sName"]
            ref = sequence["siReference"]

            RA = ref["nRightAscension"]
            DEC = ref["nDeclination"]
            events = sequence["Events"]
            filters = []
            event_data = []
            note_string = ""
            for event in events:
                filters.append(event["sSuffix"])

                event_data.append(event)
                print(event_data)
                note_string += "<br> {filter} {exp}s ({ncomplete} / {ntotal}) exposure: {total_exposure:.1f}h".format(
                    filter=event["sSuffix"],
                    exp=event["nExposureTime"],
                    ncomplete=event["nNumComplete"],
                    ntotal=event["nRepeat"],
                    total_exposure=event["nNumComplete"]
                    * event["nExposureTime"]
                    / 3600,
                )
            notes = note_string
            self.sequence[name] = dict(
                RAJ2000=RA, DECJ2000=DEC, NOTE=notes, TARGET=name, GROUP=root_name
            )
        return pd.DataFrame.from_dict(self.sequence, orient="index").reset_index(
            drop=True
        )


def object_file_reader(filename):
    if ".mdb" in filename:
        return RoboClipObjects(filename)
    # if isinstance(filename, list):
    #     for file in
    elif ".sgf" in filename:
        return SGPSequenceObjects(filename)


object_data = object_file_reader("./data/VoyRC_default.mdb")
# object_data = object_file_reader("./data/test.sgf")

# df_summary = get_exposure_summary(data_dir=f"{path_to_astrobox}/data").reset_index()


debug_status = True

show_todos = False


if not debug_status:
    show_todos = False

date_range = []


def get_site(lat=DEFAULT_LAT, lon=DEFAULT_LON, alt=290, utc_offset=DEFAULT_UTC_OFFSET):
    site = ObservingSite(lat, lon, alt, utc_offset=utc_offset)
    return site


def get_time_limits(targets, sun_alt=10):
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
):
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
    # print(sorted(target_coords.values, key=lambda x: x["alt"].argmax()))
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
        notes_text = targets[i_target].info["notes"]
        # notes_text = html.Img('<html><img src="https://www.w3schools.com/tags/smiley.gif" alt="Smiley face" width="42" height="42"></html>')
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
    return data


# region ui

# weather_dropdown = dbc.DropdownMenu(
#     children=[
#         dbc.DropdownMenuItem(
#             "Clear Outside Report",
#             id='clear_outside',
#             href=f"http://clearoutside.com/forecast/{DEFAULT_LAT}/{DEFAULT_LON}?view=current",
#             target="_blank",
#         ),
#         dbc.DropdownMenuItem(
#             "Weather",
#             id='nws_weather',
#             href=f"http://forecast.weather.gov/MapClick.php?lon={DEFAULT_LAT}&lat={DEFAULT_LON}#.U1xl5F7N7wI",
#             target="_blank",
#         ),
#         dbc.DropdownMenuItem(
#             "Satellite",
#             href="https://www.star.nesdis.noaa.gov/GOES/sector_band.php?sat=G16&sector=umv&band=11&length=12",
#             target="_blank",
#         ),
#     ],
#     nav=True,
#     in_navbar=True,
#     label="Weather",
# )

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
    ]
)

info_modal = html.Div(
    [
        dbc.Button(
            "More Info",
            id="open_info_modal",
            color="primary",
            block=False,
            className="mr-1",
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Info"),
                dbc.ModalBody(dcc.Markdown(children=markdown_info)),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="close_info_modal",
                        color="danger",
                        block=False,
                        className="mr-1",
                    ),
                ),
            ],
            id="info_modal",
            size="xl",
        ),
    ]
















)

navbar = dbc.NavbarSimple(
    
    children=[
        dbc.NavItem(
            dbc.NavLink(
                "Clear Outside Report",
                            id='clear_outside',
                href=f"http://clearoutside.com/forecast/{DEFAULT_LAT}/{DEFAULT_LON}?view=current",
                target="_blank",
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "Weather",
                            id='nws_weather',
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

banner_jumbotron = dbc.Jumbotron(
    [
        html.H2("The AstroImaging Planner", className="display-6"),
        html.Hr(className="my-2"),
        html.P(
            """This tool reads either a Voyager RoboClip target database or Sequence Generator Pro sequence file and provides data for all targets for tonight.  
            For SGP files, the sequence progress is included in the annotated card on each target trace""",
            className="lead",
        ),
        info_modal,
        # html.P(dbc.Button("GitHub", color="primary"), className="lead"),
    ],
    fluid=True,
)

date_picker = dbc.Row(
    [
        dbc.Col(html.Label("DATE: ")),
        dbc.Col(
            html.Div(
                [dcc.DatePickerSingle(id="date_picker", date=dt.now()),],
                style={"textAlign": "center"},
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
    ),
    style={"border": "0px solid"},
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
            ]
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
        ]
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
                                dcc.Input(
                                    id="input_lat",
                                    debounce=True,
                                    placeholder=DEFAULT_LAT,
                                    type="number",
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
                ]
            )
        ]
    ),
)


# exposure_summary = dash_table.DataTable(
#     id="table",
#     columns=[{"name": i, "id": i} for i in df_summary.columns],
#     data=df_summary.to_dict("records"),
#     sort_action="native",
#     # sort_mode="multi",
# )
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
# endregion

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


body = dbc.Container(
    fluid=True,
    style={"width": "90%"},
    children=[
        navbar,
        banner_jumbotron,
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
                                dbc.Row(profile_picker, justify="around"),
                                html.Br(),
                                dbc.Row(filter_picker, justify="around"),
                                html.Br(),
                                dbc.Row(date_picker, justify="around"),
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
                        html.Div(
                            id="target_graph", children=[dbc.Spinner(color="primary")]
                        ),
                        html.Br(),
                        dbc.Row(
                            html.Div(id="upload_button", children=[upload]),
                            justify="center",
                        ),
                    ],
                    width=9,
                    # style={"border": "0px solid"},
                ),
            ]
        ),
        html.Div(id="date_range", style={"display": "none"}),
    ],
)

app.layout = html.Div(
    [
        body,
        html.Div(
            id="target_data",
            children=object_data.df_objects.to_json(orient="table"),
            style={"display": "none"},
        ),
    ]
)


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if ".mdb" in filename:
            file_id = uuid.uuid1()
            local_file = f"VoyRC_{file_id}.mdb"
            with open(local_file, "wb") as f:
                f.write(decoded)
            object_data = object_file_reader(local_file)
            os.remove(local_file)
            print(object_data.df_objects.head())
        elif ".sgf" in filename:
            out_data = io.StringIO(decoded.decode("utf-8"))
            file_id = uuid.uuid1()
            local_file = f"./{filename}"
            with open(local_file, "w") as f:
                f.write(out_data.read())
            print("Done!")
            object_data = object_file_reader(local_file)
            os.remove(local_file)
            print(object_data.df_objects.head())
        else:
            return html.Div(["Unsupported file!"])
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])
    return object_data


@app.callback(
    [
        Output("target_data", "children"),
        Output("profile_selection", "options"),
        Output("profile_selection", "value"),
    ],
    [Input("upload_data", "contents")],
    [State("upload_data", "filename"), State("upload_data", "last_modified")],
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    global object_data  
    children = [object_data.df_objects.to_json(orient="table")]
    profile = object_data.profiles[0]
    options = [{"label": profile, "value": profile}]
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
    Output("info_modal", "is_open"),
    [Input("open_info_modal", "n_clicks"), Input("close_info_modal", "n_clicks")],
    [State("info_modal", "is_open")],
)
def toggle_info_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


translated_filter = {
    "ha": ["ho", "sho", "hoo", "hos", "halpha", "h-alpha"],
    "oiii": ["ho", "sho", "hoo", "hos"],
    "nb": ["ha", "oiii", "sii", "sho", "ho ", "hoo", "hos", "halpha", "h-alpha"],
    "rgb": ["osc", "bayer", "dslr", "slr", "r ", " g ", " b "],
    "lum": ["luminance", "lrgb"],
}


def update_site(lat=DEFAULT_LAT, lon=DEFAULT_LON, utc_offset=DEFAULT_UTC_OFFSET):
    if lat is None:
        lat = DEFAULT_LAT
    if lon is None:
        lon = DEFAULT_LON
    if utc_offset is None:
        utc_offset = DEFAULT_UTC_OFFSET
    site = get_site(lat=lat, lon=lon, alt=0, utc_offset=utc_offset)
    return site


@app.callback(
    Output("weather_graph", "children"),
    [
        Input("input_lat", "value"),
        Input("input_lon", "value"),
        Input("input_utc_offset", "value"),
    ],
)
def update_weather_graph(lat, lon, utc_offset):

    site = update_site(lat=lat, lon=lon, utc_offset=utc_offset)
    print(site.lat, site.lon)
    try:
        print("Trying NWS")
        nws_forecast = NWS_Forecast(site.lat, site.lon)
        df_weather = nws_forecast.parse_data()
    except:
        print("Trying Dark Sky")
        DSF_FORECAST.get_forecast_data(site.lat, site.lon)
        df_weather = DSF_FORECAST.forecast_data_to_df()["hourly"]
        df_weather = df_weather[
            df_weather.columns[df_weather.dtypes != "object"]
        ].fillna(0)

    print(df_weather.index.name, date_range)
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

    return [
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


import json


@app.callback(
    Output("target_graph", "children"),
    [
        Input("date_picker", "date"),
        Input("y_axis_type", "value"),
        Input("profile_selection", "value"),
        Input("filter_match", "value"),
        Input("input_lat", "value"),
        Input("input_lon", "value"),
        Input("input_utc_offset", "value"),
        Input("local_mpsas", "value"),
        Input("k_ext", "value"),
        Input("target_data", "children"),
    ],
)
def update_target_graph(
    date_string=dt.today(),
    value="alt",
    profile="tmb130ss qsi690",
    filters=[],
    lat=DEFAULT_LAT,
    lon=DEFAULT_LON,
    utc_offset=DEFAULT_UTC_OFFSET,
    local_mpsas=DEFAULT_MPSAS,
    k_ext=DEFAULT_K_EXTINCTION,
    json_targets=None,
):
    if json_targets is None:
        print("json_targets is None")
        json_targets = object_data.df_objects.to_json(orient="table")
    object_data.load_from_df(pd.read_json(json_targets, orient="table"))

    date = str(date_string.split("T")[0])
    targets = list(object_data.target_list[profile].values())
    if filters:
        print(filters)
        targets_with_filter = []
        for filter in filters:
            targets_with_filter += [
                target for target in targets if filter in target.info["notes"].lower()
            ]
            if filter in translated_filter:
                for t_filter in translated_filter[filter]:
                    targets_with_filter += [
                        target
                        for target in targets
                        if t_filter in target.info["notes"].lower()
                    ]
        targets = list(set(targets_with_filter))
    site = update_site(lat=lat, lon=lon, utc_offset=utc_offset)
    coords = get_coords(targets, date_string, site, time_resolution_in_sec=DEFAULT_TIME_RESOLUTION)

    data = get_data(coords, targets, value=value, local_mpsas=local_mpsas, k_ext=k_ext)

    if value == "alt":
        y_range = [0, 90]
    elif value == "airmass":
        y_range = [1, 5]
    elif value == "contrast":
        y_range = [0, 1]

    date_range = get_time_limits(coords)

    title = "Imaging Targets on {date_string}".format(date_string=date)
    return [
        dcc.Graph(
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
    ]


if __name__ == "__main__":
    app.run_server() #(debug=debug_status, host="0.0.0.0")
    # app.run_server(host="0.0.0.0")
