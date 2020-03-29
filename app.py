from datetime import datetime as dt
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
import dash_table

from plotly.subplots import make_subplots
import plotly.graph_objects as go


from astro_planner import *
import warnings
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter("ignore", category=AstropyWarning)

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# app = dash.Dash(external_stylesheets=[dbc.themes.SLATE])
app = dash.Dash(external_stylesheets=[dbc.themes.COSMO])
# app = dash.Dash(external_stylesheets=[dbc.themes.DARKLY])
# app = dash.Dash(external_stylesheets=[dbc.themes.LUX])
# app = dash.Dash(external_stylesheets=[dbc.themes.GRID])


markdown_text = """
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

## SECONDARY:
- [x] Organization of divs - use bootstrap columns
- [ ] Notes search
- [ ] Allow other sources for targets - json, astroplanner, etc.
- [ ] Allow upload of targets file
- [x] Fix double-click zoom reset
"""


DSF_FORECAST = DarkSky_Forecast(key="")


DEFAULT_LAT = 43.37
DEFAULT_LON = -88.37
DEFAULT_UTC_OFFSET = -6

date_string = datetime.datetime.now().strftime("%Y-%m-%d")
path_to_astrobox = "/Volumes/Users/gshau/Dropbox/AstroBox"
# path_to_astrobox = "/Users/gshau/Dropbox/AstroBox"
target_list = get_roboclips(filename=f"{path_to_astrobox}/roboclip/VoyRC.mdb")

profiles = list(target_list.keys())

df_summary = get_exposure_summary(data_dir=f"{path_to_astrobox}/data").reset_index()


debug_status = True

show_todos = False


if not debug_status:
    show_todos = False

date_range = []
def get_site(lat=DEFAULT_LAT, lon=DEFAULT_LON, alt=290, utc_offset=DEFAULT_UTC_OFFSET):
    print(lat, lon, utc_offset)
    site = ObservingSite(lat, lon, alt, utc_offset=utc_offset)
    return site


def get_time_limits(targets, sun_alt=10):
    sun = targets["sun"]
    # Get sun up/down
    sun_up = np.where(np.gradient((sun.alt > sun_alt).astype(int)) > 0)[0][0]
    sun_dn = np.where(np.gradient((sun.alt > sun_alt).astype(int)) < 0)[0][0]
    return sun.index[sun_dn], sun.index[sun_up]


def get_data(target_coords, targets, value="alt", sun_alt_for_twilight=-15):
    target_names = [
        name for name in list(target_coords.keys()) if name not in ["sun", "moon"]
    ]

    # this is where we sort by transit time
    # print(sorted(target_coords.values, key=lambda x: x["alt"].argmax()))
    if value == 'contrast':
        target_coords = add_contrast(target_coords, filter_bandwidth=300, mpsas=22, object_brightness=19, include_airmass=True)

    moon_data = dict(
        x=target_coords["moon"].index,
        y=target_coords["moon"]['alt'],
        text="Moon",
        opacity=1,
        line=dict(color="Gray", width=4),
        name="Moon",
    )

    sun_data = dict(
        x=target_coords["sun"].index,
        y=target_coords["sun"]['alt'],
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
        name='sun_up'
    )
    sun_dn_data = dict(
        x=[sun.index[sun_dn], sun.index[sun_dn], sun.index[0], sun.index[0]],
        y=[0, 90, 90, 0],
        mode="lines",
        line=dict(color="Orange", width=1),
        showlegend=False,
        fill="toself",
        name='sun_dn'
    )
    data = [sun_data, sun_up_data, sun_dn_data, moon_data]
    if value == 'contrast':
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


navbar = dbc.NavbarSimple(
    children=[
        # dbc.NavItem(dbc.NavLink("Clear Outside Report",
        #                         href="http://clearoutside.com/forecast/43.10/-88.40?view=current", target="_blank")),
        # dbc.NavItem(dbc.NavLink(
        #     "Weather", href="http://forecast.weather.gov/MapClick.php?lon=-88.39866&lat=43.08719#.U1xl5F7N7wI", target="_blank")),
        # dbc.NavItem(dbc.NavLink(
        #     "Satellite", href="https://www.star.nesdis.noaa.gov/GOES/sector_band.php?sat=G16&sector=umv&band=11&length=12", target="_blank")),
        dbc.DropdownMenu(
            children=[
                # dbc.DropdownMenuItem("", header=True),
                dbc.DropdownMenuItem(
                    "Clear Outside Report",
                    href="http://clearoutside.com/forecast/43.10/-88.40?view=current",
                    target="_blank",
                ),
                dbc.DropdownMenuItem(
                    "Weather",
                    href="http://forecast.weather.gov/MapClick.php?lon=-88.39866&lat=43.08719#.U1xl5F7N7wI",
                    target="_blank",
                ),
                dbc.DropdownMenuItem(
                    "Satellite",
                    href="https://www.star.nesdis.noaa.gov/GOES/sector_band.php?sat=G16&sector=umv&band=11&length=12",
                    target="_blank",
                ),
            ],
            nav=True,
            in_navbar=True,
            label="Weather",
        ),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("", header=True),
                dbc.DropdownMenuItem("Profiles", href="#"),
                dbc.DropdownMenuItem("UI Theme", href="#"),
                dbc.DropdownMenuItem("Logout", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="The AstroImaging Planner",
    brand_href="#",
    color="primary",
    dark=True,
)

banner_jumbotron = dbc.Jumbotron(
    [
        html.H2("AstroImaging Planner", className="display-5"),
        html.P(
            "This tool reads a Voyager RoboClip target database and provides ephermeris for all targets for tonight.",
            className="lead",
        ),
        html.Hr(className="my-2"),
        html.P(""),
        # html.P(dbc.Button("GitHub", color="primary"), className="lead"),
    ]
)

markdown_todos = None
if show_todos:
    markdown_todos = dbc.Row(
        dbc.Col(html.Div(dcc.Markdown(children=markdown_text)), style={"marginTop": 80})
    )

date_picker = dbc.Row([dbc.Col(html.Label("DATE: ")),
    dbc.Col(html.Div(
        [ dcc.DatePickerSingle(id="date_picker", date=dt.now()),],
        style={"textAlign": "center"},
    ))])

yaxis_map = {
    "alt": "Altitude",
    "airmass": "Airmass",
    'contrast': 'Relative Contrast'
}


yaxis_picker = dbc.Col(
    html.Div(
        [
            html.Label('Quantity to plot:'),
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
                options=[{"label": profile, "value": profile} for profile in profiles],
                value=profiles[0],
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
                        [dbc.Col(
                            html.Label("LATITUDE:  ", style={"textAlign": "right"},)),
                            dbc.Col(
                            dcc.Input(
                                id="input_lat",
                                debounce=True,
                                placeholder=DEFAULT_LAT,
                                type="number",
                            )),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(html.Label("LONGITUDE:  ", style={"textAlign": "left"},),),
                            dbc.Col(dcc.Input(
                                id="input_lon",
                                debounce=True,
                                placeholder=DEFAULT_LON,
                                type="number",
                            )),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(html.Label("UTC OFFSET:  ", style={"textAlign": "left"},),),
                            dbc.Col(dcc.Input(
                                id="input_utc_offset",
                                debounce=True,
                                placeholder=DEFAULT_UTC_OFFSET,
                                type="number",
                            ),)
                        ]
                    ),
                ]
            )
        ]
    ),
)


exposure_summary = dash_table.DataTable(
    id="table",
    columns=[{"name": i, "id": i} for i in df_summary.columns],
    data=df_summary.to_dict("records"),
    sort_action="native",
    # sort_mode="multi",
)

weather_graph = html.Div(
                            id="weather_graph", children=[dbc.Spinner(color="warning")]
                        )

weather_modal = html.Div(
    [
        dbc.Button("Show Weather Forecast", id="open", color='primary', block=True, className="mr-1"),
        dbc.Modal(
            [
                dbc.ModalHeader("Weather Forecast"),
                dbc.ModalBody(weather_graph),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close", color='danger', block=True, className="mr-1"),
                ),
            ],
            id="modal",
            size='xl',
        ),
    ]
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
                    [dbc.Container(
    fluid=True,
    style={"width": "95%"},children=[
                        dbc.Row(date_picker, justify='around'),
                        html.Br(),
                        dbc.Row(location_selection, justify='around'),
                        html.Br(),
                        dbc.Row(yaxis_picker, justify='around'),
                        html.Br(),
                        dbc.Row(profile_picker, justify='around'),
                        html.Br(),
                        dbc.Row(filter_picker, justify='around'),
                        html.Br(),
                        # dbc.Row(search_notes, justify='around'),
                        # html.Br(),
                        dbc.Row(weather_modal, justify='around'),
                        html.Br(),
                        dbc.Row(markdown_todos, justify='around'),])
                    ],
                    width=3,
                    style={"border": "2px solid"},
                ),
                dbc.Col(
                    [
                        # html.Div(
                        #     id="all_graph", children=[dbc.Spinner(color="primary")]
                        # ),
                        html.Div(
                            id="target_graph", children=[dbc.Spinner(color="primary")]
                        ),
                        # html.Div(
                        #     id="weather_graph", children=[dbc.Spinner(color="warning")]
                        # ),
                        # exposure_summary,
                    ],
                    width=9,
                    style={"border": "0px solid"},
                ),
            ]
        ),
    html.Div(id='date_range', style={'display': 'none'})
    ],
)




app.layout = html.Div([body])


@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open



translated_filter = {
    "ha": ["ho", "sho", "hoo", "hos", "halpha", "h-alpha"],
    "oiii": ["ho", "sho", "hoo", "hos"],
    "nb": ["ha", "oiii", "sii", "sho", "ho", "hoo", "hos", "halpha", "h-alpha"],
    "rgb": ["osc", "bayer", "dslr", "slr"],
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
        Input("date_picker", "date"),
        Input("input_lat", "value"),
        Input("input_lon", "value"),
        Input("input_utc_offset", "value"),
    ],
)
def update_weather_graph(date, lat, lon, utc_offset):
    # global date_range
    
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
                    legend={"x": 0, "y": 0.5},
                    xaxis={'range': date_range},
                    height=600,
                    plot_bgcolor="#ddd",
                    paper_bgcolor="#fff",
                    hovermode="closest",
                    transition={"duration": 150},
                ),
            },
        )
    ]


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
):
    date = str(date_string.split('T')[0])
    # global date_range
    targets = list(target_list[profile].values())
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
    coords = get_coords(targets, date_string, site, time_resolution_in_sec=300)

    data = get_data(coords, targets, value=value)

    if value == "alt":
        y_range = [0, 90]
    elif value == "airmass":
        y_range = [1, 5]
    elif value == "contrast":
        # y_max = max([d['y'].max() for d in data if 'sun' not in d['name']])
        # y_range = [0, y_max * 1.15]
        y_range = [0, 1]

    date_range = get_time_limits(coords)

    title = "Imaging Targets on {date_string}".format(
        date_string=date
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
                    xaxis={"title": "", "range": date_range},
                    yaxis={"title": yaxis_map[value], "range": y_range},
                    title=title,
                    margin={"l": 50, "b": 100, "t": 50, "r": 50},
                    legend={"x": 0, "y": 0.5},
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
    app.run_server(debug=debug_status, host="0.0.0.0")
    # app.run_server()

