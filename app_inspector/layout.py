import yaml
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

with open("/app/conf/config.yml", "r") as f:
    CONFIG = yaml.safe_load(f)


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
        ],
        brand="The AstroImaging Frame Inspector",
        brand_href="https://github.com/gshau/AstroPlanner/",
        color="primary",
        dark=True,
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
                        ],
                        labelStyle={"display": "block"},
                    ),
                ],
                className="dash-bootstrap",
            ),
        ]
    )

    upload = dcc.Upload(
        id="upload-data",
        children=html.Div(
            [
                dbc.Button(
                    "Drag and drop .fits file or click here",
                    color="dark",
                    className="mr-1",
                )
            ]
        ),
        multiple=True,
    )

    # target_picker = dbc.Col(
    #     [
    #         html.Div(
    #             [
    #                 html.Label("Select Target", style={"textAlign": "center"},),
    #                 dcc.Dropdown(id="target-match", options=[]),
    #             ],
    #             className="dash-bootstrap",
    #         )
    #     ]
    # )

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
                                                    # dbc.Col(target_picker, width=3),
                                                    dbc.Col(
                                                        scatter_col_picker, width=3,
                                                    ),
                                                    dbc.Col(
                                                        quick_options_col_picker,
                                                        width=3,
                                                    ),
                                                    dbc.Col(
                                                        filter_targets_check, width=3,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Row(
                                                            html.Div(
                                                                id="upload-button",
                                                                children=[upload],
                                                            ),
                                                            justify="around",
                                                        ),
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

    # alerts = html.Div(
    #     [
    #         dbc.Alert("", id="alert-auto", is_open=False, duration=1,),
    #         dcc.Interval(
    #             id="interval-component",
    #             interval=60 * 1000,  # in milliseconds
    #             n_intervals=0,
    #         ),
    #     ]
    # )

    body = dbc.Container(
        fluid=True,
        style={"width": "80%"},
        children=[
            navbar,
            html.Br(),
            # alerts,
            data_files_table_container,
        ],
    )

    layout = html.Div([body])
    return layout
