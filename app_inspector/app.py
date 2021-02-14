import base64
import dash
import dash_bootstrap_components as dbc
import os
import warnings

import numpy as np

from dash.dependencies import Input, Output, State
from image_grading.frame_analysis import (
    show_inspector_image,
    show_frame_analysis,
    show_fwhm_ellipticity_vs_r,
)
from image_grading.preprocessing import (
    process_stars,
    process_headers,
    to_numeric,
)
from layout import serve_layout
from astropy.utils.exceptions import AstropyWarning
import flask
from astro_planner.logger import log
import dash_html_components as html
import dash_table
import pandas as pd

import plotly.graph_objects as go

warnings.simplefilter("ignore", category=AstropyWarning)


L_FILTER = "L"
R_FILTER = "R"
G_FILTER = "G"
B_FILTER = "B"
HA_FILTER = "Ha"
OIII_FILTER = "OIII"
SII_FILTER = "SII"
BAYER = "OSC"
BAYER_ = "** BayerMatrix **"

FILTER_LIST = [
    L_FILTER,
    R_FILTER,
    G_FILTER,
    B_FILTER,
    HA_FILTER,
    OIII_FILTER,
    SII_FILTER,
    BAYER,
    BAYER_,
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
    BAYER_: "gray",
}


server = flask.Flask(__name__)

BS = "https://stackpath.bootstrapcdn.com/bootswatch/4.4.1/cosmo/bootstrap.min.css"
BS = dbc.themes.FLATLY
BS = dbc.themes.COSMO
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], server=server)

app.title = "The AstroImaging Frame Inspector"


def make_options(elements):
    return [{"label": element, "value": element} for element in elements]


def parse_loaded_contents(contents, filename, date):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    local_file = ""
    try:
        if ".fit" in filename.lower():
            file_root = os.path.basename(filename)
            local_file = f"./data/fits_uploads/{file_root}.fits"
            with open(local_file, "wb") as f:
                f.write(decoded)
            status = "Success"
        else:
            status = "Unsupported file!"
    except Exception as e:
        log.warning(e)
        status = "There was an error processing this file."
    return local_file, status


# Set layout
app.layout = serve_layout
result = {}
df_stars_headers = pd.DataFrame()


@app.callback(
    [
        Output("files-table", "children"),
        Output("summary-table", "children"),
        Output("header-col-match", "options"),
        Output("x-axis-field", "options"),
        Output("y-axis-field", "options"),
        Output("scatter-size-field", "options"),
    ],
    [Input("upload-data", "contents"), Input("header-col-match", "value")],
    [State("upload-data", "filename"), State("upload-data", "last_modified")],
)
def update_output_callback(
    list_of_contents, header_col_match, list_of_names, list_of_dates
):
    global result
    global df_stars_headers
    local_file = ""
    files_table = html.Div()
    summary_table = html.Div()
    header_options = make_options([])
    scatter_field_options = make_options([])

    file_list = []
    if list_of_contents is not None:
        for (c, n, d) in zip(list_of_contents, list_of_names, list_of_dates):
            local_file, status = parse_loaded_contents(c, n, d)
            file_list.append(local_file)

        result = process_stars(file_list, extract_thresh=0.25)
        df_header = process_headers(file_list)
        df_header = to_numeric(df_header)
        df_stars = result["agg_stars"]

        df_stars_headers = pd.merge(df_header, df_stars, on="filename", how="left")
        df_stars_headers["fwhm_mean_arcsec"] = (
            df_stars_headers["fwhm_mean"] * df_stars_headers["arcsec_per_pixel"]
        )
        df_stars_headers["fwhm_median_arcsec"] = (
            df_stars_headers["fwhm_median"] * df_stars_headers["arcsec_per_pixel"]
        )
        df_stars_headers["fwhm_std_arcsec"] = (
            df_stars_headers["fwhm_std"] * df_stars_headers["arcsec_per_pixel"]
        )

        df_stars_headers["frame_snr"] = (
            10 ** df_stars_headers["log_flux_mean"] * df_stars_headers["n_stars"]
        ) / df_stars_headers["bkg_val"]

        root_name = df_stars_headers["file_full_path"].apply(lambda f: f.split("/")[1])
        df_stars_headers["OBJECT"] = df_stars_headers["OBJECT"].fillna(root_name)

        columns = []
        default_cols = [
            "OBJECT",
            "DATE-OBS",
            "FILTER",
            "EXPOSURE",
            "XPIXSZ",
            "FOCALLEN",
            "arcsec_per_pixel",
            "CCD-TEMP",
            "fwhm_median_arcsec",
            "eccentricity_median",
            "star_trail_strength",
        ]
        fits_cols = [
            col
            for col in df_stars_headers.columns
            if col in header_col_match and col not in default_cols
        ]
        fits_cols = default_cols + fits_cols
        other_header_cols = [
            col for col in df_stars_headers.columns if col not in default_cols
        ]
        header_options = make_options(other_header_cols)

        for col in fits_cols:
            entry = {"name": col, "id": col, "deletable": False, "selectable": True}
            columns.append(entry)

        data = df_stars_headers.round(2).to_dict("records")
        files_table = html.Div(
            [
                dash_table.DataTable(
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
        df_stars_headers["DATE-OBS"] = pd.to_datetime(df_stars_headers["DATE-OBS"])
        df_numeric = df_stars_headers.select_dtypes(
            include=[
                "int16",
                "int32",
                "int64",
                "float16",
                "float32",
                "float64",
                "datetime64[ns]",
            ]
        )

        numeric_cols = [col for col in df_numeric.columns if "corr__" not in col]
        scatter_field_options = make_options(numeric_cols)
        df_agg = df_stars_headers.groupby(
            ["OBJECT", "FILTER", "XBINNING", "FOCALLEN", "XPIXSZ"]
        ).agg({"EXPOSURE": "sum", "CCD-TEMP": "std", "DATE-OBS": "count"})
        df_agg["EXPOSURE"] = df_agg["EXPOSURE"] / 3600
        col_map = {
            "DATE-OBS": "n_subs",
            "CCD-TEMP": "CCD-TEMP Dispersion",
            "EXPOSURE": "EXPOSURE (hour)",
        }
        df_agg = df_agg.reset_index().rename(col_map, axis=1)
        df_agg["FILTER_indx"] = df_agg["FILTER"].map(
            dict(zip(FILTER_LIST, range(len(FILTER_LIST))))
        )
        df_agg = df_agg.sort_values(by=["FILTER_indx"]).drop("FILTER_indx", axis=1)

        columns = []
        for col in df_agg.columns:
            entry = {"name": col, "id": col, "deletable": False, "selectable": True}
            columns.append(entry)
        data = df_agg.round(2).to_dict("records")
        summary_table = html.Div(
            [
                dash_table.DataTable(
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

    return (
        files_table,
        summary_table,
        header_options,
        scatter_field_options,
        scatter_field_options,
        scatter_field_options,
    )


@app.callback(
    [Output("x-axis-field", "value"), Output("y-axis-field", "value")],
    [Input("scatter-radio-selection", "value")],
)
def update_scatter_axes(value):
    x_col = "fwhm_mean_arcsec"
    y_col = "eccentricity_mean"
    if value:
        x_col, y_col = value.split(" vs. ")
    return x_col, y_col


@app.callback(
    Output("target-scatter-graph", "figure"),
    [
        Input("x-axis-field", "value"),
        Input("y-axis-field", "value"),
        Input("scatter-size-field", "value"),
        Input("header-col-match", "options"),
    ],
)
def update_scatter_plot(x_col, y_col, size_col, header_options):
    global df_stars_headers
    p = go.Figure()

    if df_stars_headers.shape[0] == 0:
        return p

    df0 = df_stars_headers.copy()

    filters = df0["FILTER"].unique()
    if not x_col:
        x_col = "fwhm_median_arcsec"
    if not y_col:
        y_col = "eccentricity_median"
    sizeref = float(2.0 * df0[size_col].max() / (5 ** 2))
    for filter in FILTER_LIST:
        if filter not in filters:
            continue
        df1 = df0[df0["FILTER"] == filter].reset_index()

        df1["text"] = df1.apply(
            lambda row: "<br>Date: "
            + str(row["DATE-OBS"])
            + f"<br>Star count: {row['n_stars']}"
            + f"<br>FWHM: {row['fwhm_median']:.2f}"
            + f"<br>Eccentricity: {row['eccentricity_median']:.2f}"
            + f"<br>{size_col}: {row[size_col]:.2f}",
            axis=1,
        )
        default_size = df1[size_col].median()
        if np.isnan(default_size):
            default_size = 1
        size = df1[size_col].fillna(default_size)

        if filter in ["L", "R", "G", "B", "OSC"]:
            symbol = "circle"
        else:
            symbol = "diamond"

        p.add_trace(
            go.Scatter(
                x=df1[x_col],
                y=df1[y_col],
                mode="markers",
                name=filter,
                hovertemplate="<b>%{text}</b><br>"
                + f"{x_col}: "
                + "%{x:.2f}<br>"
                + f"{y_col}: "
                + "%{y:.2f}<br>",
                text=df1["text"],
                marker=dict(
                    color=COLORS[filter], size=size, sizeref=sizeref, symbol=symbol
                ),
                customdata=df1["filename"],
            )
        )
    p.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        title=f"Subframe data",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return p


@app.callback(
    [
        Output("radial-frame-graph", "figure"),
        Output("xy-frame-graph", "figure"),
        Output("inspector-frame", "figure"),
    ],
    [
        Input("target-scatter-graph", "clickData"),
        Input("aberration-preview", "checked"),
        Input("frame-heatmap-dropdown", "value"),
    ],
)
def inspect_frame_analysis(
    data, as_aberration_inspector=True, frame_heatmap_col="fwhm"
):
    global result
    global df_stars_headers

    p0 = go.Figure()
    p1 = p0
    p2 = p0

    if data is None:
        return p0, p1, p2
    base_filename = data["points"][0]["customdata"]
    if not base_filename:
        log.info("Issue 1")
        return p0, p1, p2

    file_full_path = df_stars_headers[df_stars_headers["filename"] == base_filename]
    if file_full_path.shape[0] == 1:
        log.info(file_full_path)
        filename = file_full_path["file_full_path"].values[0]
    else:
        log.info("Issue 2")
        return p0, p1, p2

    log.info(f"Base filename: {base_filename}")
    df_radial = result.get("radial_frame").copy()
    log.info(list(df_radial["filename"].unique()))
    df_radial = df_radial[df_radial["filename"] == base_filename]

    df_xy = result.get("xy_frame").copy()
    log.info(list(df_xy["filename"].unique()))
    df_xy = df_xy[df_xy["filename"] == base_filename]

    log.info(f"xy: {list(df_xy.columns)}")

    p2, canvas = show_inspector_image(
        filename,
        as_aberration_inspector=as_aberration_inspector,
        with_overlay=False,
        n_cols=3,
        n_rows=3,
        border=5,
    )

    p0 = show_fwhm_ellipticity_vs_r(df_radial, filename)
    p1 = show_frame_analysis(df_xy, filename=filename, feature_col=frame_heatmap_col)

    return p0, p1, p2


if __name__ == "__main__":
    app.run_server(
        debug=True,
        host="0.0.0.0",
        port=8050,
        dev_tools_hot_reload=True,
        use_reloader=True,
    )
