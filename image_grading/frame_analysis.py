import os
import pandas as pd
import plotly.figure_factory as ff

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from itertools import product

import numpy as np

import plotly.express as px
from astro_planner.stf import auto_stf
from astropy.io.fits import getdata
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(module)s %(message)s")
log = logging.getLogger(__name__)


def show_fwhm_ellipticity_vs_r(df_radial, filename):
    p = make_subplots(specs=[[{"secondary_y": True}]])

    p.add_trace(
        go.Scatter(
            x=df_radial["chip_r_mean"],
            y=df_radial["fwhm_50_pct"],
            mode="markers",
            name="fwhm",
            error_y=dict(
                type="data",
                symmetric=False,
                array=df_radial["fwhm_75_pct"] - df_radial["fwhm_50_pct"],
                arrayminus=df_radial["fwhm_50_pct"] - df_radial["fwhm_25_pct"],
            ),
        ),
        secondary_y=False,
    )
    p.add_trace(
        go.Scatter(
            x=df_radial["chip_r_mean"],
            y=df_radial["ellipticity_50_pct"],
            mode="markers",
            name="ellipticity",
            error_y=dict(
                type="data",
                symmetric=False,
                array=df_radial["ellipticity_75_pct"] - df_radial["ellipticity_50_pct"],
                arrayminus=df_radial["ellipticity_50_pct"]
                - df_radial["ellipticity_25_pct"],
            ),
        ),
        secondary_y=True,
    )
    p.update_layout(
        yaxis_range=[0, 5],
        xaxis_title="Distance to chip center (pixels)",
        title=f"Radial analysis for<br>{os.path.basename(filename)}",
    )
    p.update_yaxes(title_text="FWHM (px)", secondary_y=False, range=[0, 5])
    p.update_yaxes(title_text="Ellipticity", secondary_y=True, range=[0, 1])

    return p


def show_frame_analysis(df_xy, filename, feature_col="fwhm"):

    df0 = df_xy.set_index(["x_bin", "y_bin"]).unstack(0).iloc[::-1]
    df1 = df0.stack()

    # Set text on hover
    df1["text"] = df1.apply(
        lambda row: f"fwhm: {row['fwhm']:.2f} px<br>\
radius: {row['chip_r']:.0f} px<br>\
ellipticity: {row['ellipticity']:.3f}<br>\
stars: {row['star_count']}<br>",
        axis=1,
    )
    df2 = df1["text"].unstack(1).iloc[::-1]

    # Add quiver for opposite direction
    df_quiver = df_xy[["x_bin", "y_bin", "vec_u", "vec_v"]]
    df_quiver["vec_u"] *= -1
    df_quiver["vec_v"] *= -1
    df_quiver = pd.concat([df_xy[["x_bin", "y_bin", "vec_u", "vec_v"]], df_quiver])

    p = ff.create_quiver(
        df_quiver["x_bin"],
        df_quiver["y_bin"],
        df_quiver["vec_u"],
        df_quiver["vec_v"],
        scale=200,
        name="quiver",
        line_width=1,
        line=dict(color="#000"),
    )
    zmax = df0[feature_col].values.max()
    if feature_col == "fwhm":
        zmin = 1
        zmax = max([5, zmax])
    elif feature_col == "ellipticity":
        zmin = 0
        zmax = max([0.5, zmax])

    p.add_trace(
        go.Heatmap(
            x=df0[feature_col].columns,
            y=df0.index,
            z=df0[feature_col].values,
            name="test",
            hovertext=df2,
            colorscale="balance",
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=feature_col.upper()),
        )
    )
    p.update_layout(title=f"Frame analysis for<br>{os.path.basename(filename)}",)
    return p


def show_inspector_image(
    filename,
    as_aberration_inspector=True,
    n_cols=3,
    n_rows=3,
    window_size=0,
    border=10,
    with_overlay=False,
    use_plotly=True,
):
    data = getdata(filename, header=False)
    nx, ny = data.shape
    base_filename = os.path.basename(filename)
    title = f"Full Preview for<br>{base_filename}"

    if as_aberration_inspector:
        if window_size == 0:
            window_size = int(ny / (n_rows * 3))

        x_skip = int((nx - window_size) // (n_cols - 1))
        x_set = []
        for i_panel in range(n_cols):
            x_set.append(
                [i_panel, i_panel * x_skip, i_panel * x_skip + window_size - 1]
            )

        y_skip = int((ny - window_size) // (n_rows - 1))
        y_set = []
        for i_panel in range(n_cols):
            y_set.append(
                [i_panel, i_panel * y_skip, i_panel * y_skip + window_size - 1]
            )

        nx_canvas = n_cols * window_size + (n_cols - 1) * border
        ny_canvas = n_rows * window_size + (n_rows - 1) * border
        canvas = np.ones((nx_canvas, ny_canvas)) * 2 ** 16

        for xlim, ylim in product(x_set, y_set):
            xlim_canvas = [
                xlim[0] * (border + window_size),
                xlim[0] * (border + window_size) + window_size - 1,
            ]
            ylim_canvas = [
                ylim[0] * (border + window_size),
                ylim[0] * (border + window_size) + window_size - 1,
            ]
            canvas[
                xlim_canvas[0] : xlim_canvas[1], ylim_canvas[0] : ylim_canvas[1]
            ] = data[xlim[1] : xlim[2], ylim[1] : ylim[2]]
        data = canvas
        title = f"Aberration Inspector for<br>{base_filename}"

    p = px.imshow(
        auto_stf(data),
        color_continuous_scale="gray",
        binary_string=True,
        binary_compression_level=4,
    )
    p.update_layout(title=title)

    return p, data
