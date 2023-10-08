import os
import sep
import skimage.measure
import statsmodels.api as sm
import numpy as np
import pandas as pd

from collections import defaultdict
from astropy.io import fits
from astro_planner.globals import EXC_INFO

from astro_planner.logger import log
from image_grading.utils import to_numeric, flatten_cols
from astro_planner.utils import timer


def process_stars_from_fits(filename, extract_thresh=3, use_simple=True):
    df_agg_stars, df_stars = pd.DataFrame(), pd.DataFrame()
    try:
        df_stars = process_image_from_filename(
            filename, extract_thresh=extract_thresh, use_simple=use_simple
        )
        df_agg_stars = aggregate_stars(df_stars)
        df_agg_stars["extract_thresh"] = extract_thresh
        return df_agg_stars, df_stars
    except KeyboardInterrupt:
        raise ("Stopping...")
    except:
        log.info(f"Problem with file {filename}", exc_info=EXC_INFO)
        return df_agg_stars, df_stars


def process_frame(file_list, extract_thresh=1.5, xy_n_bins=None, use_simple=True):
    df_lists = defaultdict(list)
    for filename in file_list:
        # file_skiplist = get_skiplist()
        # if filename in file_skiplist:
        #     continue
        log.info(f"Starting to process stars from {filename}")
        df_agg_stars, df_stars = process_stars_from_fits(
            filename, extract_thresh=extract_thresh, use_simple=use_simple
        )
        n_stars = df_stars.shape[0]
        # file_skiplist = get_skiplist()
        # if filename in file_skiplist:
        #     continue

        log.info(f"For {filename}: N-stars = {n_stars}")
        base_filename = os.path.basename(filename)
        df_stars["filename"] = base_filename
        df_stars["full_file_path"] = filename
        if n_stars == 0:
            df_lists["stars"].append(df_stars)
            df_lists["agg_stars"].append(df_agg_stars)
            df_lists["xy_frame"].append(df_stars.copy())
            df_lists["frame_gradients"].append(df_stars.copy())
            continue
        if not xy_n_bins:
            xy_n_bins = max(min(int(np.sqrt(n_stars) / 9), 10), 3)

        nx = df_stars["nx"].values[0]
        ny = df_stars["ny"].values[0]
        df_stars = preprocess_stars(df_stars, xy_n_bins=xy_n_bins, nx=nx, ny=ny)

        df_xy = bin_stars(df_stars, filename)
        trail_strength = np.sqrt(
            (df_xy["vec_u"].mean() / df_xy["vec_u"].std()) ** 2
            + (df_xy["vec_v"].mean() / df_xy["vec_v"].std()) ** 2
        )
        df_agg_stars["star_trail_strength"] = trail_strength
        df_agg_stars["star_orientation_score"] = np.abs(df_xy["dot_norm"]).median()

        df1 = df_xy[["x_bin", "y_bin", "fwhm"]].reset_index(drop=True)
        df1.columns = ["x", "y", "fwhm"]
        df1["x"] = df1["x"] / df1["y"].max()
        df1["y"] = df1["y"] / df1["y"].max()
        vx = (df1["x"] * (df1["fwhm"] - df1["fwhm"].median())).mean() / (
            df1["x"] ** 2 + 1e-9
        ).mean()
        vy = (df1["y"] * (df1["fwhm"] - df1["fwhm"].median())).mean() / (
            df1["y"] ** 2 + 1e-9
        ).mean()
        df_agg_stars["fwhm_theta"] = np.arctan2(vy, vx) * 180 / np.pi
        df_agg_stars["fwhm_slope"] = np.sqrt(vx**2 + vy**2)

        df_lists["stars"].append(df_stars)
        df_lists["agg_stars"].append(df_agg_stars)
        # df_lists["radial_frame"].append(df_radial)
        df_lists["xy_frame"].append(df_xy)

        # TODO: move
        data = fits.getdata(filename)
        res, df_binned, df_pred, df_residual = get_gradient_data(
            data.astype(float), n_samples=16
        )
        df_frame_backgrounds = (
            df_binned.stack()["value"]
            .to_frame("binned")
            .join(df_pred.stack()["value"].to_frame("fit"))
            .join(df_residual.stack()["value"].to_frame("residual"))
        )
        df_frame_backgrounds = df_frame_backgrounds.reset_index()
        df_frame_backgrounds["full_file_path"] = filename
        df_lists["frame_backgrounds"].append(df_frame_backgrounds)

        result = {}
        result["frame_rmse"] = np.sqrt(
            (
                (df_frame_backgrounds["binned"] - df_frame_backgrounds["binned"].mean())
                ** 2
            ).mean()
        )
        result["residual_rmse"] = np.sqrt(
            (
                (
                    df_frame_backgrounds["residual"]
                    - df_frame_backgrounds["residual"].mean()
                )
                ** 2
            ).mean()
        )
        result["fit_rmse"] = np.sqrt(
            (
                (df_frame_backgrounds["fit"] - df_frame_backgrounds["fit"].mean()) ** 2
            ).mean()
        )
        result["r2"] = res.rsquared
        result.update(res.params)

        result["quadratic_strength"] = np.sqrt(result["x2"] ** 2 + result["y2"] ** 2)
        result["quadratic_aspect"] = np.arctan2(result["y2"], result["x2"])
        result["quadratic_dir"] = np.arctan2(result["y2"], result["x2"])
        result["relative_quadratic_strength"] = (
            result["quadratic_strength"] / result["const"]
        )
        result["gradient_strength"] = np.sqrt(result["x"] ** 2 + result["y"] ** 2)
        result["gradient_dir"] = np.arctan2(result["y"], result["x"])
        result["relative_gradient_strength"] = (
            result["gradient_strength"] / result["const"]
        )
        df = pd.DataFrame.from_dict({filename: result}).T
        df.index.name = "full_file_path"
        df = df.reset_index()
        df["filename"] = base_filename

        df_lists["frame_gradients"].append(df)

    result = {}
    for key, l in df_lists.items():
        if l:
            result[key] = to_numeric(pd.concat(l)).reset_index()
    return result


# @timer
def get_gradient_data(data, with_quadratic=True, n_samples=50):
    bin_size = int(data.shape[0] / n_samples)
    data_reduced = skimage.measure.block_reduce(
        data, (bin_size, bin_size), np.median, cval=1e9
    )
    aspect_ratio = data.shape[0] / data.shape[1]
    ny, nx = data_reduced.shape
    y_vals = np.linspace(-1, 1, ny)
    x_vals = np.linspace(-aspect_ratio, aspect_ratio, nx)
    df1 = pd.DataFrame(data_reduced, columns=x_vals, index=y_vals)
    df1 = df1.stack().to_frame("value")
    df1.index.names = ["x", "y"]
    threshold = df1.median() + (df1.quantile(0.75) - df1.quantile(0.25)) * 1.5
    df1[df1 > threshold] = np.nan
    df_binned = df1.unstack(1)

    df2 = df1.dropna().reset_index()

    y = df2[["value"]]
    X = df2[["x", "y"]]
    if with_quadratic:
        X["x2"] = X["x"] ** 2
        X["y2"] = X["y"] ** 2
        X["xy"] = X["x"] * X["y"]

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const)
    res = model.fit()

    df_pred = pd.DataFrame(res.predict(), columns=["value"])
    df_pred["x"] = df2["x"]
    df_pred["y"] = df2["y"]

    df_pred = df_pred.set_index(["x", "y"]).unstack(1)

    df_residual = df_binned - df_pred

    return res, df_binned, df_pred, df_residual


# @timer
def process_image_data(
    data, tnpix_threshold=4, extract_thresh=3, filter_kernel=None, use_simple=True
):
    data = data.astype(float)
    ny, nx = data.shape
    if use_simple:
        bkg = np.mean(data)
        err = bkg
        globalbkg = bkg
        globalrms = 0.0  # np.std(data)
    else:
        bkg = sep.Background(data)
        err = bkg.back()
        globalbkg = bkg.globalback
        globalrms = bkg.globalrms
    data_sub = data - bkg

    try:
        objects = sep.extract(
            data_sub, extract_thresh, err=err, filter_kernel=filter_kernel
        )
    except:
        log.error("Issue with processing image")
        return pd.DataFrame()
    objects = pd.DataFrame(objects)
    objects = objects.drop(["xmin", "xmax", "ymin", "ymax"], axis=1)
    objects["r_eff"] = np.sqrt(objects["a"] ** 2 + objects["b"] ** 2) / np.sqrt(2)
    object_selection = objects["tnpix"] > tnpix_threshold
    object_selection &= objects["r_eff"] < 10
    object_selection &= objects["r_eff"] < objects["r_eff"].quantile(0.95)
    objects = objects[object_selection]

    coef = 2 * np.sqrt(2 * np.log(2))
    objects["fwhm"] = coef * objects["r_eff"]
    objects["eccentricity"] = np.sqrt(1 - (objects["b"] / objects["a"]) ** 2)
    objects["ellipticity"] = 1 - objects["b"] / objects["a"]
    objects["elongation"] = objects["a"] / objects["b"]
    objects["theta"] = -objects["theta"] * 180 / np.pi
    objects["x_ref"] = objects["x"] - nx / 2
    objects["y_ref"] = -(objects["y"] - ny / 2)
    objects["chip_theta"] = np.arctan2(objects["y_ref"], objects["x_ref"]) * 180 / np.pi
    objects["chip_r"] = np.sqrt(objects["x_ref"] ** 2 + objects["y_ref"] ** 2)
    objects["bkg_val"] = globalbkg
    objects["bkg_rms"] = globalrms
    return objects


# @timer
def process_image_from_filename(
    filename, tnpix_threshold=6, extract_thresh=3, filter_kernel=None, use_simple=True
):
    try:
        data = fits.getdata(filename, header=False)
        objects = process_image_data(
            data,
            tnpix_threshold,
            extract_thresh,
            filter_kernel=filter_kernel,
            use_simple=use_simple,
        )
        if objects.shape[0] == 0:
            log.info(f"No stars found for {filename}")
            # add_file_to_skiplist(filename)
        objects["full_file_path"] = filename
        objects["filename"] = os.path.basename(filename)
        objects["nx"] = data.shape[0]
        objects["ny"] = data.shape[1]
    except:
        log.info(
            f"Problem processing image {filename}",
            exc_info=EXC_INFO,
        )
        # add_file_to_skiplist(filename)
        return pd.DataFrame()
    return objects


def preprocess_stars(df_s, xy_n_bins=10, r_n_bins=20, nx=None, ny=None):
    df_s["chip_r_bin"] = df_s["chip_r"] // int(df_s["chip_r"].max() / r_n_bins)

    df_s["vec_u"] = np.cos(df_s["theta"] * np.pi / 180) * df_s["ellipticity"]
    df_s["vec_v"] = np.sin(df_s["theta"] * np.pi / 180) * df_s["ellipticity"]

    y_max = df_s["y_ref"].max()
    x_max = df_s["x_ref"].max()
    if ny:
        y_max = ny / 2
    if nx:
        x_max = nx / 2
    df_s["x_bin"] = np.round(
        np.round((df_s["x_ref"] / y_max) * xy_n_bins) * y_max / xy_n_bins
    )
    df_s["y_bin"] = np.round(
        np.round(((df_s["y_ref"]) / y_max) * xy_n_bins) * y_max / xy_n_bins
    )
    # df_s["x_bin"] = np.round(((df_s["x_ref"] + x_max) / (2 * x_max)) * 3)
    # df_s["y_bin"] = np.round(((df_s["y_ref"] + y_max) / (2 * y_max)) * 3)
    return df_s


def bin_stars(df_s, filename, tnpix_min=6):
    selection = df_s["tnpix"] > tnpix_min
    if df_s["is_saturated"].mean() < 1:
        selection &= ~df_s["is_saturated"]
    df_stars = df_s[selection]

    group_xy = df_stars.groupby(["x_bin", "y_bin"])
    # df_radial = flatten_cols(
    #     group_r.agg(
    #         {"fwhm": "describe", "ellipticity": "describe", "chip_r": "describe"}
    #     )
    # )

    # df_radial.columns = [col.replace("%", "_pct") for col in df_radial.columns]

    # df_radial["full_file_path"] = filename
    # df_radial["filename"] = os.path.basename(filename)

    df_xy = group_xy.agg(
        {
            "vec_u": "mean",
            "vec_v": "mean",
            "fwhm": "median",
            "eccentricity": "median",
            "tnpix": "count",
            "a": "median",
            "b": "median",
        }
    ).reset_index()
    df_xy.rename({"tnpix": "star_count"}, axis=1, inplace=True)
    df_xy["x_ref"] = df_xy["x_bin"] - df_xy["x_bin"].mean()
    df_xy["y_ref"] = df_xy["y_bin"] - df_xy["y_bin"].mean()
    df_xy["chip_theta"] = np.arctan2(df_xy["y_ref"], df_xy["x_ref"]) * 180 / np.pi
    df_xy["theta"] = np.arctan2(df_xy["vec_u"], df_xy["vec_v"]) * 180 / np.pi

    df_xy["dot"] = df_xy["x_ref"] * df_xy["vec_u"] + df_xy["y_ref"] * df_xy["vec_v"]

    df_xy["chip_r"] = np.sqrt(df_xy["x_ref"] ** 2 + df_xy["y_ref"] ** 2)
    df_xy["ellipticity"] = np.sqrt(df_xy["vec_u"] ** 2 + df_xy["vec_v"] ** 2)
    df_xy["dot_norm"] = df_xy["dot"] / df_xy["ellipticity"] / df_xy["chip_r"]

    df_xy["full_file_path"] = filename
    df_xy["filename"] = os.path.basename(filename)

    return df_xy


# @timer
def aggregate_stars(df_stars):
    if df_stars.shape[0] == 0:
        return pd.DataFrame()
    df_stars["log_flux"] = np.log10(df_stars["flux"])
    df_stars["amplitude"] = df_stars["cpeak"] / 2**16
    df_stars["is_saturated"] = df_stars["amplitude"] > 0.85
    df_stars["star_fraction_saturated"] = df_stars["is_saturated"].astype(float)
    df3 = (
        df_stars.groupby("filename").agg({"star_fraction_saturated": "mean"})
    ).astype(float)
    df_stars = df_stars[~df_stars["is_saturated"]]
    df0 = (
        df_stars[
            [
                "filename",
                "full_file_path",
                "tnpix",
                "theta",
                "log_flux",
                "fwhm",
                "eccentricity",
                "chip_theta",
            ]
        ]
        .groupby(["filename", "full_file_path"])
        .agg(["mean", "std", "median"])
    )
    df0.columns = ["_".join(col).strip() for col in df0.columns.values]
    df1 = (
        df_stars[["filename", "full_file_path", "bkg_val", "bkg_rms"]]
        .groupby(["filename", "full_file_path"])
        .mean()
    )
    df2 = (
        df_stars[["filename", "full_file_path", "bkg_val", "bkg_rms"]]
        .groupby(["filename", "full_file_path"])
        .size()
        .to_frame("n_stars")
    )
    df0 = df0.join(df1).join(df2).join(df3)

    df0.index.names = ["filename", "full_file_path"]

    return df0
