import glob
import os
import sep
import yaml
import logging
import sqlalchemy
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from sqlalchemy import Table, MetaData
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
from multiprocessing import Pool
from functools import partial
from pathlib import Path

warnings.filterwarnings("ignore", category=AstropyUserWarning)


POSTGRES_USER = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "fits_files")
PGPORT = os.getenv("PGPORT", "5432")
PGHOST = os.getenv("PGHOST", "0.0.0.0")

POSTGRES_ENGINE = sqlalchemy.create_engine(
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{PGHOST}:{PGPORT}/{POSTGRES_DB}"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(module)s %(message)s")
log = logging.getLogger(__name__)


base_dir = Path(__file__).parents[2]
with open(f"{base_dir}/conf/config.yml", "r") as f:
    CONFIG = yaml.safe_load(f)

DATA_DIR = os.getenv("DATA_DIR", "/data")
TARGET_DIR = os.getenv("TARGET_DIR", "/targets")

sep.set_extract_pixstack(1000000)

file_blacklist = []


def get_fits_file_list(data_dir, config):
    fits_file_list = []
    if "fits_file_patterns" in config:
        for file_pattern in config["fits_file_patterns"]["allow"]:
            glob_pattern = f"{data_dir}/{file_pattern}"
            fits_file_list += list(glob.iglob(glob_pattern, recursive=True))
        for reject_file_pattern in config["fits_file_patterns"]["reject"]:
            fits_file_list = [
                f for f in fits_file_list if reject_file_pattern not in f.lower()
            ]
    return fits_file_list


def aggregate_stars(df_stars):
    df_stars["log_flux"] = np.log10(df_stars["flux"])
    df0 = (
        df_stars[
            [
                "filename",
                "tnpix",
                "theta",
                "log_flux",
                "fwhm",
                "eccentricity",
                "chip_theta",
            ]
        ]
        .groupby(["filename"])
        .agg(["mean", "std"])
    )
    df0.columns = ["_".join(col).strip() for col in df0.columns.values]
    df1 = df_stars[["filename", "bkg_val", "bkg_rms"]].groupby(["filename"]).mean()
    df2 = (
        df_stars[["filename", "bkg_val", "bkg_rms"]]
        .groupby(["filename"])
        .size()
        .to_frame("n_stars")
    )
    df0 = df0.join(df1).join(df2)

    df0.index.name = "filename"

    return df0


def show_star(data, df_stars, i_row, window=20):
    star = df_stars.iloc[i_row]
    x_min = int(star["x"] - window)
    x_max = int(star["x"] + window)
    y_min = int(star["y"] - window)
    y_max = int(star["y"] + window)
    plt.imshow(data.T[x_min:x_max, y_min:y_max])


def push_rows_to_table(df0, engine, table_name, if_exists="append", index=False):
    try:
        df0.to_sql(table_name, engine, if_exists=if_exists, index=index)
        n_rows = df0.shape[0]
        log.debug(f"Added {n_rows} new entries")
    except:
        df_current = pd.read_sql(f"select * from {table_name}", engine)
        df_combined = pd.concat([df_current, df0])
        try:
            clear_table(engine, table_name)
            df_combined.to_sql(table_name, engine, if_exists="replace", index=index)
        except:
            log.info("Failed", exc_info=True)
            return None
        n_rows = df0.shape[0]
        log.debug(f"Added {n_rows} new entries")
        n_rows = df_current.shape[0]
        log.debug(f"Modified {n_rows} existing entries")


def coord_str_to_vec(coord_string):
    coord_string = str(coord_string)
    for replace_string in ["h", "m", "s", "d"]:
        coord_string = coord_string.replace(replace_string, "")
    coord_vec = coord_string.split()
    coord_vec = [float(entry) for entry in coord_vec]
    return coord_vec


def coord_str_to_float(coord_string):
    coord_vec = coord_str_to_vec(coord_string)
    result = 0
    for i, val in enumerate(coord_vec):
        result += val / 60 ** i
    return result


def process_header_from_fits(filename):
    try:
        hdul = fits.open(filename)
        header = dict(hdul[0].header)
        file_base = os.path.basename(filename)
        file_dir = os.path.dirname(filename)
        df_header = pd.DataFrame({file_base: header}).T
        df_header.index.name = "filename"
        df_header.reset_index(inplace=True)
        df_header["file_dir"] = file_dir
        df_header["file_full_path"] = filename
        df_header["filename"] = file_base
        fl_cols = ["FOCALLEN", "EFF_FL"]
        for fl_col in fl_cols:
            if fl_col in df_header.columns:
                break
        if fl_col in df_header.columns:
            df_header["arcsec_per_pixel"] = (
                df_header["XPIXSZ"] * 206 / df_header[fl_col]
            )
        else:
            df_header["arcsec_per_pixel"] = np.nan
        if "COMMENT" in df_header.columns:
            df_header["COMMENT"] = df_header["COMMENT"].apply(lambda c: str(c))
        return df_header
    except KeyboardInterrupt:
        raise ("Stopping...")
    except:
        log.info(f"Issue with file {filename}", exc_info=True)
        return pd.DataFrame()


def process_headers(file_list):
    df_header_list = []

    for file in file_list:
        log.info(f"Processing header for file {file}")
        df_header_list.append(process_header_from_fits(file))
    df_headers = pd.DataFrame()
    if df_header_list:
        df_headers = pd.concat(df_header_list)

    for col in ["OBJCTRA", "OBJCTDEC", "OBJCTALT", "OBJCTAZ"]:
        if col in df_headers.columns:
            df_headers[col] = df_headers[col].apply(coord_str_to_float)
    return df_headers


def check_status(file_list, **status_args):
    df_status_list = []
    for filename in file_list:
        status_dict = {}
        file_base = os.path.basename(filename)
        file_dir = os.path.dirname(filename)
        status_dict.update(
            dict(file_full_path=filename, file_dir=file_dir, filename=file_base)
        )
        for status, matches in status_args.items():
            for match in matches:
                is_match = match in filename
                status_dict.update({status: is_match})
        df_status_list.append(status_dict)
    df_status = pd.DataFrame()
    if df_status_list:
        df_status = pd.DataFrame(df_status_list)
    return df_status


def check_if_table_exists(engine, table_name):
    conn = engine.raw_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"""select * from information_schema.tables where table_name='{table_name}'"""
        )
        table_exists = bool(cursor.rowcount)
        cursor.close()
        conn.commit()
        return table_exists
    finally:
        conn.close()


def check_file_in_table(file_list, engine, table_name):
    try:
        df = pd.read_sql(table_name, engine)
        new_files = [
            file
            for file in file_list
            if os.path.basename(file) not in df["filename"].values
        ]
    except:
        new_files = file_list
    return new_files


def clear_table(engine, table_name):
    log.debug(f"Clearing table {table_name}")
    metadata = MetaData()
    if engine.has_table(table_name):
        t = Table(table_name, metadata)
        t.drop(engine)

    log.debug("Finished")


def process_stars_from_fits(filename, extract_thresh=3):
    df_agg_stars, df_stars = pd.DataFrame(), pd.DataFrame()
    try:
        df_stars = process_image_from_filename(filename, extract_thresh=extract_thresh)
        df_agg_stars = aggregate_stars(df_stars)
        df_agg_stars["extract_thresh"] = extract_thresh
        return df_agg_stars, df_stars
    except KeyboardInterrupt:
        raise ("Stopping...")
    except:
        log.info(f"Issue with file {filename}", exc_info=True)
        return df_agg_stars, df_stars


def process_stars(
    file_list,
    multithread_fits_read=False,
    n_threads=8,
    extract_thresh=1.5,
    xy_n_bins=None,
):
    df_lists = defaultdict(list)

    for filename in file_list:
        log.info(f"Starting to process stars from {filename}")
        df_agg_stars, df_stars = process_stars_from_fits(
            filename, extract_thresh=extract_thresh
        )
        if filename in file_blacklist:
            continue

        nx = df_stars["nx"].values[0]
        ny = df_stars["ny"].values[0]

        n_stars = df_stars.shape[0]
        log.info(f"For {filename}: N-stars = {n_stars}")

        n_stars = df_stars.shape[0]
        if n_stars == 0:
            base_filename = os.path.basename(filename)
            df_stars["filename"] = base_filename
            df_agg_stars = df_stars.copy()
            df_xy = df_stars.copy()
            df_radial = df_stars.copy()

            df_lists["stars"].append(df_stars)
            df_lists["agg_stars"].append(df_agg_stars)
            df_lists["radial_frame"].append(df_radial)
            df_lists["xy_frame"].append(df_xy)

            continue
        if not xy_n_bins:
            xy_n_bins = max(min(int(np.sqrt(n_stars) / 8), 8), 3)
        df_stars = preprocess_stars(df_stars, xy_n_bins=xy_n_bins, nx=nx, ny=ny)

        df_radial, df_xy = bin_stars(df_stars, filename)
        trail_strength = np.sqrt(
            (df_xy["vec_u"].mean() / df_xy["vec_u"].std()) ** 2
            + (df_xy["vec_v"].mean() / df_xy["vec_v"].std()) ** 2
        )
        df_agg_stars["star_trail_strength"] = trail_strength
        df_agg_stars["star_orientation_score"] = np.abs(df_xy["dot_norm"]).median()

        df_lists["stars"].append(df_stars)
        df_lists["agg_stars"].append(df_agg_stars)
        df_lists["radial_frame"].append(df_radial)
        df_lists["xy_frame"].append(df_xy)

    result = {}
    for key, l in df_lists.items():
        if l:
            result[key] = to_numeric(pd.concat(l)).reset_index()
    return result


def process_image_data(data, tnpix_threshold=4, extract_thresh=3, filter_kernel=None):
    data = data.astype(float)
    ny, nx = data.shape
    bkg = sep.Background(data)
    data_sub = data - bkg

    try:
        objects = sep.extract(
            data_sub, extract_thresh, err=bkg.back(), filter_kernel=filter_kernel
        )
    except:
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
    objects["theta"] = -objects["theta"]
    objects["x_ref"] = objects["x"] - nx / 2
    objects["y_ref"] = -(objects["y"] - ny / 2)
    objects["chip_theta"] = np.arctan2(objects["y_ref"], objects["x_ref"])
    objects["chip_r"] = np.sqrt(objects["x_ref"] ** 2 + objects["y_ref"] ** 2)
    objects["bkg_val"] = bkg.globalback
    objects["bkg_rms"] = bkg.globalrms
    return objects


def process_image_from_filename(
    filename, tnpix_threshold=6, extract_thresh=3, filter_kernel=None
):
    data = fits.getdata(filename, header=False)
    objects = process_image_data(
        data, tnpix_threshold, extract_thresh, filter_kernel=filter_kernel
    )
    if objects.shape[0] == 0:
        file_blacklist.append(filename)
    objects["filename_with_path"] = filename
    objects["filename"] = os.path.basename(filename)
    objects["nx"] = data.shape[0]
    objects["ny"] = data.shape[1]
    return objects


def init_tables():
    clear_tables(
        [
            "fits_headers",
            "fits_status",
            "aggregated_star_metrics",
            "xy_frame_metrics",
            "radial_frame_metrics",
            "targets",
            "target_status",
        ]
    )


def clear_tables(table_names):
    for table_name in table_names:
        clear_table(POSTGRES_ENGINE, table_name)


def file_has_data(filename):
    return os.stat(filename).st_size != 0


def get_file_list_with_data(file_list):
    return [filename for filename in file_list if file_has_data(filename)]


def update_fits_status(config=CONFIG, data_dir=DATA_DIR, file_list=None):
    if not file_list:
        file_list = get_fits_file_list(data_dir, config)
    new_files = check_file_in_table(file_list, POSTGRES_ENGINE, "fits_status")
    files_with_data = get_file_list_with_data(new_files)
    if files_with_data:
        for filename in files_with_data:
            log.info(f"Found new file: {filename}")
        df_status = check_status(files_with_data, reject=["reject"], cloud=["cloud"])
        df_status = to_numeric(df_status)
        push_rows_to_table(
            df_status, POSTGRES_ENGINE, table_name="fits_status", if_exists="append"
        )


def to_numeric(df0):
    for col in df0.columns:
        try:
            df0[col] = df0[col].apply(pd.to_numeric)
        except:
            continue
    return df0


def update_fits_headers(config=CONFIG, data_dir=DATA_DIR, file_list=None):
    if not file_list:
        file_list = get_fits_file_list(data_dir, config)
    new_files = check_file_in_table(file_list, POSTGRES_ENGINE, "fits_headers")
    files_with_data = get_file_list_with_data(new_files)
    n_files = len(files_with_data)
    if n_files > 0:
        log.info(f"Found {n_files} new files for headers")
    if files_with_data:
        for files in chunks(files_with_data, 100):
            df_header = process_headers(files)
            df_header = to_numeric(df_header)
            log.info("Pushing to db")
            push_rows_to_table(
                df_header,
                POSTGRES_ENGINE,
                table_name="fits_headers",
                if_exists="append",
            )
            log.info("Done")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def update_star_metrics(
    config=CONFIG, data_dir=DATA_DIR, file_list=None, n_chunk=16, extract_thresh=0.25
):
    if not file_list:
        file_list = get_fits_file_list(data_dir, config)
    new_files = check_file_in_table(
        file_list, POSTGRES_ENGINE, "aggregated_star_metrics"
    )
    files_with_data = get_file_list_with_data(new_files)

    files_to_process = [f for f in files_with_data if f not in file_blacklist]

    n_threads = int(CONFIG.get("threads_for_star_processing", 1))
    multithread_fits_read = n_threads > 1

    if files_to_process:
        for file_set in list(chunks(files_to_process, n_chunk)):
            if multithread_fits_read:
                with Pool(n_threads) as p:
                    l_result = list(
                        p.imap(
                            partial(process_stars, extract_thresh=extract_thresh),
                            [[f] for f in file_set],
                        )
                    )
                result = defaultdict(list)
                for r in l_result:
                    for k, v in r.items():
                        result[k].append(v)
                for k, v in result.items():
                    result[k] = pd.concat(v)

            else:
                result = process_stars(file_set, extract_thresh=extract_thresh)

            if "stars" not in result:
                continue
            n_stars = result["stars"].shape[0]

            log.info(f"New stars: {n_stars}")

            # Aggregate star metrics table
            push_rows_to_table(
                result["agg_stars"],
                POSTGRES_ENGINE,
                table_name="aggregated_star_metrics",
                if_exists="append",
            )

            # Radial table
            push_rows_to_table(
                result["radial_frame"],
                POSTGRES_ENGINE,
                table_name="radial_frame_metrics",
                if_exists="append",
            )

            # XY table
            push_rows_to_table(
                result["xy_frame"],
                POSTGRES_ENGINE,
                table_name="xy_frame_metrics",
                if_exists="append",
            )

            if n_stars > 0:
                log.info(
                    f"Finished pushing {n_stars} stars from {len(file_set)} files to tables"
                )


def update_db_with_matching_files(config=CONFIG, data_dir=DATA_DIR, file_list=None):
    log.debug("Checking for new files")
    update_fits_headers(config, data_dir, file_list)
    update_fits_status(config, data_dir, file_list)
    update_star_metrics(config, data_dir, file_list)


def lower_cols(df):
    df.columns = [col.lower() for col in df.columns]
    return df


def preprocess_stars(df_s, xy_n_bins=10, r_n_bins=20, nx=None, ny=None):

    df_s["chip_r_bin"] = df_s["chip_r"] // int(df_s["chip_r"].max() / r_n_bins)

    df_s["vec_u"] = np.cos(df_s["theta"]) * df_s["ellipticity"]
    df_s["vec_v"] = np.sin(df_s["theta"]) * df_s["ellipticity"]

    y_max = df_s["y_ref"].max()
    if ny:
        y_max = ny / 2
    df_s["x_bin"] = np.round(
        np.round((df_s["x_ref"] / y_max) * xy_n_bins) * y_max / xy_n_bins
    )
    df_s["y_bin"] = np.round(
        np.round(((df_s["y_ref"]) / y_max) * xy_n_bins) * y_max / xy_n_bins
    )
    return df_s


def flatten_cols(df):
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    return df


def bin_stars(df_s, filename, tnpix_min=6):
    df_stars = df_s[df_s["tnpix"] > tnpix_min]

    group_r = df_stars.groupby("chip_r_bin")
    group_xy = df_stars.groupby(["x_bin", "y_bin"])
    df_radial = flatten_cols(
        group_r.agg(
            {"fwhm": "describe", "ellipticity": "describe", "chip_r": "describe"}
        )
    )

    df_radial.columns = [col.replace("%", "_pct") for col in df_radial.columns]

    df_radial["filename_with_path"] = filename
    df_radial["filename"] = os.path.basename(filename)

    df_xy = group_xy.agg(
        {
            "vec_u": "mean",
            "vec_v": "mean",
            "fwhm": "median",
            "eccentricity": "median",
            "tnpix": "count",
        }
    ).reset_index()
    df_xy.rename({"tnpix": "star_count"}, axis=1, inplace=True)
    df_xy["x_ref"] = df_xy["x_bin"] - df_xy["x_bin"].mean()
    df_xy["y_ref"] = df_xy["y_bin"] - df_xy["y_bin"].mean()
    df_xy["chip_theta"] = np.arctan2(df_xy["y_ref"], df_xy["x_ref"])
    df_xy["theta"] = np.arctan2(df_xy["vec_u"], df_xy["vec_v"])

    df_xy["dot"] = df_xy["x_ref"] * df_xy["vec_u"] + df_xy["y_ref"] * df_xy["vec_v"]

    df_xy["chip_r"] = np.sqrt(df_xy["x_ref"] ** 2 + df_xy["y_ref"] ** 2)
    df_xy["ellipticity"] = np.sqrt(df_xy["vec_u"] ** 2 + df_xy["vec_v"] ** 2)
    df_xy["dot_norm"] = df_xy["dot"] / df_xy["ellipticity"] / df_xy["chip_r"]

    df_xy["filename_with_path"] = filename
    df_xy["filename"] = os.path.basename(filename)

    return df_radial, df_xy
