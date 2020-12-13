import glob
import os
import sep
import yaml

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sqlalchemy import Table, MetaData
from pathlib import Path
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
import logging

import sqlalchemy

import warnings

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


app_dir = str(Path(__file__).parents[1])
with open(f"{app_dir}/conf/config.yml", "r") as f:
    CONFIG = yaml.load(f, Loader=yaml.BaseLoader)
DATA_DIR = os.getenv("DATA_DIR", "/data")


sep.set_extract_pixstack(1000000)


def get_fits_file_list(data_dir, config):
    fits_file_list = []
    if "fits_file_patterns" in config:
        for file_pattern in config["fits_file_patterns"]:
            fits_file_list += list(glob.iglob(f"{data_dir}/{file_pattern}"))
    return fits_file_list


def aggregate_stars(df_stars):
    df_stars["log_flux"] = np.log10(df_stars["flux"])
    df0 = (
        df_stars[
            ["filename", "tnpix", "theta", "log_flux", "fwhm", "ecc", "chip_theta"]
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

    cols_for_corr = [
        "npix",
        "tnpix",
        "x",
        "y",
        "x2",
        "y2",
        "xy",
        "errx2",
        "erry2",
        "errxy",
        "a",
        "b",
        "theta",
        "cxx",
        "cyy",
        "cxy",
        "cflux",
        "flux",
        "cpeak",
        "peak",
        "xcpeak",
        "ycpeak",
        "xpeak",
        "ypeak",
        "fwhm",
        "ecc",
        "ellipticity",
        "elongation",
        "chip_theta",
        "chip_r",
    ]

    df_corr = df_stars[cols_for_corr].corr().stack().drop_duplicates()
    df_corr = df_corr[df_corr < 1]
    df_corr = df_corr.to_frame("corr").T
    df_corr.columns = [
        f"corr__{'__'.join(col)}".strip() for col in df_corr.columns.values
    ]
    df_corr.index = df0.index

    df0 = pd.concat([df0, df_corr], axis=1)
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
            log.info("Failed")
            return None
        n_rows = df0.shape[0]
        log.debug(f"Added {n_rows} new entries")
        n_rows = df_current.shape[0]
        log.debug(f"Modified {n_rows} existing entries")


def coord_str_to_vec(coord_string):
    coord_string = str(coord_string)
    for replace_string in ["h", "m", "s"]:
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
        header = hdul[0].header
        file_base = os.path.basename(filename)
        file_dir = os.path.dirname(filename)
        df_header = pd.DataFrame({file_base: dict(header)}).T
        df_header.index.name = "filename"
        df_header.reset_index(inplace=True)
        df_header["file_dir"] = file_dir
        df_header["file_full_path"] = filename
        df_header["filename"] = file_base
        df_header["arcsec_per_pixel"] = (
            df_header["XPIXSZ"] * 206 / df_header["FOCALLEN"]
        )
        if "COMMENT" in df_header.columns:
            df_header["COMMENT"] = df_header["COMMENT"].apply(lambda c: str(c))
        return df_header
    except KeyboardInterrupt:
        raise ("Stopping...")
    except:
        log.info(f"Issue with file {filename}", exc_info=True)


def process_headers(file_list):
    df_header_list = []

    for file in file_list:
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
        # table_exists = check_if_table_exists(engine, table_name)
        # if table_exists:
        #     log.warning("Issue reading table", exc_info=True)
        new_files = file_list
    log.debug(f"New files seen: {len(new_files)}")
    return new_files


def clear_table(engine, table_name):
    log.debug(f"Clearing table {table_name}")
    metadata = MetaData()
    if engine.has_table(table_name):
        t = Table(table_name, metadata)
        t.drop(engine)

    log.debug("Finished")


def process_stars_from_fits(filename, extract_thresh=3, with_stars=False):
    try:
        df_stars = process_image_from_filename(filename, extract_thresh=extract_thresh)
        if df_stars.shape[0] == 0:
            df_agg_stars = process_stars_from_fits(
                filename, extract_thresh=extract_thresh - 0.5, with_stars=with_stars
            )
            return df_agg_stars
        df_agg_stars = aggregate_stars(df_stars)
        df_agg_stars["extract_thresh"] = extract_thresh
        if with_stars:
            return df_agg_stars, df_stars
        return df_agg_stars
    except KeyboardInterrupt:
        raise ("Stopping...")
    except:
        log.info(f"Issue with file {filename}", exc_info=True)


def process_stars(file_list):
    df_stars_list = []
    df_stars = pd.DataFrame()
    for file in file_list:
        df_stars_list.append(process_stars_from_fits(file))
    if df_stars_list:
        df_stars = pd.concat(df_stars_list).reset_index()
    return df_stars


def process_image_data(data, header, tnpix_threshold=4, extract_thresh=3):
    data = data.astype(float)
    ny, nx = data.shape
    bkg = sep.Background(data)
    data_sub = data  # - bkg

    objects = sep.extract(data_sub, extract_thresh, err=bkg.back(), filter_kernel=None)

    objects = pd.DataFrame(objects)
    objects = objects[objects["tnpix"] > tnpix_threshold]

    coef = 2 * np.sqrt(2 * np.log(2))
    objects["fwhm"] = coef * np.sqrt(objects["a"] ** 2 + objects["b"] ** 2) / np.sqrt(2)
    objects["ecc"] = np.sqrt(1 - (objects["b"] / objects["a"]) ** 2)
    objects["ellipticity"] = 1 - objects["b"] / objects["a"]
    objects["elongation"] = objects["a"] / objects["b"]
    objects["theta"] = -objects["theta"]  # + np.pi / 2
    objects["x_ref"] = objects["x"] - nx / 2
    objects["y_ref"] = -(objects["y"] - ny / 2)
    objects["chip_theta"] = np.arctan2(objects["y_ref"], objects["x_ref"])
    objects["chip_r"] = np.sqrt(objects["x_ref"] ** 2 + objects["y_ref"] ** 2)
    objects["bkg_val"] = bkg.globalback
    objects["bkg_rms"] = bkg.globalrms
    return objects


def process_image_from_filename(filename, tnpix_threshold=4, extract_thresh=3):
    data, header = fits.getdata(filename, header=True)
    objects = process_image_data(data, header, tnpix_threshold, extract_thresh)
    objects["filename_with_path"] = filename
    objects["filename"] = os.path.basename(filename)
    return objects


def init_tables():
    clear_all_tables(["fits_headers", "fits_status", "star_metrics"])


def clear_all_tables(table_names):
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
    if files_with_data:
        df_header = process_headers(files_with_data)
        df_header = to_numeric(df_header)
        push_rows_to_table(
            df_header, POSTGRES_ENGINE, table_name="fits_headers", if_exists="append"
        )


def update_star_metrics(config=CONFIG, data_dir=DATA_DIR, file_list=None):
    if not file_list:
        file_list = get_fits_file_list(data_dir, config)
    new_files = check_file_in_table(file_list, POSTGRES_ENGINE, "star_metrics")
    files_with_data = get_file_list_with_data(new_files)
    if files_with_data:
        df_stars = process_stars(files_with_data)
        df_stars = to_numeric(df_stars)
        push_rows_to_table(
            df_stars, POSTGRES_ENGINE, table_name="star_metrics", if_exists="append"
        )


def update_db_with_matching_files(config=CONFIG, data_dir=DATA_DIR, file_list=None):
    log.debug("Updating db with matching files")
    update_fits_headers(config, data_dir, file_list)
    update_fits_status(config, data_dir, file_list)
    update_star_metrics(config, data_dir, file_list)
    log.debug("Finished")
