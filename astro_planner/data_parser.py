import os
import logging
import datetime
import glob
import yaml

from astropy.io import fits
from functools import lru_cache
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .logger import log

from .profile import cleanup_name

DATA_DIR = os.getenv("DATA_DIR", "/Volumes/Users/gshau/Dropbox/AstroBox/data/")

FILTERS = ["L", "R", "G", "B", "Ha", "OIII", "SII", "OSC"]

with open("./conf/config.yml", "r") as f:
    CONFIG = yaml.load(f, Loader=yaml.BaseLoader)


def parse_filename(file_name):
    file_root = Path(file_name).stem
    if "_LIGHT_" in file_root:
        file_root = file_root.replace("__BayerMatrix__", "_OSC_")
        target_name, metadata = file_root.split("_LIGHT_")
        metadata = metadata.split("_")
        elements = ["LIGHT", target_name] + metadata

    el_names = [
        "type",
        "target",
        "filter",
        "sub_exposure",
        "bin",
        "temp",
        "seq_num",
        "date",
        "time",
        "ms",
    ]

    d_info = dict(zip(el_names, elements))

    d_info["filename"] = file_name
    d_info["target"] = cleanup_name(target_name)
    d_info["sub_exposure"] = float(d_info["sub_exposure"].replace("s", ""))
    d_info["bin"] = int(d_info["bin"].replace("BIN", ""))

    d_info["datetime"] = datetime.datetime.strptime(
        "{}{}".format(d_info["date"], d_info["time"]), "%Y%m%d%H%M%S"
    )

    d_info.pop("date")
    d_info.pop("time")
    d_info.pop("ms")

    return d_info


def read_fits_header(fits_filename):
    hdul = fits.open(fits_filename)
    return dict(hdul[0].header)


def _parse_file(file_name, root_key):
    try:
        d_info = read_fits_header(file_name)
        d_info["filename"] = file_name
        return d_info
    except Exception:
        logging.info("Skipping {}".format(file_name), exc_info=True)


def parse_filelist(file_list, root_key="data/", verbose=False):
    d_list = []
    log.info("Reading stored fits files")
    for file_name in tqdm(file_list):
        d_list.append(_parse_file(file_name, root_key))
    d_list = [d for d in d_list if d]
    logging.info("Read {} files".format(len(d_list)))

    return d_list


def get_file_list(data_dir=DATA_DIR):
    file_list = []
    if "fits_file_patterns" in CONFIG:
        for file_pattern in CONFIG["fits_file_patterns"]:
            file_list += list(glob.iglob(f"{data_dir}{file_pattern}"))

    return tuple(file_list)


@lru_cache(maxsize=32)
def _get_data_info(file_list):
    df_list = parse_filelist(file_list)
    df_files = pd.DataFrame(df_list)
    return df_files


def get_data_info(data_dir=DATA_DIR):
    file_list = get_file_list(data_dir)
    if file_list:
        df_files = _get_data_info(file_list)
        df_files = clean_file_list(df_files)
        df_files = format_dates(df_files)
        return df_files
    return pd.DataFrame()


def format_dates(df):
    df.loc[:, "date"] = df["DATE-OBS"].apply(format_date_string)
    sel = ~df["DATE-OBS"].isnull()
    df.loc[sel, "date"] = pd.to_datetime(
        df.loc[sel, "DATE-OBS"].apply(fix_date_fmt)
    ).apply(lambda t: str(t.date() - datetime.timedelta(hours=12)))
    return df


def get_exposure_summary(df_files, filter_list=FILTERS, time_format="minutes"):
    df_exposures = df_files.groupby(["OBJECT", "FILTER"]).sum()["EXPOSURE"].to_frame()

    df = pd.pivot(
        data=df_exposures.reset_index(), columns="FILTER", index="OBJECT"
    ).fillna(0)["EXPOSURE"]

    df = df[[filter for filter in filter_list if filter in df.columns]]

    df = df[df.sum(axis=1) > 0]

    scale = 1
    if time_format == "seconds":
        scale = 1
    if time_format == "minutes":
        scale = 60
    if time_format == "hours":
        scale = 3600

    return df / scale


def format_name(name):
    name = name.lower()
    name = name.replace(" ", "_")
    if "sh2" not in name:
        name = name.replace("-", "_")
    catalogs = ["ngc", "abell", "ic", "vdb", "ldn"]

    for catalog in catalogs:
        if catalog in name[: len(catalog)]:
            if f"{catalog}_" in name:
                continue
            number = name.replace(catalog, "")
            name = f"{catalog}_{number}"
    return name


def filter_map(filter_in):
    filters_to_replace = dict(
        Clear="L", Red="R", Green="G", Blue="B", SIII="SII", Luminance="L"
    )
    if filter_in in filters_to_replace:
        return filters_to_replace[filter_in]
    return filter_in


def equinox_ccdfname_parser(string):
    try:
        split_string = string.split(".")
        r = dict(zip(["OBJECT", "IMAGETYP"], split_string[:2]))
        exposure, remain = split_string[2].split("S", 1)
        temp, remain = remain.split("X")
        bin = remain[0]
        filter = remain[1:]
        r.update(
            {
                "EXPOSURE": int(exposure),
                "CCD-TEMP": int(temp),
                "XBINNING": int(bin),
                "YBINNING": int(bin),
            }
        )
        r.update({"FILTER": filter_map(filter)})
    except:
        logging.warning(f"Error with {string}", exc_info=True)
        pass
    return r


def clean_file_list(df):
    if "CCDFNAME" in df.columns:
        sel = ~df["CCDFNAME"].isnull()
        df_rep = df.loc[sel, "CCDFNAME"].apply(equinox_ccdfname_parser).apply(pd.Series)
        df.loc[sel, df_rep.columns] = df_rep

    filters_to_replace = dict(
        Clear="L", Red="R", Green="G", Blue="B", SIII="SII", Luminance="L"
    )
    filters_to_replace.update({"** BayerMatrix **": "OSC"})
    df["FILTER"] = df["FILTER"].replace(filters_to_replace)

    df0 = df[
        [
            "filename",
            "OBJECT",
            "INSTRUME",
            "DATE-OBS",
            "IMAGETYP",
            "FILTER",
            "EXPOSURE",
            "CCD-TEMP",
            "XBINNING",
            "YBINNING",
            "XPIXSZ",
            "YPIXSZ",
            "FOCALLEN",
            "OBJCTRA",
            "OBJCTDEC",
            "AIRMASS",
            "OBJCTALT",
        ]
    ].reset_index(drop=True)

    sel = ~df0["OBJECT"].isnull()
    df0.loc[sel, "OBJECT"] = df0.loc[sel, "OBJECT"].apply(format_name)
    df0.loc[:, "INSTRUME"] = df0.loc[:, "INSTRUME"].replace(
        (
            {
                "QSI 690ws HW 12.01.00 FW 06.03.04": "QSI690-wsg8",
                "QHYCCD-Cameras-Capture": "QHY16200A",
                np.nan: "",
            }
        )
    )

    return df0


def format_date_string(x):
    try:
        if "000" == x[-3:]:
            x = x[:-4].replace(".", ":")
        return pd.to_datetime(x)
    except TypeError:
        return ""


def fix_date_fmt(datestring):
    if isinstance(datestring, int):
        return "1970-01-01"
    split_datestring = datestring.split(".")
    if len(split_datestring) > 2:
        return ":".join(split_datestring[:2])
    return datestring
