import logging
import datetime
import glob

from astropy.io import fits

import pandas as pd
from pathlib import Path

from .camera import FILTERS
from .profile import cleanup_name

DATA_DIR = "/Volumes/Users/gshau/Dropbox/AstroBox/data"


def parse_filename(file_name):
    file_root = Path(file_name).stem
    if "_LIGHT_" in file_root:
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


def parse_filelist(file_list, root_key="data/", skip_header=True, verbose=False):
    d_list = []
    for file_name in file_list:
        try:
            rel_file_path = file_name.split(root_key)[-1]
            elements = rel_file_path.split("/")
            el_names = ["target", "date_start", "filename"]

            d_info = dict(zip(el_names, elements))
            d_info.update(parse_filename(d_info["filename"]))
            if not skip_header:
                d_info.update(read_fits_header(file_name))
            d_list.append(d_info)
        except Exception as e:
            print("Skipping {}".format(file_name))
            raise (e)
            pass
    logging.info("Read {} files".format(len(d_list)))

    return pd.DataFrame(d_list)


def get_data_info(data_dir=DATA_DIR, skip_header=True):
    file_list = glob.glob("{}/*/*/*.FIT".format(data_dir))
    df_files = parse_filelist(file_list, skip_header=skip_header)
    return df_files


def get_exposure_summary(data_dir=DATA_DIR, filter_list=FILTERS, time_format="minutes"):
    df_files = get_data_info(data_dir=data_dir)
    df_exposures = (
        df_files.groupby(["target", "filter"]).sum()["sub_exposure"].to_frame()
    )

    df = pd.pivot(
        data=df_exposures.reset_index(), columns="filter", index="target"
    ).fillna(0)["sub_exposure"]

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
