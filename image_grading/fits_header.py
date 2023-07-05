import os
import pandas as pd
import numpy as np
from astropy.io import fits

from astro_planner.globals import EXC_INFO
from astro_planner.utils import get_fits_header_map
from astro_planner.logger import log
from image_grading.utils import to_numeric, coord_str_to_float

IMAGE_TYPES = ["light", "dark", "bias", "flat"]

REQUIRED_FITS_ENTRIES = [
    "OBJECT",
    "DATE-OBS",
    "CCD-TEMP",
    "FILTER",
    "OBJCTRA",
    "OBJCTDEC",
    "INSTRUME",
    "FOCALLEN",
    "EXPOSURE",
    "XBINNING",
    "XPIXSZ",
    "IMAGETYP",
]


def standardize_image_type(t):
    for image_type in IMAGE_TYPES:
        if image_type in str(t).lower():
            return image_type
    return "null"


def standardize_image_type_in_df(df_header):
    df_header["file_type"] = df_header["IMAGETYP"].apply(standardize_image_type)
    return df_header


def process_header_from_fits(filename):
    try:
        hdul = fits.open(filename)
        header = dict(hdul[0].header)
        file_base = os.path.basename(filename)
        file_dir = os.path.dirname(filename)

        if "IMAGETYP" in header:
            # try:
            file_type = standardize_image_type(header["IMAGETYP"])
        else:
            # except:
            file_type = "_UNKNOWN_"
            # raise KeyError("IMAGETYP not in header")

        if "OBJECT" not in header:
            # log.info(f'FILE TYPE: {file_type} from {header["IMAGETYP"]}')
            if file_type != "light":
                header["OBJECT"] = f"_{file_type.upper()}_"
            else:
                if "Voyager" in header.get("SWCREATE", {}):
                    header["OBJECT"] = file_base.split("_LIGHT_")[0]
                else:
                    header["OBJECT"] = "_NO_NAME_"

        if file_type != "light":
            for col in ["OBJCTDEC", "OBJCTRA"]:
                if col not in header:
                    header[col] = 0.0

        if file_type == "bias" or file_type == "dark":
            if "FILTER" not in header:
                header["FILTER"] = "NO_FILTER"

        required_numeric_cols = [
            "OBJCTRA",
            "OBJCTDEC",
            "OBJCTALT",
            "OBJCTAZ",
            "FOCUSTEM",
            "FOCUSPOS",
            "GAIN",
            "OFFSET",
        ]
        for col in required_numeric_cols:
            if col not in header:
                header[col] = np.nan

        df_header = pd.DataFrame({file_base: header}).T
        df_header.index.name = "filename"
        df_header.reset_index(inplace=True)
        df_header["file_dir"] = file_dir
        df_header["full_file_path"] = filename
        df_header["filename"] = file_base
        valid_columns = [col for col in df_header.columns if len(col) > 0]
        df_header = df_header[valid_columns]

        cols = ["filename"] + REQUIRED_FITS_ENTRIES
        missing_cols = []
        fits_header_map = get_fits_header_map()
        for col in cols:
            if col not in df_header.columns:
                found_col = False
                if col in fits_header_map.keys():
                    for trial_col in fits_header_map[col]:
                        if trial_col in df_header.columns:
                            found_col = True
                            value = df_header[trial_col]
                            df_header[col] = value
                if not found_col:
                    df_header[col] = np.nan
                    if col == "FILTER":
                        df_header[col] = "NO_FILTER"
                    else:
                        missing_cols.append(col)

        if "COMMENT" in df_header.columns:
            df_header["COMMENT"] = df_header["COMMENT"].apply(lambda c: str(c))

        if missing_cols:
            log.warn(
                f"Header for {filename} missing matching columns {missing_cols}, marked as invalid"
            )
            df_header["is_valid_header"] = False
            return df_header
        df_header["is_valid_header"] = True
        return df_header
    except KeyboardInterrupt:
        raise ("Stopping...")
    except:
        log.info(f"Problem with file {filename}", exc_info=EXC_INFO)
        return pd.DataFrame()


def process_headers(file_list):
    df_header_list = []

    for file in file_list:
        log.info(f"Processing header for file {file}")
        df_header_list.append(process_header_from_fits(file))
    df_headers = pd.DataFrame()
    if df_header_list:
        df_headers = pd.concat(df_header_list)
        if df_headers.shape[0] == 0:
            return df_headers
        df_headers = to_numeric(df_headers)

        for col in ["OBJCTRA", "OBJCTDEC", "OBJCTALT", "OBJCTAZ"]:
            if col in df_headers.columns:
                df_headers[col] = df_headers[col].apply(coord_str_to_float)
                if col == "OBJCTALT":
                    df_headers["AIRMASS"] = 1.0 / np.cos(
                        (90.0 - df_headers["OBJCTALT"]) * np.pi / 180.0
                    )

        df_headers["arcsec_per_pixel"] = (
            df_headers["XPIXSZ"] * 206 / df_headers["FOCALLEN"]
        )

    df_headers = standardize_image_type_in_df(df_headers)

    return df_headers
