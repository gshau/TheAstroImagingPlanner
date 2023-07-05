import glob
import os
import pandas as pd
import numpy as np
from astropy.io import fits

from tqdm import tqdm

from .logger import log
from astro_planner.globals import (
    EXPOSURE_COL,
    INSTRUMENT_COL,
    FOCALLENGTH_COL,
    BINNING_COL,
)

# TODO: CHANGE TO ENUM
FILE_TYPES = ["light", "dark", "bias", "flat"]


def get_lights(
    df, binning=1, filter="L", exposure=300, nx=None, focal_length=None, extra_filter={}
):
    selection = df[BINNING_COL] == binning
    selection &= df["FILTER"] == filter
    selection &= df[EXPOSURE_COL] == exposure
    selection &= df["file_type"] == "light"
    if focal_length:
        selection &= df[FOCALLENGTH_COL] == focal_length
    if nx:
        selection &= df["NAXIS1"] == nx
    if len(extra_filter) > 0:
        for k, v in extra_filter.items():
            selection &= df[k] == v
    return list(df[selection]["filename"].values)


def get_header_data(
    target_name, lights_dir, calibration_dir, ccd_temp_tolerance=5, extra_mappings={}
):

    darks_dir = f"{calibration_dir}/*/darks"
    bias_dir = f"{calibration_dir}/*/bias"
    flats_dir = f"{calibration_dir}/*/flats"

    # target_lights_dir = f"{lights_dir}/{target_name}"
    log.info(f"Lights dir: {lights_dir}")
    records = []
    for file_type, file_dir in zip(
        FILE_TYPES, [lights_dir, darks_dir, bias_dir, flats_dir]
    ):
        log.info(f'Looking in {f"{file_dir}/**/*.[Ff][Ii][Tt]"}')
        file_list = glob.glob(f"{file_dir}/**/*.[Ff][Ii][Tt]", recursive=True)
        # file_list = [file for file in file_list if "test" not in file]
        pbar = tqdm(file_list)
        pbar.set_description(file_type)
        for filename in pbar:
            pbar.set_postfix_str(filename)
            header = fits.open(filename)[0].header
            header = dict(header)
            header["filename"] = filename
            header["file_type"] = file_type
            records.append(header)
    df_header = pd.DataFrame(records)
    df_header = apply_mapping(df_header, extra_mappings)
    df_header["CCD-TEMP-ROUNDED"] = df_header["CCD-TEMP"].apply(
        lambda t: np.round(t // ccd_temp_tolerance) * ccd_temp_tolerance
    )
    return df_header


def get_calibrations(df_header):

    gain_offset_cols = [col for col in ["GAIN", "OFFSET"] if col in df_header.columns]

    df0 = df_header[
        [
            INSTRUMENT_COL,
            BINNING_COL,
            "NAXIS1",
            "file_type",
            "FILTER",
            EXPOSURE_COL,
            "DATE-OBS",
            FOCALLENGTH_COL,
            "filename",
            "CCD-TEMP-ROUNDED",
        ]
        + gain_offset_cols
    ]
    df_calibration = df0[df0["file_type"] != "light"].reset_index(drop=True)
    df_calibration[EXPOSURE_COL] = df_calibration[EXPOSURE_COL].round()
    log.info(df_calibration.shape)
    return df_calibration


def match_light_with_calibration(record, df_calibration):
    result = {}
    for calibration_type in ["bias", "dark", "flat"]:
        log.info(calibration_type)
        selection = [True] * df_calibration.shape[0]
        cols = []
        if calibration_type == "bias":
            cols = [BINNING_COL, "NAXIS1", "CCD-TEMP-ROUNDED"]
        elif calibration_type == "dark":
            cols = [BINNING_COL, "NAXIS1", EXPOSURE_COL, "CCD-TEMP-ROUNDED"]
        elif calibration_type == "flat":
            cols = [
                BINNING_COL,
                "NAXIS1",
                "FILTER",
                FOCALLENGTH_COL,
                "CCD-TEMP-ROUNDED",
            ]

        for col in ["GAIN", "OFFSET"]:
            if col in record:
                if col in df_calibration.columns:
                    cols.append(col)
                else:
                    log.warning(f"FITs entry {col} not in calibration files")
        for col in cols:
            log.info(f"{col}: {record.get(col)}")
            selection &= df_calibration[col] == record.get(col)
        df0 = df_calibration[selection]
        df1 = df0[df0["file_type"] == calibration_type]
        df1["date"] = pd.to_datetime(df1["DATE-OBS"]).dt.date
        log.info(f"{df0.shape}, {df1.shape}")
        # TODO: FIX THIS LOGIC - USE CLOSEST CALIBRATION FILE
        latest_date_selection = df1["date"] == df1["date"].max()
        calibration_files = list(df1[latest_date_selection]["filename"].values)
        result[calibration_type] = calibration_files
    return result


def get_light_specs(df_header):

    df_lights = df_header[df_header["file_type"] == "light"].reset_index()
    df_lights["date"] = pd.to_datetime(df_lights["DATE-OBS"]).dt.date
    df_lights.groupby(["date", "FILTER"]).size().unstack(1)
    df_lights = df_header[df_header["file_type"] == "light"]
    log.info(df_lights.shape)

    group_keys = [
        "OBJECT",
        INSTRUMENT_COL,
        BINNING_COL,
        "NAXIS1",
        "file_type",
        "FILTER",
        EXPOSURE_COL,
        FOCALLENGTH_COL,
        "CCD-TEMP-ROUNDED",
    ]

    for col in ["GAIN", "OFFSET"]:
        if col in df_lights.columns:
            if df_lights[col].isnull().sum() == 0:
                group_keys.append(col)

    log.info(df_lights[group_keys].head())

    light_specs = (
        df_header[df_header["file_type"] == "light"]
        .groupby(group_keys)
        .size()
        .to_frame("count")
        .reset_index()
        .to_dict(orient="records")
    )

    if len(light_specs) == 0:
        return []
    df_specs = pd.DataFrame(light_specs)
    log.info(df_specs.shape)
    log.info(light_specs)
    df_specs = df_specs.sort_values(by=[BINNING_COL, "count"], ascending=[True, False])
    records = df_specs.to_dict(orient="records")

    return records


def apply_mapping(df_header, extra_mappings):
    for key, mappings in extra_mappings.items():
        log.info(key)
        for mapping in mappings:
            log.info(mapping)
            df_header[key] = df_header[key].replace(mapping)
    return df_header
