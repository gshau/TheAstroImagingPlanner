import os
import ntpath

import datetime
import numpy as np
from astro_planner.data_parser import (
    FILTERS,
    INSTRUMENT_COL,
    EXPOSURE_COL,
    FOCALLENGTH_COL,
    BINNING_COL,
)
from astro_planner.target import normalize_target_name

VALID_STATUS = ["pending", "active", "acquired", "closed"]


def get_filename_root(filename):
    filename_root = os.path.splitext(ntpath.basename(filename))[0]
    return filename_root


def approx_ra_hr_noon(date):
    d1 = datetime.datetime.strptime("2021-03-21", "%Y-%m-%d")
    d2 = datetime.datetime.strptime(date, "%Y-%m-%d")
    days_diff = (d2 - d1).total_seconds() / (3600 * 24)
    hour_diff = np.round(days_diff / 365.25 * 24) % 24
    return int(hour_diff)


def compute_ra_order(ra, date_string):
    result = (ra - approx_ra_hr_noon(date_string)) % 24
    return result


def get_sensor_map(equipment, df0):
    sensor_map = {}
    for sensor_name in equipment.get("sensors", {}):
        for instrument in df0[INSTRUMENT_COL].unique():
            if sensor_name.lower() in instrument.lower():
                sensor_map[instrument] = sensor_name
    return sensor_map


def get_optic_map(equipment, df0):
    optic_map = {}
    for optic_name in equipment.get("telescopes", {}):
        for fl in df0[FOCALLENGTH_COL].unique():
            if (
                np.abs(int(equipment["telescopes"][optic_name]["focal_length"]) - fl)
                <= 5
            ):
                optic_map[fl] = optic_name
    return optic_map


def add_group(equipment, df0):
    sensor_map = get_sensor_map(equipment, df0)
    optic_map = get_optic_map(equipment, df0)
    df0["group"] = (
        df0[FOCALLENGTH_COL]
        .map(optic_map)
        .fillna(df0[FOCALLENGTH_COL].astype(int).astype(str) + "mm")
        + " "
        + df0[INSTRUMENT_COL].replace(sensor_map)
    )
    return df0


def merge_targets_with_stored_metadata(df_stored_data, df_targets, profile_config):

    df0 = (
        df_stored_data.groupby(
            ["OBJECT", INSTRUMENT_COL, FOCALLENGTH_COL, BINNING_COL, "FILTER"]
        ).agg({EXPOSURE_COL: "sum"})
        / 3600
    )
    df0 = df0[EXPOSURE_COL].unstack(4).fillna(0).reset_index()

    df0 = add_group(profile_config, df0)
    df_targets["OBJECT"] = df_targets["TARGET"].apply(normalize_target_name)
    df_combined = df0.set_index("OBJECT").join(
        df_targets.set_index("OBJECT"), how="outer"
    )

    filter_cols = [col for col in FILTERS if col in df0.columns]
    cols = [
        "TARGET",
        INSTRUMENT_COL,
        FOCALLENGTH_COL,
        BINNING_COL,
        "GROUP",
        "NOTE",
        "RAJ2000",
        "DECJ2000",
    ]
    cols += filter_cols

    df_combined["GROUP"] = df_combined["GROUP"].fillna(df_combined["group"])

    df_combined = df_combined[cols].round(2)

    return df_combined
