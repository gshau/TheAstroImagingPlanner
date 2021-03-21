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
from astro_planner.fast_ephemeris.distance import distance
import pandas as pd

from direct_redis import DirectRedis
import time
from .logger import log


VALID_STATUS = ["pending", "active", "acquired", "closed"]


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS = DirectRedis(host=REDIS_HOST, port=6379, db=0)


def push_df_to_redis(df, key):
    t0 = time.time()
    REDIS.set(key, df)
    t_elapsed = time.time() - t0
    log.debug(f"Pushing for {key:30s} took {t_elapsed:.3f} seconds")


def get_df_from_redis(key):
    t0 = time.time()
    df = REDIS.get(key)
    t_elapsed = time.time() - t0
    log.debug(f"Reading for {key:30s} took {t_elapsed:.3f} seconds")
    return df


def get_filename_root(filename):
    filename_root = os.path.splitext(ntpath.basename(filename))[0]
    return filename_root


def approx_ra_hr_noon(date):
    d1 = datetime.datetime.strptime("2020-03-21", "%Y-%m-%d")
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


def infer_group(equipment, df0):
    sensor_map = get_sensor_map(equipment, df0)
    optic_map = get_optic_map(equipment, df0)
    df0["inferred_group"] = (
        df0[FOCALLENGTH_COL]
        .map(optic_map)
        .fillna(df0[FOCALLENGTH_COL].astype(int).astype(str) + "mm")
        + " "
        + df0[INSTRUMENT_COL].replace(sensor_map)
    )
    return df0


def match_targets_to_data(df_targets, df_data):

    # speed-up - consider NxM array
    dfsh = df_data[
        [
            "filename",
            "OBJECT",
            "DATE-OBS",
            "OBJCTRA",
            "OBJCTDEC",
            "arcsec_per_pixel",
            "NAXIS1",
            "NAXIS2",
        ]
    ].set_index(["filename"])

    records = []
    for row in df_targets.itertuples():
        object_distance = (
            distance(row.RAJ2000, row.DECJ2000, dfsh["OBJCTRA"], dfsh["OBJCTDEC"]) * 60
        )
        fov_x = dfsh["arcsec_per_pixel"] * dfsh["NAXIS1"] / 60
        fov_y = dfsh["arcsec_per_pixel"] * dfsh["NAXIS2"] / 60
        target = row.TARGET
        matching_files = object_distance[object_distance < fov_y / 1].index
        for file in matching_files:
            object_name = dfsh.loc[file, "OBJECT"]
            records.append(
                dict(
                    TARGET=target,
                    OBJECT=object_name,
                    filename=file,
                    distance=object_distance.loc[file],
                    fov_x=fov_x.loc[file],
                    fov_y=fov_y.loc[file],
                )
            )

    df_match = pd.DataFrame(records)
    return df_match


def get_exposure_summary(df_data):
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
    group_cols = ["OBJECT"] + cols + ["FILTER"]
    df0 = df_data.groupby(group_cols).agg({EXPOSURE_COL: "sum"}) / 3600
    df0 = df0[EXPOSURE_COL].unstack(group_cols.index("FILTER")).fillna(0).reset_index()
    return df0


def merge_targets_with_stored_metadata(df_data, df_targets):

    df_match = match_targets_to_data(df_targets, df_data)

    df_merged = pd.merge(
        pd.merge(df_data, df_match.drop("OBJECT", axis=1), on="filename", how="outer"),
        df_targets,
        on="TARGET",
        how="outer",
    )

    df_merged["frame_overlap_fraction"] = np.clip(
        1 - df_merged["distance"] / (df_merged["fov_y"] / 2), 1, 0
    )

    df_exposure = get_exposure_summary(df_merged)

    log.info(df_exposure.columns)
    group = df_exposure.groupby(["OBJECT"])
    df_matching_targets_for_object = group.apply(
        lambda row: list(np.unique([t for t in row.TARGET]))
    ).to_frame("matching_targets")
    df_matching_groups_for_object = group.apply(
        lambda row: list(np.unique([g for g in row.GROUP]))
    ).to_frame("matching_groups")

    df_exposure = (
        df_exposure.drop("TARGET", axis=1).drop_duplicates().set_index("OBJECT")
    )

    df_exposure = df_exposure.join(df_matching_groups_for_object)
    df_exposure = df_exposure.join(df_matching_targets_for_object)

    df_exposure = df_exposure.reset_index()

    return df_exposure, df_merged
