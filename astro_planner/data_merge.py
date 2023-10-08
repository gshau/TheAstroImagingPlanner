import os
import datetime
import numpy as np
from astro_planner.globals import (
    INSTRUMENT_COL,
    EXPOSURE_COL,
    FOCALLENGTH_COL,
    BINNING_COL,
    EXC_INFO,
)
from .fast_ephemeris.distance import distance
import pandas as pd
from .logger import log


def get_filename_root(filename):
    filename_root = os.path.splitext(os.path.basename(filename))[0]
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
    group_cols = [col for col in group_cols if col in df_data.columns]
    df0 = df_data.groupby(group_cols).agg({EXPOSURE_COL: "sum"}) / 3600
    df0 = df0[EXPOSURE_COL].unstack(group_cols.index("FILTER")).fillna(0).reset_index()
    return df0


def merge_targets_with_stored_metadata(df_data, df_targets):
    log.info(f"Target size: {df_targets.shape[0]}")
    log.info(f"Data size: {df_data.shape[0]}")
    df_match = match_targets_to_data(df_targets, df_data)

    if "OBJECT" in df_match.columns:
        df_match = df_match.drop("OBJECT", axis=1)

    if df_match.shape[0] > 0:
        df_merged = pd.merge(
            pd.merge(df_data, df_match, on="filename", how="outer"),
            df_targets,
            on="TARGET",
            how="outer",
        )
        df_merged["frame_overlap_fraction"] = (
            1 - df_merged["distance"] / (df_merged["fov_y"] / 2)
        ).fillna(0)
        df_merged["frame_overlap_fraction"] = df_merged["frame_overlap_fraction"].clip(
            0, 1
        )
        df_exposure = get_exposure_summary(df_merged)
    else:
        log.info("No matching targets!")
        df_merged = df_data.copy()
        df_merged["TARGET"] = df_merged["OBJECT"]
        df_merged["GROUP"] = "UNMATCHED GROUP"
        df_merged["frame_overlap_fraction"] = 0.0
        df_exposure = get_exposure_summary(df_merged)

    group = df_exposure.groupby(["OBJECT"])
    if "GROUP" in df_exposure.columns:
        try:
            df_matching_groups_for_object = group.apply(
                lambda row: list(np.unique([g for g in row.GROUP]))
            )
            if df_matching_groups_for_object.shape[0] > 0:
                df_matching_groups_for_object = df_matching_groups_for_object.to_frame(
                    "matching_groups"
                )
                df_exposure = df_exposure.join(df_matching_groups_for_object)
            else:
                df_exposure["matching_groups"] = ""
        except:
            log.warning(df_exposure.head(), exc_info=EXC_INFO)
            log.warning(df_matching_groups_for_object.head(), exc_info=EXC_INFO)

    if "TARGET" in df_exposure.columns:
        try:
            df_matching_targets_for_object = group.apply(
                lambda row: list(np.unique([t for t in row.TARGET]))
            )
            df_exposure = (
                df_exposure.drop("TARGET", axis=1).drop_duplicates().set_index("OBJECT")
            )
            if df_matching_targets_for_object.shape[0] > 0:
                df_matching_targets_for_object = (
                    df_matching_targets_for_object.to_frame("matching_targets")
                )
                df_exposure = df_exposure.join(df_matching_targets_for_object)
            else:
                df_exposure["matching_targets"] = ""
        except:
            log.warning(df_exposure.head(), exc_info=EXC_INFO)
            log.warning(df_matching_targets_for_object.head(), exc_info=EXC_INFO)

    df_exposure = df_exposure.reset_index()

    return df_exposure, df_merged
