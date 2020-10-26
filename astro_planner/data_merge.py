import os
import ntpath
import yaml

import datetime
import numpy as np
import pandas as pd
from astro_planner.data_parser import FILTERS
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


def compute_ra_order(ra_string, date_string):
    try:
        ra_string = str(ra_string)
        for replace_string in ["h", "m", "s"]:
            ra_string = ra_string.replace(replace_string, "")
        ra_vec = ra_string.split()
        ra_vec[0] = str((int(ra_vec[0]) - approx_ra_hr_noon(date_string)) % 24)
        result = float(ra_vec[0]) / 24 * 360 + float(ra_vec[1]) / 24 * 6
        return result
    except:
        return np.nan


def get_sensor_map(equipment, df0):
    sensor_map = {}
    for sensor_name in equipment["sensors"]:
        for instrument in df0["INSTRUME"].unique():
            if sensor_name.lower() in instrument.lower():
                sensor_map[instrument] = sensor_name
    return sensor_map


def get_optic_map(equipment, df0):
    optic_map = {}
    for optic_name in equipment["optics"]:
        for fl in df0["FOCALLEN"].unique():
            if np.abs(int(equipment["optics"][optic_name]["focal_length"]) - fl) < 2:
                optic_map[fl] = optic_name
    return optic_map


def add_group(equipment, df0):
    sensor_map = get_sensor_map(equipment, df0)
    optic_map = get_optic_map(equipment, df0)
    df0["group"] = (
        df0["FOCALLEN"]
        .map(optic_map)
        .fillna(df0["FOCALLEN"].astype(int).astype(str) + "mm")
        + " "
        + df0["INSTRUME"].replace(sensor_map)
    )
    return df0


def merge_roboclip_stored_metadata(
    df_stored_data, df_roboclip, config, default_status="closed"
):

    df0 = (
        df_stored_data[df_stored_data["date"] > "2000-01-01"]
        .groupby(["OBJECT", "INSTRUME", "FOCALLEN", "XBINNING", "FILTER"])
        .agg({"EXPOSURE": "sum"})
        / 3600
    )
    df0 = df0["EXPOSURE"].unstack(4).fillna(0).reset_index()

    # set status
    df0["status"] = default_status

    df0 = add_group(config["equipment"], df0)
    df_roboclip["OBJECT"] = df_roboclip["TARGET"].apply(normalize_target_name)
    df_combined = df0.set_index("OBJECT").join(
        df_roboclip.set_index("OBJECT"), how="outer"
    )
    df_combined.loc[df_combined["status"].isnull(), "status"] = "pending"

    filter_cols = [col for col in FILTERS if col in df0.columns]
    cols = [
        "TARGET",
        "status",
        "INSTRUME",
        "FOCALLEN",
        "XBINNING",
        "GROUP",
        "NOTE",
        "RAJ2000",
        "DECJ2000",
    ]
    cols += filter_cols

    df_combined["GROUP"] = df_combined["GROUP"].fillna(df_combined["group"])

    df_combined = df_combined[cols].round(2)

    return df_combined


def get_targets_with_status(df_combined, status_list=["closed"]):
    targets = df_combined.index
    if status_list:
        targets = df_combined[df_combined["status"].isin(status_list)].index
    return list(set(targets))


def dump_target_status_to_file(df_combined, filename="./conf/target_status.yml"):
    target_status = {}
    groups = df_combined["GROUP"].unique()
    for group in groups:
        status_series = df_combined[df_combined["GROUP"] == group]["status"]
        target_status[group] = status_series.to_dict()
    with open(filename, "w") as f:
        yaml.dump(target_status, f)
    df_target_status = target_status_to_df(target_status)
    return df_target_status


def target_status_to_df(target_status):
    records = []
    for profile in target_status:
        for target in target_status[profile]:
            records.append(
                dict(
                    OBJECT=target, GROUP=profile, status=target_status[profile][target]
                )
            )
    df_target_status = pd.DataFrame(records)
    return df_target_status


def load_target_status_from_file(filename="./conf/target_status.yml"):
    with open(filename, "r") as f:
        target_status = yaml.load(f, Loader=yaml.BaseLoader)
    df_target_status = pd.DataFrame()
    if target_status:
        df_target_status = target_status_to_df(target_status)
    return df_target_status


def set_target_status(df_combined, df_target_status):

    status_map = {}
    if df_target_status.shape[0] > 0:
        status_map = df_target_status.set_index(["OBJECT", "GROUP"]).to_dict()["status"]
    dfc = df_combined.reset_index()
    dfc.loc[:, "status"] = dfc.set_index(["OBJECT", "GROUP"]).index.map(status_map)
    return dfc.set_index("OBJECT")


def update_targets_with_status(target_names, status, df_combined, profile):

    if status in VALID_STATUS:
        for target_name in target_names:
            df_combined.loc[
                (df_combined.index == target_name) & (df_combined["GROUP"] == profile),
                "status",
            ] = status
    df_target_status = dump_target_status_to_file(df_combined)
    return df_combined, df_target_status
