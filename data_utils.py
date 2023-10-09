import time
import sqlite3
import threading
import os

import pandas as pd
import numpy as np

from astro_planner.logger import log
from astro_planner.site import get_utc_offset
from astro_planner.target import normalize_target_name, Targets
from astro_planner.utils import timer
from astro_planner.globals import (
    EXC_INFO,
    INSTRUMENT_COL,
    FOCALLENGTH_COL,
    BINNING_COL,
    PIXELSIZE_COL,
    EXPOSURE_COL,
)


def use_planner(config):
    log.info(config.get("switch_config", {}).get("planner_switch", False))
    return config.get("switch_config", {}).get("planner_switch", False)


class ThreadWithReturnValue(threading.Thread):
    def __init__(
        self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None
    ):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        log.info(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return


def set_date_cols(df, utc_offset):
    df["date"] = df["DATE-OBS"].values
    df["date_night_of"] = (
        pd.to_datetime(df["DATE-OBS"], errors="coerce", format="ISO8601")
        + pd.Timedelta(hours=utc_offset - 12)
    ).dt.date

    return df


def get_columns(conn, table_name):
    df_columns = pd.read_sql(f"""PRAGMA table_info({table_name});""", conn)
    return list(df_columns["name"].values)


def pull_data(conn, config, targets=[], dates=[], join_type="inner"):
    columns = get_columns(conn, "fits_headers")
    extra_cols = [col for col in columns if "AOC" in col or "AIRMASS" in col]
    extra_col_queries = ""
    if len(extra_cols) > 0:
        for extra_col in extra_cols:
            extra_col_queries = f'{extra_col_queries}\n fh."{extra_col}",'
    target_query = get_target_query(targets)
    query = f"""
        select fh.full_file_path,
                fh.filename,
                fh.file_type,
                fh."OBJECT",
                strftime('%Y-%m-%d %H:%M:%S', fh."DATE-OBS") as "DATE-OBS",
                cast(fh."CCD-TEMP" as float) as "CCD-TEMP",
                fh."FILTER",
                fh."OBJCTRA",
                fh."OBJCTDEC",
                fh."OBJCTALT",
                fh."OBJCTAZ",
                fh."FOCUSTEM",
                fh."FOCUSPOS",
                fh."NAXIS1",
                fh."NAXIS2",
                fh."GAIN",
                fh."OFFSET",
                fh."INSTRUME" as "{INSTRUMENT_COL}",
                cast(fh."FOCALLEN" as float) as "{FOCALLENGTH_COL}",
                cast(fh."EXPOSURE" as float) as "{EXPOSURE_COL}",
                cast(fh."XBINNING" as float) as "{BINNING_COL}",
                fh."XPIXSZ",
                fh."IMAGETYP",
                fh.arcsec_per_pixel,
                {extra_col_queries}
                date(fh."DATE-OBS") as "date",
                asm.fwhm_mean,
                asm.fwhm_median,
                asm.fwhm_std,
                asm.fwhm_slope,
                asm.fwhm_theta,
                asm.theta_median,
                asm.eccentricity_mean,
                asm.eccentricity_median,
                asm.n_stars,
                asm.star_trail_strength,
                asm.star_orientation_score,
                asm.star_fraction_saturated,
                asm.log_flux_mean,
                asm.bkg_val,
                fh.is_valid_header,
                fg.const,
                fg.fit_rmse,
                fg.frame_rmse,
                fg.gradient_dir,
                fg.gradient_strength,
                fg.quadratic_aspect,
                fg.quadratic_dir,
                fg.quadratic_strength,
                fg.r2,
                fg.relative_gradient_strength,
                fg.relative_quadratic_strength,
                fg.residual_rmse
        from fits_headers fh
            {join_type} join aggregated_star_metrics asm
                on fh.filename  = asm.filename
            {join_type} join frame_gradients fg
                on fg.filename = fh.filename
            where fh.is_valid_header = True 
                {target_query}
    """
    try:
        with conn:
            df_data = pd.read_sql(query, conn)
    except (sqlite3.OperationalError, pd.io.sql.DatabaseError):
        log.error("Problem with reading fits_headers or aggregated_star_metrics")
        return None

    df_data = df_data.drop_duplicates(subset=["filename"])

    df_data["date"] = pd.to_datetime(df_data["date"], errors="coerce")
    utc_offset = get_utc_offset(config.get("lat"), config.get("lon"), "2022-01-01")
    df_data = set_date_cols(df_data, utc_offset=utc_offset)
    if len(dates) > 0:
        df_data = df_data[df_data["date_night_of"].astype(str).isin(dates)]

    if os.name == "nt":
        root_name = df_data["full_file_path"].apply(lambda f: f.split("\\")[1])
    else:
        root_name = df_data["full_file_path"].apply(lambda f: f.split("/")[1])

    df_data["OBJECT"] = df_data["OBJECT"].fillna(root_name)
    df_data["OBJECT"] = df_data["OBJECT"].apply(normalize_target_name)

    for col in ["fwhm_mean", "fwhm_median", "fwhm_std"]:
        df_data[f"{col}_arcsec"] = df_data[col] * df_data["arcsec_per_pixel"]

    df_data["frame_snr"] = (
        10 ** df_data["log_flux_mean"] * df_data["n_stars"]
    ) / df_data["bkg_val"]

    df_data = df_data[
        df_data["filename"].apply(
            lambda f: ("TestShot" not in f) and ("SyncVoy" not in f)
        )
    ]

    for col in ["OBJCTALT", "OBJCTAZ"]:
        if col in df_data.columns:
            df_data[col] = df_data[col].astype(float)

    # return add_sqm(conn, df_data)
    return df_data


def calculate_photon_rate(adu, qe, gain, floor=500, exposure=300):
    signal = adu - floor
    el_rate = signal * gain / exposure
    phot_rate = el_rate / qe
    return phot_rate


def calculate_scales(pixel_size, focal_ratio, central_obstruction_frac=0):
    return pixel_size**2 / focal_ratio**2 * (1 - central_obstruction_frac**2)


def calculate_mpsas(
    adu,
    qe,
    gain,
    exposure,
    focal_length,
    pixel_size,
    aperture,
    central_obstruction_frac=0,
    floor=500,
    airmass=1,
    optical_transmission=1,
):
    phot_rate = calculate_photon_rate(adu, qe, gain, floor, exposure)
    pixel_scale = 206 * pixel_size / focal_length
    phot_rate_scaled = phot_rate / (pixel_scale**2) / optical_transmission
    area = np.pi * (aperture / 2) ** 2 * (1 - central_obstruction_frac**2)

    mpsas = (
        np.log10(phot_rate_scaled / airmass / (0.205 * 1.15e11 * (area * 1e-6))) / -0.4
    )
    return mpsas


def add_sqm(conn, df_data):
    instrument_records = []
    instrument_records.append(
        dict(name="ZWO ASI2600MM Pro(FSQ106)", pixel_size=3.76, qe=0.8, gain=0.75)
    )
    instrument_records.append(
        dict(name="ZWO ASI6200MM Pro", pixel_size=3.76, qe=0.8, gain=0.75)
    )
    instrument_records.append(
        dict(name="ZWO ASI2600MM Pro", pixel_size=3.76, qe=0.8, gain=0.75)
    )
    instrument_records.append(
        dict(name="ZWO ASI2600MM Pro(Stowaway)", pixel_size=3.76, qe=0.8, gain=0.75)
    )
    instrument_records.append(
        dict(name="ASI Camera (1)", pixel_size=3.76, qe=0.8, gain=0.75)
    )
    instrument_records.append(
        dict(
            name="QSI 690ws HW 12.01.00 FW 06.03.04",
            pixel_size=3.69,
            qe=0.75,
            gain=0.16,
        )
    )
    instrument_records.append(
        dict(name="QHYCCD-Cameras-Capture", pixel_size=6, qe=0.6, gain=0.7)
    )
    df_instrument = pd.DataFrame(instrument_records)

    scope_records = []
    scope_records.append(
        dict(
            focal_length=2127,
            aperture=12.5 * 25.4,
            co_frac=0.54,
            optical_transmission=0.95,
        )
    )
    scope_records.append(
        dict(focal_length=1150, aperture=254, co_frac=0.32, optical_transmission=0.8)
    )
    scope_records.append(
        dict(focal_length=530, aperture=106, co_frac=0.0, optical_transmission=0.95)
    )
    scope_records.append(
        dict(focal_length=489, aperture=92, co_frac=0.0, optical_transmission=0.95)
    )
    scope_records.append(
        dict(focal_length=728, aperture=130, co_frac=0.0, optical_transmission=0.95)
    )
    scope_records.append(
        dict(focal_length=650, aperture=130, co_frac=0.0, optical_transmission=0.95)
    )
    scope_records.append(
        dict(focal_length=589, aperture=130, co_frac=0.0, optical_transmission=0.95)
    )
    scope_records.append(
        dict(focal_length=105, aperture=105 / 4, co_frac=0.0, optical_transmission=0.95)
    )
    scope_records.append(
        dict(focal_length=490, aperture=92, co_frac=0.0, optical_transmission=0.95)
    )
    df_scope = pd.DataFrame(scope_records)

    df_header = pd.read_sql(
        """select * from fits_headers fh where "FILTER" = 'L' AND is_valid_header = True""",
        # """select * from fits_headers fh where "FILTER" in ('L', 'R', 'G', 'B') AND is_valid_header = True""",
        conn,
    )

    dfm = pd.merge(
        df_data[
            [
                "filename",
                "full_file_path",
                "bkg_val",
                "n_stars",
                "fwhm_median_arcsec",
                "relative_gradient_strength",
            ]
        ].query("bkg_val > 500"),
        df_header[
            [
                "full_file_path",
                "OBJECT",
                "XBINNING",
                "AIRMASS",
                "AOCSKYQU",
                "FOCALLEN",
                "EXPOSURE",
                "INSTRUME",
                "CCD-TEMP",
                "FILTER",
            ]
        ],
        on="full_file_path",
    )

    df0 = pd.merge(
        dfm, df_scope, left_on="FOCALLEN", right_on="focal_length", how="outer"
    )
    df0 = pd.merge(df0, df_instrument, left_on="INSTRUME", right_on="name", how="outer")
    df0
    df0["AIRMASS"] = df0["AIRMASS"].astype(float)
    df0["mpsas_calculated"] = df0.apply(
        lambda row: calculate_mpsas(
            adu=row.bkg_val,
            qe=row.qe,
            gain=row.gain,
            exposure=row.EXPOSURE,
            focal_length=row.focal_length,
            pixel_size=row.pixel_size,
            aperture=row.aperture,
            central_obstruction_frac=row.co_frac,
            floor=500,
            optical_transmission=row.optical_transmission,
            airmass=row.AIRMASS,
        ),
        axis=1,
    )

    return pd.merge(
        df_data,
        df0[["full_file_path", "mpsas_calculated"]],
        on="full_file_path",
        how="left",
    )


def get_target_query(targets):
    target_query = ""
    if targets:
        if len(targets) > 1:
            target_query = f' and fh."OBJECT" in {tuple(targets)}'
        else:
            target_query = f" and fh.\"OBJECT\" = '{targets[0]}'"
    return target_query


@timer
def pull_target_data(conn, config):
    target_query = """select filename as target_filename,
        "TARGET",
        "GROUP",
        "RAJ2000",
        "DECJ2000",
        "NOTE"
        FROM targets
    """

    if not use_planner(config):
        return None, None, None
    try:
        with conn:
            df_targets = pd.read_sql(target_query, conn)
            df_target_status = pd.read_sql("SELECT * FROM target_status;", conn)
        df_targets["TARGET"] = df_targets["TARGET"].apply(normalize_target_name)
        df_target_status["TARGET"] = df_target_status["TARGET"].apply(
            normalize_target_name
        )
    except (sqlite3.OperationalError, pd.io.sql.DatabaseError):
        log.info("Problem with reading targets")
        df = None
        return df, df, df

    target_data = target_df_to_data(df_targets)
    return target_data, df_targets, df_target_status


def target_df_to_data(df_targets):
    target_data = Targets()
    target_data.load_from_df(df_targets)
    return target_data


@timer
def add_rejection_criteria(
    df0,
    z_score_thr=2,
    iqr_scale=1.5,
    eccentricity_median_thr=0.6,
    star_trail_strength_thr=25,
    min_star_reduction=0.5,
    gradient_thr=0.1,
    new_cols=False,
):
    df0["log_n_stars"] = np.log(df0["n_stars"])

    group_cols = [
        "OBJECT",
        "FILTER",
        INSTRUMENT_COL,
        BINNING_COL,
        FOCALLENGTH_COL,
        EXPOSURE_COL,
        "file_type",
    ]

    for col in ["OFFSET", "GAIN"]:
        if col in df0.columns:
            group_cols.append(col)

    group = df0.groupby(group_cols, dropna=True)
    df1 = df0.set_index(group_cols)

    # Calculate z-score
    df1 = df1.join(group["log_n_stars"].max().to_frame("log_n_stars_max"))
    df1 = df1.join(group["log_n_stars"].mean().to_frame("log_n_stars_mean"))
    df1 = df1.join(group["log_n_stars"].std().to_frame("log_n_stars_std"))

    df1 = df1.join(group["fwhm_median"].mean().to_frame("fwhm_median_mean"))
    df1 = df1.join(group["fwhm_median"].std().to_frame("fwhm_median_std"))

    # Calculate IQR outliers
    df1 = df1.join(group["log_n_stars"].quantile(0.25).to_frame("log_n_stars_q1"))
    df1 = df1.join(group["log_n_stars"].quantile(0.75).to_frame("log_n_stars_q3"))
    df1 = df1.join(group["fwhm_median"].quantile(0.25).to_frame("fwhm_median_q1"))
    df1 = df1.join(group["fwhm_median"].quantile(0.75).to_frame("fwhm_median_q3"))
    df1["log_n_stars_iqr"] = df1["log_n_stars_q3"] - df1["log_n_stars_q1"]
    df1["fwhm_median_iqr"] = df1["fwhm_median_q3"] - df1["fwhm_median_q1"]
    df1 = df1.reset_index()

    df1["star_count_iqr_outlier"] = (
        df1["log_n_stars"] < df1["log_n_stars_q1"] - iqr_scale * df1["log_n_stars_iqr"]
    )
    df1["star_count_z_score"] = (df1["log_n_stars"] - df1["log_n_stars_mean"]) / df1[
        "log_n_stars_std"
    ]

    df1["fwhm_iqr_outlier"] = (
        df1["fwhm_median"] > df1["fwhm_median_q3"] + iqr_scale * df1["fwhm_median_iqr"]
    )
    df1["fwhm_z_score"] = (df1["fwhm_median"] - df1["fwhm_median_mean"]) / df1[
        "fwhm_median_std"
    ]

    df1["star_count_fraction"] = np.exp(df1["log_n_stars"] - df1["log_n_stars_max"])
    df1["low_star_count_fraction"] = df1["star_count_fraction"] < min_star_reduction

    df1["bad_fwhm_z_score"] = df1["fwhm_z_score"] > z_score_thr

    df1["bad_star_count_z_score"] = df1["star_count_z_score"] < -z_score_thr
    df1["low_star_count"] = (
        df1["star_count_iqr_outlier"]
        | df1["bad_star_count_z_score"]
        | df1["low_star_count_fraction"]
    )

    df1["high_fwhm"] = df1["bad_fwhm_z_score"] | df1["fwhm_iqr_outlier"]

    df0 = df1.copy()

    df0["high_ecc"] = df0["eccentricity_median"] > eccentricity_median_thr
    df0["trailing_stars"] = df0["star_trail_strength"] > star_trail_strength_thr

    df0["bad_star_shape"] = df0["high_ecc"] | df0["trailing_stars"]

    df0["high_gradient"] = df0["relative_gradient_strength"] > gradient_thr

    df0["is_ok"] = 1
    df0.loc[
        df0["bad_star_shape"]
        | df0["low_star_count"]
        | df0["high_fwhm"]
        | df0["high_gradient"],
        "is_ok",
    ] = 0

    if "OFFSET" in df0.columns:
        df0.loc[df0["OFFSET"] == 0, "is_ok"] = 0

    status_map = {False: "&#10004;", True: "&#10006;"}

    df0["fwhm_status"] = df0["high_fwhm"].replace(status_map)
    df0["ecc_status"] = df0["high_ecc"].replace(status_map)
    df0["star_trail_status"] = df0["trailing_stars"].replace(status_map)
    df0["iqr_status"] = df0["star_count_iqr_outlier"].replace(status_map)
    df0["star_z_score_status"] = df0["bad_star_count_z_score"].replace(status_map)
    df0["fwhm_z_score_status"] = df0["bad_fwhm_z_score"].replace(status_map)
    df0["star_count_fraction_status"] = df0["low_star_count_fraction"].replace(
        status_map
    )

    df0["gradient_status"] = df0["high_gradient"].replace(status_map)

    return df0


def duration(date):
    dt = pd.to_datetime(date, errors="coerce")
    return (dt.max() - dt.min()).total_seconds()


@timer
def update_target_status_data(conn, table_data):
    df = pd.DataFrame.from_records(table_data)
    if df.shape[0] > 0:
        with conn:
            df.to_sql("target_status", conn, if_exists="replace", index=False)
    return df
