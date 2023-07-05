import os

import yaml
import click
import numpy as np

from astro_planner.globals import FOCALLENGTH_COL

from .fit_header import (
    get_header_data,
    get_light_specs,
    get_calibrations,
)
from .target import process_target
from .target_status import query_status
from .logger import log
from pathlib import Path


BASE_PATH = Path(__file__).resolve().parents[1]

# CONFIG = {}
# with open(f"{BASE_PATH}/config.yml", "r") as f:
#     CONFIG = yaml.safe_load(f)

# lights_dir = CONFIG.get("lights_dir")
# calibration_dir = CONFIG.get("calibration_dir")
# master_cal_dir = CONFIG.get("master_cal_dir")
# output_dir = CONFIG.get("output_dir")


# @click.command()
# @click.argument("target-name")
# @click.option(
#     "--lights-dir",
#     default=lights_dir,
#     help="location of stored lights files",
#     show_default=True,
# )
# @click.option(
#     "--calibration-dir",
#     default=calibration_dir,
#     help="location of stored calibration files",
#     show_default=True,
# )
# @click.option(
#     "--master-cal-dir",
#     default=master_cal_dir,
#     help="location of output master calibration files",
#     show_default=True,
# )
# @click.option("--output-dir", default=output_dir, help="location of output ")
# @click.option(
#     "--use-aip",
#     is_flag=True,
#     help="whether to use astroimaging planner criteria",
# )
# @click.option(
#     "--filter",
#     default=None,
#     multiple=True,
#     help="select only matching filter",
# )
# @click.option(
#     "--focal-length",
#     default=None,
#     multiple=True,
#     help="select only matching focal length",
#     type=int,
# )
# @click.option(
#     "--fwhm-filter",
#     default=None,
#     multiple=True,
#     help="select only top percent frames with best FWHM",
# )
# def cli(
#     target_name,
#     lights_dir,
#     calibration_dir,
#     master_cal_dir,
#     output_dir,
#     use_aip=False,
#     matching_files=None,
#     filter=None,
#     focal_length=None,
#     fwhm_filter=None,
#     query_status_kwargs={},
# ):
#     return run_auto_preproc(
#         target_name,
#         lights_dir,
#         calibration_dir,
#         master_cal_dir,
#         output_dir,
#         use_aip=use_aip,
#         matching_files=matching_files,
#         filter=filter,
#         focal_length=focal_length,
#         fwhm_filter=fwhm_filter,
#         query_status_kwargs=query_status_kwargs,
#     )


def run_auto_preproc(
    target_name,
    df_header=None,
    lights_dir=None,
    calibration_dir=None,
    master_cal_dir=None,
    output_dir=None,
    use_aip=False,
    matching_files=None,
    filter=None,
    focal_length=None,
    fwhm_filter=None,
    query_status_kwargs={},
    config={},
    # preproc_list=[],
    extra_mappings={},
    app=None,
):

    # extra_mappings = CONFIG.get("extra_mappings", {})

    stack_kwargs_list = [{}]
    if fwhm_filter is not None and len(fwhm_filter) > 0:
        for fwhm_pct in fwhm_filter:
            stack_kwargs_list.append({"filter_fwhm": fwhm_pct})

    # matching_files = None
    if use_aip:
        df_ok = query_status(target_name, **query_status_kwargs)
        n_files = df_ok.shape[0]
        if n_files > 0:
            log.info(f"Found {n_files} matching files")
            matching_files = df_ok["filename"].values
        else:
            log.warning("API did not find any files")
            exit(-1)

    ccd_temp_tolerance = config.get("ccd_temp_tolerance", 20)
    df_header["CCD-TEMP-ROUNDED"] = df_header["CCD-TEMP"].apply(
        lambda t: np.round(t // ccd_temp_tolerance) * ccd_temp_tolerance
    )
    log.info(df_header.shape)
    log.info(df_header.columns)
    # df_header = get_header_data(
    #     target_name=target_name,
    #     lights_dir=lights_dir,
    #     calibration_dir=calibration_dir,
    #     extra_mappings=extra_mappings,
    #     ccd_temp_tolerance=config.get("ccd_temp_tolerance", 20),
    # )
    # log.info(df_header.shape)
    # log.info(df_header.columns)

    df_calibration = get_calibrations(df_header)

    if matching_files is not None:
        log.info(matching_files)
        log.info(df_header["filename"].head().values)
        df_header = df_header[
            df_header["filename"]
            .apply(lambda t: os.path.basename(t))
            .isin(matching_files)
        ]
        log.info(df_header.head())
        log.info(df_header.shape)
        log.info(df_header.columns)

    records = []
    if df_header.shape[0] > 0:
        lights_records = get_light_specs(df_header)
        [log.info(light_record) for light_record in lights_records]
        if filter is not None and len(filter) > 0:
            lights_records = [
                light_record
                for light_record in lights_records
                if light_record["FILTER"] in list(filter)
            ]
        if focal_length is not None and len(focal_length) > 0:
            lights_records = [
                light_record
                for light_record in lights_records
                if int(light_record[FOCALLENGTH_COL]) in focal_length
            ]
        app.preproc_progress = 0
        app.status = f"Starting {target_name}"
        records = process_target(
            target_name,
            lights_records=lights_records,
            output_dir=output_dir,
            master_cal_dir=master_cal_dir,
            df_calibration=df_calibration,
            df_header=df_header,
            matching_files=matching_files,
            stack_kwargs=stack_kwargs_list[0],
            # preproc_list=preproc_list,
            app=app,
        )
    return records


# if __name__ == "__main__":
#     cli()
