import os
from numpy.lib.type_check import imag
import sep
import warnings

import pandas as pd

from collections import defaultdict
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
from multiprocessing import Pool
from functools import partial

from astro_planner.logger import log
from image_grading.sql_handling import check_file_in_table, push_rows_to_table
from image_grading.utils import to_numeric, chunks, to_str, get_file_list_with_data
from image_grading.star_processing import process_frame
from image_grading.fits_header import process_headers


warnings.filterwarnings("ignore", category=AstropyUserWarning)


sep.set_extract_pixstack(1000000)


# def add_file_to_skiplist(filename):
#     pass
#     # skiplist = get_skiplist()
#     # log.info(skiplist)
#     # skiplist.append(filename)
#     # skiplist = list(set(skiplist))
#     # log.info(skiplist)
#     # skiplist_filename = "data/skiplist.txt"
#     # with open(skiplist_filename, "w") as f:
#     #     for filename in skiplist:
#     #         f.write(f"{filename}\n")


# def get_skiplist():
#     return []
#     skiplist_filename = "data/skiplist.txt"
#     skiplist = []
#     if os.path.exists(skiplist_filename):
#         with open(skiplist_filename, "r") as f:
#             skiplist = f.readlines()
#         log.info(skiplist)
#         if skiplist is None:
#             skiplist = []
#     log.info(skiplist)
#     return skiplist


def check_status(file_list, **status_args):
    df_status_list = []
    # file_skiplist = get_skiplist()
    for filename in file_list:
        # if filename in file_skiplist:
        #     continue
        status_dict = {}
        file_base = os.path.basename(filename)
        file_dir = os.path.dirname(filename)
        status_dict.update(
            dict(full_file_path=filename, file_dir=file_dir, filename=file_base)
        )
        for status, matches in status_args.items():
            for match in matches:
                is_match = match in filename
                status_dict.update({status: is_match})
        df_status_list.append(status_dict)
    df_status = pd.DataFrame()
    if df_status_list:
        df_status = pd.DataFrame(df_status_list)
    return df_status


def update_fits_headers(conn, file_list):
    with conn:
        new_files = check_file_in_table(conn, file_list, "fits_headers")
    files_with_data = get_file_list_with_data(new_files)
    # file_skiplist = get_skiplist()
    # files_with_data = sorted([f for f in files_with_data if f not in file_skiplist])
    n_files = len(files_with_data)
    if n_files > 0:
        log.info(f"Found {n_files} new files for headers")
    df_header = pd.DataFrame()
    if files_with_data:
        with conn:
            for files in chunks(files_with_data, 100):
                df_header = process_headers(files)
                df_header = to_str(df_header)
                log.info("Pushing to db")
                push_rows_to_table(
                    df_header,
                    conn,
                    table_name="fits_headers",
                    if_exists="append",
                    index=True,
                    index_name="filename",
                )
                log.info("Done")
    return df_header


def update_frame_metrics(
    conn, file_list=None, n_chunk=8, extract_thresh=0.25, n_threads=2, use_simple=True
):
    with conn:
        new_files = check_file_in_table(conn, file_list, "aggregated_star_metrics")
    files_with_data = get_file_list_with_data(new_files)
    # file_skiplist = get_skiplist()
    # files_to_process = sorted([f for f in files_with_data if f not in file_skiplist])
    files_to_process = files_with_data
    multithread_fits_read = n_threads > 1

    if files_to_process:
        log.info(f"Found {len(files_to_process)} new files with data")
        for file_set in list(chunks(files_to_process, n_chunk)):
            if multithread_fits_read:
                log.info(f"Using thread count: {n_threads}")
                with Pool(n_threads) as p:
                    l_result = list(
                        p.imap(
                            partial(
                                process_frame,
                                extract_thresh=extract_thresh,
                                use_simple=use_simple,
                            ),
                            [[f] for f in file_set],
                        )
                    )
                result = defaultdict(list)
                for r in l_result:
                    for k, v in r.items():
                        result[k].append(v)
                for k, v in result.items():
                    result[k] = pd.concat(v)

            else:
                result = process_frame(
                    file_set,
                    extract_thresh=extract_thresh,
                    use_simple=use_simple,
                )

            if "stars" not in result:
                continue
            n_stars = result["stars"].shape[0]

            log.info(f"New stars: {n_stars}")
            if n_stars <= 1:
                log.warning("Too few stars found")
                continue
            with conn:
                # Aggregate star metrics table
                for result_key, table_name in zip(
                    ["agg_stars", "xy_frame", "frame_gradients"],
                    ["aggregated_star_metrics", "xy_frame_metrics", "frame_gradients"],
                ):
                    push_rows_to_table(
                        result[result_key],
                        conn,
                        table_name=table_name,
                        if_exists="append",
                        index=True,
                        index_name="filename",
                    )

            if n_stars > 0:
                log.info(
                    f"Finished pushing {n_stars} stars from {len(file_set)} files to tables"
                )

        log.info("Done processing stars")
    return len(files_to_process)


def process_files_push_to_db(
    conn, file_list=None, n_chunk=10, extract_thresh=0.5, n_threads=2, use_simple=True
):
    log.debug("Checking for new files")
    df_header = update_fits_headers(conn, file_list)

    light_file_list = get_light_files(df_header)

    n_new = update_frame_metrics(
        conn,
        light_file_list,
        n_chunk=n_chunk,
        extract_thresh=extract_thresh,
        n_threads=n_threads,
        use_simple=use_simple,
    )
    return df_header.shape[0]


def get_light_files(df_header):
    if df_header.shape[0] == 0:
        return []
    is_light = df_header["file_type"] == "light"
    file_list = df_header[is_light]["full_file_path"].values
    return file_list
