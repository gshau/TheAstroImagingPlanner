import time
import logging
import os
import glob
import pandas as pd

from image_grading.preprocessing import (
    update_db_with_matching_files,
    POSTGRES_ENGINE,
    check_file_in_table,
    get_file_list_with_data,
    push_rows_to_table,
    CONFIG,
    TARGET_DIR,
    DATA_DIR,
)

from astro_planner.target import object_file_reader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(module)s %(message)s")
log = logging.getLogger(__name__)


def update_db_with_targets(config=CONFIG, target_dir=TARGET_DIR, file_list=None):
    log.debug("Checking for new targets")
    update_targets(config, target_dir, file_list)


def update_targets(config=CONFIG, target_dir=DATA_DIR, file_list=None):
    if not file_list:
        file_list = []
        for extension in ["mdb", "sgf"]:
            file_list += glob.glob(f"{target_dir}/**/*.{extension}", recursive=True)
    new_files = list(set(check_file_in_table(file_list, POSTGRES_ENGINE, "targets")))
    n_files = len(new_files)
    if n_files > 0:
        log.info(f"Found {n_files} new files for headers")
    files_with_data = get_file_list_with_data(new_files)
    target_columns = ["filename", "TARGET", "GROUP", "RAJ2000", "DECJ2000", "NOTE"]
    if files_with_data:
        df_list = []
        for filename in files_with_data:
            try:
                objects = object_file_reader(filename)
                df = objects.df_objects
                df["filename"] = os.path.basename(filename)
                df_list.append(df[target_columns])
            except:
                log.info(f"Issue with {filename}", exc_info=True)
        df_targets = pd.concat(df_list)
        log.info("Pushing to db")
        push_rows_to_table(
            df_targets, POSTGRES_ENGINE, table_name="targets", if_exists="append",
        )
        log.info("Done")


if __name__ == "__main__":
    log.info(f"Starting watchdog on data directory")
    while True:
        update_db_with_targets(file_list=None)
        update_db_with_matching_files(file_list=None)
        time.sleep(30)
