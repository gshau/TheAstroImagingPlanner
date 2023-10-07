import asyncio

import os
import glob
import pandas as pd
from image_grading.sql_handling import (
    remove_orphaned_rows,
)
from image_grading.preprocessing import (
    process_files_push_to_db,
    get_file_list_with_data,
    push_rows_to_table,
    chunks,
    check_file_in_table,
)


from astro_planner.globals import BASE_DIR, EXC_INFO
from astro_planner.utils import get_db_conn
from astro_planner.target import (
    target_file_reader,
    normalize_target_name,
    robotarget_reader,
)

from astro_planner.logger import log


def get_table_names(conn):
    df_tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    table_names = df_tables["name"].values
    return table_names


def update_db_with_targets(
    conn,
    target_dirs,
    file_list=None,
    enable_limits=True,
    use_robotarget=False,
    robotarget_kwargs={},
):
    log.info("Checking for new targets")
    target_create_query = """
    CREATE TABLE IF NOT EXISTS targets
        (filename text, "TARGET" text, "GROUP" text, "RAJ2000" float8, "DECJ2000" float8, "NOTE" text);
        """

    with conn:
        conn.execute(target_create_query)

    update_targets(
        conn,
        target_dirs,
        file_list,
        enable_limits=enable_limits,
        use_robotarget=use_robotarget,
        robotarget_kwargs=robotarget_kwargs,
    )
    init_target_status(conn)
    log.info("done")


def init_target_status(conn):
    status_query = """
    CREATE TABLE IF NOT EXISTS target_status
        ("TARGET" varchar(512), "GROUP" varchar(512), status varchar(512), exposure_goal varchar(2048), exposure_acquired varchar(2048), priority varchar(512), metadata varchar(2048),
        UNIQUE("TARGET", "GROUP") );
        """

    with conn:
        conn.executescript(status_query)

    target_create_query = """
    CREATE TABLE IF NOT EXISTS targets
        (filename text, "TARGET" text, "GROUP" text, "RAJ2000" float8, "DECJ2000" float8, "NOTE" text);
        """
    with conn:
        conn.execute(target_create_query)

    # Set initial status as "pending"
    df_targets = pd.read_sql("select * from targets;", conn)
    df_targets["TARGET"] = df_targets["TARGET"].apply(normalize_target_name)
    df_status = pd.read_sql("select * from target_status;", conn)
    df_status["TARGET"] = df_status["TARGET"].apply(normalize_target_name)
    if df_targets.shape[0] == 0:
        return None
    df0 = df_targets.merge(df_status, on=["TARGET", "GROUP"], how="left")
    df0["status"] = df0["status"].fillna("Pending")
    df0["priority"] = df0["priority"].fillna("Low")
    df0["exposure_goal"] = df0["exposure_goal"].fillna("[]")
    df0["exposure_acquired"] = df0["exposure_acquired"].fillna("[]")
    df0["metadata"] = df0["metadata"].fillna("[]")

    df_new_status = df0[
        [
            "TARGET",
            "GROUP",
            "status",
            "priority",
            "exposure_goal",
            "exposure_acquired",
            "metadata",
        ]
    ].drop_duplicates()

    with conn:
        df_new_status.to_sql("target_status", conn, if_exists="replace", index=False)


def update_targets(
    conn,
    target_dirs,
    file_list=None,
    enable_limits=True,
    use_robotarget=False,
    robotarget_kwargs={},
):
    if not file_list:
        file_list = []
        for target_dir in target_dirs:
            for extension in ["mdb", "sgf", "xml", "ninaTargetSet"]:
                file_list += glob.glob(f"{target_dir}/**/*.{extension}", recursive=True)
    if len(file_list) == 0:
        return None
    n_files = len(file_list)
    if n_files == 0:
        log.info(f"Found {n_files} files for targets")
    files_with_data = get_file_list_with_data(file_list)
    target_columns = ["filename", "TARGET", "GROUP", "RAJ2000", "DECJ2000", "NOTE"]
    if files_with_data:
        df_list = []
        for filename in files_with_data:
            try:
                target_kwargs = {}
                if ".mdb" in filename:
                    target_kwargs = dict(mdb_export_path=f"{BASE_DIR}/")
                targets = target_file_reader(filename, **target_kwargs)
                df = targets.df_targets
                df["filename"] = os.path.basename(filename)
                df_list.append(df[target_columns])
            except:
                log.info(f"Problem with {filename}", exc_info=True)
        if use_robotarget:
            log.info("****** Robotarget running *******")
            try:
                df = asyncio.run(robotarget_reader(**robotarget_kwargs))
                df["GROUP"] = df["profilename"].apply(
                    lambda s: f'RoboTarget {s.split(".")[0]}'
                )
                df["filename"] = df["profilename"]
                df_list.append(df[target_columns])
            except Exception:
                log.info("Issue getting Robotarget data", exc_info=True)
        df_targets = pd.concat(df_list)
        if enable_limits:
            df_targets = df_targets.head(5)
    if df_targets.shape[0] > 0:
        log.debug("Pushing new targets to to db")
        try:
            push_rows_to_table(
                df_targets,
                conn,
                table_name="targets",
                if_exists="replace",
            )
        except:
            log.warning("Cannot push to targets", exc_info=EXC_INFO)
        log.debug("Done")


def get_fits_file_list(config, data_dirs, preproc_out_dirs, enable_limits=True):
    fits_file_list = []
    if "fits_file_patterns" in config:
        for data_dir in data_dirs:
            for file_pattern in config["fits_file_patterns"]["allow"]:
                glob_pattern = f"{data_dir}/{file_pattern}"
                fits_file_list += list(glob.iglob(glob_pattern, recursive=True))
            for reject_file_pattern in config["fits_file_patterns"]["reject"]:
                fits_file_list = [
                    f for f in fits_file_list if reject_file_pattern not in f.lower()
                ]
        for preproc_out_dir in preproc_out_dirs:
            glob_pattern = f"{preproc_out_dir}/**/*_master_*subs_*min.fit"
            fits_file_list += list(glob.iglob(glob_pattern, recursive=True))

    if enable_limits:
        n_file_limit_on_demo = 50
        return fits_file_list[:n_file_limit_on_demo]
    return fits_file_list


class RunFileProcessor:
    def __init__(self, config):
        self.config = config
        self.data_dirs = self.config.get("data_dirs", [])
        self.target_dirs = self.config.get("target_dirs", [])
        self.calibration_dirs = self.config.get("calibration_dirs", [])

        self.n_total = 0
        self.n_processed = 0
        self.n_removed = 0
        self.processed_file_list = []
        self.pending_file_list = []
        self.stop_loop = False
        voyager_config = self.config.get("voyager_config")

        self.robotarget_kwargs = {}
        self.use_robotarget = False
        if voyager_config is not None:
            log.info("** Got config")
            self.robotarget_kwargs["server_url"] = voyager_config.get("hostname")
            self.robotarget_kwargs["server_port"] = voyager_config.get("port")
            self.robotarget_kwargs[
                "auth_token"
            ] = f"{voyager_config.get('user')}:{voyager_config.get('password')}"
            self.use_robotarget = voyager_config.get("voyager_switch", False)
            log.info(self.use_robotarget)
            log.info(self.robotarget_kwargs)

    # TODO: ADD INCREMENTAL RUN, WHERE FILENAMES ARE EXPLICITLY PASSED, FULL RUN DONE ON STARTUP OR PROFILE CHANGE
    def run(self):
        conn = get_db_conn(self.config)
        # TODO: FIX NONE
        self.data_dirs = self.config.get("directories").get("data_dirs", [])
        self.calibration_dirs = self.config.get("directories").get(
            "calibration_dirs", []
        )
        self.target_dirs = self.config.get("directories").get("target_dirs", [])

        self.preproc_out_dirs = self.config.get("directories").get(
            "preproc_out_dirs", []
        )

        enable_limits = False

        log.info(self.target_dirs)
        update_db_with_targets(
            conn,
            target_dirs=self.target_dirs,
            enable_limits=enable_limits,
            use_robotarget=self.use_robotarget,
            robotarget_kwargs=self.robotarget_kwargs,
        )

        data_file_list = get_fits_file_list(
            self.config,
            data_dirs=self.data_dirs,
            enable_limits=enable_limits,
            preproc_out_dirs=self.preproc_out_dirs,
        )
        cal_file_list = get_fits_file_list(
            self.config,
            data_dirs=self.calibration_dirs,
            enable_limits=enable_limits,
            preproc_out_dirs=self.preproc_out_dirs,
        )
        file_list = cal_file_list + data_file_list
        log.info(f"File list length: {len(file_list)}")
        self.n_removed += remove_orphaned_rows(conn, file_list)
        # self.n_removed = 0
        log.info(f"Files removed: {self.n_removed}")

        new_files = []
        file_list = sorted(list(set(file_list)))
        with conn:
            new_files = check_file_in_table(conn, file_list, "fits_headers")
            new_files = sorted(new_files)
        log.info(f"New file length: {len(new_files)}")
        log.info(new_files)

        self.stop_loop = False
        chunk_size = self.config.get("chunk_size", 4)
        extract_thresh = self.config.get("extract_thresh", 0.5)
        if len(new_files) > 0:
            self.pending_file_list = new_files
            for file_chunk in list(chunks(new_files, chunk_size)):
                if self.stop_loop:
                    break
                n_threads = self.config.get("n_threads", 2)
                n_new = process_files_push_to_db(
                    conn,
                    file_chunk,
                    n_threads=n_threads,
                    n_chunk=chunk_size,
                    extract_thresh=extract_thresh,
                    use_simple=self.config.get("use_simple_bkg_eval", True),
                )
                self.processed_file_list += file_chunk
                self.pending_file_list = [
                    file for file in new_files if file not in self.processed_file_list
                ]
                self.n_total = len(self.pending_file_list + self.processed_file_list)
                self.n_processed = len(self.processed_file_list)
        self.stop_loop = False

    def reset(self):
        self.n_total = 0
        self.n_processed = 0
        self.n_removed = 0
        self.processed_file_list = []
        self.pending_file_list = []

    def get_update(self, clear=False):
        if clear:
            self.reset()
        return (
            self.n_total,
            self.n_processed,
            self.n_removed,
            self.processed_file_list,
            self.pending_file_list,
        )
