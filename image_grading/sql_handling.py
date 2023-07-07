import os
from astro_planner.logger import log
import pandas as pd
from astro_planner.globals import EXC_INFO


def push_rows_to_table(
    df0, conn, table_name, if_exists="append", index=False, index_name=""
):
    if index:
        # assert index_name in df0.columns
        df0 = df0.set_index(index_name)
    try:
        df0.to_sql(
            table_name, conn, if_exists=if_exists, index=index
        )  ## issue here for targets
        n_rows = df0.shape[0]
        log.debug(f"Added {n_rows} new entries")
    except:
        log.info("fallback", exc_info=EXC_INFO)
        df_current = pd.read_sql(f"select * from {table_name}", conn)
        if index:
            assert index_name in df_current.columns
            df_current = df_current.set_index(index_name)

        df_combined = pd.concat([df_current, df0])
        try:
            clear_table(conn, table_name)
            df_combined.to_sql(table_name, conn, if_exists="replace", index=index)
        except:
            log.info("Failed", exc_info=EXC_INFO)
            raise Exception
            return None
        n_rows = df0.shape[0]
        log.debug(f"Added {n_rows} new entries")
        n_rows = df_current.shape[0]
        log.debug(f"Modified {n_rows} existing entries")


def check_if_table_exists(conn, table_name):
    try:
        df = pd.read_sql(
            f"""SELECT * FROM sqlite_schema WHERE type='table' AND name='{table_name}'""",
            conn,
        )
        return df.shape[0] == 1
    finally:
        pass


def check_file_in_table(conn, file_list, table_name):
    try:
        table_exists = check_if_table_exists(conn, table_name=table_name)
        if table_exists:
            df = pd.read_sql(f"""select filename from {table_name}""", conn)
            new_files = [
                file
                for file in file_list
                if os.path.basename(file) not in df["filename"].values
            ]
            return new_files
        else:
            return file_list
    except:
        log.info(f"Problem parsing {table_name}, it may not exist", exc_info=EXC_INFO)
        new_files = file_list
    return new_files


def clear_table(conn, table_name):
    log.debug(f"Clearing table {table_name}")
    with conn:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    log.debug("Finished")


def init_tables(conn):
    clear_tables(
        conn,
        [
            "fits_headers",
            "aggregated_star_metrics",
            "xy_frame_metrics",
            # "radial_frame_metrics",
            "targets",
            "target_status",
            "frame_gradients",
        ],
    )


def clear_tables(conn, table_names):
    with conn:
        for table_name in table_names:
            clear_table(conn, table_name)


def file_is_local(filename):
    return "/Volumes/" not in filename


def check_file_orphaned_in_table(conn, file_list, table_name):
    # TODO: OPTIMIZE THIS - TAKES A LONG WHILE
    try:
        # log.info("Pre-check")
        df = pd.read_sql(f"select full_file_path from {table_name}", conn)
        orphaned_files = [
            file
            for file in df["full_file_path"].values
            if (file not in file_list) and file_is_local(file)
        ]
        # log.info("Post-check")
    except:
        orphaned_files = []
    return orphaned_files


def remove_orphaned_rows(conn, file_list):
    for table_name in [
        "aggregated_star_metrics",
        "fits_headers",
        "xy_frame_metrics",
        "frame_gradients",
    ]:
        orphaned_files = []
        with conn:
            orphaned_files = check_file_orphaned_in_table(conn, file_list, table_name)
        orphaned_files = list(set(orphaned_files))
        if orphaned_files:
            for filename in orphaned_files:
                log.info(f"Orphaned file to be removed from {table_name}: {filename}")

            drop_query = f"""DELETE FROM {table_name} WHERE full_file_path IN {tuple(orphaned_files)}"""
            if len(orphaned_files) == 1:
                drop_query = f"""DELETE FROM {table_name} WHERE full_file_path = '{orphaned_files[0]}'"""

            with conn:
                log.info(drop_query)
                conn.execute(drop_query)
    return len(orphaned_files)
