import time
import logging
import sys

import os
import glob
import pandas as pd

import paho.mqtt.client as mqtt

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

from astro_planner.target import object_file_reader, normalize_target_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(module)s %(message)s")
log = logging.getLogger(__name__)

client = mqtt.Client()


def on_connect(client, obj, flags, rc):
    log.info(f"Connecting to {client}")


def on_message(client, obj, msg):
    if msg.payload.decode("ascii") == "restart":
        log.info(msg.payload)
        sys.exit()


client.on_message = on_message
client.on_connect = on_connect
client.connect("mqtt", 1883, 60)
client.subscribe("watchdog", 0)


def update_db_with_targets(config=CONFIG, target_dir=TARGET_DIR, file_list=None):
    log.debug("Checking for new targets")
    update_targets(config, target_dir, file_list)
    init_target_status()


def init_target_status():
    status_query = """
    CREATE TABLE IF NOT EXISTS target_status
        ("TARGET" varchar(512), "GROUP" varchar(512), status varchar(512));
    ALTER TABLE target_status DROP CONSTRAINT IF EXISTS target_group_uq;
    ALTER TABLE target_status
        ADD CONSTRAINT target_group_uq
        UNIQUE ("TARGET", "GROUP") ;
        """
    with POSTGRES_ENGINE.connect() as con:
        con.execute(status_query)

    # Set initial status as "pending"
    df_targets = pd.read_sql("select * from targets;", POSTGRES_ENGINE)
    df0 = df_targets.reset_index()[["TARGET", "GROUP"]]

    df0["status"] = "pending"
    data = list(df0.values)

    query_template = """INSERT INTO target_status
        ("TARGET", "GROUP", status)
        VALUES (%s, %s, %s)
        ON CONFLICT ("TARGET", "GROUP") DO NOTHING;"""
    with POSTGRES_ENGINE.connect() as con:
        con.execute(query_template, data)


def update_targets(config=CONFIG, target_dir=DATA_DIR, file_list=None):
    if not file_list:
        file_list = []
        for extension in ["mdb", "sgf", "xml", "ninaTargetSet"]:
            file_list += glob.glob(f"{target_dir}/**/*.{extension}", recursive=True)
    new_files = list(set(check_file_in_table(file_list, POSTGRES_ENGINE, "targets")))
    n_files = len(file_list)
    if len(new_files) > 0:
        log.info(f"Found {n_files} files for targets")
    files_with_data = get_file_list_with_data(file_list)
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
        if df_targets.shape[0] > 0:
            log.debug("Pushing new targets to to db")
            push_rows_to_table(
                df_targets, POSTGRES_ENGINE, table_name="targets", if_exists="replace",
            )
            log.debug("Done")


if __name__ == "__main__":
    log.info("Starting watchdog on data directory")
    client.publish("watchdog", "running")
    t_last_update = time.time()
    update_frequency = 5
    update = True
    while True:
        client.loop()
        t_elapsed = time.time() - t_last_update
        if t_elapsed > update_frequency:
            update = True
        if update:
            update_db_with_targets(file_list=None)
            update_db_with_matching_files(file_list=None)
            t_last_update = time.time()
            update = False
        time.sleep(1)
