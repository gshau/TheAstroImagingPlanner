import time
import logging
import sys

import os
import glob
import pandas as pd

import paho.mqtt.client as mqtt
from direct_redis import DirectRedis


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
from astro_planner.target import target_file_reader, normalize_target_name


import logging.handlers

handler = logging.handlers.WatchedFileHandler(
    os.environ.get("LOGFILE", "./watchdog.log"), mode="w"
)

formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(module)s %(message)s")
handler.setFormatter(formatter)
root = logging.getLogger()

root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
root.addHandler(handler)

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(module)s %(message)s",
)

log = logging.getLogger(__name__)

client = mqtt.Client()


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS = DirectRedis(host=REDIS_HOST, port=6379, db=0)


def push_to_redis(element, key):
    t0 = time.time()
    REDIS.set(key, element)
    t_elapsed = time.time() - t0
    log.debug(f"Pushing for {key:30s} took {t_elapsed:.3f} seconds")


def get_list_from_redis(key):
    t0 = time.time()
    result = REDIS.get(key)
    t_elapsed = time.time() - t0
    log.debug(f"Reading for {key:30s} took {t_elapsed:.3f} seconds")
    return result


def on_connect(client, obj, flags, rc):
    log.info(f"Connecting to {client}")


def on_message(client, obj, msg):
    if msg.payload.decode("ascii") == "restart":
        log.info(msg.payload)
        sys.exit()


def add_column_to_table(c, table_name, column_name, column_type):
    has_column = False
    for row in c.execute(
        """
    SELECT column_name 
  FROM information_schema.columns
 WHERE table_name = '{table_name}'
 """.format(
            table_name=table_name
        )
    ):
        print(row[0])
        if row[0] == column_name:
            has_column = True
    if not has_column:
        c.execute(
            "ALTER TABLE {} ADD COLUMN {} {}".format(
                table_name, column_name, column_type
            )
        )


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
        ("TARGET" varchar(512), "GROUP" varchar(512), status varchar(512), goal varchar(2048), priority integer, metadata varchar(2048));
    ALTER TABLE target_status DROP CONSTRAINT IF EXISTS target_group_uq;
    ALTER TABLE target_status
        ADD CONSTRAINT target_group_uq
        UNIQUE ("TARGET", "GROUP") ;
        """
    with POSTGRES_ENGINE.connect() as con:
        con.execute(status_query)
        add_column_to_table(con, "target_status", "metadata", "varchar(2048)")
        add_column_to_table(con, "target_status", "priority", "integer")
        add_column_to_table(con, "target_status", "goal", "varchar(2048)")

    target_create_query = """
    CREATE TABLE IF NOT EXISTS targets
        (filename text, "TARGET" text, "GROUP" text, "RAJ2000" float8, "DECJ2000" float8, "NOTE" text);
        """
    with POSTGRES_ENGINE.connect() as con:
        con.execute(target_create_query)

    # Set initial status as "pending"
    df_targets = pd.read_sql("select * from targets;", POSTGRES_ENGINE)
    if df_targets.shape[0] == 0:
        return None
    df0 = df_targets.reset_index()[["TARGET", "GROUP"]]
    df0["TARGET"] = df0["TARGET"].apply(normalize_target_name)
    df0["status"] = "pending"
    df0["priority"] = 3
    df0["goal"] = "{}"
    df0["metadata"] = "{}"

    data = list(df0.values)

    query_template = """INSERT INTO target_status
        ("TARGET", "GROUP", status, priority, goal, metadata)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT ("TARGET", "GROUP") DO NOTHING;"""
    with POSTGRES_ENGINE.connect() as con:
        con.execute(query_template, data)


def update_targets(config=CONFIG, target_dir=DATA_DIR, file_list=None):
    if not file_list:
        file_list = []
        for extension in ["mdb", "sgf", "xml", "ninaTargetSet"]:
            file_list += glob.glob(f"{target_dir}/**/*.{extension}", recursive=True)
    if len(file_list) == 0:
        return None
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
                targets = target_file_reader(filename)
                df = targets.df_targets
                df["filename"] = os.path.basename(filename)
                df_list.append(df[target_columns])
            except:
                log.info(f"Issue with {filename}", exc_info=True)
        df_targets = pd.concat(df_list)
        # push_to_redis(df_targets, "df_targets")
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
    update_frequency = CONFIG.get("watchdog_update_frequency", 15)
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
