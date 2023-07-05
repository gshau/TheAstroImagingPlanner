import os
import yaml
import sqlite3
import functools
import time

from shutil import copyfile
from inspect import getframeinfo, stack

from astro_planner.logger import log
from astro_planner.globals import ENV, BASE_DIR


def debug_info(message):
    caller = getframeinfo(stack()[1][0])
    log.info("%s:%d - %s" % (caller.filename, caller.lineno, message))


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        caller = getframeinfo(stack()[1][0])
        log.info(
            f"({caller.filename}:{caller.lineno}) Elapsed time: {elapsed_time:0.4f} seconds for {func.__name__} "
        )
        return value

    return wrapper_timer


def get_env_dir(env=ENV):
    env_dir = f"{BASE_DIR}/data/user/{env}"
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    return env_dir


def get_config(env=ENV):
    env_dir = get_env_dir(env=env)
    config_filename = f"{env_dir}/config.yml"
    if not os.path.exists(config_filename):
        template_filename = f"{BASE_DIR}/data/_template/config_template.yml"
        copyfile(template_filename, config_filename)
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)
    config["env"] = env
    config["db_file"] = "data.db"

    save_config(config, env)
    return config


def get_db_conn(config):
    env = config.get("env", ENV)
    env_dir = get_env_dir(env=env)
    db_file = f'{env_dir}/{config["db_file"]}'
    return sqlite3.connect(db_file, check_same_thread=False)


def get_fits_header_map(base_dir=BASE_DIR):
    with open(f"{base_dir}/data/_template/fits_header.yml", "r") as f:
        fits_header_map = yaml.safe_load(f)
    return fits_header_map


def save_config(config, env):
    env_dir = get_env_dir(env=env)
    with open(f"{env_dir}/config.yml", "w") as f:
        yaml.dump(config, f)
