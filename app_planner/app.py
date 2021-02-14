import json
import base64
import time
import io
import os
import sqlalchemy
from distutils.util import strtobool

from direct_redis import DirectRedis

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

import pandas as pd
import numpy as np

import plotly.graph_objects as go

import warnings

import datetime
import yaml

import paho.mqtt.client as mqtt

from dash.dependencies import Input, Output, State
from sqlalchemy import exc
from flask import request

from scipy.interpolate import interp1d
from astro_planner.weather import NWS_Forecast
from astro_planner.target import object_file_reader, normalize_target_name, Objects
from astro_planner.contrast import add_contrast
from astro_planner.site import update_site
from astro_planner.ephemeris import get_coordinates
from astro_planner.data_parser import (
    INSTRUMENT_COL,
    EXPOSURE_COL,
    FOCALLENGTH_COL,
    BINNING_COL,
)
from astro_planner.utils import timer

from image_grading.preprocessing import clear_tables, init_tables

from astro_planner.data_merge import (
    compute_ra_order,
    merge_targets_with_stored_metadata,
)

from image_grading.frame_analysis import (
    show_inspector_image,
    show_frame_analysis,
    show_fwhm_ellipticity_vs_r,
)

from layout import serve_layout, yaxis_map
import seaborn as sns

from astropy.utils.exceptions import AstropyWarning

import flask

from astro_planner.logger import log
from pathlib import Path


warnings.simplefilter("ignore", category=AstropyWarning)


server = flask.Flask(__name__)

theme = dbc.themes.COSMO
app = dash.Dash(__name__, external_stylesheets=[theme], server=server)

app.title = "The AstroImaging Planner"


mqtt_client = mqtt.Client()
mqtt_client.connect("mqtt", 1883, 60)


base_dir = Path(__file__).parents[1]
with open(f"{base_dir}/conf/config.yml", "r") as f:
    CONFIG = yaml.safe_load(f)

HORIZON_DATA = CONFIG.get("horizon_data", {})

with open(f"{base_dir}/conf/equipment.yml", "r") as f:
    EQUIPMENT = yaml.safe_load(f)
    if EQUIPMENT is None:
        EQUIPMENT = {}


ROBOCLIP_FILE = os.getenv("ROBOCLIP_FILE", "/roboclip/VoyRC.mdb")

DEFAULT_LAT = CONFIG.get("lat", 43.37)
DEFAULT_LON = CONFIG.get("lon", -88.37)
DEFAULT_UTC_OFFSET = CONFIG.get("utc_offset", -5)
DEFAULT_MPSAS = CONFIG.get("mpsas", 20.1)
DEFAULT_BANDWIDTH = CONFIG.get("bandwidth", 120)
DEFAULT_K_EXTINCTION = CONFIG.get("k_extinction", 0.2)
DEFAULT_TIME_RESOLUTION = CONFIG.get("time_resolution", 300)
DEFAULT_MIN_MOON_DISTANCE = CONFIG.get("min_moon_distance", 30)

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
PGPORT = os.getenv("PGPORT")
PGHOST = os.getenv("PGHOST", "0.0.0.0")

POSTGRES_ENGINE = sqlalchemy.create_engine(
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{PGHOST}:{PGPORT}/{POSTGRES_DB}"
)


USE_CONTRAST = strtobool(os.getenv("USE_CONTRAST", "True")) == 1
log.debug(f"use contrast: {USE_CONTRAST}")
styles = {}
if not USE_CONTRAST:
    styles["k-ext"] = {"display": "none"}
    styles["local-mpsas"] = {"display": "none"}


L_FILTER = "L"
R_FILTER = "R"
G_FILTER = "G"
B_FILTER = "B"
HA_FILTER = "Ha"
OIII_FILTER = "OIII"
SII_FILTER = "SII"
BAYER = "OSC"
BAYER_ = "** BayerMatrix **"
NO_FILTER = "NO_FILTER"

FILTER_LIST = [
    L_FILTER,
    R_FILTER,
    G_FILTER,
    B_FILTER,
    HA_FILTER,
    OIII_FILTER,
    SII_FILTER,
    BAYER,
    BAYER_,
    NO_FILTER,
]

FILTER_MAP = {
    "Red": R_FILTER,
    "Green": G_FILTER,
    "Blue": B_FILTER,
    "Lum": L_FILTER,
    "Luminance": L_FILTER,
    "HA": HA_FILTER,
    "O3": OIII_FILTER,
    "S2": SII_FILTER,
}


COLORS = {
    L_FILTER: "black",
    R_FILTER: "red",
    G_FILTER: "green",
    B_FILTER: "blue",
    HA_FILTER: "crimson",
    SII_FILTER: "maroon",
    OIII_FILTER: "teal",
    BAYER: "gray",
    BAYER_: "gray",
    NO_FILTER: "gray",
}


TRANSLATED_FILTERS = {
    "ha": ["ho", "sho", "hoo", "hos", "halpha", "h-alpha"],
    "oiii": ["ho", "sho", "hoo", "hos"],
    "sii": ["sho", "hos"],
    "nb": ["ha", "oiii", "sii", "sho", "ho", "hoo", "hos", "halpha", "h-alpha"],
    "bb": ["luminance", "lrgb"],
    "rgb": ["osc", "bayer", "dslr", "slr", "r ", " g ", " b "],
    "lum": ["luminance"],
}

all_target_coords = pd.DataFrame()
all_targets = []

REDIS = DirectRedis(host="redis", port=6379, db=0)


def push_df_to_redis(df, key):
    t0 = time.time()
    REDIS.set(key, df)
    t_elapsed = time.time() - t0
    log.debug(f"Pushing for {key:30s} took {t_elapsed:.3f} seconds")


def get_df_from_redis(key):
    t0 = time.time()
    df = REDIS.get(key)
    t_elapsed = time.time() - t0
    log.debug(f"Reading for {key:30s} took {t_elapsed:.3f} seconds")
    return df


def set_date_cols(df, utc_offset):
    df["date"] = df["DATE-OBS"].values
    df["date_night_of"] = (
        pd.to_datetime(df["DATE-OBS"]) + pd.Timedelta(hours=utc_offset - 12)
    ).dt.date

    return df


def pull_inspector_data():

    query = "select fh.*, asm.* from fits_headers fh left join aggregated_star_metrics asm on fh.filename  = asm.filename ;"

    try:
        df_stars_headers_ = pd.read_sql(query, POSTGRES_ENGINE)
        df_stars_headers_ = df_stars_headers_.loc[
            :, ~df_stars_headers_.columns.duplicated()
        ]
    except exc.SQLAlchemyError:
        log.info(
            f"Issue with reading tables, waiting for 15 seconds for it to resolve..."
        )
        time.sleep(15)
        return None

    df_stars_headers_["fwhm_mean_arcsec"] = (
        df_stars_headers_["fwhm_mean"] * df_stars_headers_["arcsec_per_pixel"]
    )
    df_stars_headers_["fwhm_median_arcsec"] = (
        df_stars_headers_["fwhm_median"] * df_stars_headers_["arcsec_per_pixel"]
    )
    df_stars_headers_["fwhm_std_arcsec"] = (
        df_stars_headers_["fwhm_std"] * df_stars_headers_["arcsec_per_pixel"]
    )

    df_stars_headers_["frame_snr"] = (
        10 ** df_stars_headers_["log_flux_mean"] * df_stars_headers_["n_stars"]
    ) / df_stars_headers_["bkg_val"]

    root_name = df_stars_headers_["file_full_path"].apply(lambda f: f.split("/")[1])
    df_stars_headers_["OBJECT"] = df_stars_headers_["OBJECT"].fillna(root_name)

    df_stars_headers = df_stars_headers_.copy()
    df_stars_headers = set_date_cols(df_stars_headers, utc_offset=DEFAULT_UTC_OFFSET)
    return df_stars_headers.drop_duplicates(subset=["filename"])


def pull_stored_data():
    log.debug("Checking for new data")

    stored_data_query = """
        select fh.file_full_path,
        fh.filename,
        fh."OBJECT",
        fh."DATE-OBS",
        fh."FILTER",
        cast(fh."CCD-TEMP" as float),
        fh."OBJCTRA",
        fh."OBJCTDEC",
        cast(fh."AIRMASS" as float),
        fh."OBJCTALT",
        fh."INSTRUME" as "Instrument",
        cast(fh."FOCALLEN" as float) as "Focal Length",
        cast(fh."EXPOSURE" as float) as "Exposure",
        cast(fh."XBINNING" as float) as "Binning",
        date(fh."DATE-OBS") as "date",
        asm.fwhm_mean,
        asm.fwhm_median,
        asm.eccentricity_mean,
        asm.eccentricity_median,
        asm.n_stars,
        asm.star_trail_strength,
        asm.star_orientation_score,
        asm.log_flux_mean,
        asm.bkg_val
        from fits_headers fh
            left join aggregated_star_metrics asm
            on asm.filename = fh.filename;
    """

    try:
        df_stored_data = pd.read_sql(stored_data_query, POSTGRES_ENGINE)
    except exc.SQLAlchemyError:
        log.info(
            f"Issue with reading tables, waiting for 15 seconds for it to resolve..."
        )
        time.sleep(15)
        return None

    df_stored_data["date"] = pd.to_datetime(df_stored_data["date"])

    root_name = df_stored_data["file_full_path"].apply(lambda f: f.split("/")[1])
    df_stored_data["OBJECT"] = df_stored_data["OBJECT"].fillna(root_name)
    df_stored_data["OBJECT"] = df_stored_data["OBJECT"].apply(normalize_target_name)
    df_stored_data = set_date_cols(df_stored_data, utc_offset=DEFAULT_UTC_OFFSET)
    return df_stored_data.drop_duplicates(subset=["filename"])


def pull_target_data():
    target_query = """select filename,
        "TARGET",
        "GROUP",
        "RAJ2000",
        "DECJ2000",
        "NOTE"
        FROM targets
    """

    try:
        df_objects = pd.read_sql(target_query, POSTGRES_ENGINE)
        df_target_status = pd.read_sql("SELECT * FROM target_status;", POSTGRES_ENGINE)
    except exc.SQLAlchemyError:
        log.info(
            f"Issue with reading tables, waiting for 15 seconds for it to resolve..."
        )
        time.sleep(15)
        return None

    object_data = Objects()
    object_data.load_from_df(df_objects)
    return object_data, df_objects, df_target_status


def get_object_data():
    df_objects = get_df_from_redis("df_objects")
    object_data = Objects()
    object_data.load_from_df(df_objects)
    return object_data


def merge_target_with_stored_data(df_stored_data, df_objects):
    df_combined = merge_targets_with_stored_metadata(
        df_stored_data, df_objects, EQUIPMENT
    )
    return df_combined


VALID_STATUS = ["pending", "active", "acquired", "closed"]


def update_targets_with_status(target_names, status, df_combined, profile_list):

    query_template = """INSERT INTO target_status
    ("TARGET", "GROUP", status)
    VALUES (%s, %s, %s)
    ON CONFLICT ("TARGET", "GROUP") DO UPDATE set status = EXCLUDED.status;"""

    df_target_status = pd.read_sql("SELECT * FROM target_status;", POSTGRES_ENGINE)
    if status in VALID_STATUS:
        selection = df_target_status["GROUP"].isin(profile_list)
        normalized_target_matches = [normalize_target_name(t) for t in target_names]
        selection &= df_target_status["TARGET"].isin(normalized_target_matches)
        df0 = df_target_status[selection][["TARGET", "GROUP"]]
        df0["status"] = status
        if df0.shape[0] > 0:
            data = list(df0.values)
            with POSTGRES_ENGINE.connect() as con:
                con.execute(query_template, data)
        df_target_status = pd.read_sql("SELECT * FROM target_status;", POSTGRES_ENGINE)

    return df_target_status


use_planner = True


def update_data():
    df_stored_data = pull_stored_data()
    df_stars_headers = pull_inspector_data()
    if use_planner:
        object_data, df_objects, df_target_status = pull_target_data()
        df_combined = merge_target_with_stored_data(df_stored_data, df_objects)

    push_df_to_redis(df_stored_data, "df_stored_data")
    push_df_to_redis(df_stars_headers, "df_stars_headers")
    if use_planner:
        push_df_to_redis(df_combined, "df_combined")
        push_df_to_redis(df_objects, "df_objects")
        push_df_to_redis(df_target_status, "df_target_status")


update_data()


def load_custom_horizon_function(
    horizon_file, sep=" ", header=3,
):
    flat_horizon_alt = HORIZON_DATA.get("flat_horizon_alt", 0)
    flat_horizon = interp1d([0, 360], [flat_horizon_alt, flat_horizon_alt])
    if not horizon_file:
        return flat_horizon
    try:
        df_horizon = pd.read_csv(horizon_file, sep=sep, header=header)
        df_horizon.columns = ["az", "alt"]
        df_horizon = df_horizon.append(
            pd.Series(dict(az=360, alt=df_horizon.iloc[-1]["alt"])), ignore_index=True
        )
        df_horizon = df_horizon.append(
            pd.Series(dict(az=0, alt=df_horizon.iloc[0]["alt"])), ignore_index=True
        )
        df_horizon = df_horizon.drop_duplicates(["az"])
        df_horizon = df_horizon.sort_values(by="az")
        f_horizon = interp1d(df_horizon["az"], df_horizon["alt"])
    except:
        log.warning(
            f"Issue with setting custom horizon from file: {horizon_file}, using flat horizon at {flat_horizon_alt} degrees"
        )
        return flat_horizon
    return f_horizon


f_horizon = load_custom_horizon_function(
    horizon_file=HORIZON_DATA.get("horizon_file", ""),
    sep=HORIZON_DATA.get("alt_az_seperator", " "),
    header=HORIZON_DATA.get("header_size", 3),
)


def make_options(elements):
    return [{"label": element, "value": element} for element in elements]


def get_time_limits(targets, sun_alt=-5):
    sun = targets["sun"]
    # Get sun up/down
    sun_up = np.where(np.gradient((sun.alt > sun_alt).astype(int)) > 0)[0][0]
    sun_dn = np.where(np.gradient((sun.alt > sun_alt).astype(int)) < 0)[0][0]
    return sun.index[sun_dn], sun.index[sun_up]


def get_data(
    target_coords,
    df_combined,
    value="alt",
    sun_alt_for_twilight=-18,
    filter_targets=True,
    min_moon_distance=30,
):

    target_names = [
        name for name in (list(target_coords.keys())) if name not in ["sun", "moon"]
    ]

    moon_data = dict(
        x=target_coords["moon"].index,
        y=target_coords["moon"]["alt"],
        text="Moon",
        opacity=1,
        line=dict(color="Gray", width=4),
        name="Moon",
    )

    sun_data = dict(
        x=target_coords["sun"].index,
        y=target_coords["sun"]["alt"],
        text="Sun",
        opacity=1,
        line=dict(color="Orange", width=4),
        name="Sun",
    )

    # Get sun up/down
    sun = target_coords["sun"]
    sun_up = np.where(np.gradient((sun.alt > sun_alt_for_twilight).astype(int)) > 0)[0][
        0
    ]
    sun_dn = np.where(np.gradient((sun.alt > sun_alt_for_twilight).astype(int)) < 0)[0][
        0
    ]

    log.info(f"Sun down: {sun.index[sun_dn]}")
    log.info(f"Sun up: {sun.index[sun_up]}")

    duraion_sun_down = sun.index[sun_up] - sun.index[sun_dn]
    duraion_sun_down_hrs = duraion_sun_down.total_seconds() / 3600.0

    log.info(f"Sun down duration: {duraion_sun_down_hrs:.2f} hours")

    sun_up_data = dict(
        x=[sun.index[sun_up], sun.index[sun_up], sun.index[-1], sun.index[-1]],
        y=[0, 90, 90, 0],
        mode="lines",
        line=dict(color="Orange", width=1),
        showlegend=False,
        fill="toself",
        name="Sun up",
    )
    sun_dn_data = dict(
        x=[sun.index[sun_dn], sun.index[sun_dn], sun.index[0], sun.index[0]],
        y=[0, 90, 90, 0],
        mode="lines",
        line=dict(color="Orange", width=1),
        showlegend=False,
        fill="toself",
        name="Sun down",
    )
    data = [sun_data, sun_up_data, sun_dn_data, moon_data]
    if (value == "contrast") or (value == "airmass") or (value == "sky_mpsas"):
        data = [sun_up_data, sun_dn_data]
    n_targets = len(target_coords)
    colors = sns.color_palette(n_colors=n_targets).as_hex()

    if target_names:
        # Sort targets by transit time
        records = []
        for target_name in target_names:
            max_alt_time = target_coords[target_name]["alt"].idxmax()
            records.append(dict(target_name=target_name, transit=max_alt_time))
        df_transit = pd.DataFrame(records).sort_values(by="transit")
        sorted_target_names = df_transit["target_name"].values

        for i_target, (color, target_name) in enumerate(
            zip(colors, sorted_target_names)
        ):
            df = target_coords[target_name]

            if filter_targets:
                meridian_at_night = (df["alt"].idxmax() > sun.index[sun_dn]) & (
                    df["alt"].idxmax() < sun.index[sun_up]
                )
                high_at_night = (
                    df.loc[sun.index[sun_dn] : sun.index[sun_up], "alt"].max() > 60
                )
                if not (meridian_at_night or high_at_night):
                    continue
            render_target = True
            notes_text = df_combined.loc[target_name, "NOTE"]
            skip_below_horizon = True
            for horizon_status in ["above", "below"]:
                if (horizon_status == "below") and skip_below_horizon:
                    continue
                if render_target:
                    df0 = df.copy()
                    show_trace = df["alt"] >= f_horizon(np.clip(df["az"], 0, 360))
                    show_trace &= df["moon_distance"] >= min_moon_distance

                    in_legend = True
                    opacity = 1
                    width = 3
                    if horizon_status == "below":
                        show_trace = df["alt"] > -90
                        in_legend = False
                        opacity = 0.15
                        width = 1

                    if show_trace.sum() == 0:
                        render_target = False
                        continue

                    df0.loc[~show_trace, value] = np.nan

                    text = df0.apply(
                        lambda row: f"Notes: {notes_text}<br>Moon distance: {row['moon_distance']:.1f} degrees<br>Local sky brightness (experimental): {row['sky_mpsas']:.2f} mpsas",
                        axis=1,
                    )

                    data.append(
                        dict(
                            x=df0.index,
                            y=df0[value],
                            mode="lines",
                            line=dict(color=color, width=width),
                            showlegend=in_legend,
                            name=target_name,
                            connectgaps=False,
                            legend_group=target_name,
                            customdata=np.dstack(
                                (df0["moon_distance"].values, df0["sky_mpsas"].values)
                            ),
                            hovertext=text,
                            opacity=opacity,
                        )
                    )

    return data, duraion_sun_down_hrs


def parse_loaded_contents(contents, filename, date):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if ".mdb" in filename:
            file_root = filename.replace(".mdb", "")
            local_file = f"./data/uploads/{file_root}.mdb"
            with open(local_file, "wb") as f:
                f.write(decoded)
            object_data = object_file_reader(local_file)
            log.debug(object_data.df_objects.head())
            log.debug(local_file)
        elif ".sgf" in filename:
            out_data = io.StringIO(decoded.decode("utf-8"))
            file_root = filename.replace(".sgf", "")
            local_file = f"./data/uploads/{file_root}.sgf"
            with open(local_file, "w") as f:
                f.write(out_data.read())
            log.info("Done!")
            object_data = object_file_reader(local_file)
        elif ".xml" in filename:
            out_data = io.StringIO(decoded.decode("utf-8"))
            file_root = filename.replace(".xml", "")
            local_file = f"./data/uploads/{file_root}.xml"
            with open(local_file, "w") as f:
                f.write(out_data.read())
            log.info("Done!")
            object_data = object_file_reader(local_file)
        else:
            return None
    except Exception as e:
        log.warning(e)
        return None
    return object_data


def update_weather(site):
    log.debug("Trying NWS")
    nws_forecast = NWS_Forecast(site.lat, site.lon)
    df_weather = nws_forecast.parse_data()

    data = []
    for col in df_weather.columns:
        data.append(
            dict(
                x=df_weather.index,
                y=df_weather[col],
                mode="lines",
                name=col,
                opacity=1,
            )
        )

    graph_data = [
        dcc.Graph(
            config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d"],},
            figure={
                "data": data,
                "layout": dict(
                    title=df_weather.index.name,
                    margin={"l": 50, "b": 100, "t": 50, "r": 50},
                    legend={"x": 1, "y": 0.5},
                    yaxis={"range": [0, 100]},
                    height=600,
                    plot_bgcolor="#ddd",
                    paper_bgcolor="#fff",
                    hovermode="closest",
                    transition={"duration": 150},
                ),
            },
        )
    ]

    clear_outside_link = (
        f"http://clearoutside.com/forecast/{site.lat}/{site.lon}?view=current",
    )
    nws_link = (
        f"http://forecast.weather.gov/MapClick.php?lon={site.lon}&lat={site.lat}#.U1xl5F7N7wI",
    )

    goes_satellite_link = CONFIG.get(
        "goes_satellite_link",
        "https://www.star.nesdis.noaa.gov/GOES/sector_band.php?sat=G16&sector=umv&band=11&length=36",
    )

    return graph_data, clear_outside_link[0], nws_link[0], goes_satellite_link


def target_filter(targets, filters):
    targets_with_filter = []
    for filter in filters:
        for target in targets:
            if target.info["notes"]:
                if filter in target.info["notes"].lower():
                    targets_with_filter.append(target)
        if filter.lower() in TRANSLATED_FILTERS:
            for t_filter in TRANSLATED_FILTERS[filter.lower()]:
                targets_with_filter += [
                    target
                    for target in targets
                    if t_filter.lower() in target.info["notes"].lower()
                ]
    return list(set(targets_with_filter))


@timer
def get_progress_graph(
    df, date_string, days_ago, targets=[], apply_rejection_criteria=True
):

    selection = df["DATE-OBS"] < "1970-01-01"
    if days_ago > 0:
        selection |= df["DATE-OBS"] > str(
            datetime.datetime.today() - datetime.timedelta(days=days_ago)
        )
    targets_in_last_n_days = list(df[selection]["OBJECT"].unique())
    targets += targets_in_last_n_days
    if targets:
        selection |= df["OBJECT"].isin(targets)
    df0 = df[selection]
    df0 = df0.reset_index()
    df0["FILTER"] = df0["FILTER"].replace(FILTER_MAP)
    df0 = df0.dropna(
        subset=[
            "OBJECT",
            "FILTER",
            BINNING_COL,
            FOCALLENGTH_COL,
            INSTRUMENT_COL,
            "is_ok",
            EXPOSURE_COL,
            "OBJCTRA",
        ]
    )

    p = go.Figure()
    if df0.shape[0] == 0:
        return dcc.Graph(figure=p), pd.DataFrame()
    exposure_col = EXPOSURE_COL
    if "is_ok" in df0.columns and apply_rejection_criteria:
        exposure_col = "exposure_ok"
        df0[exposure_col] = df0[EXPOSURE_COL] * (2 * df0["is_ok"].astype(int) - 1)

    df_summary = (
        df0.groupby(
            ["OBJECT", "FILTER", BINNING_COL, FOCALLENGTH_COL, INSTRUMENT_COL, "is_ok"]
        )
        .agg({exposure_col: "sum"})
        .dropna()
    )

    df_summary = df_summary.unstack(1).fillna(0)
    df_summary = df_summary[exposure_col] / 3600

    cols = ["OBJECT"] + FILTER_LIST
    df_summary = df_summary[[col for col in cols if col in df_summary.columns]]

    df0["ra_order"] = df0["OBJCTRA"].apply(
        lambda ra: compute_ra_order(ra, date_string=date_string)
    )

    objects_sorted = (
        df0.dropna(subset=["OBJECT", "ra_order"])
        .groupby("OBJECT")
        .agg({"ra_order": "mean"})
        .sort_values(by="ra_order")
        .index
    )

    df0 = df_summary.reset_index()
    bin = df0[BINNING_COL].astype(int).astype(str)
    fl = df0[FOCALLENGTH_COL].astype(int).astype(str)
    df0["text"] = df0[INSTRUMENT_COL] + " @ " + bin + "x" + bin + " FL = " + fl + "mm"
    df_summary = df0[df0["OBJECT"].isin(objects_sorted)].set_index("OBJECT")
    for filter in [col for col in COLORS if col in df_summary.columns]:
        p.add_trace(
            go.Bar(
                name=f"{filter}",
                x=df_summary.index,
                y=df_summary[filter],
                hovertext=df_summary["text"],
                marker_color=COLORS[filter],
                text=np.abs(df_summary[filter].round(2)),
                textposition="auto",
            )
        )
    p.update_layout(
        barmode=CONFIG.get("progress_mode", "group"),
        yaxis_title="Total Exposure (hr)",
        xaxis_title="Object",
        title="Acquired Data",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0.02),
        title_x=0.5,
    )
    graph = dcc.Graph(
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d"]}, figure=p,
    )

    return graph, df_summary


def store_target_coordinate_data(date_string, site_data):
    object_data = get_object_data()

    t0 = time.time()
    site = update_site(
        site_data,
        default_lat=DEFAULT_LAT,
        default_lon=DEFAULT_LON,
        default_utc_offset=DEFAULT_UTC_OFFSET,
    )
    targets = []
    target_list = object_data.target_list
    for profile in target_list:
        targets += list(target_list[profile].values())

    all_target_coords = get_coordinates(
        targets, date_string, site, time_resolution_in_sec=DEFAULT_TIME_RESOLUTION
    )

    log.info(f"store_target_coordinate_data: {time.time() - t0:.3f}")

    return all_target_coords, targets


def filter_targets_for_matches_and_filters(
    targets, status_matches, filters, profile_list
):
    df_target_status = get_df_from_redis("df_target_status")
    object_data = get_object_data()

    targets = []
    for profile in profile_list:
        if profile in object_data.target_list:
            targets += list(object_data.target_list[profile].values())

    if filters:
        targets = target_filter(targets, filters)

    if status_matches:
        matching_targets = df_target_status[
            df_target_status["status"].isin(status_matches)
        ]["TARGET"].values
        targets = [target for target in targets if target.name in matching_targets]

        log.debug(f"Target matching status {status_matches}: {targets}")

    return targets


def shutdown():
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()


@app.server.route("/getLogs/<file>")
def download_log(file):
    return flask.send_file(
        f"/logs/{file}", attachment_filename=file, as_attachment=True, cache_timeout=-1
    )


def shutdown_watchdog():
    mqtt_client.publish("watchdog", "restart")


# Set layout
app.layout = serve_layout


# Callbacks


@app.callback(Output("date-picker", "date"), [Input("input-utc-offset", "value")])
def set_date(utc_offset):
    df_stars_headers = get_df_from_redis("df_stars_headers")
    df_stored_data = get_df_from_redis("df_stored_data")
    if utc_offset is None:
        utc_offset = DEFAULT_UTC_OFFSET
    date = datetime.datetime.now() + datetime.timedelta(hours=utc_offset)
    df_stars_headers = set_date_cols(df_stars_headers, utc_offset=utc_offset)
    df_stored_data = set_date_cols(df_stored_data, utc_offset=utc_offset)

    push_df_to_redis(df_stars_headers, "df_stars_headers")
    push_df_to_redis(df_stored_data, "df_stored_data")

    return date


@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal_callback(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    [Output("profile-selection", "options"), Output("profile-selection", "value")],
    [Input("upload-data", "contents")],
    [
        State("upload-data", "filename"),
        State("upload-data", "last_modified"),
        State("profile-selection", "value"),
    ],
)
def update_output_callback(
    list_of_contents, list_of_names, list_of_dates, profiles_selected
):
    object_data = get_object_data()
    if not object_data:
        return [{}], [{}]
    if not object_data.profiles:
        return [{}], [{}]

    profile = object_data.profiles[0]

    inactive_profiles = CONFIG.get("inactive_profiles", [])
    default_profiles = CONFIG.get("default_profiles", [])

    options = [
        {"label": profile, "value": profile}
        for profile in object_data.profiles
        if profile not in inactive_profiles
    ]

    default_options = profiles_selected
    if default_profiles:
        default_options = default_profiles

    if list_of_contents is not None:
        for (c, n, d) in zip(list_of_contents, list_of_names, list_of_dates):
            object_data = parse_loaded_contents(c, n, d)
            if object_data:
                for profile in object_data.profiles:
                    if profile not in inactive_profiles:
                        options.append({"label": profile, "value": profile})
        return options, default_options
    return options, default_options


@app.callback(
    [
        Output("weather-graph", "children"),
        Output("clear-outside", "href"),
        Output("nws-weather", "href"),
        Output("goes-satellite", "href"),
    ],
    [Input("store-site-data", "data")],
)
def update_weather_data_callback(site_data):
    site = update_site(
        site_data,
        default_lat=DEFAULT_LAT,
        default_lon=DEFAULT_LON,
        default_utc_offset=DEFAULT_UTC_OFFSET,
    )
    weather_graph, clear_outside_link, nws_link, goes_link = update_weather(site)
    return weather_graph, clear_outside_link, nws_link, goes_link


@app.callback(
    Output("store-site-data", "data"),
    [
        Input("input-lat", "value"),
        Input("input-lon", "value"),
        Input("input-utc-offset", "value"),
    ],
)
def update_time_location_data_callback(lat=None, lon=None, utc_offset=None):

    site_data = dict(lat=DEFAULT_LAT, lon=DEFAULT_LON, utc_offset=DEFAULT_UTC_OFFSET)
    if lat:
        site_data["lat"] = lat
    if lon:
        site_data["lon"] = lon
    if utc_offset:
        site_data["utc_offset"] = utc_offset
    return site_data


@app.callback(
    [
        Output("target-status-match", "options"),
        Output("target-status-match", "value"),
        Output("status-match", "value"),
    ],
    [Input("profile-selection", "value")],
    [State("status-match", "value")],
)
def update_target_for_status_callback(profile_list, status_match):
    df_target_status = get_df_from_redis("df_target_status")
    df_target_status = df_target_status[df_target_status["GROUP"].isin(profile_list)]
    targets = sorted(list(df_target_status["TARGET"].values))
    status_match = CONFIG.get("default_target_status", None)
    return make_options(targets), "", status_match


@app.callback(
    Output("target-status-selector", "value"),
    [Input("target-status-match", "value")],
    [State("store-target-status", "data"), State("profile-selection", "value")],
)
def update_radio_status_for_targets_callback(
    targets, target_status_store, profile_list
):
    df_target_status = get_df_from_redis("df_target_status")
    if targets and profile_list:
        selection = df_target_status["TARGET"].isin(targets)
        selection &= df_target_status["GROUP"].isin(profile_list)
        status_set = df_target_status[selection]["status"].unique()
        if len(status_set) == 1:
            log.debug(f"Fetching targets: {targets} with status of {status_set}")
            return list(status_set)[0]
        else:
            log.debug(
                f"Conflict fetching targets: {targets} with status of {status_set}"
            )


@app.callback(
    Output("store-target-status", "data"),
    [Input("target-status-selector", "value")],
    [State("target-status-match", "value"), State("profile-selection", "value")],
)
def update_target_with_status_callback(status, targets, profile_list):

    df_target_status = get_df_from_redis("df_target_status")
    if status:
        df_combined = get_df_from_redis("df_combined")
        df_target_status = update_targets_with_status(
            targets, status, df_combined, profile_list
        )
    push_df_to_redis(df_target_status, "df_target_status")
    return ""


@app.callback(
    [
        Output("tab-target-div", "style"),
        Output("tab-data-table-div", "style"),
        Output("tab-files-table-div", "style"),
        Output("tab-config-div", "style"),
    ],
    [Input("tabs", "active_tab")],
)
def render_content(tab):

    tab_names = [
        "tab-target",
        "tab-data-table",
        "tab-files-table",
        "tab-config",
    ]

    styles = [{"display": "none"}] * len(tab_names)

    indx = tab_names.index(tab)

    styles[indx] = {}
    return styles


@app.callback(
    Output("dummy-id", "children"),
    [
        Input("button-clear-tables", "n_clicks"),
        Input("button-clear-star-tables", "n_clicks"),
        Input("button-clear-header-tables", "n_clicks"),
        Input("button-clear-target-tables", "n_clicks"),
        Input("button-restart-app", "n_clicks"),
        Input("button-restart-watchdog", "n_clicks"),
    ],
)
def config_buttons(n1, n2, n3, n4, n5, n6):
    ctx = dash.callback_context

    if ctx.triggered:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "button-clear-tables":
            log.info("Clearing all tables")
            init_tables()
        if button_id == "button-clear-star-tables":
            tables_to_clear = [
                "aggregated_star_metrics",
                "xy_frame_metrics",
                "radial_frame_metrics",
            ]
            log.info(f"Clearing tables: {tables_to_clear}")
            clear_tables(tables_to_clear)
        if button_id == "button-clear-header-tables":
            tables_to_clear = [
                "fits_headers",
                "fits_status",
            ]
            log.info(f"Clearing tables: {tables_to_clear}")
            clear_tables(tables_to_clear)
        if button_id == "button-clear-target-tables":
            tables_to_clear = ["targets"]
            log.info(f"Clearing tables: {tables_to_clear}")
            clear_tables(tables_to_clear)
        if button_id == "button-restart-app":
            log.info(f"Restarting app")
            shutdown()
        if button_id == "button-restart-watchdog":
            log.info(f"Restarting watchdog")
            shutdown_watchdog()

    return ""


@app.callback(
    Output("dummy-id-target-data", "children"),
    [Input("date-picker", "date")],
    [State("store-site-data", "data")],
)
def get_target_data(
    date_string, site_data,
):
    global all_target_coords, all_targets
    t0 = time.time()

    all_target_coords, all_targets = store_target_coordinate_data(
        date_string, site_data
    )

    log.info(f"store_target_coordinate_data: {time.time() - t0}")
    return ""


@app.callback(
    Output("dummy-id-contrast-data", "children"),
    [
        Input("dummy-id-target-data", "children"),
        Input("local-mpsas", "value"),
        Input("k-ext", "value"),
    ],
)
def update_contrast(
    dummy_input, local_mpsas, k_ext,
):
    global all_target_coords

    if local_mpsas is None:
        local_mpsas = DEFAULT_MPSAS
    filter_bandwidth = DEFAULT_BANDWIDTH
    if k_ext is None:
        k_ext = DEFAULT_K_EXTINCTION

    all_target_coords = add_contrast(
        all_target_coords,
        n_thread=12,
        filter_bandwidth=filter_bandwidth,
        mpsas=local_mpsas,
        include_airmass=True,
        k_ext=k_ext,
    )

    return ""


@app.callback(
    [
        Output("store-target-data", "data"),
        Output("store-target-list", "data"),
        Output("store-target-metadata", "data"),
        Output("store-dark-sky-duration", "data"),
        Output("loading-output", "children"),
    ],
    [
        Input("dummy-id-contrast-data", "children"),
        Input("store-target-status", "data"),
        Input("y-axis-type", "value"),
        Input("filter-targets", "checked"),
        Input("status-match", "value"),
        Input("filter-match", "value"),
        Input("min-moon-distance", "value"),
    ],
    [State("profile-selection", "value")],
)
def store_data(
    dummy_input,
    target_status_store,
    value,
    filter_targets,
    status_matches,
    filters,
    min_moon_distance,
    profile_list,
):

    global all_target_coords, all_targets

    df_combined = get_df_from_redis("df_combined")

    t0 = time.time()

    if min_moon_distance is None:
        min_moon_distance = DEFAULT_MIN_MOON_DISTANCE

    targets = filter_targets_for_matches_and_filters(
        all_targets, status_matches, filters, profile_list
    )
    target_names = [t.name for t in targets]
    target_names.append("sun")
    target_names.append("moon")

    log.info(f"filter_targets_for_matches_and_filters: {time.time() - t0}")

    target_coords = dict(
        [[k, v] for k, v in all_target_coords.items() if k in target_names]
    )

    sun_down_range = get_time_limits(target_coords)
    log.info(f"get_time_limits: {time.time() - t0}")

    data, duration_sun_down_hrs = get_data(
        target_coords,
        df_combined,
        value=value,
        filter_targets=filter_targets,
        min_moon_distance=min_moon_distance,
    )
    log.info(f"get_data: {time.time() - t0}")

    dark_sky_duration_text = (
        f"Length of sky darkness: {duration_sun_down_hrs:.1f} hours"
    )

    metadata = dict(date_range=sun_down_range, value=value)
    filtered_targets = [d["name"] for d in data if d["name"]]

    return data, filtered_targets, metadata, dark_sky_duration_text, ""


@app.callback(
    [
        Output("target-graph", "children"),
        Output("progress-graph", "children"),
        Output("data-table", "children"),
    ],
    [
        Input("store-target-data", "data"),
        Input("dummy-rejection-criteria-id", "children"),
    ],
    [
        State("store-target-metadata", "data"),
        State("store-dark-sky-duration", "data"),
        State("profile-selection", "value"),
        State("status-match", "value"),
        State("date-picker", "date"),
    ],
)
def update_target_graph(
    target_data,
    rejection_criteria_change_input,
    metadata,
    dark_sky_duration,
    profile_list,
    status_list,
    date,
):
    df_target_status = get_df_from_redis("df_target_status")
    df_combined = get_df_from_redis("df_combined")
    df_reject_criteria = get_df_from_redis("df_reject_criteria_all")

    log.debug(f"Calling update_target_graph")
    if not metadata:
        metadata = {}
    try:
        value = metadata["value"]
        date_range = metadata["date_range"]
    except KeyError:
        return None, None, None

    date_string = str(date.split("T")[0])
    title = "Imaging Targets For Night of {date_string} <br> {dark_sky_duration}".format(
        date_string=date_string, dark_sky_duration=dark_sky_duration
    )

    yaxis_type = "linear"
    if value == "alt":
        y_range = [0, 90]
    elif value == "airmass":
        y_range = [0, 1]
        yaxis_type = "log"
    elif value == "sky_mpsas":
        y_range = [16, 22]
    elif value == "contrast":
        y_range = [0, 1]

    target_graph = dcc.Graph(
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d"]},
        figure={
            "data": target_data,
            "layout": dict(
                xaxis={"title": "", "range": date_range},
                yaxis={"title": yaxis_map[value], "range": y_range, "type": yaxis_type},
                title=title,
                margin={"l": 50, "b": 50, "t": 50, "r": 50},
                height=600,
                plot_bgcolor="#ccc",
                paper_bgcolor="#fff",
                hovermode="closest",
                transition={"duration": 50},
            ),
        },
    )
    selection = df_target_status["GROUP"].isin(profile_list)
    if status_list:
        selection &= df_target_status["status"].isin(status_list)
    targets = list(df_target_status[selection]["TARGET"].values)

    progress_days_ago = int(CONFIG.get("progress_days_ago", 0))

    progress_graph, df_summary = get_progress_graph(
        df_reject_criteria,
        date_string=date_string,
        days_ago=progress_days_ago,
        targets=targets,
        apply_rejection_criteria=True,
    )

    df_combined = df_combined.reset_index()

    df_combined = pd.merge(df_combined, df_target_status, on=["GROUP", "TARGET"])

    cols = [
        "OBJECT",
        "TARGET",
        "GROUP",
        "status",
        "L",
        "R",
        "G",
        "B",
        "Ha",
        "OIII",
        "SII",
        "Instrument",
        "Focal Length",
        "Binning",
        "NOTE",
        "RAJ2000",
        "DECJ2000",
    ]

    columns = []
    for col in cols:
        if col in df_combined.columns:
            entry = {"name": col, "id": col, "deletable": False, "selectable": True}
            columns.append(entry)

    # target table
    data = df_combined.to_dict("records")
    target_table = html.Div(
        [
            dash_table.DataTable(
                id="table-dropdown",
                columns=columns,
                data=data,
                editable=False,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                selected_columns=[],
                selected_rows=[],
                page_action="native",
                page_current=0,
                page_size=50,
                style_cell={"padding": "5px"},
                style_as_list_view=True,
                style_cell_conditional=[
                    {"if": {"column_id": c}, "textAlign": "left"}
                    for c in ["Date", "Region"]
                ],
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "rgb(248, 248, 248)",
                    }
                ],
                style_header={
                    "backgroundColor": "rgb(230, 230, 230)",
                    "fontWeight": "bold",
                },
            )
        ]
    )

    return target_graph, progress_graph, target_table


def add_rejection_criteria(
    df0,
    z_score_thr=2,
    iqr_scale=1.5,
    eccentricity_median_thr=0.6,
    star_trail_strength_thr=25,
    min_star_reduction=0.5,
    new_cols=False,
):
    df0["log_n_stars"] = np.log(df0["n_stars"])
    group_cols = ["OBJECT", "FILTER", "Instrument", "Binning", "Focal Length"]
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

    df0["is_ok"] = 1
    df0.loc[
        df0["bad_star_shape"] | df0["low_star_count"] | df0["high_fwhm"], "is_ok"
    ] = 0

    return df0


@app.callback(
    [
        Output("files-table", "children"),
        Output("summary-table", "children"),
        Output("header-col-match", "options"),
        Output("target-matches", "options"),
        Output("x-axis-field", "options"),
        Output("y-axis-field", "options"),
        Output("scatter-size-field", "options"),
    ],
    [
        Input("store-target-data", "data"),
        Input("header-col-match", "value"),
        Input("target-matches", "value"),
        Input("inspector-dates", "value"),
    ],
)
def update_files_table(target_data, header_col_match, target_matches, inspector_dates):
    df_stars_headers = get_df_from_redis("df_stars_headers")
    df_reject_criteria = get_df_from_redis("df_reject_criteria")

    targets = sorted(df_stars_headers["OBJECT"].unique())
    target_options = make_options(targets)

    df0 = pd.merge(df_stars_headers, df_reject_criteria, on="filename", how="left")
    if target_matches:
        log.info("Selecting target match")
        df0 = df0[df0["OBJECT"].isin(target_matches)]
    if inspector_dates:
        log.info("Selecting dates")
        df0 = df0[df0["date_night_of"].astype(str).isin(inspector_dates)]

    log.info("Done with queries")

    columns = []
    default_cols = [
        "OBJECT",
        "DATE-OBS",
        "FILTER",
        "EXPOSURE",
        "XPIXSZ",
        "FOCALLEN",
        "arcsec_per_pixel",
        "CCD-TEMP",
        "fwhm_median",
        "eccentricity_mean",
        "star_trail_strength",
        "star_orientation_score",
        "is_ok",
    ]
    if "rejected" in df0.columns:
        default_cols += ["rejected"]
    fits_cols = [
        col
        for col in df0.columns
        if col in header_col_match and col not in default_cols
    ]
    fits_cols = default_cols + fits_cols
    other_header_cols = [col for col in df0.columns if col not in default_cols]
    header_options = make_options(other_header_cols)

    for col in fits_cols:
        entry = {"name": col, "id": col, "deletable": False, "selectable": True}
        columns.append(entry)
    df0["FILTER"] = df0["FILTER"].replace(FILTER_MAP)
    df0["FILTER_indx"] = df0["FILTER"].map(
        dict(zip(FILTER_LIST, range(len(FILTER_LIST))))
    )
    df0 = df0.sort_values(by=["OBJECT", "FILTER_indx", "DATE-OBS"]).drop(
        "FILTER_indx", axis=1
    )
    data = df0.round(2).to_dict("records")
    files_table = html.Div(
        [
            dash_table.DataTable(
                columns=columns,
                data=data,
                editable=False,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                selected_columns=[],
                selected_rows=[],
                page_action="native",
                page_current=0,
                page_size=50,
                style_cell={"padding": "5px"},
                style_as_list_view=True,
                style_cell_conditional=[
                    {"if": {"column_id": c}, "textAlign": "left"}
                    for c in ["Date", "Region"]
                ],
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "rgb(248, 248, 248)",
                    }
                ],
                style_header={
                    "backgroundColor": "rgb(230, 230, 230)",
                    "fontWeight": "bold",
                },
            )
        ]
    )
    df0["DATE-OBS"] = pd.to_datetime(df0["DATE-OBS"])
    df_numeric = df0.select_dtypes(
        include=[
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
            "datetime64[ns]",
        ]
    )

    numeric_cols = [col for col in df_numeric.columns if "corr__" not in col]
    scatter_field_options = make_options(numeric_cols)

    df_agg = df0.groupby(["OBJECT", "FILTER", "XBINNING", "FOCALLEN", "XPIXSZ"]).agg(
        {
            "EXPOSURE": "sum",
            "CCD-TEMP": "std",
            "DATE-OBS": "count",
            "star_orientation_score": "mean",
        }
    )
    df_agg["EXPOSURE"] = df_agg["EXPOSURE"] / 3600
    col_map = {
        "DATE-OBS": "n_subs",
        "CCD-TEMP": "CCD-TEMP Dispersion",
        "EXPOSURE": "EXPOSURE (hour)",
    }
    df_agg = df_agg.reset_index().rename(col_map, axis=1)
    df_agg["FILTER_indx"] = df_agg["FILTER"].map(
        dict(zip(FILTER_LIST, range(len(FILTER_LIST))))
    )
    df_agg = df_agg.sort_values(by=["OBJECT", "FILTER_indx"]).drop(
        "FILTER_indx", axis=1
    )
    data = df_agg.round(2).to_dict("records")

    columns = []
    for col in df_agg.columns:
        entry = {"name": col, "id": col, "deletable": False, "selectable": True}
        columns.append(entry)
    summary_table = html.Div(
        [
            dash_table.DataTable(
                columns=columns,
                data=data,
                editable=False,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                selected_columns=[],
                selected_rows=[],
                page_action="native",
                page_current=0,
                page_size=50,
                style_cell={"padding": "5px"},
                style_as_list_view=True,
                style_cell_conditional=[
                    {"if": {"column_id": c}, "textAlign": "left"}
                    for c in ["Date", "Region"]
                ],
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "rgb(248, 248, 248)",
                    }
                ],
                style_header={
                    "backgroundColor": "rgb(230, 230, 230)",
                    "fontWeight": "bold",
                },
            )
        ]
    )

    return (
        files_table,
        summary_table,
        header_options,
        target_options,
        scatter_field_options,
        scatter_field_options,
        scatter_field_options,
    )


@app.callback(
    [Output("x-axis-field", "value"), Output("y-axis-field", "value")],
    [Input("scatter-radio-selection", "value")],
)
def update_scatter_axes(value):
    x_col = "fwhm_median"
    y_col = "eccentricity_median"
    if value:
        x_col, y_col = value.split(" vs. ")
    return x_col, y_col


@app.callback(
    [Output("inspector-dates", "options"), Output("inspector-dates", "value")],
    [Input("store-target-data", "data"), Input("target-matches", "value")],
    [State("inspector-dates", "value")],
)
def update_inspector_dates(target_data, target_matches, selected_dates):

    df_stored_data = get_df_from_redis("df_stored_data")

    df0 = df_stored_data.copy()

    if target_matches:
        normalized_target_matches = [normalize_target_name(t) for t in target_matches]
        df0 = df_stored_data[df_stored_data["OBJECT"].isin(normalized_target_matches)]

    all_dates = list(sorted(df0["date_night_of"].dropna().unique(), reverse=True))
    options = make_options(all_dates)

    default_dates = None
    if selected_dates:
        default_dates = selected_dates

    return options, default_dates


@app.callback(
    Output("dummy-rejection-criteria-id", "children"),
    [
        Input("z-score-field", "value"),
        Input("iqr-scale-field", "value"),
        Input("ecc-thr-field", "value"),
        Input("trail-thr-field", "value"),
        Input("star-frac-thr-field", "value"),
    ],
)
def rejection_criteria_callback(
    z_score_thr,
    iqr_scale,
    eccentricity_median_thr,
    star_trail_strength_thr,
    min_star_reduction,
):
    df_stored_data = get_df_from_redis("df_stored_data")

    df_reject_criteria_all = add_rejection_criteria(
        df_stored_data,
        z_score_thr=z_score_thr,
        iqr_scale=iqr_scale,
        eccentricity_median_thr=eccentricity_median_thr,
        star_trail_strength_thr=star_trail_strength_thr,
        min_star_reduction=min_star_reduction,
    )

    status_map = {False: "&#10004;", True: "&#10006;"}

    df_reject_criteria_all["fwhm_status"] = df_reject_criteria_all["high_fwhm"].replace(
        status_map
    )
    df_reject_criteria_all["ecc_status"] = df_reject_criteria_all["high_ecc"].replace(
        status_map
    )
    df_reject_criteria_all["star_trail_status"] = df_reject_criteria_all[
        "trailing_stars"
    ].replace(status_map)
    df_reject_criteria_all["iqr_status"] = df_reject_criteria_all[
        "star_count_iqr_outlier"
    ].replace(status_map)
    df_reject_criteria_all["star_z_score_status"] = df_reject_criteria_all[
        "bad_star_count_z_score"
    ].replace(status_map)
    df_reject_criteria_all["fwhm_z_score_status"] = df_reject_criteria_all[
        "bad_fwhm_z_score"
    ].replace(status_map)
    df_reject_criteria_all["star_count_fraction_status"] = df_reject_criteria_all[
        "low_star_count_fraction"
    ].replace(status_map)

    cols = [
        "filename",
        "star_count_iqr_outlier",
        "star_count_z_score",
        "fwhm_iqr_outlier",
        "fwhm_z_score",
        "high_ecc",
        "high_fwhm",
        "star_count_fraction",
        "bad_star_count_z_score",
        "bad_fwhm_z_score",
        "trailing_stars",
        "low_star_count_fraction",
        "low_star_count",
        "bad_star_shape",
        "is_ok",
        "fwhm_status",
        "ecc_status",
        "star_trail_status",
        "iqr_status",
        "star_z_score_status",
        "fwhm_z_score_status",
        "star_count_fraction_status",
    ]

    df_reject_criteria = df_reject_criteria_all[cols]

    push_df_to_redis(df_reject_criteria, "df_reject_criteria")
    push_df_to_redis(df_reject_criteria_all, "df_reject_criteria_all")

    return ""


@app.callback(
    [
        Output("target-scatter-graph", "figure"),
        Output("single-target-progress-graph", "children"),
    ],
    [
        Input("store-target-data", "data"),
        Input("inspector-dates", "value"),
        Input("target-matches", "value"),
        Input("x-axis-field", "value"),
        Input("y-axis-field", "value"),
        Input("scatter-size-field", "value"),
        Input("dummy-rejection-criteria-id", "children"),
    ],
)
@timer
def update_scatter_plot(
    target_data, inspector_dates, target_matches, x_col, y_col, size_col, dummy
):
    df_stars_headers = get_df_from_redis("df_stars_headers")
    df_reject_criteria = get_df_from_redis("df_reject_criteria")
    df_reject_criteria_all = get_df_from_redis("df_reject_criteria_all")

    p = go.Figure()
    df0 = df_stars_headers.copy()
    df0["FILTER"] = df0["FILTER"].replace(FILTER_MAP)
    df0 = pd.merge(df0, df_reject_criteria, on="filename", how="left")
    if inspector_dates:
        df0 = df0[df0["date_night_of"].astype(str).isin(inspector_dates)]
        df_reject_criteria_all = df_reject_criteria_all[
            df_reject_criteria_all["date_night_of"].astype(str).isin(inspector_dates)
        ]
    if not target_matches:
        target_matches = sorted(df0["OBJECT"].unique())
    df0 = df0[df0["OBJECT"].isin(target_matches)]
    filters = df0["FILTER"].unique()
    if not x_col:
        x_col = "fwhm_median"
    if not y_col:
        y_col = "eccentricity_median"

    normalized_target_matches = [normalize_target_name(t) for t in target_matches]
    progress_graph, df_summary = get_progress_graph(
        df_reject_criteria_all,
        date_string="2020-01-01",
        days_ago=0,
        targets=normalized_target_matches,
        apply_rejection_criteria=True,
    )

    progress_graph.figure.layout.height = 400

    df0["text"] = df0.apply(
        lambda row: "<br>Object: "
        + str(row["OBJECT"])
        + f"<br>Date: {row['DATE-OBS']}"
        + f"<br>Star count: {row['n_stars']}"
        + (f"<br>{size_col}: {row[size_col]:.2f}" if size_col else "")
        + f"<br>{row['fwhm_status']} FWHM: {row['fwhm_median']:.2f}"
        + f"<br>{row['fwhm_z_score_status']} FWHM Z-score: {row['fwhm_z_score']:.2f}"
        + f"<br>{row['ecc_status']} Eccentricity: {row['eccentricity_median']:.2f}"
        + f"<br>{row['star_trail_status']} Star trail metric: {row['star_trail_strength']:.2f}"
        + f"<br>{row['star_count_fraction_status']} Star count fraction: {row['star_count_fraction']:.2f}"
        + f"<br>{row['iqr_status']} Star IQR reject: {row['star_count_iqr_outlier']}"
        + f"<br>{row['star_z_score_status']} Star Z-score: {row['star_count_z_score']:.2f}"
        + f"<br>Accept frame: {row['is_ok']==1}",
        axis=1,
    )

    group_cols = ["FILTER", "is_ok", "low_star_count", "high_fwhm"]
    inputs = (
        (df0[group_cols].drop_duplicates())
        .sort_values(by=group_cols, ascending=[True, False, False, False])
        .values
    )

    i_filter = 0

    t0 = time.time()
    for filter, status_is_ok, low_star_count, high_fwhm in inputs:

        log.debug(
            f"{status_is_ok}, {low_star_count}, {high_fwhm}, {filter}: {time.time() - t0:.3f}"
        )
        # t0 = time.time()
        selection = df0["FILTER"] == filter
        selection &= df0["is_ok"] == status_is_ok
        selection &= df0["low_star_count"] == low_star_count
        selection &= df0["high_fwhm"] == high_fwhm
        df1 = df0[selection].reset_index()

        sizeref = 1
        size = 10
        if size_col:
            sizeref = float(2.0 * df1[size_col].max() / (5 ** 2))
            default_size = df1[size_col].median()
            if np.isnan(default_size):
                default_size = 1
            size = df1[size_col].fillna(default_size)

        legend_name = filter
        if filter in [HA_FILTER, OIII_FILTER, SII_FILTER]:
            symbol = "diamond"
        elif filter in [L_FILTER, R_FILTER, G_FILTER, B_FILTER]:
            symbol = "circle"
        else:
            symbol = "star"

        if status_is_ok:
            legend_name = f"{legend_name} &#10004; "
        else:
            symbol = "x"
            if low_star_count:
                symbol = f"{symbol}-open"
                legend_name = f"{legend_name} &#10006; - star count"
            elif high_fwhm:
                symbol = f"{symbol}-dot"
                legend_name = f"{legend_name} &#10006; - star bloat"
            else:
                symbol = f"{symbol}-open-dot"
                legend_name = f"{legend_name} &#10006; - star shape"

        if filter in COLORS:
            color = COLORS[filter]
        else:
            color = sns.color_palette(n_colors=len(filters)).as_hex()[i_filter]
            i_filter += 0

        p.add_trace(
            go.Scatter(
                x=df1[x_col],
                y=df1[y_col],
                mode="markers",
                name=legend_name,
                hovertemplate="<b>%{text}</b><br>"
                + f"{x_col}: "
                + "%{x:.2f}<br>"
                + f"{y_col}: "
                + "%{y:.2f}<br>",
                text=df1["text"],
                marker=dict(color=color, size=size, sizeref=sizeref, symbol=symbol),
                customdata=df1["filename"],
            )
        )
    p.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        title=f"Subframe data for {', '.join(normalized_target_matches)}",
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0.02),
    )

    return p, progress_graph


@app.callback(
    [
        Output("radial-frame-graph", "figure"),
        Output("xy-frame-graph", "figure"),
        Output("inspector-frame", "figure"),
        Output("loading-output-click", "children"),
    ],
    [
        Input("target-scatter-graph", "clickData"),
        Input("aberration-preview", "checked"),
        Input("frame-heatmap-dropdown", "value"),
    ],
)
def inspect_frame_analysis(data, as_aberration_inspector, frame_heatmap_col):

    df_stars_headers = get_df_from_redis("df_stars_headers")

    p0 = go.Figure()
    p1 = p0
    p2 = p0

    if data is None:
        log.info("No data")
        return p0, p1, p2, ""
    base_filename = data["points"][0]["customdata"]
    if not base_filename:
        log.info(f"No base filename found")
        return p0, p1, p2, ""

    file_full_path = df_stars_headers[df_stars_headers["filename"] == base_filename]
    if file_full_path.shape[0] != 0:
        log.info(file_full_path)
        filename = file_full_path["file_full_path"].values[0]
    else:
        log.info(f"No full path found for {base_filename}")
        return p0, p1, p2, ""

    t0 = time.time()
    radial_query = (
        f"""select * from radial_frame_metrics where filename = '{base_filename}';"""
    )
    df_radial = pd.read_sql(radial_query, POSTGRES_ENGINE)

    xy_query = f"""select * from xy_frame_metrics where filename = '{base_filename}';"""
    df_xy = pd.read_sql(xy_query, POSTGRES_ENGINE)
    df_xy.drop_duplicates(subset=["filename", "x_bin", "y_bin"], inplace=True)

    log.info(f"Time for query: {time.time() - t0:.2f} seconds")

    p2, canvas = show_inspector_image(
        filename,
        as_aberration_inspector=as_aberration_inspector,
        with_overlay=False,
        n_cols=3,
        n_rows=3,
        border=5,
    )

    p0 = show_fwhm_ellipticity_vs_r(df_radial, filename)
    p1 = show_frame_analysis(df_xy, filename=filename, feature_col=frame_heatmap_col)

    return p0, p1, p2, ""


@app.callback(
    [
        Output("alert-auto", "children"),
        Output("alert-auto", "is_open"),
        Output("alert-auto", "duration"),
        Output("alert-auto", "color"),
    ],
    [Input("interval-component", "n_intervals")],
)
def toggle_alert(n):

    df_stars_headers = get_df_from_redis("df_stars_headers")

    df_old = df_stars_headers.copy()
    update_data()
    df_new = df_stars_headers.drop(df_old.index)
    new_row_count = df_new.shape[0]

    new_files_available = new_row_count > 0

    if new_files_available:
        filenames = df_new["filename"].values
        response = [f"Detected {new_row_count} new files available:"]
        for filename in filenames:
            response.append(html.Br())
            response.append(filename)
        is_open = True
        duration = 60000
        color = "primary"
        return response, is_open, duration, color
    return "", False, 0, "primary"


@app.callback(
    [
        Output("alert-file-skiplist", "children"),
        Output("alert-file-skiplist", "is_open"),
        Output("alert-file-skiplist", "duration"),
        Output("alert-file-skiplist", "color"),
    ],
    [
        Input("button-show-file-skiplist", "n_clicks"),
        Input("button-clear-file-skiplist", "n_clicks"),
    ],
)
def toggle_file_skiplist_alert(n_show, n_clear):

    ctx = dash.callback_context

    if ctx.triggered:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "button-show-file-skiplist":
            file_skiplist = json.loads(REDIS.get("file_skiplist"))
            if file_skiplist:
                n_files = len(file_skiplist)
                response = [f"Skipped processing of {n_files} files:"]
                for filename in file_skiplist:
                    response.append(html.Br())
                    response.append(filename)
                is_open = True
                duration = 60000
                color = "warning"
                return response, is_open, duration, color
            return [f"No files skipped"], True, 5000, "info"
        if button_id == "button-clear-file-skiplist":
            REDIS.set("file_skiplist", "[]")
            response = [f"Clearing file skiplist, will re-process"]
            is_open = True
            duration = 5000
            color = "primary"
            return response, is_open, duration, color
    return "", False, 0, "primary"


@app.callback(
    [Output("glossary-modal", "is_open"), Output("glossary", "children")],
    [Input("glossary-open", "n_clicks"), Input("glossary-close", "n_clicks")],
    [State("glossary-modal", "is_open")],
)
def toggle_glossary_modal_callback(n1, n2, is_open):

    with open("/app/src/glossary.md", "r") as f:
        text = f.readlines()

    status = is_open
    if n1 or n2:
        status = not is_open

    return status, dcc.Markdown(text)


if __name__ == "__main__":
    localhost_only = strtobool(CONFIG.get("localhost_only", "True"))
    debug = strtobool(CONFIG.get("debug", "False"))
    host = "0.0.0.0"
    if localhost_only:
        host = "localhost"
    app.run_server(debug=True, host=host, dev_tools_serve_dev_bundles=True)
