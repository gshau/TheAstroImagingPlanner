import numpy as np
import astropy.units as u
import seaborn as sns
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import get_sun, get_moon
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from astropy.utils import iers

iers.conf.auto_download = False
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(module)s %(message)s")
log = logging.getLogger("app")


def dates_to_strings(dates):
    return [date.strftime("%Y%m%d %H%M") for date in dates]


def get_target_loc(target, dates, location):
    pydatetimes = dates.to_pydatetime()
    frame = AltAz(obstime=Time(pydatetimes), location=location)
    loc = target.transform_to(frame)
    df = pd.DataFrame([loc.alt.degree, loc.az.degree, loc.secz.value]).T

    df.index = dates
    df.columns = ["alt", "az", "airmass"]
    df.loc[df["airmass"] < 0, "airmass"] = np.nan
    return df


from multiprocessing import Pool
from functools import partial

import time


def get_sun_moon_loc(dates, location):
    pydatetimes = dates.to_pydatetime()
    frame = AltAz(obstime=Time(pydatetimes), location=location)
    result = {}
    for obj_name in ["sun", "moon"]:
        t0 = time.time()
        if obj_name == "moon":
            obj = get_moon(Time(pydatetimes))
        if obj_name == "sun":
            obj = get_sun(Time(pydatetimes))
        log.info(f"Time for {obj_name} is {time.time() - t0}")
        loc = obj.transform_to(frame)
        df = pd.DataFrame(
            [
                loc.alt.degree,
                loc.az.degree,
                loc.secz.value,
                obj.ra.degree,
                obj.dec.degree,
            ]
        ).T
        df.index = dates
        df.columns = ["alt", "az", "airmass", "ra", "dec"]
        result[obj_name] = df
    return result


def get_coords(targets, date_string, site, time_resolution_in_sec=60):
    log.info("Starting get_coords")

    dates = pd.date_range(
        "{} 12:00:00".format(date_string),
        freq="{}s".format(time_resolution_in_sec),
        periods=24 * (60 * 60 / time_resolution_in_sec),
        tz=site.tz,
    )
    log.info("Getting Sun/Moon")
    df_ephem = get_sun_moon_loc(dates, location=site.location)
    log.info(".")
    t0 = time.time()
    gtl = partial(get_target_loc, dates=dates, location=site.location)
    with Pool(8) as pool:
        result = pool.map(gtl, [target.target for target in targets])
    for name, df in zip([target.name for target in targets], result):
        df_ephem[name] = df
    log.info(f"Time elapsed: {time.time() - t0}")

    log.info("Done get_coords")

    return df_ephem
