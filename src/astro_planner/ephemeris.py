import numpy as np
from astropy.time import Time
from astropy.coordinates import AltAz
from astropy.coordinates import get_sun, get_moon
import pandas as pd

from astropy.utils import iers
from multiprocessing import Pool
from functools import partial

from .logger import log


iers.conf.auto_download = False


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


def get_sun_moon_loc(dates, location):
    pydatetimes = dates.to_pydatetime()
    frame = AltAz(obstime=Time(pydatetimes), location=location)
    result = {}
    for obj_name in ["sun", "moon"]:
        if obj_name == "moon":
            obj = get_moon(Time(pydatetimes))
        if obj_name == "sun":
            obj = get_sun(Time(pydatetimes))
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


def get_coordinates(targets, date_string, site, time_resolution_in_sec=60):
    log.debug("Starting get_coords")

    dates = pd.date_range(
        "{} 12:00:00".format(date_string),
        freq="{}s".format(time_resolution_in_sec),
        periods=24 * (60 * 60 / time_resolution_in_sec),
        tz=site.tz,
    )
    ephem_dict = get_sun_moon_loc(dates, location=site.location)
    gtl = partial(get_target_loc, dates=dates, location=site.location)
    with Pool(8) as pool:
        result = pool.map(gtl, [target.target for target in targets])
    for name, df in zip([target.name for target in targets], result):
        ephem_dict[name] = df

    return ephem_dict
