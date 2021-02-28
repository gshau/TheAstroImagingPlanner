import numpy as np
from astropy.time import Time
from astropy.coordinates import AltAz
from astropy.coordinates import get_sun, get_moon
import pandas as pd

from astropy.utils import iers
from .logger import log
from .utils import timer


from .fast_ephemeris.ephemeris import (
    get_moon_data,
    get_sun_data,
    get_alt_az,
    get_airmass,
)

iers.conf.auto_download = False


def dates_to_strings(dates):
    return [date.strftime("%Y%m%d %H%M") for date in dates]


@timer
def get_coordinates(targets, date_string, site, time_resolution_in_sec=60):
    log.debug("Starting get_coords with fast version")

    local_dates = pd.date_range(
        "{} 12:00:00".format(date_string),
        freq="{}s".format(time_resolution_in_sec),
        periods=24 * (60 * 60 / time_resolution_in_sec),
        # tz=site.tz,
    )

    utc_dates = local_dates - pd.Timedelta(hours=site.utc_offset)

    ephem_dict = {}
    ephem_dict["sun"] = get_sun_data(utc_dates, latitude=site.lat, longitude=site.lon)
    ephem_dict["moon"] = get_moon_data(utc_dates, latitude=site.lat, longitude=site.lon)

    for name, df in ephem_dict.items():
        df.index = local_dates
        ephem_dict[name] = df

    utc_dates_series = pd.Series(utc_dates)

    for target in targets:
        log.info(target)
        df = get_alt_az(
            utc_dates_series,
            latitude=site.lat,
            longitude=site.lon,
            ra=target.ra.value,
            dec=target.dec.value,
        )

        ephem_dict[target.name] = df

    return ephem_dict
