import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import TimezoneInfo  # Specifies a timezone
from .logger import log


import datetime
import pytz
from timezonefinder import TimezoneFinder


def get_utc_offset(lat, lon, date_string):
    tf = TimezoneFinder()
    if not lat or not lon or not date_string:
        return 0
    tz_name = tf.timezone_at(lng=lon, lat=lat)
    date = datetime.datetime.fromisoformat(date_string)
    utc_offset = float(pytz.timezone(tz_name).localize(date).strftime("%z")) / 100
    return utc_offset


def get_site(lat, lon, utc_offset):
    site = ObservingSite(lat, lon, 0, utc_offset=utc_offset)
    return site


def parse_latlon_string(string, with_hour=False):
    try:
        array = string.split()
        if len(array) >= 1:
            if len(array) == 1:
                log.warning("Only first triad given")
            result = np.float(array[0])
        if len(array) >= 2:
            result += np.float(array[1]) / 60
        if len(array) >= 3:
            result += np.float(array[2]) / 3600
        return result
    except:
        log.warning(f"Problem with conversion: {string}")
        return np.nan


class ObservingSite:
    def __init__(self, latitude, longitude, altitude, utc_offset=-6):

        self.lat = latitude
        self.lon = longitude
        if type(latitude) == str:
            self.lat = parse_latlon_string(latitude)

        if type(latitude) == str:
            self.lon = parse_latlon_string(longitude)

        self.location = EarthLocation(
            lat=self.lat * u.deg, lon=self.lon * u.deg, height=290 * u.m
        )
        self.utc_offset = utc_offset
        self.tz = TimezoneInfo(utc_offset=self.utc_offset * u.hour)  # UTC+1
