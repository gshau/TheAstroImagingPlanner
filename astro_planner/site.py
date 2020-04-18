import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation

import logging
from astropy.time import TimezoneInfo  # Specifies a timezone


def parse_latlon_string(string, with_hour=False):
    try:
        array = string.split()
        if len(array) >= 1:
            if len(array) == 1:
                logging.warning("Only first triad given")
            result = np.float(array[0])
        if len(array) >= 2:
            result += np.float(array[1]) / 60
        if len(array) >= 3:
            result += np.float(array[2]) / 3600
        return result
    except:
        raise Exception("Issue with conversion: {}".format(string))


class ObservingSite:
    def __init__(self, latitude, longitude, altitude, utc_offset=-6):

        # if type(latitude) != str or type(latitude) != str:
        #     raise Exception('Lat/Lon not string!')
        self.lat = latitude
        self.lon = longitude
        if type(latitude) == str:
            self.lat = parse_latlon_string(latitude)

        if type(latitude) == str:
            self.lon = parse_latlon_string(longitude)
        self.utc_offset = utc_offset

        self.location = EarthLocation(
            lat=self.lat * u.deg, lon=self.lon * u.deg, height=290 * u.m
        )
        self.utc_offset = utc_offset * u.hour
        self.tz = TimezoneInfo(utc_offset=self.utc_offset)  # UTC+1
