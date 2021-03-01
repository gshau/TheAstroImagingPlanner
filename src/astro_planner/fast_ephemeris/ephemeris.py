import ephem
import numpy as np
import pandas as pd
from .time import get_local_sidereal_time


DEGREE_TO_RADIAN = np.pi / 180.0


def get_airmass(alt):
    # Airmass calculation of Young 1994: https://doi.org/10.1364/AO.33.001108
    cos_z_t = np.cos((90 - alt) * DEGREE_TO_RADIAN)
    airmass = (1.002432 * cos_z_t ** 2 + 0.148386 * cos_z_t + 0.0096467) / (
        cos_z_t ** 3 + 0.149864 * cos_z_t ** 2 + 0.0102963 * cos_z_t + 0.000303978
    )
    airmass[alt < 0] = np.nan
    return airmass


def get_moon_data(time_utc, latitude, longitude):
    moon = ephem.Moon()
    home = ephem.Observer()
    home.lat = str(latitude)
    home.lon = str(longitude)
    moon_records = []
    for date in time_utc:
        home.date = date
        moon.compute(home)
        moon_records.append(
            dict(
                time=date,
                ra=moon.a_ra * 24 / (2 * np.pi),
                dec=moon.a_dec * 360 / (2 * np.pi),
                alt=moon.alt * 360 / (2 * np.pi),
                az=moon.az * 360 / (2 * np.pi),
                phase=moon.phase,
            )
        )
    df = pd.DataFrame(moon_records)
    df = get_alt_az(df["time"], latitude, longitude, df["ra"], df["dec"])
    return df


def get_sun_data(time_utc, latitude, longitude):
    sun = ephem.Sun()
    home = ephem.Observer()
    home.lat = str(latitude)
    home.lon = str(longitude)
    sun_records = []
    for date in time_utc:
        home.date = date
        sun.compute(home)
        sun_records.append(
            dict(
                time=date,
                ra=sun.a_ra * 24 / (2 * np.pi),
                dec=sun.a_dec * 360 / (2 * np.pi),
            )
        )
    df = pd.DataFrame(sun_records)
    df = get_alt_az(df["time"], latitude, longitude, df["ra"], df["dec"])
    return df


def get_alt_az(time_utc, latitude, longitude, ra, dec, as_vector=False):
    lst = get_local_sidereal_time(time_utc.dt, longitude)
    hour_angle = lst - ra
    if as_vector:
        dec_vec = dec
        ra_vec = ra
    else:
        dec_vec = np.ones(time_utc.shape) * dec
        ra_vec = np.ones(time_utc.shape) * ra
    phi = (lst - ra_vec) / 24 * 2 * np.pi
    theta = (90 - latitude) * DEGREE_TO_RADIAN
    A1 = np.array(
        [
            -np.sin(phi) * np.cos(dec_vec * DEGREE_TO_RADIAN),
            -np.cos(phi) * np.cos(dec_vec * DEGREE_TO_RADIAN),
            np.sin(dec_vec * DEGREE_TO_RADIAN),
        ]
    )
    R = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), np.sin(theta)],
            [0, -np.sin(theta), np.cos(theta)],
        ]
    )
    A2 = R.dot(A1)
    alt = np.arcsin(A2[2, :]) * 180 / np.pi
    az = np.mod(np.arctan2(A2[0, :], A2[1, :]) * 180 / np.pi, 360)
    airmass = get_airmass(alt)

    result = time_utc.values, alt, az, airmass, ra_vec, dec_vec, lst, hour_angle
    df = pd.DataFrame()
    for name, r in zip(
        ["time", "alt", "az", "airmass", "ra", "dec", "lst", "hour_angle"], result
    ):
        df[name] = r
    return df.set_index("time").astype(float)
