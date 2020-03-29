import numpy as np
import pandas as pd
from .ephemeris import get_sun, get_moon
from astropy.time import Time

from astroplan.moon import moon_illumination


def moon_phase(date_string):
    return moon_illumination(Time(date_string))


RADIANS_PER_DEGREE = np.pi / 180.0


class SkyBackgroundModel:
    def __init__(self, mpsas):
        self.mpsas = mpsas

    def _bmoon(self, phase, separation, moon_alt, sky_alt, k=0.16, aod=0):
        #     http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1991PASP..103.1033K&defaultprint=YES&filetype=.pdf
        alpha = phase
        rho = separation * RADIANS_PER_DEGREE
        Z = (90 - sky_alt) * RADIANS_PER_DEGREE
        Zm = (90 - moon_alt) * RADIANS_PER_DEGREE

        Istar = 10 ** (-0.4 * (3.84 + 0.026 * np.abs(alpha) + 4e-9 * alpha ** 4))
        X = lambda z: (1 - 0.96 * np.sin(z) ** 2) ** (-0.5)
        f = lambda rho: 10 ** 5.36 * (1.06 + np.cos(rho) ** 2) + 10 ** (
            6.15 - rho / 40.0
        )

        result = (
            f(rho) * Istar * 10 ** (-0.4 * k * X(Zm)) * (1 - 10 ** (-0.4 * k * X(Z)))
        )
        return result

    def _moon_phase(self, time):
        sun = get_sun(time)
        moon = get_moon(time)
        return sun.separation(moon).degree

    #     def sky_brightness(self, target, ):
    #         return -2.5 * np.log10((self._bmoon(x, 30, 60, 40) + self._mpsas_to_b(self.mpsas)) / self._mpsas_to_b(self.mpsas)) + self.mpsas

    def _b_to_mpsas(self, b):
        return (20.7233 - np.log(b / 34.08)) / 0.92104

    def _mpsas_to_b(self, mpsas):
        return 34.08 * np.exp(20.7233 - 0.92104 * mpsas)


def distance(target_1, target_2, lat_key="alt", long_key="az"):
    deg_to_radian = np.pi / 180.0
    # haversine formula
    dlon = target_1[long_key] - target_2[long_key]
    dlat = target_2[lat_key] - target_1[lat_key]
    a = (
        np.sin(dlat / 2 * deg_to_radian) ** 2
        + np.cos(target_1[lat_key] * deg_to_radian)
        * np.cos(target_2[lat_key] * deg_to_radian)
        * np.sin(dlon * deg_to_radian / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return c / deg_to_radian


def logistic(x, x0, tau):
    return 1 / (1 + np.exp((x - x0) / tau))


def get_sky_bkg(df_locs, target_name, mpsas):

    df_target = df_locs[target_name]
    sbm = SkyBackgroundModel(mpsas=mpsas)

    df_moon = df_locs["moon"]
    df_sun = df_locs["sun"]
    date = df_sun.index
    #     moon = get_moon(Time(date))
    phase = (distance(df_moon, df_sun, lat_key="dec", long_key="ra") + 360) % 360 - 180
    separation = distance(df_moon, df_target)

    b_moon = sbm._bmoon(phase, separation, df_moon["alt"], df_target["alt"], k=0.2)
    b_moon *= df_moon["alt"] > 0
    #     b_moon *= np.exp(-(df_moon['airmass'] - 1) / 4)
    b_moon *= logistic(df_moon["airmass"], 5, 1)

    # Ad-hoc solar model - good from -5 to -15 altitude: https://www.eso.org/~fpatat/science/skybright/twilight.pdf
    A0 = sbm._mpsas_to_b(11) / np.exp(-5)
    b_sun = A0 * np.exp(df_sun["alt"]) / 1.15

    # Ad-hoc LP model
    b_lp = 2000 * np.exp(-(df_target["alt"] / 20 - 1))

    b_all_terms = b_moon + 0 * b_lp + b_sun + sbm._mpsas_to_b(mpsas)

    sky_bkg = sbm._b_to_mpsas(b_all_terms)

    sky_bkg[df_target["alt"] < 0] = np.nan
    sky_bkg[sky_bkg < 0] = np.nan

    return sky_bkg.dropna()


def get_contrast(
    df_locs,
    target_name,
    filter_bandwidth=None,
    mpsas=20.2,
    object_brightness=19,
    include_airmass=True,
):
    sky_bkg = get_sky_bkg(df_locs, target_name, mpsas)

    vis_bw = 300

    obj = 2.5 ** (-object_brightness)
    bkg = 2.5 ** (-sky_bkg)
    dark_bkg = 2.5 ** (-mpsas)

    if filter_bandwidth:
        bkg *= filter_bandwidth / vis_bw

    contrast = obj / (bkg)
    contrast_dark = obj / (dark_bkg)
    # contrast = obj / bkg

    contrast_ratio = contrast / contrast_dark

    if include_airmass:
        contrast_ratio *= np.exp(-(df_locs[target_name]["airmass"] - 1))

    return contrast_ratio


def add_contrast(
    df_loc, filter_bandwidth=300, include_airmass=True, mpsas=20.2, object_brightness=19
):
    result = {}
    for target in df_loc.keys():
        if target in ["moon", "sun"]:
            result[target] = df_loc[target]
            continue
        df_contrast = get_contrast(
            df_loc,
            target,
            filter_bandwidth=filter_bandwidth,
            include_airmass=include_airmass,
            mpsas=mpsas,
            object_brightness=object_brightness,
        )
        df0 = df_loc[target]
        if "contrast" in df0.columns:
            df0 = df0.drop("contrast", axis=1)
        result[target] = df0.join(df_contrast.to_frame("contrast"))
    return result

