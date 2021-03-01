import time
import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool

from astropy.coordinates import get_sun, get_moon
from .logger import log


RADIANS_PER_DEGREE = np.pi / 180.0


class SkyBackgroundModel:
    def __init__(self, mpsas, k=0.16):
        self.mpsas = mpsas
        self.k = k

    def _bmoon(self, phase, separation, moon_alt, sky_alt, aod=0):
        # From http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1991PASP..103.1033K&defaultprint=YES&filetype=.pdf
        alpha = phase
        rho = separation * RADIANS_PER_DEGREE
        Z = (90 - sky_alt) * RADIANS_PER_DEGREE
        Zm = (90 - moon_alt) * RADIANS_PER_DEGREE

        Istar = 10 ** (-0.4 * (3.84 + 0.026 * np.abs(alpha) + 4e-9 * alpha ** 4))

        def X(z):
            return (1 - 0.96 * np.sin(z) ** 2) ** (-0.5)

        def f(rho_arg):
            return 10 ** 5.36 * (1.06 + np.cos(rho_arg) ** 2) + 10 ** (
                6.15 - rho_arg / 40.0
            )

        result = (
            f(rho)
            * Istar
            * 10 ** (-0.4 * self.k * X(Zm))
            * (1 - 10 ** (-0.4 * self.k * X(Z)))
        )
        return result

    def _moon_phase(self, time):
        sun = get_sun(time)
        moon = get_moon(time)
        return sun.separation(moon).degree

    def _b_to_mpsas(self, b):
        return (20.7233 - np.log(b / 34.08)) / 0.92104

    def _mpsas_to_b(self, mpsas):
        return 34.08 * np.exp(20.7233 - 0.92104 * mpsas)


def distance(target_1, target_2, lat_key="alt", long_key="az"):
    deg_to_radian = np.pi / 180.0
    hour_to_radian = 2 * np.pi / 24
    lon_scale = deg_to_radian
    if long_key == "ra":
        lon_scale = hour_to_radian
    # haversine formula
    dlon = target_1[long_key].values - target_2[long_key].values
    dlat = target_2[lat_key].values - target_1[lat_key].values
    a = (
        np.sin(dlat / 2 * deg_to_radian) ** 2
        + np.cos(target_1[lat_key].values * deg_to_radian)
        * np.cos(target_2[lat_key].values * deg_to_radian)
        * np.sin(dlon * lon_scale / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return c / deg_to_radian


def logistic(x, x0, tau):
    return 1 / (1 + np.exp((x - x0) / tau))


def get_sky_bkg(df_locs, target_name, mpsas, k_ext):

    df_target = df_locs[target_name]
    time_index = df_target.index
    sbm = SkyBackgroundModel(mpsas=mpsas, k=k_ext)

    df_moon = df_locs["moon"]
    df_sun = df_locs["sun"]
    moon_separation = pd.Series(distance(df_moon, df_target), index=df_target.index)

    b_moon = sbm._bmoon(
        df_locs["moon"]["phase"].values,
        moon_separation.values,
        df_moon["alt"].values,
        df_target["alt"].values,
    )

    b_moon *= df_moon["alt"].values > 0
    # b_moon *= np.exp(-(df_moon['airmass'] - 1) / 4)

    # Ad-hoc extinguish moon light near horizon
    b_moon *= logistic(np.clip(df_moon["airmass"].fillna(1e3).values, 1, 1e3), 10, 5)

    # opposition effect increase to _+35%
    # http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1991PASP..103.1033K&defaultprint=YES&filetype=.pdf
    b_moon *= (
        1 + np.exp(-((np.abs(df_moon["phase"].values)) ** 2) / (2 * 3 ** 2)) * 0.35
    )

    # Ad-hoc solar model - good from -5 to -15 altitude: https://www.eso.org/~fpatat/science/skybright/twilight.pdf
    A0 = sbm._mpsas_to_b(11) / np.exp(-5)
    b_sun = A0 * np.exp(df_sun["alt"].values) / 1.15

    b_zenith_lp = sbm._mpsas_to_b(mpsas)
    b_altitude_lp = b_zenith_lp * 2.512 ** (
        np.clip(df_target["airmass"].values * k_ext - 0.16, None, 20)
    )

    # For airglow impact, and scatter
    # https://arxiv.org/pdf/0905.3404.pdf
    # https://arxiv.org/pdf/0709.0813.pdf
    f = 0.6
    airglow_airmass_dependence = -2.5 * np.log(
        (1 - f) + f * df_target["airmass"].values
    ) + k_ext * (df_target["airmass"].values - 1)

    airglow_airmass_dependence = 0

    sky_bkg = pd.Series(
        sbm._b_to_mpsas(b_altitude_lp + b_moon + b_sun), index=df_target.index
    )
    sky_bkg += airglow_airmass_dependence
    sky_bkg.index = time_index

    sky_bkg.loc[df_target["alt"] < 0] = np.nan
    sky_bkg.loc[sky_bkg < 0] = np.nan

    sky_bkg_no_moon = pd.Series(sbm._b_to_mpsas(b_altitude_lp), index=df_target.index)
    sky_bkg_no_moon += airglow_airmass_dependence
    sky_bkg_no_moon.index = time_index

    sky_bkg_no_moon.loc[df_target["alt"] < 0] = np.nan
    sky_bkg_no_moon.loc[sky_bkg_no_moon < 0] = np.nan

    moon_separation.index = time_index
    moon_separation.loc[df_target["alt"] < 0] = np.nan
    moon_separation.loc[sky_bkg < 0] = np.nan

    return sky_bkg.dropna(), sky_bkg_no_moon.dropna(), moon_separation.dropna()


def get_contrast(
    df_locs,
    target_name,
    filter_bandwidth=None,
    mpsas=20.2,
    k_ext=0.2,
    include_airmass=True,
):
    sky_bkg, sky_bkg_no_moon, moon_separation = get_sky_bkg(
        df_locs, target_name, mpsas, k_ext=k_ext
    )

    bkg = 2.5 ** (-sky_bkg)
    bkg_no_moon = 2.5 ** (-sky_bkg_no_moon)
    dark_bkg = 2.5 ** (-mpsas)

    # vis_bw = 300
    # if filter_bandwidth:
    #     bkg *= filter_bandwidth / vis_bw
    #     dark_bkg *= filter_bandwidth / vis_bw

    contrast_ratio = dark_bkg / bkg
    contrast_ratio_no_moon = dark_bkg / bkg_no_moon

    fwhm_increase_from_airmass = df_locs[target_name]["airmass"] ** 0.6
    contrast_decrease_from_seeing = fwhm_increase_from_airmass ** 2
    contrast_ratio /= contrast_decrease_from_seeing

    return (
        contrast_ratio,
        contrast_ratio_no_moon,
        sky_bkg,
        sky_bkg_no_moon,
        moon_separation,
    )


def _add_contrast(
    target_df,
    df_loc=None,
    filter_bandwidth=300,
    include_airmass=True,
    mpsas=20.2,
    k_ext=0.2,
    t0=0,
):

    target, df = target_df
    log.info(f"Adding Contrast: {target}: {time.time() - t0:.3f}")
    if target in ["moon", "sun"]:
        return target, df
    (
        df_contrast,
        df_contrast_no_moon,
        sky_bkg,
        sky_bkg_no_moon,
        moon_separation,
    ) = get_contrast(
        df_loc,
        target,
        filter_bandwidth=filter_bandwidth,
        include_airmass=include_airmass,
        mpsas=mpsas,
        k_ext=k_ext,
    )
    df0 = df_loc[target]
    for col in [
        "contrast",
        "contrast_no_moon",
        "sky_mpsas",
        "sky_mpsas_no_moon",
        "moon_distance",
    ]:
        if col in df0.columns:
            df0 = df0.drop(col, axis=1)
    df0 = df0.join(df_contrast.to_frame("contrast"))
    df0 = df0.join(df_contrast_no_moon.to_frame("contrast_no_moon"))
    df0 = df0.join(sky_bkg.to_frame("sky_mpsas"))
    df0 = df0.join(sky_bkg_no_moon.to_frame("sky_mpsas_no_moon"))
    df0 = df0.join(moon_separation.to_frame("moon_distance"))
    return target, df0


def add_contrast(
    df_loc,
    n_thread=4,
    filter_bandwidth=300,
    include_airmass=True,
    mpsas=20.2,
    k_ext=0.2,
):
    result = {}
    t0 = time.time()
    df_loc["moon"]["phase"] = (
        distance(df_loc["moon"], df_loc["sun"], lat_key="dec", long_key="ra") + 360
    ) % 360 - 180

    if n_thread > 1:
        func = partial(
            _add_contrast,
            df_loc=df_loc,
            filter_bandwidth=filter_bandwidth,
            include_airmass=include_airmass,
            mpsas=mpsas,
            k_ext=k_ext,
            t0=t0,
        )
        with Pool(n_thread) as pool:
            result = dict(pool.map(func, df_loc.items()))
    else:
        for target, df in df_loc.items():
            target, df0 = _add_contrast(
                [target, df],
                df_loc=df_loc,
                filter_bandwidth=filter_bandwidth,
                include_airmass=include_airmass,
                mpsas=mpsas,
                k_ext=k_ext,
                t0=t0,
            )
            result[target] = df0

    return result
