import time
import numpy as np
from functools import partial
from multiprocessing import Pool

from .ephemeris import get_sun, get_moon
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


def get_sky_bkg(df_locs, target_name, mpsas, k_ext):

    df_target = df_locs[target_name]
    time_index = df_target.index
    sbm = SkyBackgroundModel(mpsas=mpsas, k=k_ext)

    df_moon = df_locs["moon"]
    df_sun = df_locs["sun"]
    phase = (distance(df_moon, df_sun, lat_key="dec", long_key="ra") + 360) % 360 - 180
    moon_separation = distance(df_moon, df_target)

    b_moon = sbm._bmoon(phase, moon_separation, df_moon["alt"], df_target["alt"])
    b_moon *= df_moon["alt"] > 0
    # b_moon *= np.exp(-(df_moon['airmass'] - 1) / 4)

    # Ad-hoc extinguish moon light near horizon
    b_moon *= logistic(np.clip(df_moon["airmass"], 1, 1e3), 10, 5)

    # Ad-hoc solar model - good from -5 to -15 altitude: https://www.eso.org/~fpatat/science/skybright/twilight.pdf
    A0 = sbm._mpsas_to_b(11) / np.exp(-5)
    b_sun = A0 * np.exp(df_sun["alt"]) / 1.15

    b_zenith_lp = sbm._mpsas_to_b(mpsas)
    b_altitude_lp = b_zenith_lp * 2.512 ** (df_target["airmass"] * k_ext - 0.16)

    b_all_terms = b_moon
    b_all_terms += b_altitude_lp
    b_all_terms += b_sun

    sky_bkg = sbm._b_to_mpsas(b_all_terms)
    sky_bkg.index = time_index

    sky_bkg[df_target["alt"] < 0] = np.nan
    sky_bkg[sky_bkg < 0] = np.nan

    moon_separation[df_target["alt"] < 0] = np.nan
    moon_separation[sky_bkg < 0] = np.nan
    moon_separation.index = time_index

    return sky_bkg.dropna(), moon_separation.dropna()


def get_contrast(
    df_locs,
    target_name,
    filter_bandwidth=None,
    mpsas=20.2,
    k_ext=0.2,
    include_airmass=True,
):
    sky_bkg, moon_separation = get_sky_bkg(df_locs, target_name, mpsas, k_ext=k_ext)

    bkg = 2.5 ** (-sky_bkg)
    dark_bkg = 2.5 ** (-mpsas)

    # vis_bw = 300
    # if filter_bandwidth:
    #     bkg *= filter_bandwidth / vis_bw
    #     dark_bkg *= filter_bandwidth / vis_bw

    contrast_ratio = dark_bkg / bkg

    fwhm_increase_from_airmass = df_locs[target_name]["airmass"] ** 0.6
    contrast_decrease_from_seeing = fwhm_increase_from_airmass ** 2
    contrast_ratio /= contrast_decrease_from_seeing

    return contrast_ratio, sky_bkg, moon_separation


def _add_contrast(
    l,
    # df,
    df_loc=None,
    filter_bandwidth=300,
    include_airmass=True,
    mpsas=20.2,
    k_ext=0.2,
    t0=0,
):

    target, df = l
    log.info(f"Adding Contrast: {target}: {time.time() - t0:.3f}")
    if target in ["moon", "sun"]:
        return target, df
    df_contrast, sky_bkg, moon_separation = get_contrast(
        df_loc,
        target,
        filter_bandwidth=filter_bandwidth,
        include_airmass=include_airmass,
        mpsas=mpsas,
        k_ext=k_ext,
    )
    df0 = df_loc[target]
    for col in ["contrast"]:
        if col in df0.columns:
            df0 = df0.drop(col, axis=1)
    df0 = df0.join(df_contrast.to_frame("contrast"))
    df0 = df0.join(sky_bkg.to_frame("sky_mpsas"))
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
