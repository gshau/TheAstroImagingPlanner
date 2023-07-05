import os
import requests


from functools import lru_cache

# from lxml import html
from dash import dcc

import pandas as pd
import numpy as np
import seaborn as sns

from scipy.interpolate import interp1d

from astro_planner.utils import timer
from astro_planner.logger import log
from astro_planner.weather import NWS_Forecast
from astro_planner.globals import TRANSLATED_FILTERS
from astro_planner.site import get_site
from astro_planner.ephemeris import get_coordinates

from astro_planner.sky_brightness import LightPollutionMap
from config import PlannerConfig


# TODO: REMOVE THIS IMPORT
from dash import dcc


def load_custom_horizon_function(config):
    horizon_data = config.get("horizon_data", {})
    horizon_file = horizon_data.get("horizon_file")
    if not os.path.exists(horizon_file):
        horizon_file = None
    flat_horizon_alt = horizon_data.get("flat_horizon_alt", 0)
    flat_horizon = interp1d([0, 360], [flat_horizon_alt, flat_horizon_alt])
    if horizon_file is None:
        return flat_horizon
    try:
        df_horizon = pd.read_csv(horizon_file, sep=" ", header=3)
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
            f"Problem with setting custom horizon from file: {horizon_file}, using flat horizon at {flat_horizon_alt} degrees"
        )
        return flat_horizon
    return f_horizon


@timer
def get_time_limits(targets, sun_alt=-5):
    sun = targets["sun"]
    # Get sun up/down
    sun_up = np.where(np.gradient((sun.alt > sun_alt).astype(int)) > 0)[0][0]
    sun_dn = np.where(np.gradient((sun.alt > sun_alt).astype(int)) < 0)[0][0]
    return sun.index[sun_dn], sun.index[sun_up]


@timer
def get_target_ephemeris_data_for_plotly(
    df_target_status,
    target_coords,
    df_targets,
    profile_list,
    config,
    value="alt",
    sun_alt_for_twilight=-18,
    filter_targets=True,
    min_moon_distance=30,
    color_palette="bright",
):

    df_target_status = df_target_status.query("GROUP in @profile_list").set_index(
        "TARGET"
    )

    target_names = [
        name for name in (list(target_coords.keys())) if name not in ["sun", "moon"]
    ]

    df_moon = target_coords["moon"]
    moon_transit = str(df_moon["alt"].idxmax())
    if "phase" in df_moon.columns:
        df_moon["text"] = df_moon.apply(
            lambda row: f"Moon<br>Transit: {moon_transit}<br>Phase: {row['phase']:.1f}%",
            axis=1,
        )
    else:
        df_moon["text"] = df_moon.apply(
            lambda row: f"Moon<br>Transit: {moon_transit}",
            axis=1,
        )

    moon_data = dict(
        x=df_moon.index,
        y=df_moon["alt"],
        text=df_moon["text"],
        opacity=1,
        line=dict(color="#333333", width=4),
        name="Moon",
    )
    sun_color = "#f0ad4e"
    sun_data = dict(
        x=target_coords["sun"].index,
        y=target_coords["sun"]["alt"],
        text="Sun",
        opacity=1,
        line=dict(color=sun_color, width=4),
        name="Sun",
    )

    # Get sun up/down
    sun = target_coords["sun"]
    sun_below_threshold = (sun.alt < sun_alt_for_twilight).astype(int)
    has_dark_sky = sun_below_threshold.sum() > 0

    if has_dark_sky:
        sun_up = np.where(np.gradient(sun_below_threshold) < 0)[0][0]
        sun_dn = np.where(np.gradient(sun_below_threshold) > 0)[0][0]
    else:
        sun_up = 0
        sun_dn = 0

    duraion_sun_down = sun.index[sun_up] - sun.index[sun_dn]
    duraion_sun_down_hrs = duraion_sun_down.total_seconds() / 3600.0

    log.info(
        f"Sun down from {sun.index[sun_dn]} to {sun.index[sun_up]}, duration: {duraion_sun_down_hrs:.2f} hours"
    )
    data = []
    sun_up_data = dict(
        x=[sun.index[sun_up], sun.index[sun_up], sun.index[-1], sun.index[-1]],
        y=[0, 90, 90, 0],
        mode="lines",
        line=dict(color=sun_color, width=1),
        showlegend=False,
        fill="toself",
        name="Sun up",
    )
    sun_dn_data = dict(
        x=[sun.index[sun_dn], sun.index[sun_dn], sun.index[0], sun.index[0]],
        y=[0, 90, 90, 0],
        mode="lines",
        line=dict(color=sun_color, width=1),
        showlegend=False,
        fill="toself",
        name="Sun down",
    )
    data.append(sun_up_data)
    data.append(sun_dn_data)

    data.append(sun_data)
    data.append(moon_data)

    if (value == "contrast") or (value == "airmass") or (value == "sky_mpsas"):
        if has_dark_sky:
            data = [sun_up_data, sun_dn_data]
        else:
            data = []
    n_targets = len(target_coords)
    if color_palette == "base":
        base_colors = [
            "#1f77b4",  # muted blue
            "#ff7f0e",  # safety orange
            "#2ca02c",  # cooked asparagus green
            "#d62728",  # brick red
            "#9467bd",  # muted purple
            "#8c564b",  # chestnut brown
            "#e377c2",  # raspberry yogurt pink
            "#7f7f7f",  # middle gray
            "#bcbd22",  # curry yellow-green
            "#17becf",  # blue-teal
        ]

        colors = [base_colors[i % len(base_colors)] for i in range(n_targets)]
    else:
        colors = sns.color_palette(color_palette, n_colors=n_targets).as_hex()

    targets_available = target_names
    if targets_available:
        # Sort targets by transit time
        records = []
        for target_name in targets_available:
            max_alt_time = target_coords[target_name]["alt"].idxmax()
            records.append(dict(target_name=target_name, transit=max_alt_time))
        df_transit = pd.DataFrame(records).sort_values(by="transit")
        sorted_target_names = df_transit["target_name"].values

        for i_target, (color, target_name) in enumerate(
            zip(colors, sorted_target_names)
        ):
            df = target_coords[target_name]
            if filter_targets:
                if has_dark_sky:
                    meridian_at_night = (df["alt"].idxmax() > sun.index[sun_dn]) & (
                        df["alt"].idxmax() < sun.index[sun_up]
                    )
                    high_at_night = (
                        df.loc[sun.index[sun_dn] : sun.index[sun_up], "alt"].max() > 60
                    )
                else:
                    meridian_at_night = False
                    midnight_index = sun["alt"].idxmin()
                    high_at_night = df.loc[midnight_index, "alt"] > 60

                if not (meridian_at_night or high_at_night):
                    continue
            render_target = True
            df0 = df_target_status.loc[target_name]
            if isinstance(df0, pd.DataFrame):
                status = df0["status"].values[0]
                priority = df0["priority"].values[0]
                profile_name = df0["GROUP"].values[0]
            else:
                status = df0["status"]
                priority = df0["priority"]
                profile_name = df0["GROUP"]
            dash_styles = ["", "dash", "dot", "dashdot"]

            priority_value = config.get("valid_priorities").index(priority) + 1
            notes_text = df_targets.loc[
                df_targets["TARGET"] == target_name, "NOTE"
            ].values.flatten()
            profile = df_targets.loc[
                df_targets["TARGET"] == target_name, "GROUP"
            ].values.flatten()
            skip_below_horizon = True
            f_horizon = load_custom_horizon_function(config)
            for horizon_status in ["above", "below"]:
                if (horizon_status == "below") and skip_below_horizon:
                    continue
                if render_target:
                    df0 = df.copy()
                    show_trace = df["alt"] >= f_horizon(np.clip(df["az"], 0, 360))
                    show_trace &= df["moon_distance"] >= min_moon_distance

                    in_legend = True
                    opacity = 1
                    width = 2
                    opacity = (1 + priority_value / 5.0) / 2
                    width = priority_value / 1.25
                    if profile_name in profile_list:
                        profile_index = profile_list.index(profile_name)
                    else:
                        profile_index = 0

                    if show_trace.sum() == 0:
                        render_target = False
                        continue

                    df0.loc[~show_trace, value] = np.nan
                    transit_time = str(df0["alt"].idxmax())
                    text = df0.apply(
                        lambda row: f"Target: {target_name}<br>Profile: {profile}<br>Transit: {transit_time}<br>Status: {status}<br>Priority: {priority}<br>Notes: {notes_text}<br>Moon distance: {row['moon_distance']:.1f} degrees<br>Local sky brightness (experimental): {row['sky_mpsas']:.2f} mpsas",
                        axis=1,
                    )

                    data.append(
                        dict(
                            x=df0.index,
                            y=df0[value],
                            mode="lines",
                            line=dict(
                                color=color,
                                width=width,
                                dash=dash_styles[profile_index % len(dash_styles)],
                            ),
                            showlegend=in_legend,
                            name=target_name,
                            connectgaps=False,
                            customdata=np.dstack(
                                (df0["moon_distance"].values, df0["sky_mpsas"].values)
                            ),
                            hovertext=text,
                            opacity=opacity,
                        )
                    )

    return data, duraion_sun_down_hrs


@timer
def update_weather(site, config):
    log.debug("Trying NWS")
    nws_forecast = NWS_Forecast(site.lat, site.lon)
    if nws_forecast.xmldoc:
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
                config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d"]},
                figure={
                    "data": data,
                    "layout": dict(
                        title=df_weather.index.name,
                        margin={"l": 50, "b": 100, "t": 50, "r": 50},
                        legend={"x": 1, "y": 0.5},
                        yaxis={"range": [0, 100]},
                        height=400,
                        width=750,
                        plot_bgcolor="#ccc",
                        paper_bgcolor="#fff",
                        hovermode="closest",
                        transition={"duration": 150},
                    ),
                },
            )
        ]
    else:
        graph_data = []

    clear_outside_link = (
        f"http://clearoutside.com/forecast/{site.lat}/{site.lon}?view=current",
    )
    nws_link = (
        f"http://forecast.weather.gov/MapClick.php?lon={site.lon}&lat={site.lat}#.U1xl5F7N7wI",
    )

    goes_satellite_link = config.get(
        "goes_satellite_link",
        "https://www.star.nesdis.noaa.gov/GOES/sector_band.php?sat=G16&sector=umv&band=11&length=36",
    )

    clear_outside_forecast_img = f"http://clearoutside.com/forecast_image_large/{np.round(site.lat, 2)}/{np.round(site.lon, 2)}/forecast.png"

    return (
        graph_data,
        clear_outside_link[0],
        nws_link[0],
        goes_satellite_link,
        clear_outside_forecast_img,
    )


@timer
def filter_targets(targets, filters):
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
def store_target_coordinate_data(
    target_data, date_string, lat, lon, utc_offset, config
):

    site = get_site(lat=lat, lon=lon, utc_offset=utc_offset)
    targets = []
    target_list = target_data.target_list
    for profile in target_list:
        targets += list(target_list[profile].values())

    all_target_coords = get_coordinates(
        targets,
        date_string,
        site,
        time_resolution_in_sec=config.get("planner_config", {}).get(
            PlannerConfig.TIME_RESOLUTION, 300
        ),
    )

    return all_target_coords, targets


# @lru_cache(maxsize=1000)
# def get_sqm(lat, lon):
#     url = f"http://clearoutside.com/forecast/{lat}/{lon}"
#     response = requests.get(url)
#     status_code = response.status_code
#     log.info(f"Got code: {status_code}")
#     if status_code == 200:
#         root = html.fromstring(response.text)
#         a = root.xpath('//span[contains(@class, "glyphicon-eye-open")]')
#         sqm, bortle_class = [l.text for l in a[0].xpath("//strong")][:2]
#         sqm = float(sqm)
#         return sqm, bortle_class
#     return np.nan, ""


def get_mpsas_from_lat_lon(lat, lon):
    try:
        # raise Exception
        lp_map = LightPollutionMap()
        mpsas = lp_map.mpsas_for_location(lat, lon)
    except Exception:
        # mpsas, bortle_class = get_sqm(lat, lon)
        mpsas = np.nan
    return np.round(mpsas, 2)
