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


def get_target_loc(target, dates, location):
    pydatetimes = dates.to_pydatetime()
    frame = AltAz(obstime=Time(pydatetimes), location=location)
    loc = target.transform_to(frame)
    df = pd.DataFrame([loc.alt.degree, loc.az.degree,
                       loc.secz.value]).T
    df.index = dates
    df.columns = ['alt', 'az', 'airmass']
    df.loc[df['airmass'] < 0, 'airmass'] = np.nan
    return df


def get_moon_loc(dates, location):
    pydatetimes = dates.to_pydatetime()
    frame = AltAz(obstime=Time(pydatetimes), location=location)
    obj = get_moon(Time(pydatetimes))
    loc = obj.transform_to(frame)
    df = pd.DataFrame([loc.alt.degree, loc.az.degree,
                       loc.secz.value, obj.ra.degree, obj.dec.degree]).T
    df.index = dates
    df.columns = ['alt', 'az', 'airmass', 'ra', 'dec']
    return df


def get_sun_loc(dates, location):
    pydatetimes = dates.to_pydatetime()
    frame = AltAz(obstime=Time(pydatetimes, location=location),
                  location=location)
    obj = get_sun(Time(pydatetimes))
    loc = obj.transform_to(frame)
    df = pd.DataFrame([loc.alt.degree, loc.az.degree,
                       loc.secz.value, obj.ra.degree, obj.dec.degree]).T
    df.index = dates
    df.columns = ['alt', 'az', 'airmass', 'ra', 'dec']
    return df


def get_coords(targets, date_string, site, time_resolution_in_sec=60):
    date = pd.date_range('{} 12:00:00'.format(date_string), freq='{}s'.format(
        time_resolution_in_sec), periods=24*(60 * 60 / time_resolution_in_sec), tz=site.tz)
    moon = get_moon_loc(date, location=site.location)
    sun = get_sun_loc(date, location=site.location)
    df_ephem = {'sun': sun, 'moon': moon}
    for target in targets:
        df_ephem[target.name] = get_target_loc(
            target.target, date, location=site.location)
    return df_ephem


def show_target_traces(targets, date_string, site, time_resolution_in_sec=60, value='alt'):

    df_locations = get_coords(targets, date_string,
                              site, time_resolution_in_sec=60)

    sun = df_locations['sun']
    moon = df_locations['moon']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sun.index, y=sun[value], mode='lines', line=dict(
        color='Orange', width=3), name='Sun'))

    sun_up = np.where(np.gradient((sun.alt > 0).astype(int)) > 0)[0][0]
    sun_dn = np.where(np.gradient((sun.alt > 0).astype(int)) < 0)[0][0]
    fig.add_trace(go.Scatter(x=[sun.index[sun_up], sun.index[sun_up], sun.index[-1], sun.index[-1]],
                             y=[0, 90, 90, 0],
                             mode='lines', line=dict(color='Orange', width=1), showlegend=False, fill='toself'))
    fig.add_trace(go.Scatter(x=[sun.index[sun_dn], sun.index[sun_dn], sun.index[0], sun.index[0]],
                             y=[0, 90, 90, 0],
                             mode='lines', line=dict(color='Orange', width=1), showlegend=False, fill='toself'))

    i_up = np.where(np.gradient((sun.alt > -15).astype(int)) > 0)[0][0]
    i_dn = np.where(np.gradient((sun.alt > -15).astype(int)) < 0)[0][0]
    fig.add_trace(go.Scatter(x=[sun.index[i_up], sun.index[i_up], sun.index[-1], sun.index[-1]],
                             y=[0, 90, 90, 0],
                             mode='lines', line=dict(color='Orange', width=1), showlegend=False, fill='toself'))
    fig.add_trace(go.Scatter(x=[sun.index[i_dn], sun.index[i_dn], sun.index[0], sun.index[0]],
                             y=[0, 90, 90, 0],
                             mode='lines', line=dict(color='Orange', width=1), showlegend=False, fill='toself'))

    fig.add_trace(go.Scatter(x=moon.index, y=moon[value], mode='lines', line=dict(
        color='Gray', width=2), name='Moon'))
    n_targets = len(targets)

    colors = sns.color_palette('colorblind', n_colors=n_targets).as_hex()
    for color, target in zip(colors, targets):
        df = df_locations[target.name]
        fig.add_trace(go.Scatter(x=df.index, y=df[value], mode='lines', line=dict(
            color=color, width=2), name=target.name))
        if value == 'alt':
            fig.add_trace(go.Scatter(x=[df[value].idxmax()] * 2, y=[0, df[value].max()], mode='lines',
                                     line=dict(color=color, width=2, dash='dot'), showlegend=False))
        elif value == 'airmass':
            fig.add_trace(go.Scatter(x=[df[value].idxmin()] * 2, y=[df[value].min(), 100], mode='lines',
                                     line=dict(color=color, width=2, dash='dot'), showlegend=False))

    fig.update_layout(plot_bgcolor='#ccc')
    if value == 'alt':
        fig.update_yaxes(range=[0, 90], showgrid=False)
    elif value == 'airmass':
        fig.update_yaxes(range=[1, 3], showgrid=False)
    fig.update_xaxes(showgrid=False, range=[
                     sun.index[sun_dn], sun.index[sun_up]])
    # fig.show()

    return fig


#     from astroplan.moon import moon_illumination
#     def moon_phase(date_string):
#         return moon_illumination(Time(date_string))
