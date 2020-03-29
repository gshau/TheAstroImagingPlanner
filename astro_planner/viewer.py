import numpy as np
import pandas as pd
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

# from astroplan import FixedTarget
from astropy.wcs import WCS
from astropy.coordinates import Angle
from .stf import auto_stf


def plot_finder_image(
    target,
    survey="DSS2 Red",
    fov_radius=30 * u.arcmin,
    fov=None,
    log=False,
    ax=None,
    grid=False,
    reticle=False,
    style_kwargs=None,
    reticle_style_kwargs=None,
    pixels=500,
    inverted=False,
    stf=True,
):
    """
    Plot survey image centered on ``target``.

    Survey images are retrieved from NASA Goddard's SkyView service via
    ``astroquery.skyview.SkyView``.

    If a `~matplotlib.axes.Axes` object already exists, plots the finder image
    on top. Otherwise, creates a new `~matplotlib.axes.Axes`
    object with the finder image.

    Parameters
    ----------
    target : `~astroplan.FixedTarget`, `~astropy.coordinates.SkyCoord`
        Coordinates of celestial object

    survey : string
        Name of survey to retrieve image from. For dictionary of
        available surveys, use
        ``from astroquery.skyview import SkyView; SkyView.list_surveys()``.
        Defaults to ``'DSS'``, the Digital Sky Survey.

    fov_radius : `~astropy.units.Quantity`
        Radius of field of view of retrieved image. Defaults to 10 arcmin.

    log : bool, optional
        Take the natural logarithm of the FITS image if `True`.
        False by default.

    ax : `~matplotlib.axes.Axes` or None, optional.
        The `~matplotlib.axes.Axes` object to be drawn on.
        If None, uses the current `~matplotlib.axes.Axes`.

    grid : bool, optional.
        Grid is drawn if `True`. `False` by default.

    reticle : bool, optional
        Draw reticle on the center of the FOV if `True`. Default is `False`.

    style_kwargs : dict or `None`, optional.
        A dictionary of keywords passed into `~matplotlib.pyplot.imshow`
        to set plotting styles.

    reticle_style_kwargs : dict or `None`, optional
        A dictionary of keywords passed into `~matplotlib.pyplot.axvline` and
        `~matplotlib.pyplot.axhline` to set reticle style.

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
        Matplotlib axes with survey image centered on ``target``

    hdu : `~astropy.io.fits.PrimaryHDU`
        FITS HDU of the retrieved image


    Notes
    -----
    Dependencies:
        In addition to Matplotlib, this function makes use of astroquery and WCSAxes.
    """

    import matplotlib.pyplot as plt
    from astroquery.skyview import SkyView

    coord = target if not hasattr(target, "coord") else target.coord
    position = coord.icrs
    coordinates = "icrs"
    target_name = None if isinstance(target, SkyCoord) else target.name

    if fov:
        width, height = fov
        hdu = SkyView.get_images(
            position=position,
            coordinates=coordinates,
            survey=survey,
            width=width,
            height=height,
            grid=grid,
            pixels=pixels,
        )[0][0]
    else:
        hdu = SkyView.get_images(
            position=position,
            coordinates=coordinates,
            survey=survey,
            radius=fov_radius,
            grid=grid,
            pixels=pixels,
        )[0][0]

    wcs = WCS(hdu.header)

    # Set up axes & plot styles if needed.
    if ax is None:
        ax = plt.gca(projection=wcs)
    if style_kwargs is None:
        style_kwargs = {}
    style_kwargs = dict(style_kwargs)
    if inverted:
        style_kwargs.setdefault("cmap", "Greys")
    else:
        style_kwargs.setdefault("cmap", "Greys_r")
    style_kwargs.setdefault("origin", "lower")

    if stf:
        image_data = auto_stf(hdu.data)
        style_kwargs.setdefault("vmin", 0)
        style_kwargs.setdefault("vmax", 1)

    elif log:
        image_data = np.log(hdu.data)
    else:
        image_data = hdu.data
    ax.imshow(image_data, **style_kwargs)

    # Draw reticle
    if reticle:
        pixel_width = image_data.shape[0]
        inner, outer = 0.03, 0.08

        if reticle_style_kwargs is None:
            reticle_style_kwargs = {}
        reticle_style_kwargs.setdefault("linewidth", 2)
        reticle_style_kwargs.setdefault("color", "m")

        ax.axvline(
            x=0.5 * pixel_width,
            ymin=0.5 + inner,
            ymax=0.5 + outer,
            **reticle_style_kwargs
        )
        ax.axvline(
            x=0.5 * pixel_width,
            ymin=0.5 - inner,
            ymax=0.5 - outer,
            **reticle_style_kwargs
        )
        ax.axhline(
            y=0.5 * pixel_width,
            xmin=0.5 + inner,
            xmax=0.5 + outer,
            **reticle_style_kwargs
        )
        ax.axhline(
            y=0.5 * pixel_width,
            xmin=0.5 - inner,
            xmax=0.5 - outer,
            **reticle_style_kwargs
        )

    # Labels, title, grid
    ax.set(xlabel="RA", ylabel="DEC")
    if target_name is not None:
        ax.set_title(target_name)
    ax.grid(grid)

    # Redraw the figure for interactive sessions.
    ax.figure.canvas.draw()
    return ax, hdu
