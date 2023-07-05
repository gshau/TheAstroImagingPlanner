import numpy as np

DEG_TO_RADIAN = np.pi / 180.0


def distance(ra_1, dec_1, ra_2, dec_2):
    # haversine formula
    d_ra_degrees = np.mod((ra_1 - ra_2) * 15, 360)
    d_dec = dec_1 - dec_2
    a = (
        np.sin(d_dec / 2 * DEG_TO_RADIAN) ** 2
        + np.cos(dec_1 * DEG_TO_RADIAN)
        * np.cos(dec_2 * DEG_TO_RADIAN)
        * np.sin(d_ra_degrees * DEG_TO_RADIAN / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return c / DEG_TO_RADIAN
