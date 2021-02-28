import numpy as np


def get_jd(dt):
    y = 4716
    j = 1401
    m = 2
    n = 12
    r = 4
    p = 1461
    q = 0
    v = 3
    u = 5
    s = 153
    t = 2
    w = 2
    A = 184
    B = 274277
    C = -38
    h = dt.month.values - m
    g = dt.year.values + y - ((n - h) / n).astype(int)
    f = np.mod(h - 1 + n, n)
    e = ((p * g + q) / r).astype(int) + dt.day.values - 1 - j
    J = e + ((s * f + t) / u).astype(int)
    J -= 3 * (((g + A) / 100).astype(int) / 4).astype(int) + C
    J = (
        J.astype(float)
        - 0.5
        + (dt.hour.values + dt.minute.values / 60 + dt.second.values / 3600) / 24
    )
    return J


def get_gmst(jd):
    gmst = 18.697374558 + 24.06570982441908 * (jd - 2451545)
    return np.mod(gmst, 24)


def get_lmst(gmst, longitude):
    lmst = gmst + longitude / 15
    return np.mod(lmst, 24)


def get_local_sidereal_time(date_time, longitude):
    jd = get_jd(date_time)
    lmst = get_lmst(get_gmst(jd), longitude)
    return lmst
