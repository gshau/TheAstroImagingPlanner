import numpy as np

TARGET_BKG = 0.05
SHADOWS_CLIP = -2.8


def mtf(m, x):
    return (m - 1) * x / ((2 * m - 1) * x - m)


def auto_stf(data, target_bkg=TARGET_BKG, shadow_clip=SHADOWS_CLIP):
    data = data / data.max() / 2
    median = np.median(data)
    mad = np.median(np.abs(data - np.median(data)))
    c0 = median + shadow_clip * mad
    c0 = 1 if c0 > 1 else 0 if c0 < 0 else c0
    m = mtf(target_bkg, median - c0)
    stf = mtf(m, data - c0)
    return stf
