"""Conservative luminance correction for OSNet crops."""
from math import isfinite

import cv2
import numpy as np


class ShadowGamma:
    """Brighten underexposed BGR uint8 crops, leaving unreliable extremes alone."""

    def __init__(self, gamma=0.8, dark_luma=80):
        if not isfinite(gamma) or not 0 < gamma < 1:
            raise ValueError('appearance gamma must be between 0 and 1 (exclusive)')
        if isinstance(dark_luma, bool) or not isinstance(dark_luma, int) or not 12 <= dark_luma <= 180:
            raise ValueError('appearance dark luma must be an integer from 12 to 180')
        self.tag = f'shadow-gamma-ycrcb-v1:g={gamma!r}:dark={dark_luma}'
        self.dark_luma = dark_luma
        self.lut = np.rint(255 * (np.arange(256) / 255) ** gamma).astype(np.uint8)

    def __call__(self, crop):
        if (crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3
                or not crop.size):
            raise ValueError('expected a nonempty BGR uint8 crop')
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        luma = ycrcb[:, :, 0]
        # Ignore much of the surrounding background and look at the central crop.
        h, w = luma.shape
        center = luma[h // 5:h - h // 5, w // 5:w - w // 5]
        median, upper, highlight = np.percentile(center, [50, 75, 95])
        if median >= self.dark_luma or upper <= 12 or highlight >= 240:
            return crop.copy()
        ycrcb[:, :, 0] = cv2.LUT(luma, self.lut)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
