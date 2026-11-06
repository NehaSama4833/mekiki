"""Shifts applied to 8x8 digit images (pixel range 0..16)."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, rotate


def noise(X, rng, sigma): return np.clip(X + rng.normal(0, sigma, X.shape), 0, 16)
def brightness(X, rng, f): return np.clip(X * f, 0, 16)
def dropout(X, rng, p): return X * (rng.random(X.shape) > p)
def blur(X, rng, s): return np.stack([gaussian_filter(x.reshape(8, 8), s).ravel() for x in X])
def rotation(X, rng, deg): return np.stack([np.clip(rotate(x.reshape(8, 8), deg, reshape=False, order=1, mode="constant"), 0, 16).ravel() for x in X])


FEATURE_SHIFTS = {
    "gaussian noise": (noise, [1, 2, 4, 6]),
    "brightness": (brightness, [0.4, 0.7, 1.4, 2.0]),
    "pixel dropout": (dropout, [0.1, 0.3, 0.5]),
    "blur": (blur, [0.5, 1.0, 1.5]),
    "rotation": (rotation, [10, 20, 35]),
}


def label_shift(X, y, rng, frac_low=0.9):
    """Resample so that `frac_low` of examples are digits 0-4 (features untouched)."""
    low, high = np.where(y < 5)[0], np.where(y >= 5)[0]
    n = len(y)
    idx = np.concatenate([rng.choice(low, int(n * frac_low)), rng.choice(high, n - int(n * frac_low))])
    return X[idx], y[idx]
