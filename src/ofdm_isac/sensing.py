"""Radar sensing processor: Rx/Tx division -> 2D FFT/IFFT -> range-Doppler map -> peak detection.

Sign convention (must match channel.py's phase ramps):
  - delay ramp is exp(-j*2*pi*k*Delta_f*tau)  -> IFFT over the subcarrier axis
    collapses it to a peak at bin round(R/range_resolution); range >= 0, no shift needed.
  - Doppler ramp is exp(+j*2*pi*n*T_sym*f_D)  -> FFT over the symbol axis collapses
    it to a peak; velocity can be negative, so fftshift only this axis.
Swapping IFFT/FFT or shifting the wrong axis produces a mirrored/wrong Doppler peak.
"""
import functools

import numpy as np
from scipy import ndimage

from .config import SystemConfig


def magnitude_db(x: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.abs(x) + 1e-12)


def estimate_channel(rx_grid: np.ndarray, tx_grid: np.ndarray) -> np.ndarray:
    """H_hat = Rx/Tx. Safe: normalized square QAM excludes the origin, so |tx_grid| has a
    strictly positive floor and this never divides by (near) zero."""
    return rx_grid / tx_grid


@functools.lru_cache(maxsize=4)
def _hann_window_2d(n_subcarriers: int, n_symbols: int) -> np.ndarray:
    return np.outer(np.hanning(n_subcarriers), np.hanning(n_symbols))


def range_doppler_map(h_hat: np.ndarray) -> np.ndarray:
    """Hann-windowed on both axes before the FFT/IFFT: the target's true range/velocity
    rarely land exactly on a bin, and a rectangular window's slowly-decaying sidelobes
    (Dirichlet kernel) show up as a bright cross through the peak. Hann trades a ~1.5x
    wider main lobe (resolution formulas in config.py are the rectangular-window ideal)
    for ~18 dB better sidelobe suppression (-13 dB -> -31 dB first sidelobe).

    The window only depends on h_hat's (constant, per-run) shape, so it's cached rather
    than rebuilt on every call -- this runs thousands of times across the Monte Carlo
    sweeps in experiments.py."""
    windowed = h_hat * _hann_window_2d(*h_hat.shape)

    range_domain = np.fft.ifft(windowed, axis=0, norm="ortho")
    doppler_domain = np.fft.fft(range_domain, axis=1, norm="ortho")
    return np.fft.fftshift(doppler_domain, axes=1)


def range_doppler_axes(cfg: SystemConfig) -> tuple[np.ndarray, np.ndarray]:
    range_axis = np.arange(cfg.n_subcarriers) * cfg.range_resolution_m
    velocity_axis = (np.arange(cfg.n_symbols) - cfg.n_symbols // 2) * cfg.velocity_resolution_mps
    return range_axis, velocity_axis


def find_peak(rd_map: np.ndarray, cfg: SystemConfig) -> tuple[float, float, float]:
    """Returns (R_hat, v_hat, peak_magnitude_db)."""
    range_axis, velocity_axis = range_doppler_axes(cfg)
    magnitude = np.abs(rd_map)
    idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)

    r_hat = range_axis[idx[0]]
    v_hat = velocity_axis[idx[1]]
    peak_db = magnitude_db(magnitude[idx])

    return float(r_hat), float(v_hat), float(peak_db)


def cfar_detect_2d(rd_map: np.ndarray, cfg: SystemConfig) -> list[tuple[float, float, float]]:
    """CA-CFAR (cell-averaging constant-false-alarm-rate) detection, for scenes with an
    unknown number of targets where a single argmax isn't enough.

    For every cell, estimate the local noise floor from a ring of "training" cells,
    excluding a "guard" band immediately around the cell (so a target's own sidelobe
    energy doesn't inflate its own noise estimate). Declare a detection where the cell's
    power exceeds that estimate scaled by a threshold set from the desired false-alarm
    probability (Rohling 1983, cell-averaging CFAR):

        Pfa = (1 + alpha)^(-N)  =>  alpha = Pfa^(-1/N) - 1

    where N is the number of training cells and the threshold is alpha * (sum of training
    cell powers) -- not the mean, the derivation is over the raw sum of N iid exponential
    (noise power) samples. Adjacent detection cells around the same true peak are merged
    via connected-component labeling, keeping the strongest cell per component.

    Returns a list of (R_hat, v_hat, peak_db), strongest first.
    """
    range_axis, velocity_axis = range_doppler_axes(cfg)
    magnitude = np.abs(rd_map)
    power = magnitude**2

    guard_r, guard_v = cfg.cfar_guard_cells
    train_r, train_v = cfg.cfar_training_cells
    outer_size = (2 * (guard_r + train_r) + 1, 2 * (guard_v + train_v) + 1)
    inner_size = (2 * guard_r + 1, 2 * guard_v + 1)

    # DFT bins are periodic, so wrap at the edges rather than padding with zeros.
    outer_sum = ndimage.uniform_filter(power, size=outer_size, mode="wrap") * (
        outer_size[0] * outer_size[1]
    )
    inner_sum = ndimage.uniform_filter(power, size=inner_size, mode="wrap") * (
        inner_size[0] * inner_size[1]
    )
    training_sum = outer_sum - inner_sum
    n_training = outer_size[0] * outer_size[1] - inner_size[0] * inner_size[1]

    alpha = cfg.cfar_pfa ** (-1.0 / n_training) - 1.0
    detection_mask = power > alpha * training_sum

    labeled, n_components = ndimage.label(detection_mask)
    detections = []
    for label_id in range(1, n_components + 1):
        region_mag = np.where(labeled == label_id, magnitude, -np.inf)
        idx = np.unravel_index(np.argmax(region_mag), region_mag.shape)
        detections.append((
            float(range_axis[idx[0]]),
            float(velocity_axis[idx[1]]),
            float(magnitude_db(magnitude[idx])),
        ))

    detections.sort(key=lambda d: d[2], reverse=True)
    return detections
