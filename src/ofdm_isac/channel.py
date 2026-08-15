"""Reflecting-target channel: delay/Doppler phase ramps applied to the Tx grid, plus AWGN.

A round-trip delay tau = 2R/c is a linear phase ramp across subcarriers;
a Doppler shift f_D = 2v*f_c/c is a linear phase ramp across OFDM symbols.
Applying both directly to the frequency-domain Tx grid (rather than
propagating a continuous time-domain waveform) is the standard OFDM-radar
formulation and is exactly what makes Rx_grid = Tx_grid * H work.
"""
from dataclasses import dataclass

import numpy as np

from .config import C, SystemConfig


@dataclass
class Target:
    range_m: float
    velocity_mps: float
    amplitude: complex = 1.0 + 0j


def build_channel_matrix(cfg: SystemConfig, target: Target) -> np.ndarray:
    """H[k, n] for subcarrier k, OFDM symbol n: shape (n_subcarriers, n_symbols)."""
    tau = 2 * target.range_m / C
    f_doppler = 2 * target.velocity_mps * cfg.carrier_freq_hz / C

    k = np.arange(cfg.n_subcarriers)[:, None]
    n = np.arange(cfg.n_symbols)[None, :]

    delay_ramp = np.exp(-1j * 2 * np.pi * k * cfg.subcarrier_spacing_hz * tau)
    doppler_ramp = np.exp(1j * 2 * np.pi * n * cfg.symbol_duration_total_s * f_doppler)

    return target.amplitude * delay_ramp * doppler_ramp


def two_antenna_effective_ranges(
    range_m: float, angle_rad: float, antenna_spacing_m: float
) -> tuple[float, float]:
    """Exact geometry for a 2-element receive array (elements at +-spacing/2 along the
    baseline) plus a single co-located transmitter at the array's phase center: the
    reflection's outbound leg (transmitter -> target, distance range_m) is common to both
    receive channels, and only the inbound leg (target -> antenna_i) differs by antenna
    position -- no far-field approximation, just literal Pythagorean distance, which
    reduces to the textbook d*sin(angle) approximation on its own once range_m is much
    larger than antenna_spacing_m (always true here: meters vs. millimeters).
    """
    target_x = range_m * np.sin(angle_rad)
    target_y = range_m * np.cos(angle_rad)
    inbound_0 = np.hypot(target_x + antenna_spacing_m / 2, target_y)
    inbound_1 = np.hypot(target_x - antenna_spacing_m / 2, target_y)
    return float((range_m + inbound_0) / 2), float((range_m + inbound_1) / 2)


def carrier_phase_factor(cfg: SystemConfig, effective_range_m: float) -> complex:
    """The absolute carrier-frequency phase exp(-j*2*pi*f_c*tau), tau = 2*range_m/c.
    """
    tau = 2 * effective_range_m / C
    return complex(np.exp(-1j * 2 * np.pi * cfg.carrier_freq_hz * tau))


def apply_precomputed_channel(
    tx_grid: np.ndarray,
    h_total: np.ndarray,
    signal_power: float,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Same as apply_channel, but takes an already-built channel matrix and its signal
    power instead of a target list. For callers that apply the same fixed target(s)
    across many trials (e.g. the Monte Carlo sweeps in experiments.py) -- H depends only
    on cfg and the targets, not on the trial's random bits/noise, so rebuilding it
    thousands of times from build_channel_matrix would be pure waste."""
    signal = tx_grid * h_total
    noise_var = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_var / 2) * (
        rng.standard_normal(tx_grid.shape) + 1j * rng.standard_normal(tx_grid.shape)
    )
    return signal + noise


def apply_channel(
    tx_grid: np.ndarray,
    cfg: SystemConfig,
    targets: list[Target],
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rx_grid = Tx_grid * sum(H_target) + AWGN, calibrated so snr_db is per-RE Es/N0."""
    h_total = sum(build_channel_matrix(cfg, t) for t in targets)
    signal_power = sum(abs(t.amplitude) ** 2 for t in targets)
    return apply_precomputed_channel(tx_grid, h_total, signal_power, snr_db, rng)
