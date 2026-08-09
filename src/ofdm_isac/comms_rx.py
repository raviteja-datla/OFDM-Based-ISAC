"""OFDM comms receiver: zero-forcing equalization, QAM demod, BER.

zero_force_equalize expects h_hat estimated from an *independent* reference
grid (see sensing.estimate_channel), not from the data grid's own Rx/Tx —
using a grid's own division as its own equalizer would trivially recover the
exact transmitted symbols regardless of noise.
"""
import numpy as np
from scipy.special import erfc

from .ofdm_tx import ints_to_bits, qam_constellation


def qam_demod(rx_symbols: np.ndarray, order: int) -> np.ndarray:
    """Nearest-level decision per I/Q dimension, then inverse Gray mapping."""
    levels, _ = qam_constellation(order)
    sqrt_m = levels.size
    bits_per_dim = int(np.log2(sqrt_m))

    i_idx = np.argmin(np.abs(rx_symbols.real[:, None] - levels[None, :]), axis=1)
    q_idx = np.argmin(np.abs(rx_symbols.imag[:, None] - levels[None, :]), axis=1)

    i_gray = i_idx ^ (i_idx >> 1)
    q_gray = q_idx ^ (q_idx >> 1)

    bits_2d = np.concatenate(
        [ints_to_bits(i_gray, bits_per_dim), ints_to_bits(q_gray, bits_per_dim)], axis=1
    )
    return bits_2d.reshape(-1).astype(np.int8)


def zero_force_equalize(rx_grid_data: np.ndarray, h_hat: np.ndarray) -> np.ndarray:
    return rx_grid_data / h_hat


def compute_ber(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
    return float(np.mean(bits_tx != bits_rx))


def theoretical_ber_awgn(order: int, snr_db) -> np.ndarray:
    """Gray-coded square M-QAM BER in AWGN (Proakis). snr_db is per-symbol Es/N0 in dB,
    matching the convention used in channel.apply_channel."""
    snr_db = np.asarray(snr_db, dtype=float)
    bits_per_symbol = np.log2(order)
    eb_n0 = (10 ** (snr_db / 10)) / bits_per_symbol
    sqrt_m = np.sqrt(order)

    return (
        (4 / bits_per_symbol)
        * (1 - 1 / sqrt_m)
        * 0.5
        * erfc(np.sqrt(3 * bits_per_symbol * eb_n0 / (2 * (order - 1))))
    )
