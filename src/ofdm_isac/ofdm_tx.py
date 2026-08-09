"""OFDM transmit chain: bits -> Gray-coded QAM -> resource grid -> IFFT/CP.

The QAM constellation built here (`qam_constellation`) is shared with
`comms_rx.qam_demod` so modulation and demodulation always agree on the same
bit-to-symbol mapping.
"""
import numpy as np


def generate_bits(n_bits: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=n_bits, dtype=np.int8)


def qam_constellation(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension PAM levels and a Gray-codeword -> binary-index lookup table.

    levels[i] is the PAM value assigned to binary index i (unit average symbol
    energy after scaling). gray_decode_table[g] gives the binary index i whose
    Gray codeword equals g, so a received Gray codeword can be decoded to an
    index and looked up in `levels` directly.
    """
    sqrt_m = int(round(np.sqrt(order)))
    indices = np.arange(sqrt_m)

    raw_levels = (2 * indices - (sqrt_m - 1)).astype(float)
    scale = 1.0 / np.sqrt(2 * (order - 1) / 3)  # normalizes E[|symbol|^2] = 1
    levels = raw_levels * scale

    gray_of_index = indices ^ (indices >> 1)
    gray_decode_table = np.zeros(sqrt_m, dtype=int)
    gray_decode_table[gray_of_index] = indices

    return levels, gray_decode_table


def bits_to_ints(bits_2d: np.ndarray) -> np.ndarray:
    """Rows of MSB-first bits -> integers. Inverse of ints_to_bits; shared with
    comms_rx.qam_demod so the two directions of this MSB-first convention can't drift."""
    n_bits = bits_2d.shape[1]
    weights = 1 << np.arange(n_bits - 1, -1, -1)
    return bits_2d @ weights


def ints_to_bits(ints: np.ndarray, n_bits: int) -> np.ndarray:
    """Integers -> MSB-first bit rows. Inverse of bits_to_ints."""
    shifts = np.arange(n_bits - 1, -1, -1)
    return ((ints[:, None] >> shifts) & 1).astype(np.int8)


def qam_mod(bits: np.ndarray, order: int) -> np.ndarray:
    """Gray-coded square QAM modulation. len(bits) must be a multiple of bits_per_symbol."""
    levels, gray_decode_table = qam_constellation(order)
    sqrt_m = levels.size
    bits_per_dim = int(np.log2(sqrt_m))
    bits_per_symbol = 2 * bits_per_dim

    if bits.size % bits_per_symbol != 0:
        raise ValueError(
            f"bits length {bits.size} is not a multiple of bits_per_symbol={bits_per_symbol}"
        )

    bits_2d = bits.reshape(-1, bits_per_symbol)
    i_bits, q_bits = bits_2d[:, :bits_per_dim], bits_2d[:, bits_per_dim:]

    i_idx = gray_decode_table[bits_to_ints(i_bits)]
    q_idx = gray_decode_table[bits_to_ints(q_bits)]

    return (levels[i_idx] + 1j * levels[q_idx]).astype(np.complex128)


def symbols_to_resource_grid(symbols: np.ndarray, n_subcarriers: int, n_symbols: int) -> np.ndarray:
    """Fill one OFDM symbol per column: grid[:, i] = symbols[i*n_subcarriers:(i+1)*n_subcarriers]."""
    expected = n_subcarriers * n_symbols
    if symbols.size != expected:
        raise ValueError(f"expected {expected} symbols, got {symbols.size}")
    return symbols.reshape(n_symbols, n_subcarriers).T.copy()


def resource_grid_to_symbols(grid: np.ndarray) -> np.ndarray:
    """Inverse of symbols_to_resource_grid: flattens one OFDM symbol (column) at a time."""
    return grid.T.reshape(-1)


def ifft_time_domain(grid: np.ndarray) -> np.ndarray:
    """Per-OFDM-symbol IFFT, frequency -> time. Shape unchanged."""
    return np.fft.ifft(grid, axis=0, norm="ortho")


def add_cyclic_prefix(time_grid: np.ndarray, cp_len: int) -> np.ndarray:
    """Prepend the last cp_len time samples of each OFDM symbol (column)."""
    cp = time_grid[-cp_len:, :]
    return np.vstack([cp, time_grid])
