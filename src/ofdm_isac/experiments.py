"""Monte Carlo experiment orchestration: single validation run, RMSE-vs-SNR, BER-vs-SNR sweeps.

Each experiment builds a reference/pilot resource grid and (for BER) an
independent data grid through the same physical Target but with independent
noise draws, per the "avoid circular equalization" design (see comms_rx.py).
"""
import numpy as np

from .channel import Target, apply_channel, apply_precomputed_channel, build_channel_matrix
from .comms_rx import compute_ber, qam_demod, theoretical_ber_awgn, zero_force_equalize
from .config import SystemConfig
from .ofdm_tx import generate_bits, qam_mod, resource_grid_to_symbols, symbols_to_resource_grid
from .sensing import cfar_detect_2d, estimate_channel, find_peak, range_doppler_axes, range_doppler_map


def _n_bits(cfg: SystemConfig) -> int:
    return cfg.n_subcarriers * cfg.n_symbols * cfg.bits_per_symbol


def _demo_target(cfg: SystemConfig) -> Target:
    return Target(cfg.target_range_m, cfg.target_velocity_mps, cfg.target_amplitude)


def _fresh_tx_grid(cfg: SystemConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Returns (bits, tx_grid) for a fresh random OFDM frame."""
    bits = generate_bits(_n_bits(cfg), rng)
    symbols = qam_mod(bits, cfg.qam_order)
    grid = symbols_to_resource_grid(symbols, cfg.n_subcarriers, cfg.n_symbols)
    return bits, grid


def run_single_validation(cfg: SystemConfig, rng: np.random.Generator) -> dict:
    target = _demo_target(cfg)

    _, tx_grid_ref = _fresh_tx_grid(cfg, rng)
    rx_grid_ref = apply_channel(tx_grid_ref, cfg, [target], cfg.single_run_snr_db, rng)
    h_hat = estimate_channel(rx_grid_ref, tx_grid_ref)
    rd_map = range_doppler_map(h_hat)
    r_hat, v_hat, peak_db = find_peak(rd_map, cfg)
    range_axis, velocity_axis = range_doppler_axes(cfg)

    bits_data, tx_grid_data = _fresh_tx_grid(cfg, rng)
    rx_grid_data = apply_channel(tx_grid_data, cfg, [target], cfg.single_run_snr_db, rng)
    eq_grid = zero_force_equalize(rx_grid_data, h_hat)
    bits_rx = qam_demod(resource_grid_to_symbols(eq_grid), cfg.qam_order)
    ber_point = compute_ber(bits_data, bits_rx)

    return {
        "tx_grid_ref": tx_grid_ref,
        "rx_grid_ref": rx_grid_ref,
        "h_hat": h_hat,
        "rd_map": rd_map,
        "range_axis": range_axis,
        "velocity_axis": velocity_axis,
        "r_true": cfg.target_range_m,
        "v_true": cfg.target_velocity_mps,
        "r_hat": r_hat,
        "v_hat": v_hat,
        "peak_db": peak_db,
        "ber_point": ber_point,
        "snr_db": cfg.single_run_snr_db,
    }


def run_multi_target_demo(cfg: SystemConfig, rng: np.random.Generator) -> dict:
    """Multiple simultaneous targets, detected via CFAR rather than a single argmax."""
    targets = [Target(r, v, a) for r, v, a in cfg.demo_targets]

    _, tx_grid = _fresh_tx_grid(cfg, rng)
    rx_grid = apply_channel(tx_grid, cfg, targets, cfg.multi_target_snr_db, rng)
    rd_map = range_doppler_map(estimate_channel(rx_grid, tx_grid))
    detections = cfar_detect_2d(rd_map, cfg)
    range_axis, velocity_axis = range_doppler_axes(cfg)

    return {
        "rd_map": rd_map,
        "range_axis": range_axis,
        "velocity_axis": velocity_axis,
        "true_targets": [(r, v) for r, v, _ in cfg.demo_targets],
        "detections": detections,
        "snr_db": cfg.multi_target_snr_db,
    }


def run_rmse_vs_snr(cfg: SystemConfig) -> dict:
    rng = np.random.default_rng(cfg.rng_seed + 1)
    target = _demo_target(cfg)
    # The target is fixed for the whole sweep -- only snr_db and the trial's random
    # bits/noise change per iteration -- so H is built once here rather than rebuilt
    # from scratch on every one of the ~1,800 trials below.
    h_total = build_channel_matrix(cfg, target)
    signal_power = abs(target.amplitude) ** 2

    range_rmse, velocity_rmse = [], []
    for snr_db in cfg.rmse_snr_db_sweep:
        sq_err_r = np.empty(cfg.n_monte_carlo)
        sq_err_v = np.empty(cfg.n_monte_carlo)
        for trial in range(cfg.n_monte_carlo):
            _, tx_grid = _fresh_tx_grid(cfg, rng)
            rx_grid = apply_precomputed_channel(tx_grid, h_total, signal_power, snr_db, rng)
            rd_map = range_doppler_map(estimate_channel(rx_grid, tx_grid))
            r_hat, v_hat, _ = find_peak(rd_map, cfg)
            sq_err_r[trial] = (r_hat - cfg.target_range_m) ** 2
            sq_err_v[trial] = (v_hat - cfg.target_velocity_mps) ** 2

        range_rmse.append(np.sqrt(sq_err_r.mean()))
        velocity_rmse.append(np.sqrt(sq_err_v.mean()))

    return {
        "snr_db": np.array(cfg.rmse_snr_db_sweep, dtype=float),
        "range_rmse_m": np.array(range_rmse),
        "velocity_rmse_mps": np.array(velocity_rmse),
    }


def run_ber_vs_snr(cfg: SystemConfig) -> dict:
    rng = np.random.default_rng(cfg.rng_seed + 2)
    target = _demo_target(cfg)
    # Same fixed-target hoist as run_rmse_vs_snr: H is identical for both the reference
    # and data grid at a given SNR (same target, only the noise draw differs), and
    # across all ~3,600 trials in this sweep -- build it once, not per apply_channel call.
    h_total = build_channel_matrix(cfg, target)
    signal_power = abs(target.amplitude) ** 2

    ber_sim = []
    errors_observed = []
    total_bits_per_point = cfg.n_monte_carlo * _n_bits(cfg)
    for snr_db in cfg.snr_db_sweep:
        total_errors = 0
        for _ in range(cfg.n_monte_carlo):
            _, tx_grid_ref = _fresh_tx_grid(cfg, rng)
            rx_grid_ref = apply_precomputed_channel(tx_grid_ref, h_total, signal_power, snr_db, rng)
            h_hat = estimate_channel(rx_grid_ref, tx_grid_ref)

            bits_data, tx_grid_data = _fresh_tx_grid(cfg, rng)
            rx_grid_data = apply_precomputed_channel(tx_grid_data, h_total, signal_power, snr_db, rng)
            eq_grid = zero_force_equalize(rx_grid_data, h_hat)
            bits_rx = qam_demod(resource_grid_to_symbols(eq_grid), cfg.qam_order)

            total_errors += int(np.sum(bits_data != bits_rx))

        ber_sim.append(total_errors / total_bits_per_point)
        errors_observed.append(total_errors > 0)

    snr_db_array = np.array(cfg.snr_db_sweep, dtype=float)
    return {
        "snr_db": snr_db_array,
        "ber_sim": np.array(ber_sim),
        "ber_theory": theoretical_ber_awgn(cfg.qam_order, snr_db_array),
        "errors_observed": np.array(errors_observed),
        "total_bits_per_point": total_bits_per_point,
    }
