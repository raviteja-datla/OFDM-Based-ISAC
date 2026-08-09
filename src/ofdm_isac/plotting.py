"""The deliverable figures: range-Doppler heatmaps, RMSE vs SNR, BER vs SNR.

Magnitude data (the heatmaps) uses a single-hue perceptually-uniform sequential
colormap (viridis), never a rainbow/jet map. Range RMSE (m) and velocity RMSE
(m/s) get separate subplots rather than a dual y-axis, since they're different
units/scales on one chart. The BER comparison distinguishes simulated vs.
theoretical by both color and line style, not color alone.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import SystemConfig
from .sensing import magnitude_db


def _save_and_close(fig: plt.Figure, save_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _draw_range_doppler_heatmap(fig, ax, range_axis: np.ndarray, velocity_axis: np.ndarray, mag_db: np.ndarray):
    """Shared by every range-Doppler panel (single-target, multi-target, and the
    4th panel of the pipeline walkthrough) so the heatmap's look only needs
    updating in one place."""
    mesh = ax.pcolormesh(velocity_axis, range_axis, mag_db, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="Magnitude (dB)")
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Range (m)")
    return mesh


def _scatter_true_and_estimated(
    ax, true_v, true_r, est_v, est_r, true_label: str, est_label: str, legend_fontsize: int | None = None
) -> None:
    """Shared true-target/estimate marker styling. true_v/true_r/est_v/est_r accept
    either a scalar (single target) or an array-like (multiple targets/detections)."""
    ax.scatter(
        true_v, true_r,
        marker="o", s=140, facecolors="none", edgecolors="white", linewidths=2,
        label=true_label,
    )
    if np.size(est_v) > 0:
        ax.scatter(
            est_v, est_r,
            marker="x", s=100, color="red", linewidths=2,
            label=est_label,
        )
    legend_kwargs = {"loc": "upper right", "framealpha": 0.9}
    if legend_fontsize is not None:
        legend_kwargs["fontsize"] = legend_fontsize
    ax.legend(**legend_kwargs)


def plot_range_doppler_map(result: dict, cfg: SystemConfig, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_range_doppler_heatmap(
        fig, ax, result["range_axis"], result["velocity_axis"], magnitude_db(result["rd_map"])
    )
    _scatter_true_and_estimated(
        ax, result["v_true"], result["r_true"], result["v_hat"], result["r_hat"],
        "True target", "Estimated peak",
    )

    ax.set_title(
        f"Range-Doppler Map  (ΔR={cfg.range_resolution_m:.2f} m, "
        f"Δv={cfg.velocity_resolution_mps:.2f} m/s, SNR={result['snr_db']:.0f} dB)"
    )
    _save_and_close(fig, save_path)


def plot_multi_target_map(result: dict, cfg: SystemConfig, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_range_doppler_heatmap(
        fig, ax, result["range_axis"], result["velocity_axis"], magnitude_db(result["rd_map"])
    )

    true_r, true_v = zip(*result["true_targets"])
    det_r, det_v = zip(*[(r, v) for r, v, _peak_db in result["detections"]]) if result["detections"] else ((), ())
    _scatter_true_and_estimated(ax, true_v, true_r, det_v, det_r, "True targets", "CFAR detections")

    ax.set_title(
        f"Multi-Target Range-Doppler Map  (CA-CFAR, $P_{{fa}}$={cfg.cfar_pfa:.0e}, "
        f"SNR={result['snr_db']:.0f} dB)\n"
        f"{len(result['detections'])} detection(s) for {len(result['true_targets'])} true target(s)"
    )
    _save_and_close(fig, save_path)


def plot_sensing_pipeline(result: dict, cfg: SystemConfig, save_path: Path) -> None:
    """Four-panel walkthrough of *how* sensing works, not just its final output:
    |Tx| and |Rx| look almost identical (the channel here is a pure phase rotation,
    not an amplitude change), so the target is invisible in magnitude alone. Only
    after dividing (H_hat = Rx/Tx) and looking at *phase* does the delay/Doppler
    ramp -- the target's actual signature -- become visible, which the final FFT
    panel then collapses into a single detectable peak.
    """
    tx_grid = result["tx_grid_ref"]
    rx_grid = result["rx_grid_ref"]
    phase_h = np.angle(result["h_hat"])

    vmax = max(np.abs(tx_grid).max(), np.abs(rx_grid).max())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, grid, title in (
        (axes[0, 0], np.abs(tx_grid), "1. Transmitted magnitude |Tx|\n(known exactly)"),
        (axes[0, 1], np.abs(rx_grid), "2. Received magnitude |Rx|\n(looks almost identical!)"),
    ):
        im = ax.pcolormesh(grid, cmap="viridis", vmin=0, vmax=vmax, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("OFDM symbol index")
        ax.set_ylabel("Subcarrier index")
        fig.colorbar(im, ax=ax, label="Magnitude")

    im_phase = axes[1, 0].pcolormesh(phase_h, cmap="twilight", vmin=-np.pi, vmax=np.pi, shading="auto")
    axes[1, 0].set_title("3. Phase of $\\hat{H} = Rx / Tx$\n(the target's signature)")
    axes[1, 0].set_xlabel("OFDM symbol index")
    axes[1, 0].set_ylabel("Subcarrier index")
    fig.colorbar(im_phase, ax=axes[1, 0], label="Phase (rad)")

    _draw_range_doppler_heatmap(
        fig, axes[1, 1], result["range_axis"], result["velocity_axis"], magnitude_db(result["rd_map"])
    )
    _scatter_true_and_estimated(
        axes[1, 1], result["v_true"], result["r_true"], result["v_hat"], result["r_hat"],
        "True target", "Estimated peak", legend_fontsize=8,
    )
    axes[1, 1].set_title("4. Range-Doppler map\n(2D FFT collapses the ramp into a peak)")

    fig.suptitle(
        "How sensing works: from Tx/Rx resource grids to a target detection", fontsize=14
    )
    _save_and_close(fig, save_path)


def plot_rmse_vs_snr(result: dict, cfg: SystemConfig, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].semilogy(result["snr_db"], result["range_rmse_m"], marker="o")
    axes[0].axhline(cfg.range_resolution_m, color="gray", linestyle="--", label="Range resolution ΔR")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Range RMSE (m)")
    axes[0].set_title("Range Estimation RMSE vs SNR")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogy(result["snr_db"], result["velocity_rmse_mps"], marker="o", color="tab:orange")
    axes[1].axhline(
        cfg.velocity_resolution_mps, color="gray", linestyle="--", label="Velocity resolution Δv"
    )
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Velocity RMSE (m/s)")
    axes[1].set_title("Velocity Estimation RMSE vs SNR")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    _save_and_close(fig, save_path)


def plot_ber_vs_snr(result: dict, cfg: SystemConfig, save_path: Path) -> None:
    """Points with zero observed bit errors are clipped to the Monte Carlo detection
    floor (1/total_bits_per_point) and marked as upper bounds, not measured values --
    BER=0 doesn't plot on a log axis and doesn't mean the true BER is zero, only that
    no error occurred within the bits simulated at that SNR."""
    fig, ax = plt.subplots(figsize=(7, 5))

    floor = 1.0 / result["total_bits_per_point"]
    ber_sim = result["ber_sim"]
    no_errors = result["errors_observed"] == False  # noqa: E712
    ber_sim_display = np.where(no_errors, floor, ber_sim)

    ax.semilogy(result["snr_db"], ber_sim_display, linestyle="-", color="tab:blue", zorder=2)
    ax.semilogy(
        result["snr_db"][~no_errors], ber_sim_display[~no_errors],
        marker="o", linestyle="none", color="tab:blue", label="Simulated BER", zorder=3,
    )
    if no_errors.any():
        ax.semilogy(
            result["snr_db"][no_errors], ber_sim_display[no_errors],
            marker="v", linestyle="none", markerfacecolor="none", color="tab:blue",
            label="No errors observed (upper bound)", zorder=3,
        )
    ax.semilogy(
        result["snr_db"], result["ber_theory"], linestyle="--", color="tab:orange",
        label="Theoretical AWGN BER",
    )
    ax.axhline(floor, color="gray", linestyle=":", linewidth=1, label="Monte Carlo detection floor")

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate")
    ax.set_title(f"{cfg.qam_order}-QAM BER vs SNR")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    _save_and_close(fig, save_path)
