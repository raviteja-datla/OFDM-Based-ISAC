"""Entry point: run the OFDM-ISAC pipeline end-to-end and save the deliverable plots."""
from pathlib import Path

import numpy as np

from ofdm_isac import experiments, plotting
from ofdm_isac.config import SystemConfig


def main() -> None:
    cfg = SystemConfig()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print(
        f"Bandwidth: {cfg.bandwidth_hz / 1e6:.2f} MHz | "
        f"Range resolution: {cfg.range_resolution_m:.2f} m | "
        f"Max unambiguous range: {cfg.max_unambiguous_range_m:.0f} m"
    )
    print(
        f"Observation time: {cfg.frame_duration_s * 1e3:.2f} ms | "
        f"Velocity resolution: {cfg.velocity_resolution_mps:.2f} m/s | "
        f"Max unambiguous velocity: ±{cfg.max_unambiguous_velocity_mps:.1f} m/s"
    )

    rng = np.random.default_rng(cfg.rng_seed)

    validation = experiments.run_single_validation(cfg, rng)
    print(
        f"True (R, v) = ({cfg.target_range_m:.1f} m, {cfg.target_velocity_mps:.1f} m/s) | "
        f"Estimated (R_hat, v_hat) = ({validation['r_hat']:.1f} m, {validation['v_hat']:.1f} m/s) | "
        f"Single-point BER @ {cfg.single_run_snr_db:.0f} dB: {validation['ber_point']:.4f}"
    )
    plotting.plot_range_doppler_map(validation, cfg, results_dir / "range_doppler_map.png")
    plotting.plot_sensing_pipeline(validation, cfg, results_dir / "sensing_pipeline_demo.png")

    multi_target = experiments.run_multi_target_demo(cfg, rng)
    print(
        f"Multi-target CFAR: {len(multi_target['detections'])} detection(s) for "
        f"{len(multi_target['true_targets'])} true target(s) -> {multi_target['detections']}"
    )
    plotting.plot_multi_target_map(multi_target, cfg, results_dir / "multi_target_map.png")

    aoa = experiments.run_angle_of_arrival_demo(cfg, rng)
    print(
        f"Angle of arrival: true {aoa['angle_true_deg']:.1f}° | "
        f"estimated {aoa['angle_hat_deg']:.1f}° | "
        f"antenna spacing {aoa['antenna_spacing_m'] * 1e3:.2f} mm"
    )
    plotting.plot_angle_of_arrival(aoa, cfg, results_dir / "angle_of_arrival.png")

    trajectory = experiments.run_trajectory_demo(cfg, rng)
    range_track_rmse = float(np.sqrt(np.mean((trajectory["est_ranges_m"] - trajectory["true_ranges_m"]) ** 2)))
    velocity_track_rmse = float(
        np.sqrt(np.mean((trajectory["est_velocities_mps"] - trajectory["true_velocity_mps"]) ** 2))
    )
    print(
        f"Trajectory tracking: {cfg.trajectory_n_frames} frames over "
        f"{trajectory['frame_times_s'][-1] * 1e3:.0f} ms | "
        f"range track RMSE: {range_track_rmse:.2f} m | velocity track RMSE: {velocity_track_rmse:.2f} m/s"
    )
    plotting.plot_trajectory(trajectory, cfg, results_dir / "trajectory_tracking.png")

    micro_doppler = experiments.run_micro_doppler_demo(cfg, rng)
    print(
        f"Micro-Doppler: {cfg.micro_doppler_n_frames} frames | "
        f"true oscillation {micro_doppler['true_oscillation_freq_hz']:.2f} Hz | "
        f"recovered from spectrogram: {micro_doppler['est_oscillation_freq_hz']:.2f} Hz"
    )
    plotting.plot_micro_doppler_spectrogram(
        micro_doppler, cfg, results_dir / "micro_doppler_spectrogram.png"
    )

    print("Running RMSE-vs-SNR sweep...")
    rmse_results = experiments.run_rmse_vs_snr(cfg)
    plotting.plot_rmse_vs_snr(rmse_results, cfg, results_dir / "rmse_vs_snr.png")

    print("Running BER-vs-SNR sweep...")
    ber_results = experiments.run_ber_vs_snr(cfg)
    plotting.plot_ber_vs_snr(ber_results, cfg, results_dir / "ber_vs_snr.png")

    print(f"Done. Plots saved to {results_dir.resolve()}")


if __name__ == "__main__":
    main()
