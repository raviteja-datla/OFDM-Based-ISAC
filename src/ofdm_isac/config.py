"""System parameters for the OFDM-ISAC simulation.

Everything the rest of the package needs — Tx/Rx numerology, the demo target
scenario, and Monte Carlo sweep settings — lives in `SystemConfig`. Derived
quantities (bandwidth, range/velocity resolution, unambiguous limits) are
computed properties so changing a single numerology field automatically
propagates through resolution and ambiguity everywhere else.
"""
from dataclasses import dataclass

import numpy as np

C = 2.998e8  # speed of light, m/s


@dataclass(frozen=True)
class SystemConfig:
    # --- Numerology (5G-like, mmWave) ---
    subcarrier_spacing_hz: float = 30e3
    n_subcarriers: int = 512
    n_symbols: int = 128
    cp_len: int = 64
    carrier_freq_hz: float = 28e9
    qam_order: int = 16

    # --- Demo target scenario (single target: used by the validation run + RMSE/BER sweeps,
    # which need exactly one known ground truth to measure error against) ---
    target_range_m: float = 200.0
    target_velocity_mps: float = 20.0
    target_amplitude: complex = 1.0 + 0j

    # --- Multi-target demo scenario (range_m, velocity_mps, amplitude) ---
    demo_targets: tuple = (
        (150.0, 15.0, 1.0 + 0j),
        (350.0, -25.0, 0.6 + 0j),
        (450.0, 40.0, 0.35 + 0j),
    )
    multi_target_snr_db: float = 15.0

    # --- CA-CFAR (cell-averaging constant-false-alarm-rate) detector ---
    # Guard cells shield the training window from a target's own sidelobe energy;
    # training cells estimate the local noise floor; pfa sets the false-alarm rate,
    # which (via the Rohling 1983 CA-CFAR formula) sets the detection threshold.
    cfar_guard_cells: tuple = (2, 2)       # (range axis, velocity axis)
    cfar_training_cells: tuple = (8, 8)    # (range axis, velocity axis)
    # 1e-6, not 1e-4: with ~65536 cells in the map, Pfa=1e-4 gives ~6.5 expected false
    # alarms per scan (65536 * 1e-4) -- real radar systems pick Pfa this low specifically
    # to suppress that. At 1e-6, expected false alarms drop to ~0.065 per scan.
    cfar_pfa: float = 1e-6

    # --- SNR sweeps ---
    snr_db_sweep: tuple = (-10, -5, 0, 5, 10, 15, 20, 25, 30)
    # Lower range for the RMSE sweep: the range-Doppler FFT's coherent processing gain
    # (~10*log10(n_subcarriers*n_symbols) ~= 48 dB here) makes peak detection reliable
    # well below 0 dB input SNR, so snr_db_sweep never breaks it -> RMSE looks flat.
    # This range is low enough to cross that threshold and show a real transition.
    rmse_snr_db_sweep: tuple = (-50, -45, -40, -35, -30, -25, -20, -15, -10)
    n_monte_carlo: int = 200
    single_run_snr_db: float = 20.0
    rng_seed: int = 42

    def _check_unambiguous(self, range_m: float, velocity_mps: float, label: str) -> None:
        if range_m >= self.max_unambiguous_range_m:
            raise ValueError(
                f"{label} range_m={range_m} exceeds max_unambiguous_range_m="
                f"{self.max_unambiguous_range_m:.1f}"
            )
        if abs(velocity_mps) >= self.max_unambiguous_velocity_mps:
            raise ValueError(
                f"{label} |velocity_mps|={abs(velocity_mps)} exceeds "
                f"max_unambiguous_velocity_mps={self.max_unambiguous_velocity_mps:.1f}"
            )

    def __post_init__(self) -> None:
        sqrt_m = np.sqrt(self.qam_order)
        if sqrt_m % 1 != 0:
            raise ValueError(f"qam_order={self.qam_order} must be a perfect square (square QAM)")
        if self.cp_len >= self.n_subcarriers:
            raise ValueError("cp_len must be smaller than n_subcarriers")
        self._check_unambiguous(self.target_range_m, self.target_velocity_mps, "target_range_m/target_velocity_mps")
        for range_m, velocity_mps, _amplitude in self.demo_targets:
            self._check_unambiguous(range_m, velocity_mps, "demo_targets")

    # --- Derived quantities ---
    @property
    def bandwidth_hz(self) -> float:
        return self.n_subcarriers * self.subcarrier_spacing_hz

    @property
    def sample_rate_hz(self) -> float:
        return self.bandwidth_hz

    @property
    def symbol_duration_useful_s(self) -> float:
        return 1.0 / self.subcarrier_spacing_hz

    @property
    def cp_duration_s(self) -> float:
        return self.cp_len / self.sample_rate_hz

    @property
    def symbol_duration_total_s(self) -> float:
        return self.symbol_duration_useful_s + self.cp_duration_s

    @property
    def wavelength_m(self) -> float:
        return C / self.carrier_freq_hz

    @property
    def bits_per_symbol(self) -> int:
        return int(np.log2(self.qam_order))

    @property
    def range_resolution_m(self) -> float:
        """Delta R = c / (2B): finer with more bandwidth."""
        return C / (2 * self.bandwidth_hz)

    @property
    def max_unambiguous_range_m(self) -> float:
        return C / (2 * self.subcarrier_spacing_hz)

    @property
    def velocity_resolution_mps(self) -> float:
        """Delta v = lambda / (2 * T_obs): finer with more OFDM symbols observed."""
        return self.wavelength_m / (2 * self.n_symbols * self.symbol_duration_total_s)

    @property
    def max_unambiguous_velocity_mps(self) -> float:
        return self.wavelength_m / (4 * self.symbol_duration_total_s)
