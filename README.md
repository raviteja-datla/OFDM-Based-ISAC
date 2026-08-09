# OFDM-Based ISAC

A simulation of **Integrated Sensing and Communication (ISAC)**: one OFDM waveform, used at
the same time to (1) carry data and (2) act as a radar that estimates a target's range and
velocity — with no extra transmission, no separate radar hardware, just signal processing
applied to a signal you were already sending.

---

## 1. The idea, from scratch

Normally a radio system picks one job. A comms system (WiFi, LTE, 5G) sends bits and tries to
recover them at the other end. A radar system sends a known pulse and listens for the echo to
figure out where something is. They're built, tuned, and licensed completely separately.

ISAC notices something: a comms transmitter already knows *exactly* what it sent, on every
subcarrier, every symbol. If some of that transmission reflects off a physical object and comes
back, comparing "what I received" to "what I know I sent" tells you everything about what
happened to the signal in between — including the presence, distance, and speed of whatever it
bounced off. You get radar sensing as a byproduct of a communications link you were running
anyway.

OFDM (the modulation behind WiFi/LTE/5G) makes this especially convenient, because it already
represents the signal as a grid of known values — one QAM symbol per subcarrier, per time slot —
rather than a continuous waveform you'd have to correlate against a template.

### The two physical effects a reflection introduces

Say a target is at range `R`, moving at velocity `v`, relative to the radar:

- **Delay.** The signal travels there and back, a distance `2R`, taking time `τ = 2R / c`.
  A delay in time is a **linear phase shift across frequency** — each subcarrier's phase gets
  rotated by an amount proportional to its frequency and to `τ`.
- **Doppler shift.** If the target is moving, the reflected signal's carrier frequency shifts by
  `f_D = 2v·f_c / c` (`f_c` = carrier frequency). A frequency shift is a **linear phase rotation
  across time** — each successive OFDM symbol's phase advances a little further than the last.

So a reflection turns the transmitted resource grid `Tx` into a received grid
`Rx = Tx ⊙ H + noise`, where `H[k, n]` (subcarrier `k`, OFDM symbol `n`) is a 2D grid of pure
phase ramps:

```
H[k, n] = amplitude · exp(-j·2π·k·Δf·τ) · exp(+j·2π·n·T_sym·f_D)
          \_______________ delay ramp ______________/  \____ Doppler ramp ____/
```

### Recovering range and velocity: undo the ramps with an FFT

A linear phase ramp across an axis is exactly what a **DFT (FFT/IFFT)** is built to collapse into
a single peak at a specific frequency bin. So:

1. Divide: `H_hat = Rx / Tx` — since `Tx` is known exactly, this isolates the channel's effect at
   every single resource element. No correlation, no matched filter needed.
2. **IFFT across the subcarrier axis** collapses the delay ramp into a peak at the bin
   corresponding to the target's range → the **range profile**.
3. **FFT across the OFDM-symbol axis** collapses the Doppler ramp into a peak at the bin
   corresponding to the target's velocity → the **Doppler profile**.
4. Do both together → a 2D **range-Doppler map**. A bright spot's `(x, y)` position directly
   gives you `(velocity, range)`.

Meanwhile, the *same* `Rx` grid also gets run through an ordinary OFDM receiver — equalize using
a channel estimate, demap the QAM symbols back to bits — for the communications side, computing
bit error rate (BER) just like any comms link.

That's the whole idea: **one transmission, one Rx/Tx division, two different FFT directions,
two completely different outputs (bits and a radar image).**

---

## 2. How the full system works, end to end

This is the same idea as above, but traced through as actual code, actual function calls, and
actual array shapes — the version you'd want if you were about to read or modify the source.
Numbers below use the default config: 512 subcarriers, 128 OFDM symbols, 16-QAM, one target at
200 m / 20 m/s.

**1. Build the transmitted grid — [`ofdm_tx.py`](src/ofdm_isac/ofdm_tx.py)**
- `generate_bits(n_bits, rng)` → random bits. `n_bits = n_subcarriers × n_symbols × bits_per_symbol
  = 512 × 128 × 4 = 262,144` for this config.
- `qam_mod(bits, qam_order)` → Gray-coded 16-QAM symbols, shape `(65536,)` complex128, built from
  `qam_constellation()`'s shared PAM-levels/Gray-decode lookup table (the same table
  `comms_rx.qam_demod` uses later, so modulation and demodulation can never silently disagree).
- `symbols_to_resource_grid(symbols, N, M)` → reshapes into the `(512, 128)` `Tx` grid, one OFDM
  symbol per column.
- `ifft_time_domain` + `add_cyclic_prefix` also exist here and build the literal time-domain
  waveform a real transmitter would send, but sit off the main pipeline's critical path — the
  frequency-domain grid is what everything downstream actually uses.

**2. Simulate the reflection — [`channel.py`](src/ofdm_isac/channel.py)**
- `build_channel_matrix(cfg, target)` computes `τ = 2R/c`, `f_D = 2v·f_c/c`, then the `(512, 128)`
  matrix `H[k,n] = amplitude · exp(-j2π·k·Δf·τ) · exp(+j2π·n·T_sym·f_D)`.
- `apply_channel(tx_grid, cfg, [target], snr_db, rng)` computes `Rx = Tx ⊙ H + noise`, with the
  noise variance derived from `snr_db` so it's calibrated as per-resource-element `Es/N0`. For
  multiple targets, `H` is just summed across each `Target` in the list first (linear
  superposition — see §5's "multi-target" notes on what that assumes away).

**3. Sensing path — [`sensing.py`](src/ofdm_isac/sensing.py)**
- `estimate_channel(Rx, Tx)` = `Rx / Tx` → `Ĥ`, the (noisy) channel estimate. Safe because
  normalized QAM symbols are bounded away from zero.
- `range_doppler_map(Ĥ)`: multiply by a 2D Hann window, `IFFT` along the subcarrier axis, `FFT`
  along the symbol axis, `fftshift` the symbol axis only → the `(512, 128)` range-Doppler map.
- Either `find_peak(rd_map, cfg)` (one known target, plain argmax — used for the validation run
  and the RMSE sweep, where ground truth is known) or `cfar_detect_2d(rd_map, cfg)` (unknown
  number of targets, CA-CFAR — used for the multi-target demo) turns that map into `(R̂, v̂)`
  estimates.

**4. Comms path — [`comms_rx.py`](src/ofdm_isac/comms_rx.py)**, using a *second, independent*
Tx/Rx grid pair through the same physical target (see §7's circular-equalization note for why
this has to be a separate grid, not the sensing grid reused)
- `zero_force_equalize(rx_grid_data, Ĥ)` = `rx_grid_data / Ĥ` → equalized symbols.
- `qam_demod(...)` → recovered bits, via the same constellation table `qam_mod` built.
- `compute_ber(bits_tx, bits_rx)` → the actual error rate; `theoretical_ber_awgn` gives the
  closed-form AWGN comparison curve plotted alongside it.

**5. Orchestration — [`experiments.py`](src/ofdm_isac/experiments.py)** repeats steps 1–4 under
different conditions: once for the single validation run (`run_single_validation`), once with
`demo_targets` + CFAR for the multi-target demo (`run_multi_target_demo`), once across
`trajectory_n_frames` consecutive frames for the trajectory-tracking demo
(`run_trajectory_demo` — see §5's `trajectory_tracking.png` section), once across
`micro_doppler_n_frames` frames for the micro-Doppler demo (`run_micro_doppler_demo` — see
§5's `micro_doppler_spectrogram.png` section), and `n_monte_carlo` times per SNR point for the
RMSE and BER sweeps (`run_rmse_vs_snr`, `run_ber_vs_snr`).

**6. Plotting — [`plotting.py`](src/ofdm_isac/plotting.py)** turns each experiment's returned
dict into one of the seven PNGs in `results/`.

**7. [`main.py`](main.py)** sequences all of the above in order and prints the derived
numbers plus a couple of headline results to the console.

The one-sentence version: **two completely independent analyses (sensing, comms) branch off the
exact same `Rx`/`Tx` pair**, differing only in which axis gets Fourier-transformed and what the
result gets compared against.

---

## 3. Project layout

```
main.py                         entry point — runs the full pipeline, saves the deliverable plots
pyproject.toml                  dependencies (numpy, scipy, matplotlib)
results/                        generated PNGs (gitignored, created by main.py)
src/ofdm_isac/
  config.py       SystemConfig — all system parameters + derived resolution/ambiguity limits
  ofdm_tx.py       bits -> Gray-coded QAM -> resource grid -> IFFT/CP (the Tx chain)
  channel.py       Target + delay/Doppler phase-ramp channel model, AWGN
  sensing.py       Rx/Tx division -> windowed 2D FFT/IFFT -> range-Doppler map ->
                     single-target peak detection + multi-target CA-CFAR detection
  comms_rx.py      equalization, QAM demod, BER, theoretical AWGN BER curve
  experiments.py   the Monte Carlo experiments (validation run, multi-target demo,
                     trajectory-tracking demo, micro-Doppler demo, RMSE sweep, BER sweep)
  plotting.py      the deliverable figures
```

Each module maps directly onto one stage of the pipeline described above — `ofdm_tx.py` builds
`Tx`, `channel.py` builds `Rx` from it, `sensing.py` does the range-Doppler side of step 2,
`comms_rx.py` does the bits side of step 2.

---

## 4. Running it

Install the project (editable) into your virtual environment — this pulls in `numpy`, `scipy`,
and `matplotlib` from `pyproject.toml` automatically:

```bash
uv pip install -e .
```

Then run the whole pipeline:

```bash
python main.py
```

This prints the system's derived numbers (bandwidth, range/velocity resolution, unambiguous
limits), runs the single-target validation pass, the multi-target CFAR demo, the trajectory-
tracking demo, the micro-Doppler demo, then two Monte Carlo sweeps, and saves seven PNGs to
`results/`. Takes about 35 seconds.

---

## 5. Reading the deliverables

### `sensing_pipeline_demo.png` — how sensing actually works, step by step

This is the figure to look at first, because it shows *why* the math works rather than just its
final output. Four panels, using the same single validation run:

1. **`|Tx|`** — the magnitude of every resource element we transmitted. Just QAM constellation
   amplitudes scattered across the grid — no visible structure, because random data has no
   structure.
2. **`|Rx|`** — the magnitude of what came back after the reflection. **It looks basically
   identical to panel 1.** This is the whole point: the channel here (`channel.build_channel_matrix`)
   is a pure phase rotation — delay and Doppler only rotate each resource element's phase, they
   don't change its amplitude. If you only ever looked at received signal *strength*, you would
   see no evidence a target exists at all.
3. **`phase(Rx / Tx)`** — the phase of the same two grids after dividing out the known
   transmitted symbols. Suddenly there's obvious diagonal fringe structure across the whole grid.
   That's the delay ramp (varying down the subcarrier axis) and the Doppler ramp (varying across
   the OFDM-symbol axis) superimposed — this diagonal interference pattern *is* the target's
   signature. It was always there in panels 1 and 2, just invisible, because it only shows up in
   phase, not magnitude.
4. **Range-Doppler map** — the same figure as `range_doppler_map.png`: the 2D FFT/IFFT collapses
   that diagonal phase ramp into a single sharp peak at the target's actual range and velocity.

In short: **you cannot see the target by comparing Tx and Rx by eye — you have to divide, then
look at phase, then Fourier-transform.** That's the entire "aha" of OFDM sensing in one figure.

### `range_doppler_map.png`
A heatmap of received signal magnitude (dB) over range (y-axis) and velocity (x-axis). A single
bright spot should appear at the demo target's true `(range, velocity)` — the white circle marks
the true position, the red X marks where peak detection actually found it. The map is
Hann-windowed on both axes before the FFT/IFFT to suppress spectral leakage sidelobes (a
rectangular window would otherwise smear a faint cross through the peak, since the target's true
range/velocity rarely land exactly on a bin).

### `multi_target_map.png` — detecting more than one object

Everything above uses a single known target, found with a plain argmax over the range-Doppler
map — that works because there's exactly one peak and we know it. Real scenes have an unknown
number of targets buried in noise, so `main.py` also runs a 3-target scenario
(`SystemConfig.demo_targets`) detected with a proper **CA-CFAR** (cell-averaging
constant-false-alarm-rate) detector (`sensing.cfar_detect_2d`) instead.

For every cell in the map, CA-CFAR estimates the local noise floor from a ring of "training"
cells around it (skipping a "guard" band immediately adjacent, so a target's own sidelobe energy
doesn't inflate its own noise estimate), then declares a detection if that cell's power exceeds
the noise estimate by a threshold set from a target false-alarm probability `Pfa` (the
Rohling 1983 CA-CFAR formula: `Pfa = (1 + α)^-N`, solved for the threshold multiplier `α`).
Adjacent detected cells around the same true peak are merged via connected-component labeling.

`Pfa` matters more than it might look: with ~65,536 cells in the map, `Pfa = 1e-4` yields roughly
`65536 × 1e-4 ≈ 6.5` *expected* false alarms per scan just from noise — which is exactly what an
early version of this plot showed. Dropping to `Pfa = 1e-6` (this project's default) brings that
down to about 0.065 expected false alarms, and the three real targets (including the weakest
one, deliberately given a smaller reflection amplitude) are still detected cleanly. This
threshold/false-alarm tradeoff — not "can you see a bright spot," but "how do you draw a
principled line between signal and noise" — is the actual hard problem in radar detection.

### `trajectory_tracking.png` — watching a target actually move

Every plot above is a single snapshot: one OFDM frame, one moment in time. Real targets move
between observations, so `main.py` also runs `experiments.run_trajectory_demo`: one target at a
constant velocity (`SystemConfig.trajectory_velocity_mps`), observed across
`trajectory_n_frames` (200, by default) separate, independent OFDM frames spaced
`frame_duration_s` apart, with the target's true range advancing between frames
(`R(t) = R0 + v·t`). Each frame runs through the exact same per-frame pipeline as every other
plot here — nothing new algorithmically, just repeated over time and plotted as a track instead
of a single point.

Two things to notice in the result: the **range track is a visible staircase**, not a smooth
line — each flat plateau is the estimate sitting in one range bin (`ΔR ≈ 9.76 m`) before jumping
to the next as the target physically crosses that boundary, a direct, visual demonstration of
range quantization that's easy to miss in a single static snapshot. The **velocity track stays
essentially flat and accurate throughout** (RMSE well under `Δv`), because velocity here is
genuinely constant — a target with non-constant velocity is exactly what the next demo covers.

### `micro_doppler_spectrogram.png` — a stationary target's periodic motion

This project started from an ISAC angle — one waveform doing radar and comms — but the
same underlying trick (compare received to known-transmitted, look at what's left) is also
the mechanism behind **WiFi sensing** (IEEE 802.11bf), where a repurposed comms signal detects
human presence, gestures, or breathing by watching how the channel fluctuates over time. This
demo is the bridge between the two: `experiments.run_micro_doppler_demo` simulates a
*stationary* body (fixed range, zero net translation) whose instantaneous velocity oscillates —
`v(t) = bulk + amplitude·sin(2π·freq·t)` — modeling something like a repeated gesture or limb
swing rather than bulk motion. Because real micro-motion frequencies (~1–2 Hz) are far slower
than one OFDM frame's duration (a few ms), each individual frame's constant-velocity assumption
stays physically valid; the oscillation only becomes visible by stacking many frames' Doppler
profiles into a spectrogram.

No new sensing algorithm is needed for this — `range_doppler_map` already computes a full
Doppler profile (magnitude across the velocity axis) per frame. This demo simply keeps that
whole profile, evaluated at the target's fixed range bin, instead of collapsing it to one number
via `find_peak`, and stacks those profiles across `micro_doppler_n_frames` (600, by default) as
columns of an image — velocity axis vs. frame time. The result is a proper **micro-Doppler
spectrogram**: the oscillating limb/gesture traces a wavy bright band instead of a straight line.

The demo closes the loop the same way everything else in this project verifies its own claims:
it also runs a second, slower FFT — literally the real technique, not a simplification of it —
across the recovered per-frame velocity estimates, and recovers the oscillation frequency itself
(printed to console, and in the plot title: true vs. recovered). The two won't match exactly —
frequency resolution here is `1/(micro_doppler_n_frames × frame_duration_s)`, about 0.35 Hz at
the defaults — but landing in the right bin is the same bin-quantization story that shows up
everywhere else in this project (RMSE floors, range staircases), not a new kind of error.

### `rmse_vs_snr.png`
Two subplots: range estimation RMSE and velocity estimation RMSE, each swept over SNR, each with
a horizontal dashed line marking the theoretical resolution (`ΔR`, `Δv`). This sweep uses a much
lower SNR range (−50 to −10 dB) than the BER sweep, and for a specific reason: the range-Doppler
FFT is a coherent sum over `n_subcarriers × n_symbols` samples, which gives roughly
`10·log10(512×128) ≈ 48 dB` of processing gain. That means peak detection stays essentially
perfect all the way down to a much lower *input* SNR than you'd naively expect — sweeping only
down to −10 dB would just show a flat line at the resolution floor with no visible transition.
At −50 to −10 dB you can actually see the cliff where detection breaks down.

### `ber_vs_snr.png`
Simulated bit error rate vs. theoretical AWGN BER for the QAM order in use, both on a log scale.
Points where zero bit errors were observed across the full Monte Carlo run (at high SNR) are
marked as open triangles — an upper bound, not a measured value, since "0 errors in N bits
simulated" only proves the true BER is below `1/N`, not that it's exactly zero. A dotted line
marks that Monte Carlo detection floor.

---

## 6. Tuning parameters, field by field

Everything lives in `SystemConfig` (`src/ofdm_isac/config.py`) as a frozen dataclass: raw
numerology as fields, everything derivable — bandwidth, resolution, ambiguity limits — as
computed `@property` values, so changing one field propagates everywhere automatically. Invalid
combinations raise `ValueError` at construction time (`__post_init__`), not partway through a
30-second run — see the validation notes below for exactly which combinations that covers.

### Core numerology — these define the whole system's physics

| Field | Default | What changing it does |
|---|---|---|
| `subcarrier_spacing_hz` (`Δf`) | 30 kHz | Wider spacing → more bandwidth for a fixed `n_subcarriers` (`B = N·Δf`) → finer range resolution, **but** also *shrinks* `max_unambiguous_range_m = c/(2Δf)` — the two move in opposite directions from this one knob. |
| `n_subcarriers` (`N`) | 512 | More subcarriers → more bandwidth (finer range resolution) **without** shrinking unambiguous range (unlike `Δf`) — this is the "free" way to improve range resolution. Also more bits per OFDM symbol (higher comms throughput) and a larger range-Doppler map (more compute: the 2D FFT is `O(N log N · M)`). |
| `n_symbols` (`M`) | 128 | More symbols → longer observation time → finer velocity resolution (`Δv = λ/(2·M·T_sym)`), and more coherent processing gain for detection (roughly `10·log10(N·M)` dB — see the RMSE-sweep note in §5). Linear cost in runtime and bits transmitted. |
| `cp_len` | 64 | Cyclic prefix length, samples. Must be `< n_subcarriers` (enforced). Not used by the frequency-domain sensing/comms pipeline at all — it only matters if you exercise `ofdm_tx.ifft_time_domain`/`add_cyclic_prefix` directly, where in a real system it would need to exceed the expected multipath delay spread. |
| `carrier_freq_hz` (`f_c`) | 28 GHz | Sets wavelength `λ = c/f_c`, which sets **both** velocity resolution and unambiguous velocity (`Δv` and `v_max` both scale with `λ`). Lower carrier (e.g. sub-6 GHz) → coarser velocity resolution but tolerates much faster targets before aliasing; mmWave (here) → the opposite. Classic radar-band tradeoff, not free either direction. |
| `qam_order` | 16 | Bits per symbol = `log2(qam_order)`. Higher order (64, 256-QAM) → more comms throughput, but constellation points pack closer together → worse BER at the same SNR. Must be a perfect square (`4, 16, 64, 256, ...`) — enforced, since I/Q are modulated as two independent PAM dimensions. |

### Single-target demo scenario — drives `range_doppler_map.png`, `sensing_pipeline_demo.png`, and both Monte Carlo sweeps (they need exactly one known ground truth to measure error against)

| Field | Default | Notes |
|---|---|---|
| `target_range_m`, `target_velocity_mps` | 200 m, 20 m/s | Must stay strictly under `max_unambiguous_range_m` / `max_unambiguous_velocity_mps` (enforced) — past that, the target would alias and appear at the *wrong* position, which is a real radar failure mode (range/Doppler ambiguity), not just a config error, but this project treats it as one so you notice immediately rather than silently getting a wrong-looking plot. |
| `target_amplitude` | `1.0 + 0j` | Reflection strength. Feeds into the noise calibration in `channel.apply_channel` (`noise_var = Σ|amplitude|² / SNR`), so changing it while holding `snr_db` fixed doesn't actually change detectability — SNR is defined relative to this amplitude, not in absolute terms. |

### Multi-target demo scenario — drives `multi_target_map.png`

| Field | Default | Notes |
|---|---|---|
| `demo_targets` | 3 targets at (150 m, 15 m/s), (350 m, −25 m/s), (450 m, 40 m/s), with amplitudes 1.0, 0.6, 0.35 | Each `(range_m, velocity_mps, amplitude)` tuple is validated the same way as the single-target scenario. Add/remove tuples freely to change the scene. Keep targets separated by more than roughly `(2·guard_cells + 1)` resolution bins on each axis (see CFAR row below) or CA-CFAR's connected-component step will merge two real targets into a single detection. |
| `multi_target_snr_db` | 15 dB | Lower this to see the weakest target (amplitude 0.35) approach the CFAR detection threshold and eventually get missed — a direct, hands-on view of the detection-probability side of the CFAR tradeoff described next. |

### Trajectory-tracking demo — drives `trajectory_tracking.png`

| Field | Default | Notes |
|---|---|---|
| `trajectory_start_range_m`, `trajectory_velocity_mps` | 100 m, 50 m/s | The start position and (constant) velocity. Both the start *and* end-of-trajectory range (`start + velocity × n_frames × frame_duration_s`) are validated against `max_unambiguous_range_m`/`max_unambiguous_velocity_mps` — a trajectory that would drift past the unambiguous range partway through is rejected at construction time, not partway through a run. |
| `trajectory_n_frames` | 200 | How many independent frames to track across. More frames → longer observed trajectory (`n_frames × frame_duration_s`) and more visible range-bin transitions in the plot; linearly more runtime (one full sensing pipeline pass per frame). |
| `trajectory_snr_db` | 20 dB | SNR for every frame in the sweep. Lower it to see the per-frame range/velocity scatter widen around the true track — the single-frame analog of the RMSE-vs-SNR sweep, but visualized as a track instead of an aggregate statistic. |

### Micro-Doppler demo — drives `micro_doppler_spectrogram.png`

| Field | Default | Notes |
|---|---|---|
| `micro_doppler_range_m` | 200 m | Fixed range — this target has zero net translation, only oscillatory motion, so range doesn't need per-frame validation the way the trajectory demo's does. |
| `micro_doppler_bulk_velocity_mps` | 0.0 | Constant component of velocity, e.g. non-zero to model a body walking *while* gesturing. Kept at 0 by default to isolate the oscillation cleanly rather than compounding it with the range-staircase effect already shown in `trajectory_tracking.png`. |
| `micro_doppler_amplitude_mps` | 3.0 m/s | Peak oscillation amplitude. Needs to be large relative to `velocity_resolution_mps` (~1.12 m/s) to show up clearly in the spectrogram — this is roughly limb-swing scale; something breathing-scale (mm/s) would be far below this system's velocity resolution and invisible. |
| `micro_doppler_freq_hz` | 1.5 Hz | Oscillation rate (roughly gesture/limb-swing cadence). Higher → faster wobble in the spectrogram; must stay well below `1/(2·frame_duration_s)` for the per-frame constant-velocity assumption to hold (trivially true at human motion rates here). |
| `micro_doppler_n_frames` | 600 | Total observation length. More frames → finer frequency resolution in the recovered-oscillation-frequency FFT (`1/(n_frames × frame_duration_s)`) and more visible oscillation cycles in the plot; linearly more runtime. |
| `micro_doppler_snr_db` | 20 dB | SNR for every frame. Lower it to see the bright band in the spectrogram widen/fade and the recovered oscillation frequency degrade. |

### CA-CFAR detector — drives what counts as a "detection" in `cfar_detect_2d`

| Field | Default | Notes |
|---|---|---|
| `cfar_guard_cells` | `(2, 2)` (range, velocity) | Shields the training window from a target's own sidelobe energy. Too small → the target's own energy leaks into its noise estimate, inflating the threshold and risking a missed detection on itself. Too large → wastes cells without much downside otherwise. |
| `cfar_training_cells` | `(8, 8)` (range, velocity) | How many neighboring cells estimate the local noise floor. More cells → a more accurate, stable noise estimate (closer to the theoretical exponential-noise assumption the Rohling formula relies on) — but a wider window is also more likely to accidentally include *another* real target's energy if the scene is crowded, corrupting that estimate (classic CFAR "masking" failure mode when targets sit close together). |
| `cfar_pfa` | `1e-6` | Sets the detection threshold via the Rohling formula. This is *the* fundamental radar detection knob: lower `Pfa` → fewer false alarms but a higher chance of missing real, weak targets (the threshold rises); higher `Pfa` → catches weaker targets but risks false alarms (§5's `multi_target_map.png` section walks through exactly this at `1e-4` vs `1e-6`). There is no free lunch here — it's a real ROC-curve tradeoff, not a bug to be tuned away. |

### SNR sweeps and Monte Carlo settings

| Field | Default | Notes |
|---|---|---|
| `snr_db_sweep` | −10 to 30 dB, 5 dB steps | Range for `ber_vs_snr.png`. Finer steps or a wider range → smoother/more informative curve, linearly more runtime. |
| `rmse_snr_db_sweep` | −50 to −10 dB, 5 dB steps | Separate, lower range for `rmse_vs_snr.png` — see §5's explanation of why the RMSE sweep needs to go much lower than the BER sweep to actually show a transition. If you increase `n_subcarriers`/`n_symbols`, processing gain goes up, and this range would need to shift even lower to keep showing the cliff. |
| `n_monte_carlo` | 200 | Trials averaged per SNR point. Controls the smoothness of both sweep curves *and* — critically for `ber_vs_snr.png` — the Monte Carlo detection floor, `1 / (n_monte_carlo × n_bits_per_trial)`. More trials pushes that floor down, letting you resolve lower BER values before hitting the "zero errors observed" upper-bound case (§5), at the cost of linearly more runtime. |
| `single_run_snr_db` | 20 dB | SNR used for the one-off validation run (`range_doppler_map.png`, `sensing_pipeline_demo.png`) and the single BER number printed to the console. |
| `rng_seed` | 42 | Master seed. Changing it gives a different concrete realization (exact peak location, exact BER numbers) without changing the underlying statistical trends — useful for confirming a result isn't a seed-specific fluke. |

With the numerology defaults: **15.36 MHz** bandwidth → **9.76 m** range resolution, **5 km**
unambiguous range; **4.8 ms** observation time → **1.12 m/s** velocity resolution, **±71 m/s**
unambiguous velocity.

---

## 7. Issues faced during development

None of these were caught by reading the code back over — each one showed up as something
concrete looking wrong in an actual run (a bad number, a broken-looking plot, a statistically
implausible result), and got traced back to its cause from there.

**Circular equalization gave a fake zero BER.** The original plan was to reuse one grid's own
`Rx/Tx` division as both the sensing channel estimate *and* the comms equalizer. Read literally,
that's circular: `equalized = Rx / (Rx/Tx) = Tx` exactly, regardless of noise, so BER vs. SNR
would have been zero at every single SNR point — a completely fake result that would have looked
fine until someone (or an interviewer) asked why noise didn't matter. The fix: generate two
independent resource grids through the same physical target — a reference/pilot grid whose
`Rx/Tx` supplies both the range-Doppler input *and* the channel estimate, and a separate data
grid with fresh random bits, equalized using that reference's estimate and compared against its
own known bits for BER. See `experiments.run_ber_vs_snr` and the docstring on
`comms_rx.zero_force_equalize`.

**Spectral leakage smeared a cross through the range-Doppler peak.** The first version of
`range_doppler_map.png` showed a target's true peak correctly located, but with a faint, full-
width bright line running through it in both directions — because the target's true range/
velocity essentially never land exactly on a bin, and a rectangular (unwindowed) FFT's sidelobes
decay very slowly (a Dirichlet kernel), showing up as a "picket fence" across the whole axis.
Fixed by Hann-windowing both axes before the FFT/IFFT in `sensing.range_doppler_map` — trades a
~1.5× wider main lobe for roughly 18 dB better sidelobe suppression.

**`BER = 0` broke the log-scale BER plot.** At high SNR, Monte Carlo trials sometimes produced
*zero* observed bit errors, and `0` can't be plotted on a `semilogy` axis (`log10(0) = -inf`),
which showed up as the plotted line plunging off the bottom of the frame instead of leveling
off. The deeper issue: "zero errors in N simulated bits" doesn't mean the true BER is zero, only
that it's below the detectable floor of `1/N` for that many bits. Fixed by clipping those points
to that Monte Carlo detection floor and marking them as explicit upper bounds (open triangle
markers) rather than plotting them as measured values — see `plotting.plot_ber_vs_snr`.

**The RMSE-vs-SNR sweep was uninformatively flat.** The first version swept the same −10 to
30 dB range used for the BER plot, and range/velocity RMSE came out completely flat — sitting
right at the resolution floor at every single SNR point, with no visible trend. The cause: the
range-Doppler FFT is a coherent sum over `n_subcarriers × n_symbols` samples (~65,536 here),
giving roughly 48 dB of processing gain, so peak detection was already essentially perfect even
at the sweep's lowest tested SNR. Fixed by adding a second, much lower dedicated sweep range
(−50 to −10 dB, `rmse_snr_db_sweep`) just for this experiment, which reveals the real cliff where
detection actually breaks down (see `experiments.run_rmse_vs_snr`).

**CFAR's default false-alarm rate produced real false alarms.** The first version of the
multi-target demo used `Pfa = 1e-4` and reliably returned 9–10 detections for 3 true targets —
not a bug in the detector, but exactly what that `Pfa` predicts: with ~65,536 cells in the map,
`65536 × 1e-4 ≈ 6.5` false alarms are *expected* from noise alone per scan, and the extra
detections' peak magnitudes (~−9 dB, far below the three real targets' 27–34 dB) confirmed they
were noise, not a broken algorithm. Fixed by dropping to `Pfa = 1e-6` (real radar systems pick it
this low for exactly this reason), which brought expected false alarms down to ~0.065/scan while
still cleanly detecting all three real targets, including the deliberately weak one.
