# gCLM blow-up profiles with a PINN

Can a physics-informed neural network (PINN) in self-similar coordinates find the
blow-up profiles of the generalized Constantin–Lax–Majda (gCLM) model? This
includes unstable profiles. How does success depend on the advection parameter `a`?

**Short answer:** The PINN recovers the stable profile cleanly at `a = 0`. At
`a = 0.25` it misses one of the two criteria by a small margin, and at `a = 0.5` it
fails. It found no credible unstable profile. The method stops working before
the unstable profiles become reachable.

## Problem

gCLM on the real line:

```
ω_t + a·u·ω_x = u_x·ω,        u_x = H[ω]    (Hilbert transform)
```

Self-similar ansatz with `τ = T − t`:

```
ω = τ^(-1) Ω(ξ),   ξ = x / τ^c,   u = τ^(c-1) U(ξ)
R = Ω + c·ξ·Ω' + a·U·Ω' − H[Ω]·Ω = 0,     U' = H[Ω],  U(0) = 0
```

The unknowns are the profile `Ω` and the scaling exponent `c`. `Ω` is **odd** and
normalized by `Ω'(0) = −1`. The reasons are in [Deviations](#deviations-from-the-original-plan).

## Method

`gclm_blowup.py` runs as a single script: about 58 min on one Kaggle T4, in FP64.

| Step | What it does |
|---|---|
| 0 | sympy check of the profile equation. Hilbert transform on ℝ via the map `ξ = L·tan(θ/2)` and an FFT on a uniform θ grid (N = 2048). Gate: `H[1/(1+x²)] = x/(1+x²)` to < 1e-8; measured error 5e-16. The residual pipeline is also tested against the exact CLM profile. |
| 1 | Reference `c_ref(a)` from a pseudospectral time integration of the periodic gCLM. Initial data `ω0 = −cos x`, RK4, `dt = min(0.02/max|ω|, advection CFL)`. `T` comes from the linear fit of `1/max|ω|` against `t`, and `c` from fitting kernel width `~ (T−t)^c`. |
| 2 | PINN (MLP 4×64, tanh, features `cos kθ`, k = 1..4) with an exact far-field term. Scan over 17 fixed `c` in [0.2, 3] plus `c_ref` (Adam 3000 + L-BFGS 300). Then a run with learnable `c` from every local loss minimum. FP32 control scan at `a = 0`. |
| 3 | Evaluation. Stable profile confirmed if `|c − c_ref| < 0.02` **and** the profile matches the rescaled time integral (relative L2 < 0.05 on `|ξ| ≤ 10`). |

## Results

### Reference from time integration

| a | c_ref | T | check: c from `ω_x(x0)` | max\|ω\| reached |
|---|---|---|---|---|
| 0 | 1.007 | 2.000 | 1.011 | 601 (N = 65536) |
| 0.25 | 0.685 | 2.409 | 0.687 | 2281 (N = 16384) |
| 0.5 | 0.335 | 3.142 | 0.336 | 1e4 (N = 16384) |

In all cases `max|ω| ~ τ^(-1.00)`, and the rescaled profiles are self-similar to
relative L2 ≤ 0.002. At `a = 0` the theoretical values are `c = 1` and `T = 2`. At
`a = 0.5` the values `T ≈ π` and `c ≈ 1/3` look like closed-form values; this
has not been checked against the literature.

### PINN

| a | learned c (Δ to c_ref) | profile rel. L2 | stable profile confirmed | further minima |
|---|---|---|---|---|
| 0 | 0.990 (−0.017) | 0.024 (exact CLM: 0.015) | **yes** | none |
| 0.25 | 0.675 (−0.010) | 0.059 | **no** (profile 0.059 > 0.05) | none |
| 0.5 | 0.364 (+0.029) | 0.038 | **no** (c off by 0.029) | c = 0.05, 0.14 (artefacts, see below) |

![loss vs c](results/loss_vs_c.png)
![profiles](results/profiles.png)

- **a = 0:** One sharp minimum, at `c_ref`. The FP32 control shows exactly the
  same single minimum. Theory (NOTES.md §4) says CLM has exactly one smooth
  profile, so this case is a clean positive control.
- **a = 0.25:** The run with `c` fixed at `c_ref` matches the time-integral profile
  to 0.008. In the scan, `c_ref` (loss 9.7e-6) was not a local minimum; the grid
  point `c = 0.625` (9.2e-6) was marginally lower. So the learnable-c run started
  there and ended at a slightly worse profile.
- **a = 0.5:** The loss landscape is flat and rugged, and its minimum is about 50×
  higher than for `a ≤ 0.25`. With `c` fixed at `c_ref` the PINN converges to a
  wrong profile (rel. L2 0.62). Starting from `c_ref`, learnable `c` drifts away
  to 0.14. The extra minima at `c = 0.05` (the lower clamp of `c`) and `c = 0.14`
  are strongly localized, decay far too fast and have about 100× larger `dR/dξ`
  residuals. They are not reported as unstable profiles.
- Caveat for all runs: the learned far-field exponent `s` never matched `1/c`
  (e.g. 0.85 vs 1.01 at `a = 0`). The far field is represented only approximately.

Full tables, all runs and per-criterion yes/no: [results/SUMMARY.md](results/SUMMARY.md)
(German). Raw data: `results/results.csv` (one row per run) and `results/sim_summary.csv`.

## Deviations from the original plan

All deviations are derived and documented in [results/NOTES.md](results/NOTES.md)
(German).

1. **Ω is odd, not even.** For even `Ω`, the equation splits into an even part
   `Ω + cξΩ' = 0` and an odd part. The even part has only the singular solution
   `|ξ|^(-1/c)`. For `ω0 = −cos x` the blow-up is at `x0 = −π/2`, where the data is odd.
2. **Normalization `Ω'(0) = −1`** instead of `Ω(1) = Ω(0)/2`, which degenerates
   for odd `Ω`.
3. **Relative loss** `mean(R²)/mean(Ω²) + 0.1·mean((dR/dξ)²)/mean(Ω²) + (Ω'(0)+1)²`.
   The absolute `mean(R²)` can be driven to zero for *every* `c` by shrinking the
   profile, as seen in the smoke test.
4. **Exact far-field term** `A·Im((L/(L−iξ))^s)` with learnable `s` and a known
   Hilbert transform. It replaces the planned input feature `|cos(θ/2)|^s`, whose
   FFT Hilbert transform is up to ~8 % off at `c = 3`.
5. **Time integration:** advection CFL added for `a > 0`. The first Kaggle run (v1)
   used only `dt = 0.02/max|ω|` and blew up numerically after 3–6 steps for
   `a > 0`. The `a = 0` results of v1 and v2 are identical. The resolution was
   raised to N = 65536 (`a = 0`) and N = 16384 (`a > 0`) beside the planned 4096,
   because 4096 resolves only up to `max|ω| ≈ 40`.

## Reproduce

```bash
python gclm_blowup.py --smoke          # ~30 s on CPU, a = 0, 3 c-values, N = 256
python gclm_blowup.py                  # full run, ~60 min on a Kaggle T4
python gclm_blowup.py --replot results # only redraw loss_vs_c.png from the CSVs
```

Kaggle: `kaggle_kernel/` contains the notebook (script embedded via `%%writefile`)
and the metadata. Push it with
`kaggle kernels push -p kaggle_kernel --accelerator NvidiaTeslaT4`; on Windows, set
`PYTHONUTF8=1` first. Outputs go to `/kaggle/working/results/`. The script picks
CPU or GPU after a short FP64 benchmark. The published results come from kernel
version 2; the committed script differs from it only in the plot label position and
the `--replot` option. The log is in `results/kaggle_v2.log`.

Dependencies: Python ≥ 3.10, torch, numpy, scipy, sympy, pandas, matplotlib. No
network access is needed.

## Limitations

- Three values of `a`, one network architecture, no hyperparameter sweep (by design).
- The success criteria (0.02 in `c`, 0.05 rel. L2) are this project's own choices.
- The analytic statements in NOTES.md (uniqueness at `a = 0`, Hilbert map constant)
  are own derivations, not checked against the literature.

## License

MIT, see [LICENSE](LICENSE). All code is original; it uses only the libraries listed above.
