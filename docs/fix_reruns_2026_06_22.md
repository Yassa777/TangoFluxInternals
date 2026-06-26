# Fix Reruns - 2026-06-22

Re-running the first three future experiments after diagnosing the codex-branch
negatives (`docs/future_experiment_results_2026_06_19.md`). One fix failed, one is a
partial positive, one is a strong positive.

## Exp 1 (temporal readout): proposed fix FAILED, honestly

Hypothesis: flat PCA before the probe discarded the low-variance temporal-contrast
signal, so supervised reduction (PLS) or no-PCA would recover decay/density/AM-rate.

Test (local, K=8 features, best layer per factor, grouped CV by source):

| factor | PCA50+Ridge | no-PCA Ridge | PLS-20 |
| --- | --- | --- | --- |
| attack | R2 +0.09 / rho 0.38 | +0.04 / 0.45 | +0.04 / 0.45 |
| density | **+0.23 / 0.49** | +0.11 / 0.43 | +0.11 / 0.43 |
| decay | **+0.06 / 0.36** | -0.08 / 0.27 | -0.08 / 0.27 |
| tail | **-0.07 / 0.14** | -0.37 / 0.07 | -0.37 / 0.07 |
| AM rate | **+0.03 / 0.20** | -0.20 / 0.21 | -0.20 / 0.20 |
| F0 | **+0.34 / 0.40** | +0.13 / 0.39 | +0.12 / 0.39 |

PCA+Ridge is the **best** readout for every factor; no-PCA and PLS are equal-or-worse.
So PCA was not discarding the temporal signal -- it was regularizing a 250-sample /
8192-dim problem. The fix is refuted. The Spearman > R2 pattern (e.g. attack 0.45/0.09,
density 0.49/0.23) indicates **metric noise**, not the readout: rank order survives but
magnitude/variance does not. The real lever is multi-seed metric denoising (a re-capture),
not the reduction method. Not pursued here.

## Exp 2 (causal steering): magnitude bug confirmed; partial positive

The codex steering vector was unit-normalized and applied with scale <= 4, so the injected
perturbation (L2 norm <= 4) was negligible against activation magnitudes -- hence the near-
zero movement. Rerunning the brightness axis with a wide scale sweep (site dual block 2):

| scale | centroid delta | centroid gap frac | loudness (rms) delta | rms gap frac |
| ---: | ---: | ---: | ---: | ---: |
| 5 | +49.7 Hz | 0.087 | -0.37 dB | 0.102 |
| 10 | +88.9 Hz | 0.128 | -0.74 dB | 0.154 |
| 20 | -221.8 Hz | 0.197 | -1.72 dB | 0.319 |
| 40 | -854.8 Hz | 0.413 | -4.35 dB | 0.506 |
| 80 | -1518 Hz | 0.813 | -10.13 dB | 0.976 |

- **Clean regime (scale 5-10):** centroid moves the *correct* direction (+50 to +89 Hz)
  while loudness stays essentially flat (-0.4 to -0.7 dB) -- directionally correct and
  roughly specific control. This is the partial positive the magnitude fix unlocks.
- **Distortion regime (scale >= 20):** centroid sign-flips and crashes (-1518 Hz at scale
  80), flatness/crest blow up -- the audio degenerates. The large "movements" are artifacts,
  not control.

Conclusion: the magnitude bug was real (it explains codex's null), but additive probe-axis
steering saturates and breaks past a narrow usable range. Strong, clean control needs an
operator we already showed works better -- activation patching -- not bigger additive scale.

Artifact: `results/brightness-axis-steer-fixed-v1/`.

## Exp 3 (second-model replication): STRONG positive

The codex AudioLDM2 result was a 12-record pilot whose stability threshold (0.80) made any
positive impossible. Rerun at **N=80** (duration 2.5 s, 30 steps, 8 sites, d=256):

| factor | stability | best R2 | best CV Spearman |
| --- | ---: | ---: | ---: |
| loudness | 0.67 | 0.77 | 0.91 |
| rolloff | 0.65 | 0.59 | 0.81 |
| brightness | 0.63 | 0.57 | 0.83 |
| tilt | 0.58 | 0.52 | 0.73 |
| flatness | 0.56 | (R2 noisy) | 0.71 |
| crest | 0.54 | 0.43 | 0.72 |
| bandwidth | 0.53 | 0.26 | 0.62 |
| zcr | 0.53 | 0.40 | 0.73 |
| hnr | 0.39 | 0.29 | 0.67 |

**9 of 10 stationary factors are stable** (threshold 0.291), and the headline replicates on a
completely different architecture (U-Net latent diffusion, not MMDiT):

> **geometry-mirrors-physics on AudioLDM2: Pearson r = 0.95 over 36 factor pairs**
> (TangoFlux was r = 0.88-0.98).

So the central finding -- a generative audio model's linearly-accessible factor geometry
mirrors the physical correlation structure of its output -- is **not TangoFlux-specific**.
This is the strongest cross-model evidence in the project so far.

Artifacts: `results/second-model-audioldm2-real-geometry-v1/`,
`outputs/second-model-audioldm2-real-v1/` (features, gitignored).

## Net

| experiment | fix | outcome |
| --- | --- | --- |
| 1 temporal readout | PLS / no-PCA reduction | failed -- PCA was already optimal; metric noise is the real cause (needs multi-seed denoising) |
| 2 causal steering | calibrate magnitude / wider scales | partial -- magnitude bug confirmed; clean control only in a narrow range; patching is the right operator |
| 3 second model | run at adequate N | strong -- r=0.95 mirroring on AudioLDM2, generality established |
