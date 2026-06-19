# Future Experiment Results - 2026-06-19

This note records the four follow-ups from `docs/future_experiments.md`.
It is intentionally conservative: completed artifacts are listed separately
from interpretation, and the second-model run is reported as a pilot rather
than a full replication.

## Summary

| experiment | status | headline |
| --- | --- | --- |
| 1. K=16-64 temporal resolution | complete for temporal six-factor map | F0 remains the cleanest recovered temporal factor; density recovers at K=16/K=32 but weakens at K=64 under the current PCA/CV setting. Attack and AM rate are partial, decay and tail remain weak. |
| 2. Spectral/loudness causal steering | complete | Brightness steering moves centroid monotonically but is not specific; loudness steering is modestly more target-selective. |
| 3. Second-model replication | pilot-complete only | AudioLDM2 runtime, capture, and geometry plumbing work on a 12-record pilot, but no factor clears stability and this is not a full cross-model replication. |
| 4. Time-localized steering | complete | Token-window brightness steering moves the target window, but the edit bleeds outside the window; the strongest case is second-half, scale 4. |

## Experiment 1: K=16-64 Temporal Resolution

The K>=16 return-size blocker was cleared by allowing `probe_capture` to write
large per-record feature payloads through the `tangoflux-activations` volume and
assemble the final local NPZ from volume paths. This avoids Modal's local return
blob limit for high time-bin captures.

Feature captures completed for all three K values. The full 18-factor maps were
too slow for fast iteration, so the committed high-K summaries use the six
temporal factors from the experiment brief:

- attack: `onset_strength_max`
- density: `onset_rate_per_second`
- decay: `decay_time_to_minus_20db_ms`
- tail: `tail_energy_fraction_500ms`
- AM rate: `am_rate_hz`
- F0: `f0_median_hz`

The map uses grouped CV by source, PCA below the train-fold size, and log
transforms for decay and tail. The analyzer was also tightened so R2 and held-out
correlations are computed from the same grouped CV fit loop rather than fitting
the same Ridge/PCA model twice per site/factor.

| K | PCA | stability threshold | stable factors |
| ---: | ---: | ---: | --- |
| 16 | 40 | 0.2523 | density, F0 |
| 32 | 50 | 0.2257 | attack, density, F0 |
| 64 | 64 | 0.1995 | F0 |

| factor | K=16 R2 | K=16 Spearman | K=16 stability | K=16 status | K=32 R2 | K=32 Spearman | K=32 stability | K=32 status | K=64 R2 | K=64 Spearman | K=64 stability | K=64 status |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| attack | 0.0356 | 0.3281 | 0.2420 | partial | 0.0486 | 0.3400 | 0.2636 | partial | 0.0397 | 0.3352 | 0.1583 | partial |
| density | 0.1946 | 0.4651 | 0.2789 | recovered | 0.2113 | 0.4758 | 0.2368 | recovered | 0.1918 | 0.4707 | 0.1744 | partial |
| decay | 0.0797 | 0.3403 | 0.1083 | weak | 0.0706 | 0.3325 | 0.0749 | weak | 0.0567 | 0.3144 | 0.0468 | weak |
| tail | -0.0748 | 0.0971 | 0.1167 | weak | -0.0870 | 0.0930 | 0.0703 | weak | -0.0790 | 0.1005 | 0.0561 | weak |
| AM rate | 0.0375 | 0.2283 | 0.1836 | weak | 0.0758 | 0.2793 | 0.1718 | partial | 0.0747 | 0.2631 | 0.1552 | partial |
| F0 | 0.3156 | 0.3606 | 0.4453 | recovered | 0.2999 | 0.3610 | 0.4258 | recovered | 0.3108 | 0.3558 | 0.2762 | recovered |

Interpretation: the K=4 -> K=8 monotonic optimism does not continue cleanly to
K=64 for every factor. F0 is robust. Density is readable and stable at K=16/K=32
but does not clear the K=64 stability threshold. Attack and AM rate are above
zero and monotonic in held-out rank order, but not clean stable recoveries.
Decay and tail remain weak, suggesting metric noise, inadequate linear readout,
or a representation that requires a more local temporal model than flat PCA +
Ridge over concatenated bins.

Artifacts:

- `results/diverse-corpus-geometry-tb16-temporal/`
- `results/diverse-corpus-geometry-tb32-temporal/`
- `results/diverse-corpus-geometry-tb64-temporal/`

## Experiment 2: Causal Spectral and Loudness Steering

Brightness and loudness directions were derived from the realized-factor geometry
bundle and applied during generation with audio-token-only additive steering.
The summary normalizes target and off-target movement by the realized population
gap.

Brightness axis target: `spectral_centroid_mean_hz`.

| scale | centroid delta mean | centroid gap fraction |
| ---: | ---: | ---: |
| 0.5 | +8.89 Hz | 0.0049 |
| 1.0 | +20.56 Hz | 0.0106 |
| 2.0 | +30.49 Hz | 0.0241 |
| 4.0 | +62.95 Hz | 0.0465 |

At scale 4, the max off-target normalized movement was 0.0872 and the target
movement was 0.0465, giving specificity ratio 0.533. Centroid moves
monotonically, but the effect is not cleanly target-specific.

Loudness axis target: `rms_dbfs`.

| scale | RMS delta mean | RMS gap fraction |
| ---: | ---: | ---: |
| 0.5 | +0.0227 dB | 0.0100 |
| 1.0 | +0.0629 dB | 0.0147 |
| 2.0 | +0.3690 dB | 0.0466 |
| 4.0 | +0.1718 dB | 0.0878 |

At scale 4, the max off-target normalized movement was 0.0776 and the target
movement was 0.0878, giving specificity ratio 1.132. Loudness steering is a
modestly positive specificity contrast, though not a clean isolated control.

Artifacts:

- `results/brightness-axis-steer-v1/`
- `results/loudness-axis-steer-exp2-tb1-v1/`

## Experiment 3: Second-Model Replication Pilot

The second-model path uses AudioLDM2 (`cvssp/audioldm2`) because Stable Audio Open
was not the clean fast path for this run. The implementation adds a separate
AudioLDM2 hook/feature recorder and second-model local entrypoints rather than
mixing the architecture-specific logic into the TangoFlux recorder.

Completed:

- runtime feasibility report
- small hook smoke test
- 12-record pilot capture with `duration=1`, `steps=4`, `site_limit=4`
- geometry analysis for CFG row 0 and CFG row 1

Pilot feature shape: `(12, 4, 2, 256)`.

| factor | row 0 best R2 | row 0 Spearman | row 0 stability | row 1 best R2 | row 1 Spearman | row 1 stability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| brightness | 0.1606 | 0.5035 | 0.4492 | 0.1579 | 0.5035 | 0.4021 |
| onset_rate | 0.1331 | 0.4126 | 0.1833 | 0.1423 | 0.4126 | 0.2017 |
| decay | -4.6336 | -0.6573 | 0.4841 | -4.6336 | -0.6573 | 0.4964 |
| loudness | 0.1703 | 0.6503 | 0.5428 | 0.2037 | 0.6503 | 0.5321 |

Stability threshold is 0.7979 for this tiny pilot; no factor clears it. This is
therefore not evidence for or against cross-model geometry mirroring. It does
show that the capture/analyze pipeline can be ported to a non-TangoFlux
architecture and that row choice is not materially changing the pilot conclusion.

The full default diverse-corpus AudioLDM2 replication was not run here. The
12-record, 1-second, 4-step pilot took long enough that a 250-record, 3.5-second,
50-step replication is a separate long-running experiment, not part of this
fast-completion batch.

Artifacts:

- `docs/second_model_replication_plan.md`
- `results/second-model-audioldm2-pilot-geometry-v1/`
- `results/second-model-audioldm2-pilot-geometry-cfgrow1-v1/`

## Experiment 4: Time-Localized Brightness Steering

`SteeringApplier` now supports token-range masking. The time-localized entrypoint
derives a brightness vector from the feature bundle, applies it only over a named
audio-token window, and computes spectral centroid over output quarters.

Run settings:

- 5 prompts
- token windows: `first_half`, `middle_half`, `second_half`
- output segments: q1-q4
- scales: 0, 1, 2, 4
- metric: `spectral_centroid_mean_hz`

Overall localization summary:

| region | mean delta | median delta | n |
| --- | ---: | ---: | ---: |
| inside steered windows | +38.06 Hz | +39.12 Hz | 90 |
| outside steered windows | +32.21 Hz | +32.74 Hz | 90 |

Overall localization gap: +5.85 Hz.

Best localized condition:

| token window | scale | inside mean | outside mean | gap | bleed ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| second_half | 4.0 | +97.75 Hz | +56.01 Hz | +41.74 Hz | 0.573 |

Interpretation: token-window steering can bias the intended portion of the
clip, but the DiT/global attention path smears a large part of the edit outside
the target window. This is a weak localization result rather than a clean
"edit only this interval" demonstration.

Artifacts:

- `results/time-localized-brightness-steer-v1/`
- `results/time-localized-brightness-steer-v1-pilot/`

## Validation

Final local checks:

```bash
python -m py_compile modal_app.py src/tangoflux_lab/*.py
git diff --check
```
