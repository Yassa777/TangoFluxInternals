# Representation Geometry of TangoFlux Against Physical Ground Truth

This document consolidates the representation-geometry findings — the most developed,
replicated result in this repository. The premise: a text-to-audio model is a rare
interpretability setting where the output's generative factors are **exactly measurable**
(spectral centroid, loudness, harmonicity, ...), so we can test representation-geometry
claims a language model cannot settle.

## Method

1. **Realized-corpus binning, not designed sweeps.** Designed text "sweeps" do not give
   clean single-factor control in TangoFlux (it follows "bright" loosely and "fast"/"short"
   barely). Instead we generate a diverse 250-prompt corpus (40 source groups) and regress
   activations against the **realized measured** factor per clip. This decouples ground truth
   from prompt control.
2. **Pooled features per site.** For each of the 24 DiT blocks we capture the audio-token
   activations, mean-pool over tokens, and average over denoising steps -> one vector per
   (clip, site). The conditional CFG row is used.
3. **Controls that the field usually skips.**
   - grouped cross-validation by source (no source leakage);
   - out-of-sample projection monotonicity (Spearman of `cross_val_predict`), not in-sample;
   - within-factor **direction stability** (split-half cosine on source-disjoint halves);
   - a **random-vector baseline** `E|cos| = sqrt(2/(pi*d_eff))`; a factor's direction is only
     trusted if stability > 2x that baseline;
   - PCA reduction so directions are estimable, and a non-linear (gradient-boosting) probe
     alongside linear ridge to test whether a factor is *unencoded* vs *non-linearly* encoded.

## Headline result: the representation's factor geometry mirrors the physics

Over an 18-factor panel, **12-13 factors have trustworthy (stable) directions**, and they are
all *stationary* timbre properties: the spectral family (centroid, rolloff, spectral tilt,
bandwidth, contrast, flatness, zero-crossing rate), plus voicing, harmonic-to-noise ratio,
amplitude-modulation depth, loudness, and crest factor.

For these stable factors, the model's internal **direction-cosine matrix tracks the physical
correlation matrix of the factors**:

- 6-factor panel: Pearson r = **0.98** (over 6 pairs).
- 18-factor panel: Pearson r = **0.88** (over 66 pairs).

Concretely, the physically-collinear spectral metrics collapse onto a single shared
"brightness" direction (pairwise cosine 0.87-0.97), while loudness — physically near-
independent of spectrum — occupies a distinct axis (cosine 0.29-0.48). The linearly-accessible
representation is **low-dimensional**: a spectral axis plus a loudness axis, with structure
that copies the output-space statistics. The cleanest *linear* (calibrated-magnitude) encoding
sits in the early **dual** (text-conditioned) stream (best brightness R^2 at
`transformer_blocks.3`).

## Secondary findings

- **Linear, not merely ordinal.** With adequate data, brightness is linearly encodable
  (R^2 +0.59), correcting an earlier small-N artifact that had suggested an ordinal-only code.
- **Non-linearity is not the missing ingredient.** A gradient-boosting probe does not beat
  linear ridge for any factor (non-linearity gains <= 0); the linearly-accessible structure is
  genuinely linear.

## The temporal boundary: resolution-limited, not absent

The factors that are not recoverable from token-mean pooling are exactly the *time-varying*
ones: attack salience, decay time, onset density, tail energy, amplitude-modulation **rate**,
and pitch (F0). The split is sharp — amplitude-modulation *depth* (a magnitude) is stable
while *rate* (a frequency) is at the noise floor.

A time-binned pooling sweep (split the audio tokens into K temporal bins) shows these factors
are **resolution-limited, not absent** — their direction stability rises monotonically as the
temporal resolution increases:

| factor | K=4 (~875 ms bins) | K=8 (~440 ms bins) |
| --- | ---: | ---: |
| attack | 0.20 | 0.25 |
| decay | 0.19 | 0.25 |
| density | 0.29 | 0.31 |
| tail | 0.05 | 0.09 |
| AM rate | 0.16 | 0.22 |
| F0 | 0.37 | **0.54** (R^2 0.16 -> 0.35) |

(stability vs random threshold 0.291). Every temporal factor improves with finer bins; F0
crosses cleanly into "stable, linearly encodable", and the envelope-scale factors (decay,
density, AM rate) trend toward the threshold. The fastest event (attack, ~20 ms) responds
least at these still-coarse resolutions. We have also ruled out non-linearity (a
gradient-boosting probe does not help any temporal factor).

The K>=16 Modal return-size blocker has now been cleared by writing high-K feature payloads
through the `tangoflux-activations` volume before assembling the local NPZ. A focused
six-factor temporal map was run at K=16, 32, and 64:

| factor | K=16 stability | K=16 status | K=32 stability | K=32 status | K=64 stability | K=64 status |
| --- | ---: | --- | ---: | --- | ---: | --- |
| attack | 0.242 | partial | 0.264 | partial | 0.158 | partial |
| density | 0.279 | recovered | 0.237 | recovered | 0.174 | partial |
| decay | 0.108 | weak | 0.075 | weak | 0.047 | weak |
| tail | 0.117 | weak | 0.070 | weak | 0.056 | weak |
| AM rate | 0.184 | weak | 0.172 | partial | 0.155 | partial |
| F0 | 0.445 | recovered | 0.426 | recovered | 0.276 | recovered |

Conclusion: temporal structure *is* linearly present in the activations but the K>8 result is
more mixed than a simple monotonic-recovery story. F0 is robust across high K. Density is a
real recovery at K=16/K=32 but does not clear the K=64 stability threshold under the current
PCA/CV setting. Attack and AM rate show positive held-out rank order but remain only partial;
decay and tail stay weak. The next step is not just "increase K" but use a more local temporal
readout or average seeds to reduce noisy realized attack/decay estimates.

## Causal and Cross-Model Follow-Ups

The first causal steering and generality follow-ups are now mixed rather than decisive.
Brightness steering moves spectral centroid monotonically, but off-target movement is larger
than target movement at the max scale. Loudness steering is modestly more target-selective.
Token-window brightness steering changes the target window, but the edit bleeds outside the
window. The AudioLDM2 second-model path is implemented and pilot-tested, but the 12-record
pilot has no stable factors and should not be treated as a full replication.

See `docs/future_experiment_results_2026_06_19.md` for the command-level result tables and
artifact paths.

## Limitations

Single model, single corpus; cosine is sign-blind; loudness has silent-clip outliers; the
temporal factors are only partly resolved. The current causal steering results show movement
but limited specificity, and the second-model work is still pilot-only rather than a
cross-model geometry-mirroring result.

## Artifacts

- `results/diverse-corpus-geometry-v1/` — 6-factor panel (r=0.98).
- `results/diverse-corpus-geometry-v2/` — 18-factor panel (r=0.88), linear vs non-linear.
- `results/diverse-corpus-geometry-tb4/`, `-tb8/` — K=4 and K=8 time-binned
  (temporal factors resolution-limited; stability rises with K, F0 recovers).
- `results/diverse-corpus-geometry-tb16-temporal/`, `-tb32-temporal/`,
  `-tb64-temporal/` — high-K six-factor temporal maps.
- `results/brightness-axis-steer-v1/`, `results/loudness-axis-steer-exp2-tb1-v1/` —
  spectral/loudness additive steering specificity tests.
- `results/time-localized-brightness-steer-v1/` — token-window brightness steering.
- `results/second-model-audioldm2-pilot-geometry-v1/` — AudioLDM2 pilot geometry map.
- `results/factor-geometry-v1/`, `factor-geometry-v2/` — earlier designed-sweep attempts
  (superseded; document why designed sweeps fail).
