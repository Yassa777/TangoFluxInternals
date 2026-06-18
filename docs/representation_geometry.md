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

## The temporal boundary (open problem)

The factors that are NOT recoverable are exactly the *time-varying* ones: attack salience,
decay time, onset density, tail energy, and amplitude-modulation **rate**. The split is sharp
— amplitude-modulation *depth* (a magnitude) is stable while *rate* (a frequency) is at the
noise floor. We have ruled out two explanations:

- not non-linearity (gradient boosting does not help), and
- not simple time-averaging at coarse resolution (K=4 temporal-binned pooling does not help).

Remaining hypotheses: temporal resolution still too coarse, and/or noisy single-clip temporal
metrics. (Finer-binning result recorded below.)

## Limitations

Single model, single corpus; cosine is sign-blind; loudness has silent-clip outliers; the
temporal factors are unresolved. Natural extensions: causal confirmation (steer along the
spectral axis, confirm it moves centroid without moving loudness), and replication of the
geometry-mirrors-physics correlation on a second audio model.

## Artifacts

- `results/diverse-corpus-geometry-v1/` — 6-factor panel (r=0.98).
- `results/diverse-corpus-geometry-v2/` — 18-factor panel (r=0.88), linear vs non-linear.
- `results/diverse-corpus-geometry-tb4/` — K=4 time-binned (temporal negative).
- `results/factor-geometry-v1/`, `factor-geometry-v2/` — earlier designed-sweep attempts
  (superseded; document why designed sweeps fail).
