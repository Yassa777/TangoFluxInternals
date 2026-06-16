# Results Index

This directory contains compact, tracked experiment artifacts. Raw generated audio is intentionally
kept out of git; local WAV downloads live under `outputs/*-download-subset/`, and full runs live in
the Modal `tangoflux-outputs` volume.

## Bright/Dark

Top-level files:

- `bright-dark-v1-manifest.csv`: 40 generated baseline records.
- `bright-dark-v1-spectral-centroids.csv`: per-file centroid metrics.
- `bright-dark-v1-spectral-summary.json`: paired bright/dark centroid summary.
- `bright-dark-v1-summary.json`: generation-run summary.

Key result:

- Bright mean centroid: 1851.57 Hz.
- Dark mean centroid: 1308.57 Hz.
- Mean paired difference: +543.00 Hz.
- Bright higher than dark: 14/20 pairs.

## Brightness Patching

Directories:

- `brightness-layer-sweep-final/`: combined 960-row DiT layer-only patching sweep.
- `brightness-centroid-movement-v1/`: top-site centroid movement test with patched audio metadata.

Key result:

- Layer sweep: 960/960 valid rows, mean signed brightness effect +498.12 Hz, positive effect in 658/960 rows.
- Movement test: 160/160 valid rows, 155/160 moved toward the source centroid, 149/160 moved closer to source.

## Acoustic Concept Screens

Each concept directory contains:

- `<run>-manifest.csv`: generation manifest.
- `<run>-audio-metrics.csv`: row-level signal metrics.
- `<run>-audio-metrics-paired.csv`: within-pair positive-negative metric differences.
- `<run>-audio-metrics-summary.json`: grouped and paired metric summaries.
- `<run>-summary.json`: generation summary.

Concept directories:

- `clear-muffled-v1/`
- `sharp-soft-v1/`
- `dry-reverberant-v1/`
- `percussive-sustained-v1/`

Summary reports:

- `concept-tests-clear-muffled-sharp-soft-v1-summary.md`
- `concept-tests-dry-reverberant-percussive-sustained-v1-summary.md`

Best current non-brightness target:

- `percussive-sustained-v1`, with onset strength max in the expected direction for 20/20 pairs and strong decay/tail movement.

## Percussive/Sustained Linear Probe Map

Directory: `percussive-sustained-probe-v1/`

- `probe-map-rows.csv`: per-site decodability (logistic accuracy) and predictability
  (ridge R²) for onset, decay, tail, centroid, and high/low-energy targets.
- `probe-map-summary.json`: CFG-row choice, best sites, and dual-vs-single means.

Captured at all 24 DiT sites for the 20 percussive/sustained pairs (audio-token pooling,
conditional CFG row, `GroupKFold` by `pair_id`, pair 19 excluded from metric targets).

Key result:

- Decodability of the prompt label is high and roughly flat across the whole stack
  (dual mean 0.91, single mean 0.90; best 0.95 at `transformer_blocks.3`). The model
  represents the concept almost everywhere.
- Predictability of the realized *onset strength* is negative/near-zero in the early
  dual blocks and only becomes positive after the streams merge (dual mean −0.38 vs
  single mean +0.23; best 0.35 at `single_transformer_blocks.8`).
- Together this is a measurable semantic→acoustic handoff: the label is linearly present
  early, but the realized percussive attack only becomes linearly readable in the merged
  single stream.
- Decay/tail targets are log1p-transformed (heavy-tailed in ms). Decay then becomes
  predictable in the single stream (best R² +0.33 at `single_transformer_blocks.17`),
  while tail energy stays not linearly accessible (best R² −0.20).
- Best predictability per target: onset +0.35 (`single_blocks.8`), spectral centroid +0.55
  (`single_blocks.14`), high/low energy +0.60 (`transformer_blocks.4`), decay +0.33
  (`single_blocks.17`).

## Percussive/Sustained Causal Patch Cross-Check

Directory: `percussive-sustained-patch-v1/` (6 pairs, both directions, alpha 1.0).

Patches source-side activations into the target generation at the probe's top sites and
measures audio-metric movement toward the source (`toward_source` = fraction of the
source-target gap covered; 1.0 = fully reached the source).

Key result:

- **Onset is causally controllable**: onset-strength moved toward the source in **12/12**
  generations at every tested site (mean fraction 0.90–0.97). The sites that *predict*
  onset also *cause* it — the probe map and the causal map agree.
- **Stream specificity**: single-stream blocks move *everything* (onset and off-target
  metrics both ≈0.98; specificity gap ≈0), because the merged stream carries the whole
  representation. The dual block (`transformer_blocks.4`) is more targeted — onset moves
  more than off-target metrics (gap +0.07).

### 20-pair confirmation and per-metric specificity (`percussive-sustained-patch-v2/`)

Wider run: 20 pairs x 5 sites (dual 3/4, single 8/14/15), both directions.

- Onset causality holds at scale: onset moves 0.86–0.99 toward source at every site, with
  dual blocks the most onset-specific (`transformer_blocks.3` gap +0.08).
- Re-summarizing the same rows for other on-targets (`patch-summary-<metric>.json`):
  spectral centroid is the most *specifically* controllable metric and peaks in the dual
  stream (`transformer_blocks.3` gap +0.13); decay resists specific control (negative gaps
  except `transformer_blocks.4`). The site that best *predicts* a metric is not always the
  site where patching most *specifically* moves it.

## Percussive/Sustained Steering Specificity

Directory: `percussive-sustained-steer-v1/` (5 sustained prompts, scales 1 and 2).

Builds per-site steering vectors (percussive − sustained) from the captured features and
adds them during generation, measuring movement relative to the population gap.

Key result (a deliberate negative/contrast finding):

- Simple additive steering with the mean-difference vector does **not** specifically move
  onset (toward-source ≈0.06–0.08) while it moves off-target spectral/energy metrics a lot
  (≈0.85–0.91). The linear difference direction is dominated by spectral content, not the
  transient onset structure.
- Contrast with patching: replacing the activation trajectory moves onset strongly, but a
  constant additive bias broadcast over tokens/timesteps does not. Onset looks like a
  trajectory/temporal property rather than a single additive direction at these sites.
- Caveat: only scales 1–2 and a token-broadcast scheme were tested. Audio-token-only
  application, a wider scale sweep, or per-timestep steering are open follow-ups before
  concluding onset is un-steerable.

### Audio-token-only steering follow-up (`percussive-sustained-steer-v2-audio/`)

Restricting steering to the trailing audio tokens and sweeping scale 0.5–4 (the open
follow-up above) gives onset a fair shot. Onset response roughly doubled (toward-source
0.06→0.18 at `single_blocks.8`) but stays dominated by off-target movement at every scale:
at scale 0.5, onset moves 0.28 while spectral centroid moves 0.72; at larger scales centroid
overshoots (≈1.5) while onset stays ≤0.3. There is no scale where onset moves cleanly on its
own. Interpretation: the linear percussive−sustained activation direction is essentially a
*spectral* (brightness) direction — additive steering drives centroid, while the transient
onset is only moved by full-trajectory patching. This sharpens, rather than overturns, the
earlier steering result.

## Flow-Time Commitment (Method + First Result)

Directory: `percussive-sustained-commitment-v1/` (5 pairs, 2 sites, 50 steps, both directions).

New primitive: `ActivationPatcher` can gate patching to a denoising-step window, so we
patch a site's source trajectory only during prefix `[0,k)` or suffix `[k,T)` flow-steps
and measure how much of the attribute still moves. Sweeping the window recovers *when* in
the generation trajectory an attribute is causally malleable. This is flow-time-resolved
causal tracing — the temporal axis most static-pass interpretability ignores.

Movement-toward-source by window (fraction of source-target gap; T=50):

| site / attribute | [0,12) | [0,25) | [0,50) | [12,50) | [25,50) | [38,50) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single_8 onset | 0.76 | 0.92 | 0.95 | 0.75 | 0.52 | 0.26 |
| single_8 centroid | 0.84 | 0.93 | 0.99 | 0.75 | 0.49 | 0.31 |
| dual_3 onset | 0.58 | 0.82 | 0.86 | 0.74 | 0.52 | 0.26 |
| dual_3 centroid | 0.66 | 0.86 | 0.91 | 0.74 | 0.50 | 0.32 |

Findings (honest):

- **Commitment is early-weighted and graded, not a sharp point of no return.** Patching only
  the first quarter of steps `[0,12)` already secures ~76–84% of the full effect; the first
  half `[0,25)` reaches ~92–93%. Late-only patching tapers smoothly (`[25,50)` ≈ 0.5,
  `[38,50)` ≈ 0.3). The high-noise early phase carries most of the causal weight.
- **The coarse-to-fine *ordering* hypothesis is NOT supported at this resolution.** Onset
  (expected "late/fine") and spectral centroid (expected "early/global") commit on nearly
  identical schedules at both sites. The earlier PoNR difference (onset 25 vs centroid 12)
  was a threshold artifact (suffix `[25,50)` = 0.52 vs 0.49). This challenges, rather than
  confirms, the image-diffusion folklore that fine detail is always committed late — at
  least for these two acoustic attributes at these sites.

Implications for next experiments:

- Most of the action is inside `[0,12)`; we did not resolve *within* the early phase.
  A finer early sweep (cuts at ~2,4,6,9,12) is needed to see any attribute ordering.
- Activation-level windowed patching conflates layer and flow-time. A **latent-level**
  swap (replace the latent at step k and continue) would isolate pure flow-time and give a
  clean, architecture-free point-of-no-return.

## Representation-Geometry Testbed (Ground-Truth Factors)

Directory: `factor-geometry-v1/` (60 prompts: brightness sweep + onset-rate sweep,
6 levels x 5 sources each; 24 sites; grouped CV by source).

Method: use *physically measurable* generative factors as ground truth to test
representation-geometry claims a language model can't settle. Per layer we report
grouped-CV linear R^2, out-of-sample projection monotonicity (Spearman of
`cross_val_predict`), within-factor direction stability (split-half cosine), and the
cross-factor cosine — each against a random-vector baseline `E|cos| = sqrt(2/(pi*d)) =
0.025` for d=1024.

Manipulation check (essential): brightness level -> realized centroid Spearman 0.56
(range 646-5754 Hz) — a usable factor. Onset-rate level -> realized rate Spearman 0.20 —
the text **did not** reliably control rate, so onset-rate results are treated as a failed
manipulation, not a model claim.

Findings (honest):

- **Brightness is encoded as a robust *ordinal* axis at every layer**: held-out projection
  Spearman 0.71-0.84 across all 24 sites (source-generalizing). The direction is
  reproducible above chance everywhere (split-half cosine 0.07-0.21 vs 0.025 random),
  peaking at dual `transformer_blocks.3` (0.21, ~8x random).
- **Ordinal, but not linearly calibrated.** Linear R^2 is mostly <=0 (the magnitude in Hz
  is not captured) except the early dual blocks (`transformer_blocks.3/4`, R^2 ~0.2). So
  the model represents brightness *ordering* strongly while affine-linear *magnitude* is
  legible only in the early dual (text-conditioned) stream. This monotonic-not-linear
  distinction is only visible because we regress against the exact physical scale — the
  kind of claim LM interpretability cannot make.
- **Disentanglement is at the noise floor and underpowered.** Cross-factor cosine (0.037)
  ~ random (0.025), and within-factor stability for onset-rate (0.05) is barely above
  random — so no disentanglement claim can be made at N=60 in 1024-d. The *method*
  (within- vs cross-factor cosine vs random baseline) is the right test; the data is too
  small and onset-rate too poorly manipulated.

What a properly powered version needs: many more sources/levels per factor, PCA before
probing (reduce 1024-d), a text-manipulation that actually moves onset-rate (or a
non-text control of it), and a third clean factor. The contribution is the *testbed +
controls*, plus the concrete finding that brightness is represented ordinally with
affine-linear legibility localized to the early dual stream.

## Representation-Geometry v2 (Powered: PCA + controls)

Directory: `factor-geometry-v2/` (96 prompts: brightness + decay sweeps, 8 sources x 6
levels each; PCA-20 before probing; grouped CV by source; decay log1p-transformed).
Random |cos| baseline at PCA-20 is 0.178.

Manipulation check (honest, and a problem): brightness level -> realized centroid
Spearman 0.30; **decay level -> realized decay 0.00 (control failed entirely)**. TangoFlux
does not reliably realize "short vs sustained" text, and the new heterogeneous brightness
sources add baseline variance. Realized factors still vary (centroid 570-5340 Hz, decay
12-3460 ms) and are mutually independent (0.13), so realized-factor encodability is still
testable.

Findings:

- **Brightness replicates and strengthens as a reproducible *ordinal* axis.** With PCA-20
  the direction is highly stable across source-disjoint halves (split-half cosine
  0.62-0.78, mean 0.67, vs 0.178 random ~3.8x) and monotonically recovered out-of-sample
  (Spearman 0.57-0.75, slightly stronger in the dual stream). But R^2 < 0 everywhere: the
  model encodes brightness *ordering*, not a calibrated linear magnitude. This is the same
  monotonic-not-linear result as v1, now with proper power and dimensionality control.
- **Decay is not recoverable** (stability 0.22 ~ random, Spearman ~0) -- a consequence of
  the failed manipulation and a noisy metric, not a model fact.
- **Disentanglement cannot be claimed**: cross-factor cosine (0.206) ~ random (0.178), and
  because decay's own direction is at the noise floor the cosine is uninformative. A clean
  disentanglement test needs two factors that each have stable directions.

Methodological conclusion (the actionable lesson): designed text "sweeps" do not give
clean single-factor control in TangoFlux (brightness weakly, decay not at all). The fix is
to **decouple ground truth from prompt control** -- generate a large, diverse prompt corpus
and bin/regress activations by the *realized measured* factor, rather than trying to make
prompts sweep a factor. Brightness already shows that realized variation is robustly
encoded even when the manipulation is weak, which validates the realized-binning approach.

## Representation-Geometry v3 (Realized-Corpus, 8 factors)

Directory: `diverse-corpus-geometry-v1/` (250 diverse prompts, 40 source groups; 8 realized
factors measured per clip; PCA-20; grouped CV by source; decay log1p). This uses the
realized-binning approach (measure factors on a diverse corpus) instead of designed text
sweeps. Random |cos| baseline 0.178; stability threshold 2x = 0.357.

Per-factor (direction stability / best linear R^2 / best out-of-sample Spearman):

| factor (metric) | stability | best R^2 | best CV Spearman |
| --- | ---: | ---: | ---: |
| brightness (centroid) | 0.85 | +0.59 | 0.82 |
| rolloff_85 | 0.76 | +0.42 | 0.74 |
| tilt (high/low dB) | 0.73 | +0.49 | 0.76 |
| loudness (rms dBFS) | 0.69 | +0.15 | 0.56 |
| density (onset rate) | 0.34 | +0.19 | 0.49 |
| attack (onset strength) | 0.30 | +0.03 | 0.40 |
| decay (log) | 0.26 | +0.03 | 0.29 |
| tail energy | 0.13 | -0.08 | 0.08 |

Findings:

- **Correction of the v1/v2 claim**: with 250 diverse samples brightness is *linearly*
  encodable (R^2 **+0.59**, dual block 3), not merely ordinal. The earlier negative R^2 was
  a small-N / source-confound artifact, not a property of the model. Science self-correcting.
- **Only 4 factors have trustworthy directions** (stability > 0.357): the three spectral
  metrics + loudness. The temporal/transient/envelope factors (attack, decay, density, tail)
  are not robustly linearly recoverable here — either the model does not linearly encode them
  or the metrics are too noisy on diverse audio (not yet separable).
- **The model's factor geometry mirrors the physical correlation structure.** Among the
  stable factors, direction-cosine vs realized-Spearman:

  ```
  direction cosine (model)      realized |corr| (physical)
            bri  rol  til  lou            bri  rol  til  lou
  bright   1.00 0.92 0.97 0.29   bright  1.00 0.95 0.89 0.24
  rolloff  0.92 1.00 0.87 0.48   rolloff 0.95 1.00 0.79 0.30
  tilt     0.97 0.87 1.00 0.35   tilt    0.89 0.79 1.00 0.20
  loudness 0.29 0.48 0.35 1.00   loudness0.24 0.30 0.20 1.00
  ```

  The three physically-collinear spectral factors (|r| 0.79-0.95) are encoded along a single
  shared direction (cosine 0.87-0.97) — the model has one "brightness" latent, not three
  independent encodings. Loudness, physically near-independent of spectrum (|r| 0.2-0.3, and
  negative in sign), occupies a distinct axis (cosine 0.29-0.48). So the linearly-accessible
  representation is low-dimensional: a spectral axis + a loudness axis, with internal geometry
  that tracks the output-space statistics. Spectral structure is most legible in the early
  dual stream (best R^2 at `transformer_blocks.3`).
- **Quantitatively, the internal geometry mirrors physics**: across the 6 off-diagonal factor
  pairs, the model direction-cosine correlates with the physical |Spearman| at **Pearson
  r = 0.98**. The representation's pairwise factor structure is nearly a copy of the output's.

Limitations: cosine is sign-blind; temporal factors may need denoised metrics before any
claim; loudness has silent-clip outliers; single model / single corpus. Next: denoise the
temporal metrics (or pick cleaner temporal factors) to test whether transient/envelope axes
exist separately, and replicate on a second model.

## Representation-Geometry v4 (18 factors, linear vs non-linear)

Directory: `diverse-corpus-geometry-v2/` (same 250 clips, 18 realized factors spanning
spectral / level / envelope / pitch / harmonicity / modulation; PCA-20; grouped CV;
gradient-boosting non-linear estimator alongside linear ridge).

Findings:

- **12 of 18 factors have stable directions** (split-half cosine > 2x random = 0.357), and
  every one of them is a *stationary* timbre property: the spectral family (brightness,
  rolloff, tilt, bandwidth, contrast, flatness, ZCR), plus voicing, HNR, AM depth, loudness,
  crest. The 6 unstable factors are exactly the *time-varying* ones: attack, decay, onset
  density, tail, AM rate, and (sparse) F0.
- **Non-linearity is not the missing ingredient.** A gradient-boosting probe does not beat
  linear ridge for the temporal factors (non-linearity gains <= 0). So they are not
  "non-linearly encoded" -- they are absent from the token-mean-pooled features.
- **The cause is pooling, not the model or noise.** We mean-pool over audio tokens, which
  discards the time axis, so only time-invariant factors survive. The cleanest evidence is
  the modulation family splitting exactly on this line: AM *depth* (a magnitude) is stable
  (0.60) while AM *rate* (a frequency, purely temporal) is at the noise floor (0.21).
  Multi-seed denoising would not fix this; **time-aware pooling** (time-binned or unpooled
  token features) is the right fix, and it also sets up the revived flow-time commitment work.
- **Geometry-mirrors-physics replicates at scale**: across all 66 stable-factor pairs the
  model direction-cosine correlates with the physical |Spearman| at **Pearson r = 0.88**
  (was 0.98 on 6 pairs). The spectral family clusters (mean cosine 0.65); loudness/crest/AM
  depth are more separate (0.35-0.40 to the spectral cluster). The representation's factor
  geometry robustly tracks the output-space statistics.

## Time-Binned Pooling Test (temporal factors)

Directory: `diverse-corpus-geometry-tb4/` (same 250 clips, captured with `--time-bins 4`:
audio tokens split into 4 temporal bins -> `[batch, 4*1024]` features; PCA-30).

Hypothesis: token-mean pooling discards time, so temporal factors fail; preserving coarse
time (4 bins) should bring them back.

Result -- the hypothesis is **not** supported at K=4:

- The temporal factors stay at the noise floor: attack stability 0.20, decay 0.19, tail
  0.05, AM-rate 0.16 (threshold 0.291); density 0.29 and F0 0.37 are borderline. Best linear
  R^2 stays ~0 for attack/decay/tail. K=4 binning did not rescue them (they were also below
  threshold at K=1).
- The stationary geometry is unaffected: 13 stable factors, and geometry-mirrors-physics
  still holds (Pearson r = 0.80 over 78 pairs vs 0.88 at K=1).

Interpretation / leading explanations (now narrowed):
- **K=4 is likely too coarse**: 4 bins over 3.5 s = ~875 ms each, far coarser than an attack
  (~20 ms) or inter-onset spacing (~140 ms at 7 events/s). Finer binning (K=16-32, ~110-220 ms)
  is the obvious next test for decay/density.
- Combined with the earlier non-linear test (gradient boosting did not help either), the live
  hypotheses for the temporal factors are now: **too-coarse temporal resolution** and/or
  **noisy single-clip temporal metrics** -- not non-linearity, and not simple time-averaging.
- It remains possible these fast temporal properties are not linearly read out from the
  audio-token activations at all; finer bins + denoised metrics are needed to decide.

Takeaway: the durable, replicated result is the stationary-factor geometry (spectral family +
loudness/harmonicity/voicing, mirroring physics at r=0.8-0.98). The temporal factors are a
clean open problem with two concrete remaining tests (finer K, denoised metrics).

## Artifact Policy

Tracked:

- prompt JSONL files
- metric CSV/JSON summaries
- research notes and plans
- Modal/code utilities

Not tracked:

- generated WAV files
- local PDF copies of papers
- Python caches
- Modal/Hugging Face caches
- activation tensor dumps
