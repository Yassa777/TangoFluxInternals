# Future Experiments

Planned follow-ups to the representation-geometry work (`docs/representation_geometry.md`).
Each entry: what we're after, how it could fail, the nuances that matter, and the steps to
finish. Kept deliberately lightweight.

Results update: these four follow-ups were run as the 2026-06-19 fast-completion batch.
The detailed writeup is in `docs/future_experiment_results_2026_06_19.md`.

High-level status:

- K=16/32/64 temporal maps completed for the six temporal factors; F0 is robustly
  recovered, density recovers at K=16/K=32, attack and AM rate are partial, decay/tail remain
  weak.
- Spectral/loudness steering completed; centroid moves monotonically under brightness
  steering but not specifically, while loudness steering is modestly target-selective.
- AudioLDM2 second-model path is pilot-complete only; the full diverse-corpus replication
  remains a long-running follow-up.
- Time-localized brightness steering completed; the intended window moves, but the effect
  bleeds substantially outside the target window.

---

## 1. Push temporal resolution: K = 16-64

**Overview.** The K-sweep showed temporal factors (attack, decay, onset density, AM rate) are
*resolution-limited, not absent* — their linear readability rose monotonically from K=4 to
K=8, and F0 crossed into "stable". The goal is to push the time-binning finer (K=16, 32, 64;
down to ~55-220 ms bins, and ultimately per-token) and find the resolution at which
decay/density/AM-rate fully recover (stability >> random, R^2 clearly positive), or plateau.

**Blocker to clear first.** K>=16 features exceed Modal's 2 MB per-call return limit
("BlobGet not implemented" in this environment). Fix: have `capture_probe_features` **write
features to the `tangoflux-activations` volume** instead of returning them, and have
`probe_capture` read them back — decoupling feature size from the return path.

**Failure modes.**
- *Plateau below threshold*: stability stops rising and stays under ~0.3 → the factor may be
  genuinely weakly/non-linearly represented (re-check with the non-linear probe at high K).
- *Overfitting*: high-K features are very high-dimensional (K x 1024); PCA components must stay
  well below the 250-sample / ~200-train-fold count, or R^2 collapses.
- *Metric noise*: single-clip attack/decay estimates are noisy; a flat result could be the
  metric, not the model — pair with multi-seed averaging (below) before concluding.

**Nuances.**
- The audio tokens are ~5.4 ms each; attack (~20 ms) needs near-per-token resolution, decay /
  density / AM-rate (~100-500 ms) should recover at K=16-32. Expect a factor-specific
  recovery resolution — that ordering is itself a result.
- Consider **per-token (unpooled) probing** with a small temporal model as the K->645 limit.
- Optional denoising: average the realized metric (and activations) over a few seeds per prompt
  to separate "model doesn't encode it" from "metric is noisy".

**Steps.**
1. Add volume-write + read path to `capture_probe_features` / `probe_capture`.
2. Capture the diverse corpus at K in {16, 32, 64}.
3. Run `geometry_analyze` (PCA tuned to K) for each; track temporal-factor stability/R^2 vs K.
4. Report the recovery-resolution per factor; if attack stays flat, test per-token probing.

---

## 2. Causal confirmation: steer the spectral axis, move centroid not loudness

**Overview.** Probing shows *correlation* (brightness is linearly readable; its direction is
separate from loudness). This tests *causation + specificity*: add the brightness direction to
the activations during generation and confirm the realized **centroid moves monotonically with
scale while loudness stays put** — closing the loop between geometry and behaviour.

**Setup.** Use the brightness direction from `build_geometry_map` (best stable layer, e.g.
`transformer_blocks.3`), apply it with `concept_steer_generate` (audio-token-only,
`SteeringApplier`), sweep scale, and measure the full metric panel on the output.

**Failure modes.**
- *Off-target drag*: centroid and loudness move together → the geometry's "separate axes" claim
  is weaker causally than observationally (report honestly; quantify on-vs-off-target movement).
- *No movement / artefacts*: additive steering may be too weak, or large scales just distort the
  audio (recall additive steering failed for onset). Sweep scale and watch a quality guard
  (clip fraction, crest) to distinguish real control from breakage.
- *Direction sign*: cosine is sign-blind; pick the sign by which way centroid moves.

**Nuances.**
- This is the *positive* counterpart to the earlier onset steering negative — brightness is
  stationary and linearly encoded, so it *should* be the case that steers cleanly. If it does
  not, that gap (linear-readable but not linear-steerable) is itself interesting.
- Normalise movement by the population gap so "specificity" = centroid-movement / centroid-gap
  vs loudness-movement / loudness-gap (reuse `summarize_intervention_rows`).

**Steps.**
1. Extract brightness (and loudness, for contrast) directions at the best stable layer.
2. Steer along brightness at scales {0, 0.5, 1, 2, 4}; measure centroid + loudness + guards.
3. Report specificity (centroid moves, loudness flat) and repeat steering along loudness as a
   cross-check (should move loudness, not centroid).

---

## 3. Second-model replication

**Overview.** The headline claim — *a generative audio model's linearly-accessible factor
geometry mirrors the physical correlation structure (r ~ 0.9)* — is currently single-model.
Replicate the geometry pipeline on a second text-to-audio model (e.g. Stable Audio Open,
AudioLDM 2, or a music model) to show it is a property of audio generators, not of TangoFlux.

**Failure modes.**
- *Different architecture, no clean dual/single split*: the stream-localisation finding may not
  transfer; keep that claim model-specific and lead with the mirroring result.
- *Different latent / token-to-time mapping*: the pooling + K logic assumes tokens map to time;
  verify the new model's latent layout before reusing the recorder.
- *Weaker text control*: realized-corpus binning should still work (it does not rely on control),
  but factor spread must be checked per model.

**Nuances.**
- The reusable contribution is the *method + controls*; the cross-model number to report is the
  geometry-vs-physics correlation r and which factors are stable in each model.
- A *shared* low-dimensional spectral+loudness structure across models would be the strong claim;
  divergence is also publishable (architecture shapes geometry).

**Steps.**
1. Stand up the second model behind the same `_generate_wave` / hook interface.
2. Confirm token->time layout; adapt `ProbeFeatureRecorder` site list if needed.
3. Capture the diverse corpus, run `geometry_analyze`, compare stable factors + r to TangoFlux.

---

## 4. Time-localized steering

**Overview.** Combine the time axis (exposed by K-binning / per-token features) with causal
steering: inject a factor direction **only in specific time bins** and confirm the output
changes **only in the corresponding part of the clip** — e.g. brighten the first half, leave
the second unchanged. This is the natural payoff of preserving temporal structure: a handle on
*when*, not just *whether*.

**Setup.** Extend `SteeringApplier` to accept a token range (analogous to the existing
`suffix_tokens` / `step_window`), apply the brightness direction to only the first N audio
tokens, and measure centroid in time-windowed segments of the output.

**Failure modes.**
- *Bleed across time*: the DiT attends globally, so a token-local edit may smear across the
  whole clip → measure the centroid time-profile, not just the clip mean, to detect bleed.
- *Token<->time misalignment*: confirm the audio-token order actually corresponds to output time
  before interpreting (a quick check: steer only the first tokens, see if only early audio moves).
- *Weak/again-only-spectral*: as with global steering, expect this to work for stationary-ish
  spectral factors first; temporal factors will need the high-K representation from #1.

**Nuances.**
- Requires a *time-resolved* output metric (centroid per output window), not the single
  clip-level number used in the geometry probes.
- Strongest demonstration: a clean "edit only [t0, t1]" with a measured centroid bump localized
  to that window and flat elsewhere.

**Steps.**
1. Add a token-range mask to `SteeringApplier`; expose it through a steering entrypoint.
2. Steer brightness in the first-half tokens only; compute windowed centroid over the output.
3. Show the centroid rises in the steered window and is flat outside; sweep the window position.
4. Once #1 yields high-K temporal directions, repeat for a temporal factor (e.g. local density).

---

## Suggested order

1 (unblocks temporal + feeds 4) -> 2 (cheap causal win, validates geometry) -> 4 (time-localized
control) -> 3 (generality). 2 and 3 can run in parallel with 1.
