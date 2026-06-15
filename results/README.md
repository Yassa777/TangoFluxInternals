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
