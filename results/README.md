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
