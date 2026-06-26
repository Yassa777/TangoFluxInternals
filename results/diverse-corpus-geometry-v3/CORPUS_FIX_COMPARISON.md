# Corpus duration-bug fix: geometry impact (v2 contaminated -> v3 fixed)

Both analyses use **identical** `geometry_analyze` params (full 18-factor panel, cfg-row 0, log-factors decay). The only difference is the underlying corpus:

- **v2 (old, contaminated):** `diverse_corpus_v1.jsonl`, duration=3.5s, comma-stack prompts. Truncation silenced late-onset clips, so realized acoustic factors (centroid/onset/loudness/...) were computed on partly-silent audio.
- **v3 (fixed):** `diverse_corpus_v2.jsonl`, duration=10s, natural captions, CFG 4.5, 50 steps.

## Analysis layer selected
- v2: `transformer.transformer_blocks.0` (dual) — **block 0**, i.e. the analysis found no meaningful deep structure and fell back to the input-adjacent layer.
- v3: `transformer.single_transformer_blocks.15` (single) — a sensible mid-deep layer.

## Per-factor encodability / rank / direction stability

| factor | R² old | R² new | Δ | ρ old | ρ new | stab old | stab new |
|---|---:|---:|---:|---:|---:|---:|---:|
| brightness | 0.29 | 0.94 | +0.65 | 0.78 | 0.99 | 0.15 | 0.54 |
| rolloff | 0.24 | 0.96 | +0.72 | 0.76 | 0.98 | 0.21 | 0.55 |
| bandwidth | -0.02 | 0.97 | +0.98 | 0.67 | 0.98 | 0.19 | 0.59 |
| contrast | -0.03 | 0.89 | +0.91 | 0.67 | 0.96 | 0.16 | 0.35 |
| zcr | 0.03 | 0.88 | +0.85 | 0.62 | 0.96 | 0.06 | 0.46 |
| loudness | -1.08 | 0.80 | +1.88 | 0.51 | 0.91 | 0.09 | 0.28 |
| crest | -0.79 | 0.53 | +1.32 | 0.48 | 0.82 | 0.05 | 0.21 |
| hnr | -0.45 | 0.55 | +1.00 | 0.48 | 0.83 | 0.08 | 0.18 |
| voiced | -0.41 | 0.76 | +1.17 | 0.54 | 0.91 | 0.06 | 0.19 |
| tilt | 0.19 | 0.57 | +0.39 | 0.69 | 0.84 | 0.05 | 0.15 |
| flatness | -2.20 | -0.03 | +2.17 | 0.52 | 0.73 | 0.13 | 0.11 |
| density | -0.92 | 0.01 | +0.93 | 0.33 | 0.57 | 0.05 | 0.13 |
| attack | -1.35 | -0.53 | +0.83 | 0.31 | 0.54 | 0.07 | 0.11 |
| decay | -1.06 | -0.55 | +0.51 | 0.23 | 0.20 | 0.04 | 0.05 |
| tail | -1.37 | -0.99 | +0.38 | 0.14 | 0.20 | 0.03 | 0.03 |
| f0 | -0.95 | -0.49 | +0.46 | 0.19 | 0.43 | 0.03 | 0.08 |
| am_depth | -1.83 | -0.78 | +1.05 | 0.47 | 0.66 | 0.12 | 0.16 |
| am_rate | -1.90 | -0.91 | +0.99 | 0.20 | 0.27 | 0.02 | 0.05 |

R² is best-layer cross-validated linear encodability (negative = worse than predicting the mean). ρ = best-layer CV Spearman. stab = split-half direction cosine on source-disjoint halves.

## Key direction cosines (|cos|, lower = more separable; random baseline = 0.025)

| factor pair | old | new | Δ |
|---|---:|---:|---:|
| brightness × loudness | 0.221 | 0.071 | -0.149 |
| density × loudness | 0.145 | 0.037 | -0.108 |
| brightness × decay | 0.136 | 0.036 | -0.101 |
| brightness × density | 0.059 | 0.040 | -0.019 |
| decay × loudness | 0.064 | 0.152 | +0.087 |
| brightness × rolloff | 0.751 | 0.823 | +0.071 |

## Takeaways

- The v2 corpus produced **mostly negative R²** (loudness −1.08, density −0.92, flatness −2.20) and layer-0 fallback — hallmarks of contaminated realized factors from silenced clips.
- The fixed corpus yields **strong, stable, linearly-encodable spectral/loudness geometry** at a mid-deep layer: brightness R² 0.29→0.94, loudness −1.08→0.80, rolloff 0.24→0.96, bandwidth −0.02→0.97.
- **brightness×loudness direction cosine 0.221→0.071** (near the 0.025 random baseline): brightness and loudness are encoded along near-orthogonal directions — this *strengthens* the 'patch centroid without moving loudness' specificity story rather than resting on a contaminated number.
- Fine temporal-envelope factors (decay, attack, tail, AM rate) remain weakly encodable even after the fix — an honest limitation: TangoFlux's mid-layer geometry represents spectral/loudness structure well but fine envelope timing poorly.
