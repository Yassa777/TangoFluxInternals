# Clear/Muffled and Sharp/Soft Concept Test Summary

## Setup

- Model: `declare-lab/TangoFlux`
- Duration: 3 seconds
- Inference steps: 50
- Guidance scale: 4.0
- Pairs per concept: 20
- Generations per concept: 40
- Labels: `positive` vs `negative`

## Artifacts

Clear/muffled:

- Prompts: `prompts/clear_muffled_pairs.jsonl`
- Manifest: `outputs/clear-muffled-v1-manifest.csv`
- Metrics: `outputs/clear-muffled-v1-audio-metrics.csv`
- Paired metrics: `outputs/clear-muffled-v1-audio-metrics-paired.csv`
- Summary: `outputs/clear-muffled-v1-audio-metrics-summary.json`
- Downloaded subset: `outputs/clear-muffled-v1-download-subset/`

Sharp/soft:

- Prompts: `prompts/sharp_soft_pairs.jsonl`
- Manifest: `outputs/sharp-soft-v1-manifest.csv`
- Metrics: `outputs/sharp-soft-v1-audio-metrics.csv`
- Paired metrics: `outputs/sharp-soft-v1-audio-metrics-paired.csv`
- Summary: `outputs/sharp-soft-v1-audio-metrics-summary.json`
- Downloaded subset: `outputs/sharp-soft-v1-download-subset/`

## Clear/Muffled Result

Clear/muffled shows a real but moderate spectral signal.

| Metric | Mean paired difference | Median paired difference | Expected direction |
| --- | ---: | ---: | ---: |
| Spectral centroid mean | +193.995 Hz | +394.878 Hz | 13/20 |
| Rolloff 85 mean | +268.791 Hz | +321.460 Hz | 13/20 |
| High/low energy ratio | +6.574 dB | +6.686 dB | 11/20 |
| High/total energy ratio | +0.051 | +0.015 | 11/20 |

Interpretation: usable, but less clean than the earlier bright/dark set. The positive side is brighter/clearer on average, but too many pairs move the wrong way for this to be a strong first-pass causal-steering target without prompt refinement.

## Sharp/Soft Result

Sharp/soft is cleaner on transient and onset metrics.

| Metric | Mean paired difference | Median paired difference | Expected direction |
| --- | ---: | ---: | ---: |
| Onset strength p95 | +0.536 | +0.284 | 16/20 |
| Onset strength max | +3.423 | +3.815 | 13/20 |
| Transient energy fraction | +0.036 | +0.020 | 13/20 |
| Transient energy per onset mean | +59.380 | +23.648 | 17/20 |
| First attack time | -43.828 ms | -20.317 ms | 12/20 |
| High/low energy ratio | +7.820 dB | +6.089 dB | 15/20 |

Interpretation: this is a stronger candidate for the next activation-patching or linear-probe concept. The strongest signal is in transient energy per onset and onset strength p95.

## Quality Guards

Clear/muffled:

- Valid rows: 40/40
- Mean clip fraction: 0.0000595
- Max clip fraction: 0.000930
- Mean peak level: -8.754 dBFS

Sharp/soft:

- Valid rows: 40/40
- Mean clip fraction: 0.000278
- Max clip fraction: 0.006878
- Mean peak level: -11.372 dBFS

The runs are much less clipped than the earlier bright/dark music-heavy subset.

## Recommendation

Use sharp/soft as the next concept for a layer-only activation patching sweep. Keep clear/muffled, but refine the prompt set toward cleaner same-source pairs if we want it to become a high-confidence steering target.

