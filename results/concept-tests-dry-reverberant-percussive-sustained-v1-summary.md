# Dry/Reverberant and Percussive/Sustained Concept Test Summary

## Setup

- Model: `declare-lab/TangoFlux`
- Duration: 3.5 seconds
- Inference steps: 50
- Guidance scale: 4.0
- Pairs per concept: 20
- Generations per concept: 40
- Labels:
  - dry/reverberant: `positive=dry`, `negative=reverberant`
  - percussive/sustained: `positive=percussive`, `negative=sustained`

## Artifacts

Dry/reverberant:

- Prompts: `prompts/dry_reverberant_pairs.jsonl`
- Manifest: `outputs/dry-reverberant-v1-manifest.csv`
- Metrics: `outputs/dry-reverberant-v1-audio-metrics.csv`
- Paired metrics: `outputs/dry-reverberant-v1-audio-metrics-paired.csv`
- Summary: `outputs/dry-reverberant-v1-audio-metrics-summary.json`
- Downloaded subset: `outputs/dry-reverberant-v1-download-subset/`

Percussive/sustained:

- Prompts: `prompts/percussive_sustained_pairs.jsonl`
- Manifest: `outputs/percussive-sustained-v1-manifest.csv`
- Metrics: `outputs/percussive-sustained-v1-audio-metrics.csv`
- Paired metrics: `outputs/percussive-sustained-v1-audio-metrics-paired.csv`
- Summary: `outputs/percussive-sustained-v1-audio-metrics-summary.json`
- Downloaded subset: `outputs/percussive-sustained-v1-download-subset/`

## Dry/Reverberant Result

Dry/reverberant shows a clear but moderate signal. The most interpretable metric is direct-to-late energy in dB, where dry clips should have more direct energy relative to the late tail.

| Metric | Mean paired difference | Median paired difference | Expected direction |
| --- | ---: | ---: | ---: |
| Direct-to-late ratio dB | +4.705 dB | +4.008 dB | 16/20 |
| Reverb proxy score | -0.228 | -0.221 | 15/20 |
| Decay time to -20 dB | -339.882 ms | -249.615 ms | 14/20 |
| Tail energy fraction after 500 ms | -0.208 | -0.159 | 14/20 |
| Late energy fraction after 700 ms | -0.179 | -0.148 | 14/20 |
| Onset strength max | +2.304 | +2.130 | 18/20 |

Interpretation: usable for steering/probing, but not as clean as percussive/sustained. Raw `direct_to_late_ratio` has outliers, so use `direct_to_late_db` as the primary direct/late metric.

## Percussive/Sustained Result

Percussive/sustained is very strong across onset, decay, tail, and spectral measures.

| Metric | Mean paired difference | Median paired difference | Expected direction |
| --- | ---: | ---: | ---: |
| Onset strength max | +10.894 | +12.066 | 20/20 |
| Decay time to -20 dB | -1367.075 ms | -986.848 ms | 17/20 |
| Tail energy fraction after 500 ms | -0.291 | -0.238 | 17/20 |
| Reverb proxy score | -0.279 | -0.292 | 17/20 |
| Direct-to-late ratio dB | +11.281 dB | +7.296 dB | 14/20 |
| Spectral centroid mean | +1068.367 Hz | +1083.924 Hz | 18/20 |
| High/low energy ratio | +26.668 dB | +23.323 dB | 19/20 |

Interpretation: this is one of the cleanest concepts so far. It is a strong candidate for activation patching, linear probing, and later steering validation.

## Quality Guards

Dry/reverberant:

- Valid rows: 40/40
- Mean clip fraction: 0.000408
- Max clip fraction: 0.005708
- Mean peak level: -9.909 dBFS
- Highest clipping row: pair 03 negative, `a reverberant snare drum hit in a gymnasium`

Percussive/sustained:

- Valid rows: 40/40
- Mean clip fraction: 0.002414
- Max clip fraction: 0.084645
- Mean peak level: -4.414 dBFS
- Highest clipping row: pair 19 negative, `a sustained wind chime tone rings softly`

The percussive/sustained concept is strong, but pair 19 should be treated cautiously because the sustained wind-chime output clips heavily and can inflate spectral/tail metrics.

## Recommendation

Use percussive/sustained as the next high-confidence concept. Dry/reverberant is worth keeping, especially for late-energy and direct-to-late tests, but it should probably be refined before we use it as the main causal-steering proof.

