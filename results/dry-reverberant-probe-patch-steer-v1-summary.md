# Dry/Reverberant Probe -> Patch -> Steer v1

This run repeats the percussive/sustained protocol for the 20-pair
`dry_reverberant` concept screen. Positive side is dry; negative side is
reverberant. The main on-target metric is `direct_to_late_db`, with
`reverb_proxy_score`, late-energy fractions, tail energy, decay time, high/low
energy, and onset tracked as specificity checks.

## Artifacts

- Probe features: `outputs/dry-reverberant-probe-v1/probe-features.npz`
  (gitignored; 40 observations x 24 sites x 2 CFG rows x 1024 hidden dims).
- Probe map: `results/dry-reverberant-probe-v1/`
- Patch cross-check: `results/dry-reverberant-patch-v1/`
- Audio-token-only steering sweep: `results/dry-reverberant-steer-v1/`

## Probe

The label is linearly readable, but realized dry/reverb audio metrics are not
cleanly predicted by grouped ridge probes.

- Best dry-vs-reverberant label decodability: 0.95 at
  `transformer.single_transformer_blocks.6`.
- `cfg_row=1` was selected by automatic label decodability.
- All grouped metric R2 scores are negative. Least-bad targets:
  `decay_time_to_minus_20db_ms` -0.30 at `single_transformer_blocks.7`,
  `direct_to_late_db` -0.65 at `transformer_blocks.3`,
  and `reverb_proxy_score` -0.81 at `single_transformer_blocks.3`.

Interpretation: the model carries the semantic dry/reverberant distinction, but
this 20-pair prompt set does not give a stable linear map from these pooled
features to the realized reverb/tail metrics.

## Patch

Patch run: 20 pairs x 6 sites x 2 directions = 240 valid patched generations.
Sites were selected from the top label site and the least-bad reverb/decay/tail
metric sites.

For `direct_to_late_db`, most rows move toward the source, but the mean is
flipped negative by large outliers:

- Directional fraction moved toward source: 0.825-0.925 across sites.
- Median movement toward source: 0.76-0.97 across sites.
- Mean movement toward source: -1.89 to -0.99 across sites.

Other reverb-adjacent metrics move strongly too. Off-target median movement is
roughly comparable to on-target movement, so this is broad trajectory replacement
rather than specific `direct_to_late_db` control.

## Steer

Steering run: four sites x five reverberant prompts x scales 0.5, 1, 2, 4, with
scale 0 baselines. Steering uses the dry - reverberant mean activation vector and
applies it only to trailing audio tokens for single-stream blocks.

Best steering site:

- `transformer.single_transformer_blocks.6`: `direct_to_late_db` movement
  mean 1.07, median 1.03, fraction moved 0.90.
- Same site off-target movement is also high: mean 1.38, median 1.10.
- Scale 2 is strongest for `direct_to_late_db` at this site
  (mean 1.58, median 1.46).

Other sites move `direct_to_late_db` in the dry direction, but less strongly:
mean 0.32-0.59, median 0.38-0.64.

Interpretation: additive steering can push reverberant prompts toward dry on the
main metric, unlike the percussive onset result, but it is not specific. The
direction appears to be a broad dry/brightness/energy direction rather than a
clean reverberation control knob.

## Bottom Line

Dry/reverberant is weaker than percussive/sustained as a clean causal-tracing
case. It is semantically decodable and steerable in a broad sense, but not a
stable metric-predictive or metric-specific concept under this first 20-pair
protocol. The next useful step would be prompt refinement or a larger dataset
before treating dry/reverberant as a main proof target.
