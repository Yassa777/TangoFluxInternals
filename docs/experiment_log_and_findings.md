# TangoFlux Internals Experiment Log and Findings

This document is a consolidated record of what has been attempted in this
codebase, what each experiment was meant to test, what artifacts were produced,
and what the current evidence says. It is intentionally result-first: if a claim
is not backed by a committed CSV, JSON, Markdown summary, or explicit repo
workflow, it is treated as a working interpretation rather than a result.

## Executive Summary

The codebase has evolved from a Modal-backed TangoFlux generation harness into a
small mechanistic-interpretability lab for text-to-audio concepts. The strongest
current direction is metric-grounded semantic-to-acoustic causal tracing:

1. Generate paired contrastive prompts with shared settings and seeds.
2. Measure realized waveform properties, not just prompt labels.
3. Capture pooled DiT activations at dual-stream and single-stream sites.
4. Train grouped linear probes by `pair_id`.
5. Patch full activation trajectories and measure movement in waveform metrics.
6. Try additive steering vectors and compare on-target against off-target
   movement.

The strongest concept is still `percussive_sustained`. It separates cleanly in
audio metrics, has strong label decodability, has positive metric predictability
for onset and spectral targets, and activation patching moves onset strongly.
However, additive steering mostly moves spectral/energy properties rather than
onset specifically.

`dry_reverberant` is useful but weaker. It is semantically decodable and can be
steered in a broad sense, but grouped audio-metric R2 is negative and patching or
steering does not isolate a clean reverberation control direction.

Brightness remains the clearest activation-patching proof of concept: centroid
patching moves audio toward the counterfactual source in most rows.

The 2026-06-19 future-experiment batch extends the representation-geometry work:
K=16/32/64 temporal maps now run through volume-backed feature capture, spectral
and loudness additive steering has causal specificity measurements, AudioLDM2 has
an end-to-end second-model pilot, and token-window brightness steering has a first
localization result. The result is mixed: F0 is robustly recovered at high temporal
resolution, density is recovered at K=16/K=32, brightness/loudness steering moves
the intended metrics but with limited specificity, AudioLDM2 is pilot-only, and
time-localized steering bleeds across windows. Detailed tables are in
`docs/future_experiment_results_2026_06_19.md`.

## Repository State and Core Components

| Area | Files | Purpose |
| --- | --- | --- |
| Modal app | `modal_app.py` | Defines Modal image, GPU functions, volumes, and local entrypoints. |
| Prompt parsing | `src/tangoflux_lab/records.py` | Expands single and contrastive prompt JSONL rows into deterministic generation records. |
| Generation utilities | `src/tangoflux_lab/generation.py` | Run prefixes, output paths, CSV/JSON writing, and WAV path conventions. |
| Audio metrics | `src/tangoflux_lab/audio_features.py` | Spectral, onset, transient, decay, tail, direct-to-late, and quality metrics. |
| Metric summaries | `src/tangoflux_lab/metric_summaries.py` | Group and paired summaries for positive-vs-negative concept screens. |
| Hook utilities | `src/tangoflux_lab/hooks.py` | Activation capture, suffix-token patching, mean-difference steering, and hook specs. |
| Probes | `src/tangoflux_lab/probing.py` | Pair-grouped logistic label probes, ridge metric probes, and intervention summaries. |
| Prompt sets | `prompts/*.jsonl` | Contrastive concept definitions. |
| Results | `results/` | Tracked compact CSV, JSON, and Markdown artifacts. |
| Raw outputs | `outputs/`, Modal volumes | Generated WAVs and activation feature bundles; intentionally mostly untracked. |

The Modal app uses three persistent volumes:

| Volume | Mount | Role |
| --- | --- | --- |
| `tangoflux-hf-cache` | `/cache/huggingface` | Model and Hugging Face cache. |
| `tangoflux-outputs` | `/outputs` | Generated WAVs and remote output records. |
| `tangoflux-activations` | `/activations` | Activation payloads and steering vectors. |

Default remote runtime:

| Setting | Value |
| --- | --- |
| Model | `declare-lab/TangoFlux` |
| Default GPU | `L40S` |
| Python | 3.11 in Modal image |
| Torch | 2.4.0 |
| Torchaudio | 2.4.0 |
| Transformers | 4.44.0 |
| Diffusers | 0.30.0 |

## Experimental Design Principles

The recurring protocol has four guardrails:

| Principle | Implementation |
| --- | --- |
| Preserve pair structure | Contrastive rows share settings and seeds; summaries use within-pair differences. |
| Avoid row-wise leakage | Probe splits use `GroupKFold` by `pair_id`. |
| Prefer realized audio metrics | Acoustic targets come from waveform analysis, not only prompt labels. |
| Separate decodability from causality | Linear probes answer what is readable; activation patching tests causal movement. |

The key distinction is between prompt-label decodability and audio-metric
predictability. A site can encode "this prompt asked for dry audio" without
linearly predicting how much late energy the generated waveform actually has.
The strongest research claim depends on where those two maps diverge or align.

## Timeline of Attempts

| Stage | What was attempted | Result |
| --- | --- | --- |
| Environment scaffold | Built a Modal-backed TangoFlux lab with CUDA/GPU support and persistent caches. | Working remote environment with cached model/runtime and repeatable local entrypoints. |
| Smoke generation | Generated short test WAVs through Modal. | Confirmed model execution and output volume wiring. |
| Bright/dark screen | Generated 20 bright/dark pairs and measured spectral centroid. | Bright prompts were higher in 14/20 pairs, mean difference +543 Hz. |
| Brightness patching | Patched layer activations across dual and single DiT sites. | 960/960 valid rows; mean signed brightness effect +498 Hz. |
| Brightness movement subset | Regenerated top-site patched audio and measured source movement. | 155/160 rows moved toward source; 149/160 moved closer to source. |
| Metric suite expansion | Added shared spectral, onset, transient, decay, tail, direct-to-late, and quality metrics. | Enabled concept screens beyond brightness. |
| Clear/muffled screen | Generated and measured 20 clear/muffled pairs. | Moderate spectral signal; not clean enough as a first causal target. |
| Sharp/soft screen | Generated and measured 20 sharp/soft pairs. | Stronger transient/onset signal; second-best non-brightness candidate. |
| Dry/reverberant screen | Generated and measured 20 dry/reverberant pairs. | Direct-to-late and reverb proxy are usable but outlier-sensitive. |
| Percussive/sustained screen | Generated and measured 20 percussive/sustained pairs. | Cleanest concept; onset max separates 20/20 pairs. |
| Probe methodology | Chose grouped pair splits and prompt-vs-realized metric distinction. | Avoids pair leakage and keeps decodability separate from acoustic realization. |
| Percussive/sustained probes | Captured audio-token-pooled features at 24 DiT sites and trained grouped probes. | Label decodable broadly; onset predictability becomes positive in single stream. |
| Percussive/sustained patching | Patched top probe sites and measured metric movement. | Onset moves strongly toward source at all tested sites. |
| Percussive/sustained steering | Applied mean percussive-minus-sustained vectors. | Additive steering mostly moves spectral/energy metrics, not onset specifically. |
| Dry/reverberant probes | Repeated grouped probe protocol with dry/reverb targets. | Label decodable, but all grouped metric R2 values are negative. |
| Dry/reverberant patching | Patched label and reverb/tail candidate sites. | Many rows move toward source by median/count, but effects are broad and outlier-sensitive. |
| Dry/reverberant steering | Steered reverberant prompts toward dry using audio-token-only vectors. | `direct_to_late_db` moves toward dry, strongest at `single_blocks.6`, but off-target movement is high. |
| Novelty check | Compared broad audio steering/patching claim against adjacent work. | Broad "audio activation steering is novel" is unsafe; metric-grounded semantic-to-acoustic tracing is more defensible. |

## Prompt Sets

| Prompt file | Concept | Positive side | Negative side | Pairs | Typical duration/steps |
| --- | --- | --- | --- | ---: | --- |
| `prompts/bright_dark_pairs.jsonl` | brightness | bright | dark | 20 | short paired generation |
| `prompts/clear_muffled_pairs.jsonl` | clarity | clear | muffled | 20 | 3 s, 50 steps |
| `prompts/sharp_soft_pairs.jsonl` | transients | sharp | soft | 20 | 3 s, 50 steps |
| `prompts/dry_reverberant_pairs.jsonl` | reverberation | dry | reverberant | 20 | 3.5 s, 50 steps |
| `prompts/percussive_sustained_pairs.jsonl` | onset/decay | percussive | sustained | 20 | 3.5 s, 50 steps |

## Audio Metrics Implemented

| Metric family | Example columns | Main use |
| --- | --- | --- |
| Waveform quality | `rms_dbfs`, `peak_dbfs`, `clip_fraction`, `silence_fraction` | Detect bad runs and clipping outliers. |
| Spectral centroid | `spectral_centroid_mean_hz`, `spectral_centroid_median_hz` | Brightness, clarity, spectral shift. |
| Spectral rolloff | `rolloff_85_mean_hz`, `rolloff_95_mean_hz` | Bandwidth and muffling. |
| Frequency energy | `high_to_low_db`, `high_to_total_ratio` | Brightness, sharpness, muffling. |
| Onset/transient | `onset_strength_max`, `onset_strength_p95`, `transient_energy_fraction` | Percussive and sharp concepts. |
| Attack timing | `attack_time_ms_first`, `attack_time_ms_median` | Sharp/soft attack comparisons. |
| Tail/decay | `tail_energy_fraction_500ms`, `decay_time_to_minus_20db_ms` | Sustained and reverberant concepts. |
| Direct-to-late | `direct_to_late_db`, `direct_to_late_ratio` | Dry/reverberant contrast. |
| Reverb proxy | `reverb_proxy_score`, `late_energy_fraction_300ms`, `late_energy_fraction_700ms` | Relative reverberation/tail evidence. |

Important caveat: direct-to-late and reverb proxy metrics are relative proxies
from a single generated waveform. They are useful within paired prompts, but
they are not physical room measurements.

## Baseline Concept Screen Results

### Bright/Dark

| Metric | Bright mean | Dark mean | Mean paired diff | Median paired diff | Expected direction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spectral centroid mean Hz | 1851.6 | 1308.6 | +543.0 | +298.9 | 14/20 |

Interpretation: brightness is usable as a first activation-patching target. The
paired result is not perfect, but the mean shift is large and patching later
confirmed causal movement.

### Clear/Muffled

| Metric | Mean diff | Median diff | Expected direction |
| --- | ---: | ---: | ---: |
| `spectral_centroid_mean_hz` | +194.0 | +394.9 | 13/20 (0.65) |
| `rolloff_85_mean_hz` | +268.8 | +321.5 | 13/20 (0.65) |
| `high_to_low_db` | +6.574 | +6.686 | 11/20 (0.55) |
| `high_to_total_ratio` | +0.051 | +0.015 | 11/20 (0.55) |

Interpretation: clear/muffled has a moderate spectral signal, but too many
pairs move the wrong way for a high-confidence first causal target.

### Sharp/Soft

| Metric | Mean diff | Median diff | Expected direction |
| --- | ---: | ---: | ---: |
| `onset_strength_p95` | +0.536 | +0.284 | 16/20 (0.80) |
| `onset_strength_max` | +3.423 | +3.815 | 13/20 (0.65) |
| `transient_energy_fraction` | +0.036 | +0.020 | 13/20 (0.65) |
| `transient_energy_per_onset_mean` | +59.38 | +23.65 | 17/20 (0.85) |
| `attack_time_ms_first` | -43.83 | -20.32 | 12/20 (0.60) |
| `high_to_low_db` | +7.820 | +6.089 | 15/20 (0.75) |

Interpretation: sharp/soft is cleaner than clear/muffled. It remains a plausible
next candidate, especially if the target is transient energy per onset rather
than raw onset maximum.

### Dry/Reverberant

| Metric | Mean diff | Median diff | Expected direction |
| --- | ---: | ---: | ---: |
| `direct_to_late_db` | +4.705 | +4.008 | 16/20 (0.80) |
| `reverb_proxy_score` | -0.228 | -0.221 | 15/20 (0.75) |
| `decay_time_to_minus_20db_ms` | -339.9 | -249.6 | 14/20 (0.70) |
| `tail_energy_fraction_500ms` | -0.208 | -0.159 | 14/20 (0.70) |
| `late_energy_fraction_700ms` | -0.179 | -0.148 | 14/20 (0.70) |
| `onset_strength_max` | +2.304 | +2.130 | 18/20 (0.90) |

Interpretation: dry/reverberant is usable but not clean. `direct_to_late_db` and
`reverb_proxy_score` are the preferred metrics. Raw `direct_to_late_ratio` is
too outlier-prone.

### Percussive/Sustained

| Metric | Mean diff | Median diff | Expected direction |
| --- | ---: | ---: | ---: |
| `onset_strength_max` | +10.89 | +12.07 | 20/20 (1.00) |
| `decay_time_to_minus_20db_ms` | -1367.1 | -986.8 | 17/20 (0.85) |
| `tail_energy_fraction_500ms` | -0.291 | -0.238 | 17/20 (0.85) |
| `reverb_proxy_score` | -0.279 | -0.292 | 17/20 (0.85) |
| `direct_to_late_db` | +11.28 | +7.296 | 14/20 (0.70) |
| `spectral_centroid_mean_hz` | +1068.4 | +1083.9 | 18/20 (0.90) |
| `high_to_low_db` | +26.67 | +23.32 | 19/20 (0.95) |

Interpretation: this is the cleanest non-brightness concept. It is the best
current target for probe -> patch -> steer experiments.

## Quality Guards

| Run | Valid rows | Mean clip fraction | Max clip fraction | Mean peak dBFS | Highest clipping row |
| --- | ---: | ---: | ---: | ---: | --- |
| `clear-muffled-v1` | 40/40 | 0.000060 | 0.000930 | -8.754 | pair 09 negative, `a muffled typewriter carriage bell dings behind a closed door` |
| `sharp-soft-v1` | 40/40 | 0.000278 | 0.006878 | -11.372 | pair 02 positive, `a sharp wooden block is struck once` |
| `dry-reverberant-v1` | 40/40 | 0.000408 | 0.005708 | -9.909 | pair 03 negative, `a reverberant snare drum hit in a gymnasium` |
| `percussive-sustained-v1` | 40/40 | 0.002414 | 0.084645 | -4.414 | pair 19 negative, `a sustained wind chime tone rings softly` |

Pair 19 negative in `percussive-sustained-v1` is the main quality outlier. It
was excluded from ridge metric targets in the percussive/sustained probe map.

## Brightness Activation Patching

Brightness was used to prove that the hook and patch machinery could move a
measured waveform property.

### Layer Sweep

| Artifact | Rows | Sites | Valid rows | Mean signed effect | Median signed effect |
| --- | ---: | ---: | ---: | ---: | ---: |
| `results/brightness-layer-sweep-final/` | 960 | 24 | 960 | +498.1 Hz | +284.0 Hz |

### Top-Site Movement Test

| Artifact | Rows | Sites | Valid rows | Mean signed effect | Median signed effect | Toward source | Closer to source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `results/brightness-centroid-movement-v1/` | 160 | 4 | 160 | +519.1 Hz | +296.6 Hz | 155/160 | 149/160 |

Interpretation: activation replacement can causally move generated audio toward
the source-side spectral centroid. The effect is broad and sometimes overshoots,
but the directionality is robust enough to support the rest of the protocol.

## Linear Probe Protocol

Probe capture stores `features[N, n_sites, cfg_batch, d_model]` from all 24 DiT
sites:

- Six dual-stream sites: `transformer.transformer_blocks.0` through `.5`
- Eighteen single-stream sites: `transformer.single_transformer_blocks.0`
  through `.17`

Feature extraction pools audio tokens. For single-stream blocks, text and audio
tokens share a stream, so the trailing audio-token slice is used to keep prompt
length differences from dominating the representation.

Probe families:

| Probe | Target | Score | Purpose |
| --- | --- | --- | --- |
| Logistic label probe | positive vs negative side | cross-validated accuracy | Tests semantic decodability. |
| Ridge metric probe | realized audio metric | grouped cross-validated R2 | Tests whether waveform property is linearly predictable. |

## Percussive/Sustained Probe Results

| Field | Value |
| --- | --- |
| Observations | 40 |
| Pairs | 20 |
| Sites | 24 |
| CFG row | 0 |
| Excluded metric pair | 19 |
| Best label decodability | 0.95 at `transformer.transformer_blocks.3` |
| Dual mean label accuracy | 0.9125 |
| Single mean label accuracy | 0.9042 |

| Target metric | Best site | Best grouped R2 |
| --- | --- | ---: |
| `decay_time_to_minus_20db_ms` | `transformer.single_transformer_blocks.17` | +0.333 |
| `high_to_low_db` | `transformer.transformer_blocks.4` | +0.598 |
| `onset_strength_max` | `transformer.single_transformer_blocks.8` | +0.348 |
| `spectral_centroid_mean_hz` | `transformer.single_transformer_blocks.14` | +0.545 |
| `tail_energy_fraction_500ms` | `transformer.single_transformer_blocks.0` | -0.199 |

Interpretation: the label is readable almost everywhere. The realized onset
metric becomes meaningfully predictable in the merged single stream, while
spectral/high-low content is also linearly predictable. This is the clearest
semantic-to-acoustic handoff result so far.

## Percussive/Sustained Patch Results

The larger confirmation run patched 20 pairs at five sites in both directions.
The on-target metric was `onset_strength_max`.

| Site | On-target mean | Off-target mean | Specificity gap |
| --- | ---: | ---: | ---: |
| `transformer.single_transformer_blocks.14` | 0.992 | 0.972 | +0.019 |
| `transformer.single_transformer_blocks.15` | 0.989 | 0.873 | +0.115 |
| `transformer.single_transformer_blocks.8` | 0.958 | 0.954 | +0.005 |
| `transformer.transformer_blocks.3` | 0.862 | 0.786 | +0.076 |
| `transformer.transformer_blocks.4` | 0.884 | 0.867 | +0.017 |

Values are fractions of the source-target metric gap covered by patched audio.
`1.0` means the patched output reached the source value. Negative values mean
the output moved away from the source.

Interpretation: onset is causally controllable by full-trajectory activation
replacement. The tested single-stream sites move many metrics at once; the dual
sites are somewhat more specific but still not perfectly isolated.

## Percussive/Sustained Steering Results

Steering uses the mean activation difference vector `percussive - sustained`.
The first steering run broadcast the vector over tokens/timesteps. The follow-up
restricted single-stream steering to trailing audio tokens and swept scales
0.5, 1, 2, and 4.

| Run | Site | On-target mean | Off-target mean | Specificity gap |
| --- | --- | ---: | ---: | ---: |
| `percussive-sustained-steer-v1` | `transformer.single_transformer_blocks.8` | 0.061 | 0.905 | -0.844 |
| `percussive-sustained-steer-v1` | `transformer.transformer_blocks.4` | 0.085 | 0.849 | -0.764 |
| `percussive-sustained-steer-v2-audio` | `transformer.single_transformer_blocks.8` | 0.180 | 0.768 | -0.588 |
| `percussive-sustained-steer-v2-audio` | `transformer.transformer_blocks.4` | 0.140 | 0.817 | -0.677 |

Interpretation: additive steering does not reproduce the patch result for
onset. It moves off-target spectral and energy properties much more than onset.
The current best explanation is that onset is a trajectory or temporal property:
full activation replacement moves it, but a constant mean-difference vector does
not isolate it.

## Dry/Reverberant Probe Results

The dry/reverberant protocol used `direct_to_late_db` as the primary on-target
metric and included `reverb_proxy_score`, late energy, tail energy, decay time,
high/low energy, and onset as specificity checks.

| Field | Value |
| --- | --- |
| Observations | 40 |
| Pairs | 20 |
| Sites | 24 |
| CFG row | 1 |
| Excluded metric pairs | none |
| Best label decodability | 0.95 at `transformer.single_transformer_blocks.6` |
| Dual mean label accuracy | 0.8542 |
| Single mean label accuracy | 0.8833 |

| Target metric | Best site | Best grouped R2 |
| --- | --- | ---: |
| `decay_time_to_minus_20db_ms` | `transformer.single_transformer_blocks.7` | -0.304 |
| `direct_to_late_db` | `transformer.transformer_blocks.3` | -0.647 |
| `high_to_low_db` | `transformer.single_transformer_blocks.7` | -0.226 |
| `late_energy_fraction_300ms` | `transformer.single_transformer_blocks.8` | -1.548 |
| `late_energy_fraction_700ms` | `transformer.single_transformer_blocks.9` | -1.690 |
| `onset_strength_max` | `transformer.single_transformer_blocks.5` | -0.520 |
| `reverb_proxy_score` | `transformer.single_transformer_blocks.3` | -0.809 |
| `tail_energy_fraction_500ms` | `transformer.single_transformer_blocks.9` | -1.343 |

Interpretation: the semantic label is linearly readable, but realized
dry/reverberant metrics are not stable linear targets in this 20-pair dataset.
This is an important negative result.

## Dry/Reverberant Patch Results

Patch run: 20 pairs x 6 sites x 2 directions = 240 valid patched generations.
The on-target metric was `direct_to_late_db`.

| Site | On mean | On median | Off mean | Off median | Gap | Median gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `transformer.single_transformer_blocks.3` | -1.068 | 0.818 | 1.714 | 0.896 | -2.782 | -0.078 |
| `transformer.single_transformer_blocks.6` | -1.189 | 0.896 | 1.722 | 0.919 | -2.911 | -0.023 |
| `transformer.single_transformer_blocks.7` | -1.893 | 0.916 | 1.453 | 0.945 | -3.346 | -0.029 |
| `transformer.single_transformer_blocks.8` | -1.113 | 0.974 | 1.809 | 0.956 | -2.922 | +0.018 |
| `transformer.single_transformer_blocks.9` | -1.207 | 0.973 | 1.793 | 0.968 | -2.999 | +0.005 |
| `transformer.transformer_blocks.3` | -0.988 | 0.759 | 0.895 | 0.790 | -1.883 | -0.031 |

Interpretation: by fraction/count and median, most patched rows move toward the
source on `direct_to_late_db`. The mean is flipped negative by large outliers,
which confirms the heavy-tail warning around dry/reverb metrics. Off-target
metrics move about as much as the target, so this is broad trajectory movement,
not specific reverberation control.

## Dry/Reverberant Steering Results

Steering run: four sites x five reverberant prompts x scales 0.5, 1, 2, 4, plus
scale 0 baselines. Steering used the `dry - reverberant` mean vector and applied
it only to trailing audio tokens for single-stream sites.

| Site | On mean | On median | Off mean | Off median | Gap | Median gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `transformer.single_transformer_blocks.3` | 0.323 | 0.385 | 0.715 | 0.669 | -0.392 | -0.284 |
| `transformer.single_transformer_blocks.6` | 1.070 | 1.033 | 1.385 | 1.101 | -0.315 | -0.068 |
| `transformer.single_transformer_blocks.9` | 0.587 | 0.639 | 1.177 | 0.843 | -0.590 | -0.203 |
| `transformer.transformer_blocks.3` | 0.410 | 0.569 | 1.007 | 0.792 | -0.598 | -0.223 |

Per-scale detail for the best site:

| Site | Scale | On-target mean | On-target median | Fraction moved toward dry |
| --- | ---: | ---: | ---: | ---: |
| `transformer.single_transformer_blocks.6` | 0.5 | 0.608 | 0.348 | 1.00 |
| `transformer.single_transformer_blocks.6` | 1.0 | 0.810 | 0.799 | 0.80 |
| `transformer.single_transformer_blocks.6` | 2.0 | 1.582 | 1.462 | 1.00 |
| `transformer.single_transformer_blocks.6` | 4.0 | 1.279 | 1.020 | 0.80 |

Interpretation: dry/reverberant steering is more responsive than
percussive/sustained onset steering. It can move reverberant prompts toward dry
on `direct_to_late_db`, especially at `single_transformer_blocks.6`. It is still
not specific, because off-target movement remains high.

## Concept Ranking

| Rank | Concept | Evidence | Current use |
| ---: | --- | --- | --- |
| 1 | Percussive/sustained | Strong metric screen, useful grouped probes, strong patch result. | Main causal-tracing target. |
| 2 | Bright/dark | Strong centroid patching proof of concept. | Best hook/patch validation target. |
| 3 | Sharp/soft | Strong transient-energy screen, not yet probed/patched. | Good next concept after percussive/sustained. |
| 4 | Dry/reverberant | Usable screen and steerable broad direction; poor grouped metric R2. | Secondary/negative-control concept unless prompts are refined. |
| 5 | Clear/muffled | Moderate spectral screen. | Keep as prompt-refinement candidate. |

## Novelty and Paper Positioning

The broad claim "activation steering for audio is novel" is not safe. Adjacent
work already includes activation patching, steering, concept localization, and
network dissection for audio or music generation.

| Adjacent area/work surfaced | Implication |
| --- | --- |
| Activation patching for music/audio generation | Do not claim activation patching itself is new. |
| TADA and audio diffusion control work | Do not frame all audio steering as unexplored. |
| Fine-grained control over music generation with activation steering | Music-generation steering is directly adjacent. |
| ConceptCaps | Concept-level control/localization is adjacent. |
| Audio Network Dissection and related AudioLLM analysis | Mechanistic audio concept analysis is not empty territory. |

The defensible angle is narrower and stronger:

> A metric-grounded semantic-to-acoustic causal tracing protocol for short
> text-to-audio concepts, comparing prompt-label decodability, realized
> audio-metric predictability, activation-patch causality, and additive steering
> specificity.

The percussive/sustained result is the core supporting case:

- The prompt label is decodable broadly.
- Realized onset predictability appears in the single stream.
- Full activation patching moves onset strongly.
- Additive steering fails to isolate onset, revealing a difference between
  trajectory replacement and mean-direction steering.

The dry/reverberant result is useful as a contrast:

- The label is decodable.
- Realized reverb metrics are not linearly predictable under grouped CV.
- Steering can move `direct_to_late_db`, but broadly and not specifically.

## Known Failure Modes and Fixes

| Failure mode | Cause | Current handling |
| --- | --- | --- |
| Probe scores look too good | Row-wise split leaks pair identity and seed/source fingerprints. | Use `GroupKFold` by `pair_id`. |
| Regression dominated by heavy tails | Decay and tail metrics can have outliers. | Use `log1p` for selected strictly positive targets. |
| Dry/reverb means disagree with counts | `direct_to_late_db` and tail metrics remain outlier-sensitive. | Report medians and fractions moved toward source, not only means. |
| Single-stream patching breaks when text lengths differ | Single stream contains text and audio tokens together. | Use suffix-token patching for trailing audio tokens. |
| Pair 19 affects percussive/sustained metrics | Sustained wind-chime output clips heavily. | Exclude pair 19 from percussive/sustained metric probes. |
| Additive steering underperforms patching | Mean vector is not equivalent to full activation trajectory replacement. | Treat steering and patching as different interventions, not interchangeable tests. |

## Current Artifact Map

| Artifact | Meaning |
| --- | --- |
| `results/bright-dark-v1-*` | Baseline brightness generation and centroid summaries. |
| `results/brightness-layer-sweep-final/` | 960-row brightness activation patch sweep. |
| `results/brightness-centroid-movement-v1/` | Top-site patched brightness movement test. |
| `results/clear-muffled-v1/` | Clear/muffled baseline metrics. |
| `results/sharp-soft-v1/` | Sharp/soft baseline metrics. |
| `results/dry-reverberant-v1/` | Dry/reverberant baseline metrics. |
| `results/percussive-sustained-v1/` | Percussive/sustained baseline metrics. |
| `results/percussive-sustained-probe-v1/` | Percussive/sustained grouped probe map. |
| `results/percussive-sustained-patch-v1/` | Initial six-pair patch cross-check. |
| `results/percussive-sustained-patch-v2/` | 20-pair patch confirmation and specificity checks. |
| `results/percussive-sustained-steer-v1/` | Initial additive steering test. |
| `results/percussive-sustained-steer-v2-audio/` | Audio-token-only steering scale sweep. |
| `results/dry-reverberant-probe-v1/` | Dry/reverberant grouped probe map. |
| `results/dry-reverberant-patch-v1/` | Dry/reverberant patch cross-check. |
| `results/dry-reverberant-steer-v1/` | Dry/reverberant audio-token-only steering sweep. |
| `results/dry-reverberant-probe-patch-steer-v1-summary.md` | Short dry/reverb protocol summary. |

## Recommended Next Work

1. Scale `percussive_sustained` beyond 20 pairs before making paper-level claims.
2. Regenerate or replace the clipped pair 19 sustained prompt.
3. Add a sharp/soft probe map, prioritizing transient energy per onset and
   onset p95 rather than onset max alone.
4. Refine dry/reverberant prompts around tighter same-source events and less
   confounded brightness/energy changes.
5. Test timestep-specific or denoising-window-specific steering, because onset
   looks trajectory-dependent.
6. Keep reporting off-target movement for every intervention. The most useful
   negative results so far came from specificity checks, not from on-target
   movement alone.
