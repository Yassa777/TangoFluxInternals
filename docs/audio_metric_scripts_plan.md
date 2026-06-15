# Audio Metric Scripts Plan

Status: planning document. The current implemented workflow runs metric extraction through
`modal run modal_app.py::metrics` and shared functions in `src/tangoflux_lab/audio_features.py`
and `src/tangoflux_lab/metric_summaries.py`. The separate `scripts/audio_metrics/*` layout below
is a proposed future refactor, not the current public API.

## Goal

Build a modular metric suite for TangoFlux audio generations so we can test concepts beyond bright/dark with measurable audio-side evidence. The suite should work for:

- baseline generated audio batches such as `outputs/bright-dark-v1-manifest.csv`
- downloaded subsets for listening and local checks
- activation-patched or steered audio from movement tests
- future concept prompt-pair files with positive/negative labels

The immediate implementation target is a set of separate command-line scripts, one per requested metric, backed by shared loading, framing, summary, and paired-comparison utilities.

## Design Principles

1. Keep each metric script separate and inspectable.
   Each script should have one primary responsibility and write one CSV. We can add a `run_all.py` convenience wrapper later, but the individual scripts should remain the source of truth.

2. Share audio loading and feature primitives.
   Repeated logic such as path resolution, mono conversion, STFT framing, onset detection, paired summaries, and JSON/CSV writing should live in `src/tangoflux_lab/audio_features.py` and `src/tangoflux_lab/metric_summaries.py`.

3. Preserve paired experimental structure.
   Every metric row should keep `pair_id`, `side` or `label`, `prompt`, `job_id`, `audio_path`, `sample_rate`, and duration metadata. Summaries should report both group means and within-pair signed differences.

4. Separate signal metrics from semantic agreement metrics.
   Features such as centroid, rolloff, onset count, and loudness are direct signal measurements. CLAP/audio-classifier agreement is a separate semantic validation layer and should not be mixed into low-level audio-feature scripts.

5. Treat reverb, direct-to-reverb ratio, and noisiness as proxies.
   These are useful for relative comparisons, especially within the same prompt pair and generation settings, but they are not ground-truth physical measurements from a single generated waveform.

6. Make defaults suitable for short clips.
   Most clips will be 3-4 seconds. Defaults should use enough temporal resolution for attacks and onsets while staying stable on short files.

## Proposed File Layout

```text
src/tangoflux_lab/
  audio_features.py
  metric_summaries.py

scripts/audio_metrics/
  spectral_centroid.py
  frequency_energy_ratio.py
  spectral_rolloff.py
  onset_strength.py
  attack_time.py
  transient_energy.py
  noisiness.py
  decay_tail.py
  f0_estimate.py
  onset_rate.py
  event_tempo.py
  energy_after_onset.py
  reverb_proxy.py
  loudness.py
  high_frequency_attenuation.py
  direct_to_reverb_ratio_proxy.py
  onset_count.py
  entropy.py
  clap_classifier_agreement.py
  run_all.py
```

`run_all.py` is optional and should only call the individual scripts or shared functions. It should not duplicate metric logic.

## Standard CLI Shape

Each metric script should accept the same core arguments:

```bash
python scripts/audio_metrics/spectral_centroid.py \
  --manifest outputs/bright-dark-v1-manifest.csv \
  --output-dir outputs/bright-dark-v1/metrics \
  --pair-label-a bright \
  --pair-label-b dark
```

Common arguments:

- `--manifest`: CSV containing at least `audio_path` or Modal/local output fields sufficient to resolve the WAV.
- `--output-dir`: directory for metric CSV/JSON outputs.
- `--audio-root`: optional root path used when manifest paths are relative.
- `--pair-label-a` and `--pair-label-b`: optional labels for paired signed summaries.
- `--positive-label`: optional label expected to have a higher metric value.
- `--negative-label`: optional label expected to have a lower metric value.
- `--sample-rate`: optional analysis sample rate. Default should preserve original rate unless a metric/model requires resampling.
- `--limit`: optional development/debug limit.
- `--overwrite`: replace existing outputs.

Common outputs:

```text
outputs/<run>/metrics/<metric>.csv
outputs/<run>/metrics/<metric>-summary.json
outputs/<run>/metrics/<metric>-paired.csv
```

Common row columns:

- `metric`
- `audio_path`
- `pair_id`
- `side`
- `label`
- `prompt`
- `job_id`
- `sample_rate`
- `duration_seconds`
- metric-specific value columns
- `status`
- `error`

Common summary fields:

- `n_rows`
- `n_valid_rows`
- `group_means`
- `group_medians`
- `n_pairs`
- `mean_paired_difference`
- `median_paired_difference`
- `positive_higher_count`
- `positive_higher_fraction`

## Shared Audio Feature Core

`src/tangoflux_lab/audio_features.py` should provide:

- `load_audio(path, target_sample_rate=None)`: returns waveform, sample rate, duration, channels.
- `to_mono(waveform)`: mean across channels.
- `safe_rms(waveform)`: RMS with silence guards.
- `peak_dbfs(waveform)`: peak level for quality diagnostics.
- `frame_signal(...)`: frame-based helper for envelope and energy features.
- `stft_magnitude(...)`: shared STFT helper with consistent windowing.
- `mel_spectrogram(...)`: optional helper for entropy/classifier features.
- `onset_envelope(...)`: shared onset-strength envelope.
- `detect_onsets(...)`: peak-picking wrapper with configurable thresholds.
- `hpss(...)`: harmonic/percussive split if using librosa.
- `finite_or_none(value)`: stable JSON/CSV values.

Default STFT settings:

- General timbre metrics: `n_fft=2048`, `hop_length=512`, Hann window.
- Attack/onset metrics: `n_fft=1024`, `hop_length=128` or `256`.
- Short clips: require no minimum longer than 1 second; report `status=too_short` if a metric cannot be estimated.

## Metric Scripts

### 1. Spectral Centroid

Script: `scripts/audio_metrics/spectral_centroid.py`

Purpose:
Measure the frequency center of mass. Useful for bright/dark, clear/muffled, metallic/dull.

Primary values:

- `spectral_centroid_mean_hz`
- `spectral_centroid_median_hz`
- `spectral_centroid_std_hz`

Formula:

`centroid[t] = sum(freq * magnitude[f,t]) / sum(magnitude[f,t])`

Interpretation:
Higher usually means brighter or more high-frequency content, but can also rise with hiss/noise.

### 2. Frequency Energy Ratio

Script: `scripts/audio_metrics/frequency_energy_ratio.py`

Purpose:
Compare energy in high-frequency bands against low or total energy. Useful for clear/muffled, high-frequency attenuation, and bright/dark.

Primary values:

- `high_band_energy`
- `low_band_energy`
- `total_energy`
- `high_to_low_ratio`
- `high_to_total_ratio`
- `high_to_low_db`

Default bands:

- low: `20-2000 Hz`
- high: `4000-11025 Hz`

Arguments:

- `--low-band 20 2000`
- `--high-band 4000 11025`

Interpretation:
Higher values indicate more high-frequency energy relative to low frequencies.

### 3. Spectral Rolloff

Script: `scripts/audio_metrics/spectral_rolloff.py`

Purpose:
Find the frequency below which a percentage of spectral energy lies. Useful for brightness, muffling, and bandwidth.

Primary values:

- `rolloff_85_mean_hz`
- `rolloff_85_median_hz`
- `rolloff_95_mean_hz`
- `rolloff_95_median_hz`

Defaults:

- rolloff thresholds: `0.85`, `0.95`

Interpretation:
Lower rolloff suggests darker or more bandwidth-limited audio.

### 4. Onset Strength

Script: `scripts/audio_metrics/onset_strength.py`

Purpose:
Measure transient/onset salience. Useful for sharp/soft, percussive/sustained, crisp/muffled.

Primary values:

- `onset_strength_mean`
- `onset_strength_median`
- `onset_strength_max`
- `onset_strength_p95`

Implementation:
Use librosa onset strength or a shared spectral-flux implementation.

Interpretation:
Higher max or p95 values imply sharper, more salient attacks.

### 5. Attack Time

Script: `scripts/audio_metrics/attack_time.py`

Purpose:
Estimate how quickly a sound reaches peak level after an onset. Useful for sharp/soft and percussive/sustained.

Primary values:

- `attack_time_ms_first`
- `attack_time_ms_median`
- `attack_time_ms_min`
- `attack_time_ms_valid_count`

Method:

1. Detect onsets from onset envelope.
2. Around each onset, compute amplitude envelope.
3. Attack time is time from 10% to 90% of local peak envelope.

Interpretation:
Shorter attack time means sharper onset.

Limitations:
Requires a clear onset. Sustained ambience or noisy clips may return `status=no_clear_onset`.

### 6. Transient Energy

Script: `scripts/audio_metrics/transient_energy.py`

Purpose:
Estimate energy concentrated around detected transients. Useful for sharp/soft and percussive/sustained.

Primary values:

- `transient_energy_total`
- `transient_energy_fraction`
- `transient_energy_per_onset_mean`
- `non_transient_energy_total`

Method:
Detect onsets, take windows around each onset, and compute energy fraction inside transient windows.

Default windows:

- pre-onset: `20 ms`
- post-onset: `120 ms`

Interpretation:
Higher fraction means more of the clip energy is transient/percussive.

### 7. Noisiness

Script: `scripts/audio_metrics/noisiness.py`

Purpose:
Estimate noise-like versus tonal structure. Useful for noisy/clean, breathy/tonal, mechanical/natural in some cases.

Primary values:

- `spectral_flatness_mean`
- `spectral_flatness_median`
- `harmonic_energy_fraction`
- `percussive_energy_fraction`
- `noise_proxy_score`

Method:
Use spectral flatness as the main proxy. Optionally include HPSS fractions.

Interpretation:
Higher flatness means more noise-like or hiss-like audio. This should not be treated as semantic "bad quality" by itself.

### 8. Decay Tail

Script: `scripts/audio_metrics/decay_tail.py`

Purpose:
Measure how much energy remains after a peak/onset. Useful for sustained/percussive, reverberant/dry, metallic/wooden.

Primary values:

- `tail_energy_fraction_250ms`
- `tail_energy_fraction_500ms`
- `tail_energy_fraction_1000ms`
- `decay_time_to_minus_20db_ms`
- `decay_slope_db_per_second`

Method:
Find strongest onset or peak, compute post-onset envelope decay, and measure residual energy after fixed windows.

Interpretation:
Higher tail fractions and slower decay slopes indicate more sustain or reverberation.

### 9. F0 Estimate

Script: `scripts/audio_metrics/f0_estimate.py`

Purpose:
Estimate fundamental frequency for tonal sounds. Useful for high/low pitch concepts.

Primary values:

- `f0_mean_hz`
- `f0_median_hz`
- `f0_min_hz`
- `f0_max_hz`
- `voiced_fraction`
- `f0_confidence_proxy`

Method:
Use `librosa.pyin` or `librosa.yin` on mono waveform.

Defaults:

- `fmin=50 Hz`
- `fmax=2000 Hz`

Interpretation:
Use only when prompts are tonal: bells, whistles, sustained notes, sirens, voices. For noisy impacts, report low voiced fraction and avoid strong conclusions.

### 10. Onset Rate

Script: `scripts/audio_metrics/onset_rate.py`

Purpose:
Measure event density per second. Useful for fast/slow repetition and sparse/dense.

Primary values:

- `onset_count`
- `onset_rate_per_second`
- `mean_inter_onset_interval_ms`
- `median_inter_onset_interval_ms`

Method:
Peak-pick the onset envelope and normalize by clip duration.

Interpretation:
Higher onset rate means faster repetition or denser events.

### 11. Event Tempo

Script: `scripts/audio_metrics/event_tempo.py`

Purpose:
Estimate periodic event tempo when repeated events exist. Useful for fast/slow rhythm-like prompts.

Primary values:

- `tempo_bpm`
- `tempo_confidence_proxy`
- `beat_count`

Method:
Use onset envelope autocorrelation or librosa beat tracking.

Interpretation:
Only reliable for repeated events. Single impacts should return `status=insufficient_repetition`.

### 12. Energy After Onset

Script: `scripts/audio_metrics/energy_after_onset.py`

Purpose:
Measure energy retained after the first or strongest onset. Useful for sustain, reverberation, and muffling.

Primary values:

- `post_onset_energy_100ms`
- `post_onset_energy_250ms`
- `post_onset_energy_500ms`
- `post_onset_energy_1000ms`
- `post_onset_to_pre_onset_ratio`

Method:
Detect onset, compute energy in fixed windows after it, optionally compare with a short pre-onset window.

Interpretation:
Higher post-onset energy suggests sustain, ringing, echo, or ongoing event texture.

### 13. Reverb Estimators / Proxies

Script: `scripts/audio_metrics/reverb_proxy.py`

Purpose:
Approximate dry/reverberant differences from a single waveform.

Primary values:

- `late_energy_fraction_300ms`
- `late_energy_fraction_700ms`
- `estimated_decay_slope_db_per_second`
- `reverb_proxy_score`

Method:
Use onset-aligned energy decay. Late energy after the direct transient is treated as a reverb/sustain proxy.

Interpretation:
Higher score suggests more reverberant or sustained audio. Metallic ringing can also raise this score, so use prompt controls carefully.

### 14. Loudness

Script: `scripts/audio_metrics/loudness.py`

Purpose:
Measure perceived/physical level. Useful for close/distant and quality checks.

Primary values:

- `rms_dbfs`
- `peak_dbfs`
- `lufs_integrated` if `pyloudnorm` is available
- `crest_factor_db`
- `clip_fraction`

Method:
Use RMS and peak by default. Add LUFS if dependency is installed.

Interpretation:
Higher loudness can indicate closer or more intense audio, but model output normalization can confound this.

### 15. High Frequency Attenuation

Script: `scripts/audio_metrics/high_frequency_attenuation.py`

Purpose:
Measure loss of high-frequency content. Useful for close/distant and clear/muffled.

Primary values:

- `hf_attenuation_db`
- `hf_energy_fraction`
- `low_mid_to_high_ratio_db`

Method:
Compute high-band energy relative to low/mid-band energy. This overlaps with frequency energy ratio but reports attenuation-oriented column names and default bands.

Default bands:

- low/mid: `300-3000 Hz`
- high: `5000-12000 Hz`

Interpretation:
Higher attenuation means less high-frequency energy.

### 16. Direct-To-Reverb Ratio Proxy

Script: `scripts/audio_metrics/direct_to_reverb_ratio_proxy.py`

Purpose:
Approximate close/distant and dry/reverberant differences.

Primary values:

- `direct_energy`
- `late_energy`
- `direct_to_late_ratio`
- `direct_to_late_ratio_db`

Method:

1. Detect strongest onset.
2. Direct window: onset to `50 ms`.
3. Late window: `80-600 ms` after onset.
4. Compute direct/late energy ratio.

Interpretation:
Higher ratio means more direct/dry/close. Lower ratio means more late energy, often more distant or reverberant.

Limitations:
This is not a true DRR estimate without a known impulse response.

### 17. Onset Count

Script: `scripts/audio_metrics/onset_count.py`

Purpose:
Count discrete detected events. Useful for sparse/dense, one/many, fast/slow.

Primary values:

- `onset_count`
- `strong_onset_count`
- `weak_onset_count`
- `onset_density_per_second`

Method:
Peak-pick onset envelope with configurable thresholds.

Interpretation:
Higher count suggests more events, but noisy continuous sounds may create false positives.

### 18. Entropy

Script: `scripts/audio_metrics/entropy.py`

Purpose:
Measure distributional spread/complexity in time-frequency energy. Useful for sparse/dense, simple/complex, noisy/clean.

Primary values:

- `spectral_entropy_mean`
- `spectral_entropy_median`
- `temporal_energy_entropy`
- `mel_band_entropy_mean`

Method:
Normalize energy distributions and compute Shannon entropy.

Interpretation:
Higher entropy means energy is spread more broadly across frequency or time. This can indicate complexity, noise, or dense ambience.

### 19. CLAP / Audio Classifier Agreement

Script: `scripts/audio_metrics/clap_classifier_agreement.py`

Purpose:
Evaluate semantic agreement between generated audio and prompts or concept labels.

Primary values:

- `clap_prompt_similarity`
- `clap_positive_prompt_similarity`
- `clap_negative_prompt_similarity`
- `clap_margin_positive_minus_negative`
- `classifier_top1_label`
- `classifier_top1_score`
- `classifier_expected_label_score`

Method:

1. Load audio at the model-required sample rate.
2. Compute CLAP audio embedding.
3. Compare against original prompt, positive concept prompt, and negative concept prompt.
4. Optionally run an audio classifier such as PANNs, AudioSet, or another available model.

Interpretation:
Use this for semantic concepts that do not have a clean signal metric, such as natural/mechanical, indoor/outdoor, crowd/single source, or object identity. For acoustic adjectives, use it as secondary evidence rather than the main metric.

Implementation caution:
CLAP can reward prompt-text overlap and broad scene similarity. It should not be the only proof of a causal steering effect.

## Concept-To-Metric Map

| Concept | Primary metrics | Secondary metrics |
| --- | --- | --- |
| bright/dark | spectral centroid, rolloff, frequency energy ratio | CLAP margin |
| clear/muffled | high-frequency attenuation, frequency energy ratio, rolloff | centroid |
| sharp/soft | onset strength, attack time, transient energy | direct-to-late ratio |
| percussive/sustained | transient energy, decay tail, energy after onset | onset count |
| metallic/wooden/cloth | centroid, rolloff, decay tail | noisiness, entropy |
| high/low pitch | F0 estimate | centroid |
| fast/slow repetition | onset rate, event tempo, onset count | temporal entropy |
| dry/reverberant | decay tail, energy after onset, reverb proxy | direct-to-late ratio |
| close/distant | loudness, high-frequency attenuation, direct-to-late ratio | reverb proxy |
| sparse/dense | onset count, onset rate, entropy | CLAP/classifier agreement |
| natural/mechanical | CLAP/classifier agreement | entropy, noisiness |

## Paired Analysis Method

For a concept prompt set, each pair should identify the expected positive and negative side. Example:

```json
{"pair_id":"01","positive":"a crisp cymbal tap","negative":"a muted cymbal tap","concept":"sharp_soft","duration":3,"seed":201}
```

For each metric:

1. Compute row-level metric values for every audio file.
2. Join rows by `pair_id`.
3. Compute `positive_value - negative_value`.
4. Report mean paired difference, median paired difference, and count of pairs in expected direction.
5. For activation-patched audio, compute movement:
   - baseline target value
   - source value
   - patched value
   - signed movement toward source
   - closer-to-source boolean
   - overshot-source boolean

This mirrors the centroid movement test but generalizes it to every metric.

## Quality Guard Metrics

Several scripts should also expose quality guard columns, either directly or through `loudness.py`:

- `clip_fraction`
- `peak_dbfs`
- `rms_dbfs`
- `duration_seconds`
- `silence_fraction`
- `nan_or_inf_detected`

These guards are important because activation patching can increase centroid or onset measures by adding harsh artifacts rather than moving the intended concept cleanly.

## Modal Integration

Initial scripts can run locally on downloaded audio and local manifests. After that, add a Modal entrypoint:

```bash
modal run modal_app.py::metrics \
  --manifest outputs/bright-dark-v1-manifest.csv \
  --output-prefix bright-dark-v1 \
  --metrics spectral_centroid,rolloff,onset_strength
```

Preferred design:

- Local scripts are the canonical metric implementations.
- Modal entrypoint calls the same Python functions inside the Modal image.
- Results are written to `/outputs/<output-prefix>/metrics/` and downloaded locally when requested.

## Implementation Phases

### Phase 1: Core Infrastructure

- Add `src/tangoflux_lab/audio_features.py`.
- Add `src/tangoflux_lab/metric_summaries.py`.
- Add manifest path resolution that supports current generated manifests and downloaded subsets.
- Add a small local test over `outputs/bright-dark-v1-download-subset`.

### Phase 2: Direct Spectral Metrics

Implement:

- `spectral_centroid.py`
- `frequency_energy_ratio.py`
- `spectral_rolloff.py`
- `high_frequency_attenuation.py`

These are the fastest and most directly useful for bright/dark and clear/muffled.

### Phase 3: Onset, Transient, and Temporal Metrics

Implement:

- `onset_strength.py`
- `attack_time.py`
- `transient_energy.py`
- `onset_rate.py`
- `onset_count.py`
- `event_tempo.py`

These support sharp/soft, percussive/sustained, sparse/dense, and fast/slow concepts.

### Phase 4: Decay, Reverb, and Distance Proxies

Implement:

- `decay_tail.py`
- `energy_after_onset.py`
- `reverb_proxy.py`
- `direct_to_reverb_ratio_proxy.py`
- `loudness.py`

These support dry/reverberant and close/distant concepts, with explicit caveats.

### Phase 5: Pitch, Noise, and Complexity

Implement:

- `f0_estimate.py`
- `noisiness.py`
- `entropy.py`

These support high/low pitch, noisy/clean, and simple/complex or sparse/dense concepts.

### Phase 6: Semantic Agreement

Implement:

- `clap_classifier_agreement.py`

This may require adding dependencies or using already installed CLAP/PANNs-compatible packages. It should be kept optional because model downloads and GPU/CPU requirements are different from the lightweight signal metrics.

### Phase 7: Reporting

Add a combined report generator:

- aggregate all metric summaries for a run
- rank concepts by effect size and expected-direction count
- flag metrics likely contaminated by clipping or silence
- produce a compact Markdown report per run

## Validation Checklist

Before trusting the metric suite:

- Run all direct metrics on the existing 10-file baseline subset.
- Confirm spectral centroid reproduces the previous bright/dark values within tolerance.
- Check that clipping-heavy files are flagged by loudness quality guards.
- Hand-inspect at least five metric CSV rows against the source WAV paths.
- For onset scripts, verify onset count on a few obvious one-hit and repeated-event clips.
- For F0, verify `voiced_fraction` is low on noisy impacts and higher on tonal bells/whistles.
- For CLAP/classifier, verify prompt similarity is higher for original prompts than unrelated prompts on a small sanity set.

## Definition Of Done

The metric suite is ready when:

- every requested metric has a separate script
- all scripts share the same manifest and output conventions
- each script produces row-level CSV and summary JSON
- paired positive/negative summaries work for concept prompt files
- movement summaries work for patched/steered audio
- quality guards are available to detect artifact-driven effects
- README includes example commands for local and Modal usage
