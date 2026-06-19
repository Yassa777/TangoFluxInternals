# Second-Model Replication Plan

## Status

Smoke path completed; full replication is not complete. I added a minimal runnable AudioLDM2
adapter, ran a one-prompt generation/hook smoke, and ran a two-record NPZ capture smoke. I did
not run the full second-model corpus capture or `geometry_analyze`, so this is not yet evidence
for or against the cross-model geometry claim.

The fastest viable route is `cvssp/audioldm2`, not Stable Audio Open, for the first replication
run:

- `cvssp/audioldm2` is visible and non-gated on Hugging Face metadata.
- Diffusers `AudioLDM2Pipeline` is available in the existing Modal image.
- The model exposes hookable U-Net latent activations through `pipe.unet.named_modules()`.
- Its tensors are latent spectrogram-like features, not TangoFlux audio-token streams. This is
  still suitable for K=1 geometry replication because `geometry_analyze` only requires
  per-example site vectors plus realized audio metrics. Time-bin claims should remain
  TangoFlux-specific until the AudioLDM2 latent time axis is validated.

Stable Audio Open is the cleaner architectural match because it is a DiT over audio latents, but
Hub metadata reports `gated=auto`. It is therefore a poor "fast parallel worker" target unless
the coordinator confirms account access/license acceptance and allows a larger first-download
smoke.

## Local and Runtime Constraints Checked

- Local Python has `modal==1.4.2`.
- Local `diffusers==0.27.2` is broken with the installed `huggingface_hub==0.33.2`
  (`cached_download` import error), so local Diffusers imports are not valid evidence.
- Modal profile is configured as `yasiru-rasintha2`.
- Existing Modal runtime reports:
  - GPU: NVIDIA L40S
  - `torch==2.4.0+cu121`
  - `torchaudio==2.4.0+cu121`
  - `transformers==4.44.0`
  - `diffusers==0.30.0`

## Code Added

- `src/tangoflux_lab/second_model.py`
  - `load_audioldm2_pipeline()`
  - `second_model_wave()`
  - `SecondModelFeatureRecorder`
  - AudioLDM2 site selection helpers
- Isolated `modal_app.py` section:
  - `second_model_runtime`
  - `second_model_smoke`
  - `second_model_probe_capture`
  - remote helpers behind those local entrypoints

## Hook Geometry Feasibility

AudioLDM2 U-Net sites are hookable, but their geometry is not a drop-in TangoFlux token geometry.
The adapter handles:

- 3D tensors `[batch, tokens, channels]`: mean-pool tokens, or split tokens into K bins.
- 4D tensors `[batch, channels, freq, time]`: split the final axis into K time bins, average over
  frequency and within-bin time, concatenate channel vectors.

For the headline replication, start with `time_bins=1`. That tests whether realized acoustic
factors have a stable low-dimensional latent geometry in a second generator. Do not reuse the
TangoFlux dual/single stream localization claim for AudioLDM2.

## Validation Run So Far

Cheap runtime check:

```bash
modal run modal_app.py::local_env
```

Summary:

- Modal app built and ran.
- CUDA was available on an L40S.
- The remote image had the pinned heavy stack listed above.

Candidate/import check without loading model weights:

```bash
modal run modal_app.py::second_model_runtime
```

Summary:

- `AudioLDM2Pipeline`: ok
- `StableAudioPipeline`: ok
- `cvssp/audioldm2`: private false, gated false
- `stabilityai/stable-audio-open-1.0`: private false, gated auto

Smallest AudioLDM2 generation plus hook smoke:

```bash
modal run modal_app.py::second_model_smoke --duration 1 --steps 2 --site-limit 2
```

Summary:

- `status=ok`
- waveform shape `[1, 16000]`
- sample rate `16000`
- hooked sites:
  - `down_blocks.0.resnets.0`
  - `down_blocks.0.resnets.1`
- latent shapes at both sites: `[2, 128, 25, 16]`
- pooled feature shapes at both sites: `[2, 128]`
- each site was called 2 times, matching the 2 denoising steps.

Two-record NPZ capture smoke:

```bash
modal run modal_app.py::second_model_probe_capture \
  --max-records 2 \
  --duration 1 \
  --steps 2 \
  --site-limit 2 \
  --output-prefix second-model-audioldm2-smoke-v1
```

Summary:

- wrote `outputs/second-model-audioldm2-smoke-v1/probe-features.npz`
- features shape `(2, 2, 2, 128)`
- realized metrics shape `(2, 18)`
- sites `['down_blocks.0.resnets.0', 'down_blocks.0.resnets.1']`
- grouped sources `['dog_bark', 'kick_drum_thump']`

## Next Commands for Coordinator

Run a slightly larger but still cheap capture to verify grouped-CV plumbing before the full
250-prompt corpus:

```bash
modal run modal_app.py::second_model_probe_capture \
  --max-records 12 \
  --duration 1 \
  --steps 4 \
  --site-limit 4 \
  --output-prefix second-model-audioldm2-pilot-v1
```

Then run a low-stakes geometry analyzer check. With 12 records, treat the numbers as plumbing
only, not science:

```bash
modal run modal_app.py::geometry_analyze \
  --features-path outputs/second-model-audioldm2-pilot-v1/probe-features.npz \
  --output-prefix second-model-audioldm2-pilot-geometry-v1 \
  --factors brightness:spectral_centroid_mean_hz,onset_rate:onset_rate_per_second,decay:decay_time_to_minus_20db_ms,loudness:rms_dbfs \
  --cfg-row 0 \
  --n-splits 3 \
  --pca-components 4 \
  --log-factors decay
```

If that works, run the full replication:

```bash
modal run modal_app.py::second_model_probe_capture \
  --prompts-path prompts/diverse_corpus_v1.jsonl \
  --output-prefix second-model-audioldm2-diverse-v1 \
  --site-limit 6 \
  --time-bins 1
```

```bash
modal run modal_app.py::geometry_analyze \
  --features-path outputs/second-model-audioldm2-diverse-v1/probe-features.npz \
  --output-prefix second-model-audioldm2-geometry-v1 \
  --factors brightness:spectral_centroid_mean_hz,onset_rate:onset_rate_per_second,decay:decay_time_to_minus_20db_ms,loudness:rms_dbfs \
  --cfg-row 0 \
  --pca-components 16 \
  --log-factors decay
```

## Exact Boundary to a Completed Replication

A completed replication requires generated AudioLDM2 features for the corpus and a
`geometry_analyze` result. The smoke proves the candidate is runnable and hookable, and the NPZ
path is valid, but the two-record smoke is far too small for grouped CV or geometry-vs-physics
correlation. The valid next decision gate is the 12-record pilot command above.
