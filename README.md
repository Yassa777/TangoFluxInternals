# TangoFlux Internals

This repository contains a Modal-backed research environment for TangoFlux text-to-audio generation,
audio-metric concept screening, activation patching, and steering/probing experiments.

## What is included

- `modal_app.py`: Modal image, GPU functions, Volumes, and local entrypoints.
- `src/tangoflux_lab/records.py`: JSONL prompt-pair parsing and deterministic job expansion.
- `src/tangoflux_lab/hooks.py`: module inspection, activation capture, patch hooks, and steering hooks.
- `src/tangoflux_lab/audio_features.py`: signal metrics for spectral, onset, transient, decay, and tail analysis.
- `src/tangoflux_lab/metric_summaries.py`: grouped and paired metric summaries.
- `prompts/`: contrastive prompt-pair files for the current concept screens.
- `results/`: compact tracked experiment artifacts. Raw WAV files are not committed.
- `docs/`: implementation plans and research-positioning notes.

The Modal app uses three persistent Volumes:

- `tangoflux-hf-cache` mounted at `/cache/huggingface`
- `tangoflux-outputs` mounted at `/outputs`
- `tangoflux-activations` mounted at `/activations`

## Environment

The Modal image is built from Debian slim with Python 3.11 and installs:

- `torch==2.4.0`, `torchaudio==2.4.0`, `torchvision==0.19.0`
- `transformers==4.44.0`, `diffusers==0.30.0`, `accelerate==0.34.2`
- TangoFlux from `git+https://github.com/declare-lab/TangoFlux`
- audio and research helpers such as `soundfile`, `librosa`, `pandas`, `pyarrow`, `safetensors`, and `huggingface_hub[hf_transfer]`

Default GPU is `L40S`. Override it locally before running Modal commands:

```bash
export TANGOFLUX_MODAL_GPU=A100
```

## Quickstart From Fresh Clone

Prerequisites:

- Python 3.11+
- Modal CLI authenticated with access to a GPU-capable Modal workspace
- `ffmpeg` available locally if you plan to inspect or play downloaded audio

Install the local package and the Modal client:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify Modal authentication:

```bash
modal profile current
modal run modal_app.py::env
```

The heavy audio/model dependencies are installed inside the Modal image, not your local Python
environment. Local commands mostly dispatch jobs, write manifests, and download selected files.

## Run

Check the remote environment:

```bash
modal run modal_app.py::local_env
```

Generate one short audio file:

```bash
modal run modal_app.py::smoke --duration 3 --steps 10
```

Run the example contrastive batch:

```bash
modal run modal_app.py::batch --prompts-path prompts/contrastive_pairs.example.jsonl --samples-per-prompt 1
```

Run the bright/dark centroid set:

```bash
modal run modal_app.py::batch --prompts-path prompts/bright_dark_pairs.jsonl --output-prefix bright-dark-v1 --samples-per-prompt 1
modal run modal_app.py::analyze --prompts-path prompts/bright_dark_pairs.jsonl --output-prefix bright-dark-v1
modal run modal_app.py::download_subset --prompts-path prompts/bright_dark_pairs.jsonl --output-prefix bright-dark-v1 --pairs 5
```

This writes all 40 WAVs to the `tangoflux-outputs` Modal Volume and downloads only the first 5 prompt pairs to `outputs/bright-dark-v1-download-subset/`.

Run a metric-grounded concept screen:

```bash
modal run modal_app.py::batch --prompts-path prompts/percussive_sustained_pairs.jsonl --output-prefix percussive-sustained-v1 --samples-per-prompt 1
modal run modal_app.py::metrics --prompts-path prompts/percussive_sustained_pairs.jsonl --output-prefix percussive-sustained-v1 --samples-per-prompt 1
modal run modal_app.py::download_subset --prompts-path prompts/percussive_sustained_pairs.jsonl --output-prefix percussive-sustained-v1 --pairs 5
```

Run the layer-only brightness patching sweep:

```bash
modal run modal_app.py::layer_patch_probe --pair-id 01 --site-limit 1 --output-prefix brightness-layer-probe-v1
modal run modal_app.py::layer_patch_sweep --output-prefix brightness-layer-sweep-v1
```

For the single-stream blocks, use suffix-token patching so only fixed audio tokens are patched when prompt text token counts differ:

```bash
modal run modal_app.py::layer_patch_sweep --output-prefix brightness-layer-sweep-v2-single --stack-filter single
```

The final combined 960-row sweep artifact is tracked in `results/brightness-layer-sweep-final/`.

Inspect TangoFlux module names before choosing hook sites:

```bash
modal run modal_app.py::inspect --pattern "transformer_blocks" --limit 200
```

## Local Entrypoints

| Entrypoint | Purpose | Example |
| --- | --- | --- |
| `env` | Print Modal GPU/runtime environment. | `modal run modal_app.py::env` |
| `smoke` | Generate one short smoke-test WAV. | `modal run modal_app.py::smoke --duration 3 --steps 10` |
| `batch` | Generate a prompt JSONL batch into the Modal output volume. | `modal run modal_app.py::batch --prompts-path prompts/percussive_sustained_pairs.jsonl --output-prefix percussive-sustained-v1` |
| `analyze` | Compute the original bright/dark spectral-centroid summary. | `modal run modal_app.py::analyze --prompts-path prompts/bright_dark_pairs.jsonl --output-prefix bright-dark-v1` |
| `metrics` | Compute shared audio metrics and paired summaries for a concept run. | `modal run modal_app.py::metrics --prompts-path prompts/sharp_soft_pairs.jsonl --output-prefix sharp-soft-v1` |
| `download_subset` | Download the first N prompt pairs as local WAVs. | `modal run modal_app.py::download_subset --prompts-path prompts/sharp_soft_pairs.jsonl --output-prefix sharp-soft-v1 --pairs 5` |
| `inspect` | Inspect TangoFlux module names for hook selection. | `modal run modal_app.py::inspect --pattern transformer_blocks` |
| `layer_patch_probe` | Probe one or a few patch sites on one pair. | `modal run modal_app.py::layer_patch_probe --pair-id 01 --site-limit 1` |
| `layer_patch_sweep` | Run a layer-only activation patching sweep. | `modal run modal_app.py::layer_patch_sweep --output-prefix brightness-layer-sweep-v1` |
| `centroid_movement_test` | Regenerate patched audio for top brightness sites and measure movement. | `modal run modal_app.py::centroid_movement_test --output-prefix brightness-centroid-movement-v1` |
| `probe_capture` | Capture audio-token-pooled DiT features at all 24 sites for a prompt set. | `modal run modal_app.py::probe_capture --prompts-path prompts/percussive_sustained_pairs.jsonl` |
| `probe_train` | Fit pair-grouped logistic + ridge probes and write the decodability/R² map. | `modal run modal_app.py::probe_train` |

These commands can incur Modal GPU costs. Start with `smoke`, small `--max-pairs`, or short
prompt files before launching full sweeps.

## Prompt JSONL Schema

Single-prompt rows are accepted:

```json
{"id":"single_test","prompt":"A glass bell rings in a quiet hall","duration":5,"steps":25,"seed":42}
```

Contrastive rows are expanded into two generation records with the same seed per sample:

```json
{"pair_id":"room_size","a":"A hand clap in a dry booth","b":"A hand clap in a cathedral","duration":5,"steps":25,"seed":1000}
```

You can also use `prompt_a`/`prompt_b`, `positive`/`negative`, or `left`/`right`.

Rows may include `concept`; it is copied into generation manifests and metric outputs:

```json
{"pair_id":"01","positive":"a percussive wooden block is struck once","negative":"a sustained bowed wooden instrument note","concept":"percussive_sustained","duration":3.5,"steps":50,"seed":601}
```

## Activation workflow

The intended workflow is:

1. Run `inspect` and pick stable module-name patterns.
2. Use `capture_activations.remote(...)` from a notebook or small driver script for the positive and negative prompts.
3. Use `make_steering_vector.remote(...)` to compute `positive - negative` vectors.
4. Use `generate_with_steering.remote(...)` to apply a steering vector at selected hook sites.

The hook utilities are deliberately shape-conservative. Start with one or two module sites and short duration/step counts before scaling capture jobs.

## Linear Probe Map

The probe pipeline asks, at each of the 24 DiT block sites, two independent questions:

- **Decodability**: is the prompt label (`percussive=1` / `sustained=0`) linearly readable
  from the site's activations? Scored by cross-validated accuracy (logistic probe).
- **Predictability**: is the *realized* audio metric (onset, decay, tail, ...) linearly
  predictable from the same activations? Scored by cross-validated R² (ridge probe).

Comparing where these two maps peak across the dual→single stream boundary localizes the
semantic-to-acoustic handoff.

```bash
# 1. Capture audio-token-pooled features at all 24 sites for the 20 pairs.
modal run modal_app.py::probe_capture \
  --prompts-path prompts/percussive_sustained_pairs.jsonl \
  --output-prefix percussive-sustained-probe-v1

# 2. Fit pair-grouped probes locally (CPU) and write the map.
modal run modal_app.py::probe_train --output-prefix percussive-sustained-probe-v1
```

Implementation notes:

- Features are pooled over the **audio tokens only** (dual blocks expose the audio stream
  directly; single blocks keep the trailing audio tokens), so prompts of different text
  length stay comparable.
- The CFG batch is preserved; `probe_train --cfg-row auto` selects the conditional row by
  picking whichever batch row is most decodable on average.
- Cross-validation is **grouped by `pair_id`** (`GroupKFold`) so the two sides of a pair never
  straddle the train/test split, which would let a probe memorize a shared seed/source
  fingerprint instead of generalizing the concept.
- The clipped pair (`--exclude-pairs 19`) is dropped from the ridge metric targets.
- Raw pooled features land in `outputs/<prefix>/probe-features.npz` (gitignored); the compact
  map is written to `results/<prefix>/probe-map-rows.csv` and `probe-map-summary.json`.

The local probe trainer needs the `analysis` extra: `pip install -e ".[dev,analysis]"`.

## Current Results

See `results/README.md` for the committed artifact index.

Current strongest signals:

- Brightness: centroid patching moves generated audio toward the counterfactual source in most top-site tests.
- Percussive/sustained: onset, decay, tail, spectral, and high-frequency metrics separate strongly across prompt pairs.

The next planned experiment is a grouped linear-probe map for percussive/sustained, comparing prompt-label decodability against realized audio-metric predictability.

## Artifact Policy

The repository intentionally ignores:

- raw generated WAV files under `outputs/`
- activation tensor dumps under `activations/`
- local paper PDFs and cache directories

Compact CSV/JSON/Markdown summaries are copied into `results/` for version control.

## License note

This repository's lab code is released under the MIT License. The upstream TangoFlux model,
generated outputs, and source datasets are governed by their own licenses and usage terms.

The upstream TangoFlux repository describes the model as research/non-commercial under the Stability AI Community License, with additional dataset-license constraints. Confirm the intended use fits those terms before using generated outputs outside research.

See `CITATION.md` for upstream TangoFlux citation details.
