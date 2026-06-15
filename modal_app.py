from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import modal

LOCAL_SRC = Path(__file__).parent / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from tangoflux_lab.generation import (  # noqa: E402
    SAMPLE_RATE,
    json_output_path,
    result_manifest_row,
    run_prefix,
    wav_output_path,
    write_json,
    write_manifest_csv,
)
from tangoflux_lab.records import expand_prompt_rows, load_jsonl, normalize_generation_record  # noqa: E402
from tangoflux_lab.analysis import (  # noqa: E402
    movement_row,
    summarize_layer_patch_rows,
    summarize_movement_rows,
)


APP_NAME = "tangoflux-lab"
MODEL_NAME = os.environ.get("TANGOFLUX_MODEL", "declare-lab/TangoFlux")
GPU_TYPE = os.environ.get("TANGOFLUX_MODAL_GPU", "L40S")

HF_CACHE_DIR = "/cache/huggingface"
OUTPUT_DIR = "/outputs"
ACTIVATION_DIR = "/activations"

hf_cache = modal.Volume.from_name("tangoflux-hf-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("tangoflux-outputs", create_if_missing=True)
activation_volume = modal.Volume.from_name("tangoflux-activations", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.4.0",
        "torchaudio==2.4.0",
        "torchvision==0.19.0",
        "torchlibrosa==0.1.0",
        "transformers==4.44.0",
        "diffusers==0.30.0",
        "accelerate==0.34.2",
        "datasets==2.21.0",
        "huggingface_hub[hf_transfer]",
        "safetensors",
        "librosa",
        "soundfile",
        "numpy",
        "pandas",
        "pyarrow",
        "pyyaml",
        "tqdm",
        "click",
        "gradio",
        "wandb",
        "einops",
    )
    .pip_install("git+https://github.com/declare-lab/TangoFlux")
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .env(
        {
            "PYTHONPATH": "/root/src",
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": f"{HF_CACHE_DIR}/hub",
            "HUGGINGFACE_HUB_CACHE": f"{HF_CACHE_DIR}/hub",
            "TRANSFORMERS_CACHE": f"{HF_CACHE_DIR}/transformers",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

app = modal.App(APP_NAME)

COMMON_VOLUMES = {
    HF_CACHE_DIR: hf_cache,
    OUTPUT_DIR: output_volume,
    ACTIVATION_DIR: activation_volume,
}

_RUNNER: Any | None = None
_RUNNER_MODEL_NAME: str | None = None


def dit_layer_patch_sites() -> list[dict[str, Any]]:
    sites: list[dict[str, Any]] = []
    for block in range(6):
        sites.append(
            {
                "site": f"transformer.transformer_blocks.{block}",
                "stack": "dual",
                "block": block,
                "output_index": 1,
            }
        )
    for block in range(18):
        sites.append(
            {
                "site": f"transformer.single_transformer_blocks.{block}",
                "stack": "single",
                "block": block,
                "output_index": 0,
            }
        )
    return sites


def _site_slug(site: str) -> str:
    return site.replace("transformer.", "").replace(".", "-")


def _load_runner(model_name: str = MODEL_NAME):
    global _RUNNER, _RUNNER_MODEL_NAME
    if _RUNNER is not None and _RUNNER_MODEL_NAME == model_name:
        return _RUNNER

    import torch
    from tangoflux import TangoFluxInference

    torch.set_float32_matmul_precision("high")
    runner = TangoFluxInference(name=model_name)
    runner.model.eval()
    runner.vae.eval()
    _RUNNER = runner
    _RUNNER_MODEL_NAME = model_name
    return runner


def _generate_wave(runner: Any, record: dict[str, Any]):
    import torch

    prompt = record["prompt"]
    duration = record["duration"]
    steps = record["steps"]
    guidance_scale = record["guidance_scale"]
    seed = record["seed"]

    with torch.inference_mode():
        latents = runner.model.inference_flow(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            duration=duration,
            seed=seed,
            disable_progress=True,
        )
        wave = runner.vae.decode(latents.transpose(2, 1)).sample.cpu()[0]
        sample_rate = int(getattr(runner.vae.config, "sampling_rate", SAMPLE_RATE))
        waveform_end = int(duration * sample_rate)
        return wave[:, :waveform_end].contiguous(), sample_rate


def _save_wav(path: str, audio: Any, sample_rate: int) -> None:
    import torchaudio

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(path, audio, sample_rate=sample_rate)


def _records_from_prompt_file(
    prompts_path: str,
    *,
    samples_per_prompt: int = 1,
    default_duration: float = 3.0,
    default_steps: int = 25,
    default_guidance_scale: float = 4.5,
    default_seed: int = 0,
) -> list[dict[str, Any]]:
    rows = load_jsonl(prompts_path)
    return expand_prompt_rows(
        rows,
        samples_per_prompt=samples_per_prompt,
        default_duration=default_duration,
        default_steps=default_steps,
        default_guidance_scale=default_guidance_scale,
        default_seed=default_seed,
    )


def _baseline_centroids_from_csv(path: str) -> dict[str, float]:
    import csv

    baseline: dict[str, float] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            pair_id = row["pair_id"]
            label = row.get("label") or row.get("side")
            value = row.get("spectral_centroid_mean_hz")
            if not pair_id or not label or not value:
                continue
            baseline[f"{pair_id}|{label}"] = float(value)
    return baseline


def _records_by_pair(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    pairs: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        pairs.setdefault(str(record["pair_id"]), []).append(record)
    return [pairs[pair_id] for pair_id in sorted(pairs)]


def _top_sites_from_summary(
    path: str,
    *,
    top_sites: int,
    include_best_dual: bool = True,
) -> list[dict[str, Any]]:
    import csv

    all_sites = {site["site"]: site for site in dit_layer_patch_sites()}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))

    selected_names: list[str] = []
    for row in summary_rows:
        name = row["site"]
        if name in all_sites and name not in selected_names:
            selected_names.append(name)
        if len(selected_names) >= top_sites:
            break

    if include_best_dual and not any(all_sites[name]["stack"] == "dual" for name in selected_names):
        for row in summary_rows:
            name = row["site"]
            if name in all_sites and all_sites[name]["stack"] == "dual":
                selected_names.append(name)
                break

    return [all_sites[name] for name in selected_names]


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes=COMMON_VOLUMES,
    timeout=20 * 60,
    scaledown_window=300,
)
def env_report() -> dict[str, Any]:
    import diffusers
    import torch
    import torchaudio
    import transformers

    return {
        "app": APP_NAME,
        "gpu_type": GPU_TYPE,
        "model_name": MODEL_NAME,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": transformers.__version__,
        "diffusers": diffusers.__version__,
        "hf_home": os.environ.get("HF_HOME"),
        "output_dir": OUTPUT_DIR,
        "activation_dir": ACTIVATION_DIR,
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes=COMMON_VOLUMES,
    timeout=60 * 60,
    scaledown_window=600,
    max_containers=20,
)
def generate_one(record: dict[str, Any], output_prefix: str = "manual") -> dict[str, Any]:
    record = normalize_generation_record(record)
    started = time.time()
    runner = _load_runner(MODEL_NAME)
    audio, sample_rate = _generate_wave(runner, record)
    wav_path = wav_output_path(OUTPUT_DIR, output_prefix, record)
    _save_wav(wav_path, audio, sample_rate)
    output_volume.commit()
    return {
        "status": "ok",
        "job_id": record["job_id"],
        "wav_path": wav_path,
        "sample_rate": sample_rate,
        "elapsed_seconds": round(time.time() - started, 3),
        "model_name": MODEL_NAME,
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes=COMMON_VOLUMES,
    timeout=60 * 60,
    scaledown_window=600,
)
def inspect_model_modules(
    patterns: list[str] | None = None,
    regex: bool = True,
    limit: int = 300,
) -> list[dict[str, str]]:
    from tangoflux_lab.hooks import find_module_names

    runner = _load_runner(MODEL_NAME)
    return find_module_names(runner.model, patterns=patterns, regex=regex, limit=limit)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes=COMMON_VOLUMES,
    timeout=60 * 60,
    scaledown_window=600,
)
def capture_activations(
    record: dict[str, Any],
    module_patterns: list[str],
    output_prefix: str,
    *,
    regex: bool = True,
    output_index: int = 0,
    max_calls: int | None = None,
    capture: str = "full",
) -> dict[str, Any]:
    import torch
    from tangoflux_lab.hooks import ActivationRecorder, compile_spec

    record = normalize_generation_record(record)
    runner = _load_runner(MODEL_NAME)
    spec = compile_spec(
        module_patterns,
        regex=regex,
        output_index=output_index,
        max_calls=max_calls,
        capture=capture,
    )
    with ActivationRecorder(runner.model, spec) as recorder:
        audio, sample_rate = _generate_wave(runner, record)

    wav_path = wav_output_path(OUTPUT_DIR, output_prefix, record)
    _save_wav(wav_path, audio, sample_rate)

    activation_path = json_output_path(ACTIVATION_DIR, output_prefix, f"{record['job_id']}.pt")
    Path(activation_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        recorder.payload(
            {
                "record": record,
                "model_name": MODEL_NAME,
                "wav_path": wav_path,
                "sample_rate": sample_rate,
            }
        ),
        activation_path,
    )
    output_volume.commit()
    activation_volume.commit()
    return {
        "status": "ok",
        "job_id": record["job_id"],
        "wav_path": wav_path,
        "activation_path": activation_path,
        "matched_modules": sorted(recorder.module_types),
        "calls": recorder.calls,
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes=COMMON_VOLUMES,
    timeout=60 * 60,
    scaledown_window=600,
)
def make_steering_vector(
    positive_activation_path: str,
    negative_activation_path: str,
    output_prefix: str,
    *,
    reduce_dims: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    import torch
    from tangoflux_lab.hooks import mean_activation_difference

    activation_volume.reload()
    positive_payload = torch.load(positive_activation_path, map_location="cpu")
    negative_payload = torch.load(negative_activation_path, map_location="cpu")
    vectors = mean_activation_difference(
        positive_payload,
        negative_payload,
        reduce_dims=reduce_dims,
    )
    vector_path = json_output_path(ACTIVATION_DIR, output_prefix, "steering_vectors.pt")
    Path(vector_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "positive_activation_path": positive_activation_path,
            "negative_activation_path": negative_activation_path,
            "reduce_dims": reduce_dims,
            "vectors": vectors,
        },
        vector_path,
    )
    activation_volume.commit()
    return {
        "status": "ok",
        "vector_path": vector_path,
        "modules": sorted(vectors),
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes=COMMON_VOLUMES,
    timeout=60 * 60,
    scaledown_window=600,
)
def generate_with_steering(
    record: dict[str, Any],
    vector_path: str,
    module_patterns: list[str],
    output_prefix: str,
    *,
    scale: float,
    regex: bool = True,
    output_index: int = 0,
    max_calls: int | None = None,
) -> dict[str, Any]:
    import torch
    from tangoflux_lab.hooks import SteeringApplier, compile_spec

    activation_volume.reload()
    payload = torch.load(vector_path, map_location="cpu")
    vectors = payload["vectors"]
    record = normalize_generation_record(record)
    runner = _load_runner(MODEL_NAME)
    spec = compile_spec(
        module_patterns,
        regex=regex,
        output_index=output_index,
        max_calls=max_calls,
        capture="full",
    )
    with SteeringApplier(runner.model, spec, vectors, scale=scale):
        audio, sample_rate = _generate_wave(runner, record)
    wav_path = wav_output_path(OUTPUT_DIR, output_prefix, record)
    _save_wav(wav_path, audio, sample_rate)
    output_volume.commit()
    return {
        "status": "ok",
        "job_id": record["job_id"],
        "wav_path": wav_path,
        "vector_path": vector_path,
        "scale": scale,
    }


@app.function(
    image=image,
    volumes={OUTPUT_DIR: output_volume},
    timeout=30 * 60,
    scaledown_window=300,
)
def spectral_centroids(records: list[dict[str, Any]], output_prefix: str) -> dict[str, Any]:
    import torch
    import torchaudio
    from tangoflux_lab.analysis import finite_or_none, summarize_bright_dark_centroids

    output_volume.reload()
    rows: list[dict[str, Any]] = []
    for raw_record in records:
        record = normalize_generation_record(raw_record)
        wav_path = wav_output_path(OUTPUT_DIR, output_prefix, record)
        base = {
            "job_id": record["job_id"],
            "pair_id": record.get("pair_id", ""),
            "label": record.get("side", "single"),
            "side": record.get("side", "single"),
            "sample_index": record.get("sample_index", 0),
            "seed": record["seed"],
            "duration": record["duration"],
            "steps": record["steps"],
            "guidance_scale": record["guidance_scale"],
            "prompt": record["prompt"],
            "wav_path": wav_path,
        }
        if not Path(wav_path).exists():
            rows.append(
                {
                    **base,
                    "status": "missing",
                    "error": f"Missing WAV at {wav_path}",
                    "sample_rate": None,
                    "spectral_centroid_mean_hz": None,
                    "spectral_centroid_median_hz": None,
                }
            )
            continue

        waveform, sample_rate = torchaudio.load(wav_path)
        mono = waveform.mean(dim=0).float()
        if mono.numel() == 0:
            rows.append(
                {
                    **base,
                    "status": "empty",
                    "error": "Empty waveform",
                    "sample_rate": int(sample_rate),
                    "spectral_centroid_mean_hz": None,
                    "spectral_centroid_median_hz": None,
                }
            )
            continue

        n_fft = min(2048, max(256, int(2 ** max(8, (mono.numel() // 8).bit_length() - 1))))
        hop_length = max(128, n_fft // 4)
        window = torch.hann_window(n_fft)
        spectrum = torch.stft(
            mono,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        ).abs()
        freqs = torch.linspace(0, sample_rate / 2, spectrum.shape[0])
        denominator = spectrum.sum(dim=0).clamp_min(1e-12)
        centroid = (freqs[:, None] * spectrum).sum(dim=0) / denominator
        rows.append(
            {
                **base,
                "status": "ok",
                "error": "",
                "sample_rate": int(sample_rate),
                "spectral_centroid_mean_hz": finite_or_none(float(centroid.mean().item())),
                "spectral_centroid_median_hz": finite_or_none(float(centroid.median().item())),
            }
        )

    summary = summarize_bright_dark_centroids(rows)
    analysis_dir = Path(OUTPUT_DIR) / output_prefix
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_manifest_csv(str(analysis_dir / "spectral-centroids.csv"), rows)
    write_json(str(analysis_dir / "spectral-summary.json"), summary)
    output_volume.commit()
    return {
        "status": "ok",
        "output_prefix": output_prefix,
        "centroids_path": str(analysis_dir / "spectral-centroids.csv"),
        "summary_path": str(analysis_dir / "spectral-summary.json"),
        "rows": rows,
        "summary": summary,
    }


@app.function(
    image=image,
    volumes={OUTPUT_DIR: output_volume},
    timeout=30 * 60,
    scaledown_window=300,
)
def concept_audio_metrics(
    records: list[dict[str, Any]],
    output_prefix: str,
    positive_label: str = "positive",
    negative_label: str = "negative",
) -> dict[str, Any]:
    import torchaudio
    from tangoflux_lab.audio_features import all_core_metrics
    from tangoflux_lab.metric_summaries import summarize_metric_rows

    output_volume.reload()
    rows: list[dict[str, Any]] = []
    for raw_record in records:
        record = normalize_generation_record(raw_record)
        metadata = dict(record.get("metadata", {}))
        wav_path = wav_output_path(OUTPUT_DIR, output_prefix, record)
        base = {
            "job_id": record["job_id"],
            "pair_id": record.get("pair_id", ""),
            "concept": metadata.get("concept", ""),
            "label": record.get("side", "single"),
            "side": record.get("side", "single"),
            "sample_index": record.get("sample_index", 0),
            "seed": record["seed"],
            "duration": record["duration"],
            "steps": record["steps"],
            "guidance_scale": record["guidance_scale"],
            "prompt": record["prompt"],
            "wav_path": wav_path,
        }
        if not Path(wav_path).exists():
            rows.append(
                {
                    **base,
                    "status": "missing",
                    "error": f"Missing WAV at {wav_path}",
                }
            )
            continue

        try:
            waveform, sample_rate = torchaudio.load(wav_path)
            rows.append(
                {
                    **base,
                    "status": "ok",
                    "error": "",
                    **all_core_metrics(waveform, int(sample_rate)),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    **base,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    summary = summarize_metric_rows(
        rows,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    analysis_dir = Path(OUTPUT_DIR) / output_prefix
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_manifest_csv(str(analysis_dir / "audio-metrics.csv"), rows)
    write_manifest_csv(str(analysis_dir / "audio-metrics-paired.csv"), summary["paired_rows"])
    write_json(str(analysis_dir / "audio-metrics-summary.json"), summary)
    output_volume.commit()
    return {
        "status": "ok",
        "output_prefix": output_prefix,
        "metrics_path": str(analysis_dir / "audio-metrics.csv"),
        "paired_path": str(analysis_dir / "audio-metrics-paired.csv"),
        "summary_path": str(analysis_dir / "audio-metrics-summary.json"),
        "rows": rows,
        "summary": summary,
    }


@app.function(
    image=image,
    volumes={OUTPUT_DIR: output_volume},
    timeout=10 * 60,
    scaledown_window=120,
)
def read_audio_files(wav_paths: list[str]) -> list[dict[str, Any]]:
    output_volume.reload()
    files: list[dict[str, Any]] = []
    for wav_path in wav_paths:
        path = Path(wav_path)
        if not path.exists():
            files.append({"status": "missing", "wav_path": wav_path, "data": b""})
            continue
        files.append(
            {
                "status": "ok",
                "wav_path": wav_path,
                "filename": path.name,
                "data": path.read_bytes(),
            }
        )
    return files


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes=COMMON_VOLUMES,
    timeout=30 * 60,
    scaledown_window=300,
)
def probe_layer_patch_site_shapes(
    record: dict[str, Any],
    sites: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    import torch

    runner = _load_runner(MODEL_NAME)
    normalized = normalize_generation_record(record)
    site_by_name = {site["site"]: site for site in sites}
    rows: dict[str, dict[str, Any]] = {}
    handles = []

    def shape_payload(value: Any) -> Any:
        if torch.is_tensor(value):
            return {
                "type": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        if isinstance(value, tuple):
            return {"type": "tuple", "items": [shape_payload(item) for item in value]}
        if isinstance(value, list):
            return {"type": "list", "items": [shape_payload(item) for item in value]}
        return {"type": type(value).__name__}

    def selected_shape(output: Any, output_index: int) -> Any:
        if torch.is_tensor(output):
            if output_index != 0:
                return {"error": "tensor output only supports output_index=0"}
            return shape_payload(output)
        if isinstance(output, (tuple, list)):
            if output_index >= len(output):
                return {"error": f"output_index={output_index} out of range"}
            return shape_payload(output[output_index])
        return {"error": f"unsupported output type {type(output).__name__}"}

    def make_hook(name: str):
        def hook(module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if name in rows:
                return
            spec = site_by_name[name]
            rows[name] = {
                "site": name,
                "stack": spec["stack"],
                "block": spec["block"],
                "output_index": spec["output_index"],
                "module_type": module.__class__.__name__,
                "output": shape_payload(output),
                "selected": selected_shape(output, int(spec["output_index"])),
            }

        return hook

    for name, module in runner.model.named_modules():
        if name in site_by_name:
            handles.append(module.register_forward_hook(make_hook(name)))
    try:
        _generate_wave(runner, normalized)
    finally:
        for handle in handles:
            handle.remove()

    return [rows[site["site"]] for site in sites if site["site"] in rows]


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes=COMMON_VOLUMES,
    timeout=2 * 60 * 60,
    scaledown_window=600,
    max_containers=20,
)
def layer_patch_pair_sweep(
    pair_records: list[dict[str, Any]],
    sites: list[dict[str, Any]],
    baseline_centroids: dict[str, float],
    output_prefix: str,
    *,
    alpha: float = 1.0,
    save_audio: bool = False,
) -> list[dict[str, Any]]:
    from tangoflux_lab.analysis import spectral_centroid_from_waveform
    from tangoflux_lab.hooks import ActivationPatcher, HookSpec, MultiSiteActivationRecorder

    runner = _load_runner(MODEL_NAME)
    records = [normalize_generation_record(record) for record in pair_records]
    by_side = {str(record["side"]): record for record in records}
    if set(by_side) != {"bright", "dark"}:
        raise ValueError(f"Expected bright/dark records, got sides: {sorted(by_side)}")

    site_output_indices = {str(site["site"]): int(site["output_index"]) for site in sites}
    needs_audio_suffix = any(str(site.get("stack")) == "single" for site in sites)
    audio_token_site = "transformer.transformer_blocks.0"
    if needs_audio_suffix and audio_token_site not in site_output_indices:
        site_output_indices[audio_token_site] = 1
    site_meta = {str(site["site"]): site for site in sites}
    activations_by_side: dict[str, dict[str, list[Any]]] = {}
    capture_calls_by_side: dict[str, dict[str, int]] = {}
    audio_tokens_by_side: dict[str, int] = {}

    for side in ("bright", "dark"):
        record = by_side[side]
        with MultiSiteActivationRecorder(
            runner.model,
            site_output_indices,
            capture="full",
        ) as recorder:
            _generate_wave(runner, record)
        activations_by_side[side] = recorder.activations
        capture_calls_by_side[side] = recorder.calls
        if needs_audio_suffix:
            audio_values = recorder.activations.get(audio_token_site, [])
            if not audio_values:
                raise ValueError(f"Could not infer audio-token count from {audio_token_site}")
            audio_tokens_by_side[side] = int(audio_values[0].shape[1])

    directions = [
        ("bright_to_dark", "bright", "dark"),
        ("dark_to_bright", "dark", "bright"),
    ]
    rows: list[dict[str, Any]] = []
    for site in sites:
        site_name = str(site["site"])
        output_index = int(site["output_index"])
        for direction, source_side, target_side in directions:
            source_record = by_side[source_side]
            target_record = by_side[target_side]
            source_values = activations_by_side[source_side].get(site_name, [])
            suffix_tokens = (
                min(audio_tokens_by_side[source_side], audio_tokens_by_side[target_side])
                if site_meta[site_name]["stack"] == "single"
                else None
            )
            target_key = f"{target_record['pair_id']}|{target_side}"
            source_key = f"{source_record['pair_id']}|{source_side}"
            target_baseline = baseline_centroids.get(target_key)
            source_baseline = baseline_centroids.get(source_key)
            started = time.time()
            try:
                spec = HookSpec(
                    patterns=(site_name,),
                    regex=False,
                    output_index=output_index,
                    max_calls=None,
                    capture="full",
                )
                with ActivationPatcher(
                    runner.model,
                    spec,
                    {site_name: source_values},
                    alpha=alpha,
                    suffix_tokens=suffix_tokens,
                ) as patcher:
                    audio, sample_rate = _generate_wave(runner, target_record)
                centroid = spectral_centroid_from_waveform(audio, sample_rate)
                patched_mean = centroid["mean_hz"]
                patched_median = centroid["median_hz"]
                if save_audio:
                    patch_record = dict(target_record)
                    patch_record["job_id"] = (
                        f"{target_record['pair_id']}-{direction}-{_site_slug(site_name)}"
                    )
                    patch_record["side"] = direction
                    wav_path = wav_output_path(OUTPUT_DIR, output_prefix, patch_record)
                    _save_wav(wav_path, audio, sample_rate)
                else:
                    wav_path = ""

                target_delta = (
                    float(patched_mean) - float(target_baseline)
                    if patched_mean is not None and target_baseline is not None
                    else None
                )
                if direction == "bright_to_dark":
                    signed_effect = target_delta
                elif target_delta is not None:
                    signed_effect = -target_delta
                else:
                    signed_effect = None

                denominator = (
                    float(source_baseline) - float(target_baseline)
                    if source_baseline is not None and target_baseline is not None
                    else None
                )
                source_shift_fraction = (
                    float(target_delta) / denominator
                    if target_delta is not None
                    and denominator is not None
                    and abs(denominator) > 1e-9
                    else None
                )

                rows.append(
                    {
                        "status": "ok",
                        "error": "",
                        "pair_id": target_record["pair_id"],
                        "site": site_name,
                        "site_slug": _site_slug(site_name),
                        "stack": site_meta[site_name]["stack"],
                        "block": site_meta[site_name]["block"],
                        "output_index": output_index,
                        "direction": direction,
                        "source_label": source_side,
                        "target_label": target_side,
                        "source_prompt": source_record["prompt"],
                        "target_prompt": target_record["prompt"],
                        "seed": target_record["seed"],
                        "duration": target_record["duration"],
                        "steps": target_record["steps"],
                        "guidance_scale": target_record["guidance_scale"],
                        "alpha": alpha,
                        "sample_rate": sample_rate,
                        "source_activation_calls": len(source_values),
                        "patch_calls": patcher.calls.get(site_name, 0),
                        "suffix_tokens": suffix_tokens,
                        "source_baseline_centroid_mean_hz": source_baseline,
                        "target_baseline_centroid_mean_hz": target_baseline,
                        "patched_centroid_mean_hz": patched_mean,
                        "patched_centroid_median_hz": patched_median,
                        "target_delta_centroid_hz": target_delta,
                        "signed_brightness_effect_hz": signed_effect,
                        "source_shift_fraction": source_shift_fraction,
                        "elapsed_seconds": round(time.time() - started, 3),
                        "wav_path": wav_path,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "status": "error",
                        "error": repr(exc),
                        "pair_id": target_record["pair_id"],
                        "site": site_name,
                        "site_slug": _site_slug(site_name),
                        "stack": site_meta[site_name]["stack"],
                        "block": site_meta[site_name]["block"],
                        "output_index": output_index,
                        "direction": direction,
                        "source_label": source_side,
                        "target_label": target_side,
                        "source_prompt": source_record["prompt"],
                        "target_prompt": target_record["prompt"],
                        "seed": target_record["seed"],
                        "duration": target_record["duration"],
                        "steps": target_record["steps"],
                        "guidance_scale": target_record["guidance_scale"],
                        "alpha": alpha,
                        "source_activation_calls": len(source_values),
                        "suffix_tokens": suffix_tokens,
                        "elapsed_seconds": round(time.time() - started, 3),
                    }
                )

    if save_audio:
        output_volume.commit()
    return rows


@app.local_entrypoint(name="env")
def local_env() -> None:
    print(json.dumps(env_report.remote(), indent=2, sort_keys=True))


@app.local_entrypoint()
def smoke(
    prompt: str = "A wooden mallet softly taps a table in a quiet room",
    duration: float = 3.0,
    steps: int = 10,
    guidance_scale: float = 4.5,
    seed: int = 0,
    output_prefix: str = "",
) -> None:
    prefix = output_prefix or run_prefix("smoke")
    record = {
        "job_id": "smoke",
        "prompt": prompt,
        "duration": duration,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
    }
    result = generate_one.remote(record, prefix)
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def batch(
    prompts_path: str = "prompts/contrastive_pairs.example.jsonl",
    output_prefix: str = "",
    samples_per_prompt: int = 1,
    default_duration: float = 10.0,
    default_steps: int = 25,
    default_guidance_scale: float = 4.5,
    default_seed: int = 0,
) -> None:
    prefix = output_prefix or run_prefix(Path(prompts_path).stem)
    records = _records_from_prompt_file(
        prompts_path,
        samples_per_prompt=samples_per_prompt,
        default_duration=default_duration,
        default_steps=default_steps,
        default_guidance_scale=default_guidance_scale,
        default_seed=default_seed,
    )
    print(f"Dispatching {len(records)} generations to Modal with prefix {prefix!r}")

    results: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for record, result in zip(
        records,
        generate_one.map(records, kwargs={"output_prefix": prefix}, order_outputs=True),
        strict=True,
    ):
        results.append(result)
        manifest_rows.append(result_manifest_row(record, result))
        print(json.dumps(result, sort_keys=True))

    local_manifest = Path("outputs") / f"{prefix}-manifest.csv"
    local_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_manifest_csv(str(local_manifest), manifest_rows)
    local_summary = Path("outputs") / f"{prefix}-summary.json"
    write_json(
        str(local_summary),
        {
            "output_prefix": prefix,
            "records": len(records),
            "results": results,
            "local_manifest": str(local_manifest),
        },
    )
    print(f"Wrote local manifest: {local_manifest}")
    print(f"Wrote local summary: {local_summary}")


@app.local_entrypoint()
def inspect(pattern: str = "transformer", limit: int = 200, regex: bool = True) -> None:
    rows = inspect_model_modules.remote([pattern], regex=regex, limit=limit)
    print(json.dumps(rows, indent=2, sort_keys=True))


@app.local_entrypoint()
def analyze(
    prompts_path: str = "prompts/bright_dark_pairs.jsonl",
    output_prefix: str = "bright-dark-v1",
    samples_per_prompt: int = 1,
) -> None:
    records = _records_from_prompt_file(prompts_path, samples_per_prompt=samples_per_prompt)
    result = spectral_centroids.remote(records, output_prefix)
    local_centroids = Path("outputs") / f"{output_prefix}-spectral-centroids.csv"
    local_summary = Path("outputs") / f"{output_prefix}-spectral-summary.json"
    local_centroids.parent.mkdir(parents=True, exist_ok=True)
    write_manifest_csv(str(local_centroids), result["rows"])
    write_json(str(local_summary), result["summary"])
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"Wrote local centroids: {local_centroids}")
    print(f"Wrote local summary: {local_summary}")


@app.local_entrypoint()
def metrics(
    prompts_path: str,
    output_prefix: str,
    samples_per_prompt: int = 1,
    positive_label: str = "positive",
    negative_label: str = "negative",
) -> None:
    records = _records_from_prompt_file(prompts_path, samples_per_prompt=samples_per_prompt)
    result = concept_audio_metrics.remote(
        records,
        output_prefix,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    local_metrics = Path("outputs") / f"{output_prefix}-audio-metrics.csv"
    local_paired = Path("outputs") / f"{output_prefix}-audio-metrics-paired.csv"
    local_summary = Path("outputs") / f"{output_prefix}-audio-metrics-summary.json"
    local_metrics.parent.mkdir(parents=True, exist_ok=True)
    write_manifest_csv(str(local_metrics), result["rows"])
    write_manifest_csv(str(local_paired), result["summary"]["paired_rows"])
    write_json(str(local_summary), result["summary"])
    print(json.dumps(result["summary"]["paired_summaries"], indent=2, sort_keys=True))
    print(f"Wrote local metrics: {local_metrics}")
    print(f"Wrote local paired metrics: {local_paired}")
    print(f"Wrote local summary: {local_summary}")


@app.local_entrypoint()
def download_subset(
    prompts_path: str = "prompts/bright_dark_pairs.jsonl",
    output_prefix: str = "bright-dark-v1",
    pairs: int = 5,
    samples_per_prompt: int = 1,
) -> None:
    records = _records_from_prompt_file(prompts_path, samples_per_prompt=samples_per_prompt)
    pair_ids = sorted({str(record["pair_id"]) for record in records})[:pairs]
    selected = [record for record in records if str(record["pair_id"]) in pair_ids]
    wav_paths = [wav_output_path(OUTPUT_DIR, output_prefix, record) for record in selected]
    files = read_audio_files.remote(wav_paths)

    download_dir = Path("outputs") / f"{output_prefix}-download-subset"
    manifest_rows: list[dict[str, Any]] = []
    for record, file_payload in zip(selected, files, strict=True):
        local_path = (
            download_dir
            / str(record["pair_id"])
            / str(record.get("side", "single"))
            / Path(file_payload.get("filename") or f"{record['job_id']}.wav").name
        )
        if file_payload["status"] == "ok":
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(file_payload["data"])
        manifest_rows.append(
            {
                "status": file_payload["status"],
                "pair_id": record["pair_id"],
                "label": record.get("side", "single"),
                "prompt": record["prompt"],
                "remote_wav_path": wav_output_path(OUTPUT_DIR, output_prefix, record),
                "local_wav_path": str(local_path) if file_payload["status"] == "ok" else "",
            }
        )

    manifest_path = download_dir / "manifest.csv"
    write_manifest_csv(str(manifest_path), manifest_rows)
    print(f"Downloaded {sum(1 for row in manifest_rows if row['status'] == 'ok')} files")
    print(f"Wrote subset manifest: {manifest_path}")


@app.local_entrypoint()
def layer_patch_probe(
    prompts_path: str = "prompts/bright_dark_pairs.jsonl",
    baseline_path: str = "outputs/bright-dark-v1-spectral-centroids.csv",
    output_prefix: str = "brightness-layer-probe-v1",
    pair_id: str = "01",
    site_limit: int = 1,
    alpha: float = 1.0,
    save_audio: bool = True,
) -> None:
    records = _records_from_prompt_file(prompts_path)
    pair_groups = _records_by_pair(records)
    selected_pair = None
    for group in pair_groups:
        if str(group[0]["pair_id"]) == pair_id:
            selected_pair = group
            break
    if selected_pair is None:
        raise ValueError(f"Pair {pair_id!r} was not found in {prompts_path}")

    sites = dit_layer_patch_sites()[:site_limit]
    shapes = probe_layer_patch_site_shapes.remote(selected_pair[0], sites)
    print("Patch-site shape probe:")
    print(json.dumps(shapes, indent=2, sort_keys=True))

    baseline = _baseline_centroids_from_csv(baseline_path)
    rows = layer_patch_pair_sweep.remote(
        selected_pair,
        sites,
        baseline,
        output_prefix,
        alpha=alpha,
        save_audio=save_audio,
    )
    summary = summarize_layer_patch_rows(rows)
    out_dir = Path("outputs") / output_prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "layer-patch-probe-rows.csv"
    summary_path = out_dir / "layer-patch-probe-summary.json"
    write_manifest_csv(str(rows_path), rows)
    write_json(str(summary_path), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote probe rows: {rows_path}")
    print(f"Wrote probe summary: {summary_path}")


@app.local_entrypoint()
def layer_patch_sweep(
    prompts_path: str = "prompts/bright_dark_pairs.jsonl",
    baseline_path: str = "outputs/bright-dark-v1-spectral-centroids.csv",
    output_prefix: str = "brightness-layer-sweep-v1",
    alpha: float = 1.0,
    save_audio: bool = False,
    max_pairs: int = 0,
    stack_filter: str = "all",
) -> None:
    records = _records_from_prompt_file(prompts_path)
    pair_groups = _records_by_pair(records)
    if max_pairs > 0:
        pair_groups = pair_groups[:max_pairs]
    sites = dit_layer_patch_sites()
    if stack_filter != "all":
        allowed = {part.strip() for part in stack_filter.split(",") if part.strip()}
        unknown = allowed - {"dual", "single"}
        if unknown:
            raise ValueError(f"Unknown stack_filter values: {sorted(unknown)}")
        sites = [site for site in sites if site["stack"] in allowed]
    baseline = _baseline_centroids_from_csv(baseline_path)

    expected_rows = len(pair_groups) * len(sites) * 2
    print(
        f"Dispatching layer-only patch sweep: {len(pair_groups)} pairs, "
        f"{len(sites)} sites, {expected_rows} patched generations."
    )
    print(f"stack_filter={stack_filter!r}; save_audio={save_audio}; output_prefix={output_prefix!r}")

    all_rows: list[dict[str, Any]] = []
    for pair_rows in layer_patch_pair_sweep.map(
        pair_groups,
        kwargs={
            "sites": sites,
            "baseline_centroids": baseline,
            "output_prefix": output_prefix,
            "alpha": alpha,
            "save_audio": save_audio,
        },
        order_outputs=True,
    ):
        all_rows.extend(pair_rows)
        completed_pairs = len({row["pair_id"] for row in all_rows if "pair_id" in row})
        ok_rows = sum(1 for row in all_rows if row.get("status") == "ok")
        print(
            f"Completed {completed_pairs}/{len(pair_groups)} pairs; "
            f"{ok_rows}/{len(all_rows)} rows ok."
        )

    summary = summarize_layer_patch_rows(all_rows)
    out_dir = Path("outputs") / output_prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "layer-patch-rows.csv"
    summary_path = out_dir / "layer-patch-summary.json"
    site_summary_path = out_dir / "layer-patch-site-summary.csv"
    site_direction_summary_path = out_dir / "layer-patch-site-direction-summary.csv"
    write_manifest_csv(str(rows_path), all_rows)
    write_json(str(summary_path), summary)
    write_manifest_csv(str(site_summary_path), summary["site_rows"])
    write_manifest_csv(str(site_direction_summary_path), summary["site_direction_rows"])
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote sweep rows: {rows_path}")
    print(f"Wrote sweep summary: {summary_path}")
    print(f"Wrote site summary: {site_summary_path}")
    print(f"Wrote site-direction summary: {site_direction_summary_path}")


@app.local_entrypoint()
def centroid_movement_test(
    prompts_path: str = "prompts/bright_dark_pairs.jsonl",
    baseline_path: str = "outputs/bright-dark-v1-spectral-centroids.csv",
    site_summary_path: str = "outputs/brightness-layer-sweep-final/layer-patch-site-summary.csv",
    baseline_output_prefix: str = "bright-dark-v1",
    output_prefix: str = "brightness-centroid-movement-v1",
    top_sites: int = 3,
    download_pairs: int = 5,
    alpha: float = 1.0,
    include_best_dual: bool = True,
) -> None:
    records = _records_from_prompt_file(prompts_path)
    pair_groups = _records_by_pair(records)
    sites = _top_sites_from_summary(
        site_summary_path,
        top_sites=top_sites,
        include_best_dual=include_best_dual,
    )
    baseline = _baseline_centroids_from_csv(baseline_path)

    print("Selected patch sites:")
    for site in sites:
        print(f"- {site['site']} ({site['stack']}, block {site['block']})")
    print(
        f"Regenerating patched audio for {len(pair_groups)} pairs x "
        f"{len(sites)} sites x 2 directions."
    )

    raw_rows: list[dict[str, Any]] = []
    for pair_rows in layer_patch_pair_sweep.map(
        pair_groups,
        kwargs={
            "sites": sites,
            "baseline_centroids": baseline,
            "output_prefix": output_prefix,
            "alpha": alpha,
            "save_audio": True,
        },
        order_outputs=True,
    ):
        raw_rows.extend(pair_rows)
        completed_pairs = len({row["pair_id"] for row in raw_rows if "pair_id" in row})
        print(f"Completed {completed_pairs}/{len(pair_groups)} pairs.")

    rows = [movement_row(row) for row in raw_rows]
    summary = summarize_movement_rows(rows)

    out_dir = Path("outputs") / output_prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "centroid-movement-rows.csv"
    summary_path = out_dir / "centroid-movement-summary.json"
    site_summary_out = out_dir / "centroid-movement-site-summary.csv"
    site_direction_out = out_dir / "centroid-movement-site-direction-summary.csv"
    write_manifest_csv(str(rows_path), rows)
    write_json(str(summary_path), summary)
    write_manifest_csv(str(site_summary_out), summary["site_rows"])
    write_manifest_csv(str(site_direction_out), summary["site_direction_rows"])

    selected_pair_ids = {str(group[0]["pair_id"]) for group in pair_groups[:download_pairs]}
    patched_for_download = [
        row for row in rows if row.get("status") == "ok" and str(row["pair_id"]) in selected_pair_ids
    ]
    baseline_records = [
        record for record in records if str(record["pair_id"]) in selected_pair_ids
    ]
    download_specs: list[dict[str, Any]] = []
    for record in baseline_records:
        download_specs.append(
            {
                "kind": "baseline",
                "pair_id": record["pair_id"],
                "label": record["side"],
                "site": "",
                "direction": "",
                "prompt": record["prompt"],
                "remote_wav_path": wav_output_path(OUTPUT_DIR, baseline_output_prefix, record),
            }
        )
    for row in patched_for_download:
        download_specs.append(
            {
                "kind": "patched",
                "pair_id": row["pair_id"],
                "label": row["target_label"],
                "site": row["site"],
                "direction": row["direction"],
                "prompt": row["target_prompt"],
                "remote_wav_path": row["wav_path"],
            }
        )

    files = read_audio_files.remote([spec["remote_wav_path"] for spec in download_specs])
    download_dir = out_dir / "download-subset"
    download_manifest: list[dict[str, Any]] = []
    for spec, file_payload in zip(download_specs, files, strict=True):
        if spec["kind"] == "baseline":
            local_path = (
                download_dir
                / "baseline"
                / str(spec["pair_id"])
                / str(spec["label"])
                / Path(file_payload.get("filename") or "audio.wav").name
            )
        else:
            local_path = (
                download_dir
                / "patched"
                / str(spec["pair_id"])
                / str(spec["site"]).replace("transformer.", "").replace(".", "-")
                / str(spec["direction"])
                / Path(file_payload.get("filename") or "audio.wav").name
            )
        if file_payload["status"] == "ok":
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(file_payload["data"])
        download_manifest.append(
            {
                **spec,
                "status": file_payload["status"],
                "local_wav_path": str(local_path) if file_payload["status"] == "ok" else "",
            }
        )
    download_manifest_path = download_dir / "manifest.csv"
    write_manifest_csv(str(download_manifest_path), download_manifest)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote movement rows: {rows_path}")
    print(f"Wrote movement summary: {summary_path}")
    print(f"Wrote site summary: {site_summary_out}")
    print(f"Wrote site-direction summary: {site_direction_out}")
    print(
        f"Downloaded {sum(1 for row in download_manifest if row['status'] == 'ok')} "
        f"subset audio files to {download_dir}"
    )
