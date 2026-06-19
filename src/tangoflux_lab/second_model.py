from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch

from tangoflux_lab.hooks import _get_tensor


SELECTED_MODEL_ID = "cvssp/audioldm2"
SELECTED_MODEL_FAMILY = "audioldm2"
NEGATIVE_PROMPT = "Low quality."


@dataclass(frozen=True)
class SecondModelSite:
    site: str
    stack: str
    block: int
    output_index: int = 0


def load_audioldm2_pipeline(model_id: str = SELECTED_MODEL_ID) -> Any:
    from diffusers import AudioLDM2Pipeline

    pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.unet.eval()
    pipe.vae.eval()
    return pipe


def second_model_wave(pipe: Any, record: dict[str, Any]) -> tuple[torch.Tensor, int]:
    generator = torch.Generator(device="cuda").manual_seed(int(record["seed"]))
    with torch.inference_mode():
        output = pipe(
            prompt=str(record["prompt"]),
            negative_prompt=NEGATIVE_PROMPT,
            audio_length_in_s=float(record["duration"]),
            num_inference_steps=int(record["steps"]),
            guidance_scale=float(record["guidance_scale"]),
            num_waveforms_per_prompt=1,
            generator=generator,
            output_type="pt",
        )
    audio = output.audios[0]
    waveform = normalize_waveform(audio)
    sample_rate = int(getattr(pipe.vocoder.config, "sampling_rate", 16000))
    waveform_end = int(float(record["duration"]) * sample_rate)
    return waveform[:, :waveform_end].contiguous().cpu(), sample_rate


def normalize_waveform(audio: Any) -> torch.Tensor:
    tensor = torch.as_tensor(audio).detach().float()
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == 2:
        # Diffusers audio pipelines commonly return [samples, channels] for stereo
        # or [batch, samples] for mono. Prefer the longer dimension as time.
        if tensor.shape[0] > tensor.shape[1]:
            return tensor.transpose(0, 1).contiguous()
        return tensor.contiguous()
    if tensor.ndim == 3:
        return normalize_waveform(tensor[0])
    raise ValueError(f"Unsupported audio tensor shape: {tuple(tensor.shape)}")


def default_audioldm2_sites(pipe: Any, *, limit: int = 12) -> list[SecondModelSite]:
    rows: list[SecondModelSite] = []
    for name, module in pipe.unet.named_modules():
        if not name:
            continue
        module_type = module.__class__.__name__
        if module_type not in {"ResnetBlock2D", "Transformer2DModel"}:
            continue
        block = _block_index(name)
        stack = _stack_name(name)
        rows.append(SecondModelSite(site=name, stack=stack, block=block, output_index=0))
        if len(rows) >= limit:
            break
    return rows


def describe_audioldm2_sites(pipe: Any, *, limit: int = 24) -> list[dict[str, Any]]:
    rows = []
    for site in default_audioldm2_sites(pipe, limit=limit):
        module = dict(pipe.unet.named_modules())[site.site]
        rows.append(
            {
                "site": site.site,
                "stack": site.stack,
                "block": site.block,
                "output_index": site.output_index,
                "type": module.__class__.__name__,
            }
        )
    return rows


def _block_index(name: str) -> int:
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return -1


def _stack_name(name: str) -> str:
    if name.startswith("down_blocks"):
        return "down"
    if name.startswith("mid_block"):
        return "mid"
    if name.startswith("up_blocks"):
        return "up"
    return "unet"


class SecondModelFeatureRecorder(AbstractContextManager["SecondModelFeatureRecorder"]):
    """Capture pooled features from AudioLDM2 U-Net sites.

    AudioLDM2 exposes diffusion activations as latent spectrogram-like tensors.
    For 4D tensors ``[batch, channels, freq, time]``, this recorder splits the
    final axis into K temporal windows, averages over frequency and window time,
    and concatenates channel vectors. For 3D tensors it uses the TangoFlux token
    convention ``[batch, tokens, channels]``.
    """

    def __init__(self, model: torch.nn.Module, sites: list[SecondModelSite], *, time_bins: int = 1):
        self.model = model
        self.sites = sites
        self.time_bins = max(1, int(time_bins))
        self.handles: list[Any] = []
        self.calls: dict[str, int] = {}
        self.sums: dict[str, torch.Tensor] = {}
        self.shapes: dict[str, tuple[int, ...]] = {}

    def __enter__(self) -> "SecondModelFeatureRecorder":
        modules = dict(self.model.named_modules())
        missing = [site.site for site in self.sites if site.site not in modules]
        if missing:
            raise ValueError(f"Missing AudioLDM2 U-Net sites: {missing}")
        for site in self.sites:
            module = modules[site.site]
            self.handles.append(module.register_forward_hook(self._hook(site)))
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, site: SecondModelSite):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            try:
                tensor = _get_tensor(output, site.output_index).detach().float()
            except (IndexError, TypeError):
                return
            if tensor.ndim not in {3, 4}:
                return
            pooled = pool_latent_features(tensor, time_bins=self.time_bins).to(device="cpu")
            if site.site in self.sums:
                self.sums[site.site] = self.sums[site.site] + pooled
            else:
                self.sums[site.site] = pooled
            self.calls[site.site] = self.calls.get(site.site, 0) + 1
            self.shapes[site.site] = tuple(int(dim) for dim in tensor.shape)

        return hook

    def features(self) -> dict[str, torch.Tensor]:
        return {
            name: self.sums[name] / max(self.calls.get(name, 1), 1) for name in self.sums
        }


def pool_latent_features(tensor: torch.Tensor, *, time_bins: int = 1) -> torch.Tensor:
    if tensor.ndim == 3:
        if time_bins > 1 and tensor.shape[1] >= time_bins:
            return torch.cat([chunk.mean(dim=1) for chunk in torch.chunk(tensor, time_bins, dim=1)], dim=1)
        return tensor.mean(dim=1)
    if tensor.ndim == 4:
        if time_bins > 1 and tensor.shape[-1] >= time_bins:
            chunks = torch.chunk(tensor, time_bins, dim=-1)
            return torch.cat([chunk.mean(dim=(-2, -1)) for chunk in chunks], dim=1)
        return tensor.mean(dim=(-2, -1))
    raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(tensor.shape)}")
