from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import re
from typing import Any

import torch


@dataclass(frozen=True)
class HookSpec:
    """Selects module outputs for capture, patching, or steering."""

    patterns: tuple[str, ...]
    regex: bool = True
    output_index: int = 0
    max_calls: int | None = None
    capture: str = "full"


def compile_spec(patterns: list[str] | tuple[str, ...], **kwargs: Any) -> HookSpec:
    if not patterns:
        raise ValueError("At least one module pattern is required.")
    return HookSpec(patterns=tuple(patterns), **kwargs)


def _matches(name: str, spec: HookSpec) -> bool:
    for pattern in spec.patterns:
        if spec.regex:
            if re.search(pattern, name):
                return True
        elif pattern == name:
            return True
    return False


def find_module_names(
    model: torch.nn.Module,
    patterns: list[str] | tuple[str, ...] | None = None,
    *,
    regex: bool = True,
    limit: int | None = None,
) -> list[dict[str, str]]:
    if patterns:
        spec = HookSpec(tuple(patterns), regex=regex)
    else:
        spec = None

    rows: list[dict[str, str]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if spec is not None and not _matches(name, spec):
            continue
        rows.append({"name": name, "type": module.__class__.__name__})
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _get_tensor(output: Any, index: int) -> torch.Tensor:
    if torch.is_tensor(output):
        if index != 0:
            raise IndexError("Tensor outputs only support output_index=0.")
        return output
    if isinstance(output, tuple):
        value = output[index]
        if not torch.is_tensor(value):
            raise TypeError(f"Tuple output at index {index} is not a tensor.")
        return value
    if isinstance(output, list):
        value = output[index]
        if not torch.is_tensor(value):
            raise TypeError(f"List output at index {index} is not a tensor.")
        return value
    raise TypeError(f"Unsupported hook output type: {type(output)!r}")


def _replace_tensor(output: Any, index: int, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        if index != 0:
            raise IndexError("Tensor outputs only support output_index=0.")
        return replacement
    if isinstance(output, tuple):
        values = list(output)
        values[index] = replacement
        return tuple(values)
    if isinstance(output, list):
        values = list(output)
        values[index] = replacement
        return values
    raise TypeError(f"Unsupported hook output type: {type(output)!r}")


def _capture_tensor(tensor: torch.Tensor, mode: str) -> torch.Tensor:
    detached = tensor.detach()
    if mode == "full":
        captured = detached
    elif mode == "mean":
        captured = detached.mean()
    elif mode == "mean_batch":
        captured = detached.mean(dim=0)
    elif mode == "mean_tokens":
        if detached.ndim < 3:
            captured = detached.mean(dim=0)
        else:
            captured = detached.mean(dim=1)
    else:
        raise ValueError(f"Unknown capture mode: {mode}")
    return captured.to(device="cpu", dtype=torch.float16)


class ActivationRecorder(AbstractContextManager["ActivationRecorder"]):
    def __init__(self, model: torch.nn.Module, spec: HookSpec):
        self.model = model
        self.spec = spec
        self.handles: list[Any] = []
        self.calls: dict[str, int] = {}
        self.activations: dict[str, list[torch.Tensor]] = {}
        self.module_types: dict[str, str] = {}

    def __enter__(self) -> "ActivationRecorder":
        for name, module in self.model.named_modules():
            if not name or not _matches(name, self.spec):
                continue
            self.module_types[name] = module.__class__.__name__
            self.handles.append(module.register_forward_hook(self._hook(name)))
        if not self.handles:
            raise ValueError(f"No modules matched patterns: {self.spec.patterns}")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            call_index = self.calls.get(name, 0)
            self.calls[name] = call_index + 1
            if self.spec.max_calls is not None and call_index >= self.spec.max_calls:
                return
            tensor = _get_tensor(output, self.spec.output_index)
            self.activations.setdefault(name, []).append(_capture_tensor(tensor, self.spec.capture))

        return hook

    def payload(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "metadata": metadata or {},
            "spec": {
                "patterns": list(self.spec.patterns),
                "regex": self.spec.regex,
                "output_index": self.spec.output_index,
                "max_calls": self.spec.max_calls,
                "capture": self.spec.capture,
            },
            "module_types": self.module_types,
            "calls": self.calls,
            "activations": self.activations,
        }


class MultiSiteActivationRecorder(AbstractContextManager["MultiSiteActivationRecorder"]):
    def __init__(
        self,
        model: torch.nn.Module,
        site_output_indices: dict[str, int],
        *,
        max_calls: int | None = None,
        capture: str = "full",
    ):
        self.model = model
        self.site_output_indices = site_output_indices
        self.max_calls = max_calls
        self.capture = capture
        self.handles: list[Any] = []
        self.calls: dict[str, int] = {}
        self.activations: dict[str, list[torch.Tensor]] = {}
        self.module_types: dict[str, str] = {}

    def __enter__(self) -> "MultiSiteActivationRecorder":
        for name, module in self.model.named_modules():
            if name not in self.site_output_indices:
                continue
            self.module_types[name] = module.__class__.__name__
            self.handles.append(module.register_forward_hook(self._hook(name)))
        missing = sorted(set(self.site_output_indices) - set(self.module_types))
        if missing:
            raise ValueError(f"Missing module sites: {missing}")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            call_index = self.calls.get(name, 0)
            self.calls[name] = call_index + 1
            if self.max_calls is not None and call_index >= self.max_calls:
                return
            tensor = _get_tensor(output, self.site_output_indices[name])
            self.activations.setdefault(name, []).append(_capture_tensor(tensor, self.capture))

        return hook


class ProbeFeatureRecorder(AbstractContextManager["ProbeFeatureRecorder"]):
    """Capture audio-token-pooled features at many sites in a single forward pass.

    For each hooked site the recorder selects the configured output tensor, keeps
    only the audio tokens, mean-pools over them, and accumulates a running mean
    across denoising-step calls. The batch dimension is preserved so the caller
    can later select the classifier-free-guidance row that carries the prompt
    conditioning (TangoFlux batches the conditional and unconditional passes).

    Audio-token handling:
      - ``dual`` blocks expose the audio stream directly (output_index=1), so the
        whole token axis is audio. The audio length is recorded from the first
        dual site seen (dual blocks run before single blocks in the forward pass).
      - ``single`` blocks output the merged ``[text || audio]`` sequence, so the
        trailing ``audio_tokens`` tokens are pooled. Text-token counts differ
        between prompts, so pooling the suffix keeps features comparable.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        site_output_indices: dict[str, int],
        site_stacks: dict[str, str],
        *,
        audio_tokens: int | None = None,
        time_bins: int = 1,
    ):
        self.model = model
        self.site_output_indices = site_output_indices
        self.site_stacks = site_stacks
        self.audio_tokens = audio_tokens
        self.time_bins = max(1, int(time_bins))
        self.handles: list[Any] = []
        self.calls: dict[str, int] = {}
        self.sums: dict[str, torch.Tensor] = {}
        self.token_dims: dict[str, int] = {}
        self.module_types: dict[str, str] = {}

    def __enter__(self) -> "ProbeFeatureRecorder":
        for name, module in self.model.named_modules():
            if name not in self.site_output_indices:
                continue
            self.module_types[name] = module.__class__.__name__
            self.handles.append(module.register_forward_hook(self._hook(name)))
        missing = sorted(set(self.site_output_indices) - set(self.module_types))
        if missing:
            raise ValueError(f"Missing module sites: {missing}")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _audio_slice(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        stack = self.site_stacks.get(name)
        if stack == "dual":
            if self.audio_tokens is None:
                self.audio_tokens = int(tensor.shape[1])
            return tensor
        # single (merged text+audio): pool the trailing audio tokens
        n_audio = self.audio_tokens
        if n_audio is not None and tensor.shape[1] >= n_audio:
            return tensor[:, -n_audio:, :]
        return tensor

    def _hook(self, name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = _get_tensor(output, self.site_output_indices[name]).detach().float()
            if tensor.ndim != 3:
                return
            audio = self._audio_slice(name, tensor)
            self.token_dims[name] = int(audio.shape[1])
            # Pool over the audio-token (time) axis. With time_bins>1, split the audio
            # tokens into K consecutive temporal bins and concatenate their means, so the
            # feature preserves coarse temporal structure: [batch, time_bins * d_model].
            if self.time_bins > 1 and audio.shape[1] >= self.time_bins:
                chunks = torch.chunk(audio, self.time_bins, dim=1)
                pooled = torch.cat([chunk.mean(dim=1) for chunk in chunks], dim=1)
            else:
                pooled = audio.mean(dim=1)  # [batch, d_model]
            pooled = pooled.to(device="cpu")
            if name in self.sums:
                self.sums[name] = self.sums[name] + pooled
            else:
                self.sums[name] = pooled
            self.calls[name] = self.calls.get(name, 0) + 1

        return hook

    def features(self) -> dict[str, torch.Tensor]:
        """Return ``name -> [batch, d_model]`` features averaged over calls."""
        return {
            name: self.sums[name] / max(self.calls.get(name, 1), 1) for name in self.sums
        }


class ActivationPatcher(AbstractContextManager["ActivationPatcher"]):
    def __init__(
        self,
        model: torch.nn.Module,
        spec: HookSpec,
        source_activations: dict[str, list[torch.Tensor]],
        *,
        alpha: float = 1.0,
        suffix_tokens: int | None = None,
        step_window: tuple[int, int] | None = None,
    ):
        self.model = model
        self.spec = spec
        self.source_activations = source_activations
        self.alpha = float(alpha)
        self.suffix_tokens = suffix_tokens
        self.step_window = step_window
        self.handles: list[Any] = []
        self.calls: dict[str, int] = {}

    def __enter__(self) -> "ActivationPatcher":
        for name, module in self.model.named_modules():
            if not name or not _matches(name, self.spec):
                continue
            if name in self.source_activations:
                self.handles.append(module.register_forward_hook(self._hook(name)))
        if not self.handles:
            raise ValueError("No modules matched both the patch spec and activation payload.")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            call_index = self.calls.get(name, 0)
            self.calls[name] = call_index + 1
            if self.spec.max_calls is not None and call_index >= self.spec.max_calls:
                return output
            # Flow-time gate: each forward call is one denoising step (CFG is batched),
            # so call_index == step index. Outside the window we pass through unpatched.
            if self.step_window is not None:
                start, end = self.step_window
                if not (start <= call_index < end):
                    return output
            source_values = self.source_activations.get(name, [])
            if call_index >= len(source_values):
                return output
            current = _get_tensor(output, self.spec.output_index)
            source = source_values[call_index].to(device=current.device, dtype=current.dtype)
            if source.shape == current.shape:
                patched = current.lerp(source, self.alpha)
                return _replace_tensor(output, self.spec.output_index, patched)
            if self.suffix_tokens is not None:
                patched = _patch_suffix(current, source, self.suffix_tokens, self.alpha)
                return _replace_tensor(output, self.spec.output_index, patched)
            if source.shape != current.shape:
                source = source.expand_as(current)
            patched = current.lerp(source, self.alpha)
            return _replace_tensor(output, self.spec.output_index, patched)

        return hook


def _patch_suffix(
    current: torch.Tensor,
    source: torch.Tensor,
    suffix_tokens: int,
    alpha: float,
) -> torch.Tensor:
    if current.ndim < 2 or source.ndim < 2:
        raise ValueError("suffix patching requires tensors with a token dimension")
    if current.shape[0] != source.shape[0] or current.shape[2:] != source.shape[2:]:
        raise ValueError(
            "suffix patching requires matching batch and hidden dimensions: "
            f"current={tuple(current.shape)} source={tuple(source.shape)}"
        )
    if suffix_tokens <= 0:
        raise ValueError("suffix_tokens must be positive")
    if current.shape[1] < suffix_tokens or source.shape[1] < suffix_tokens:
        raise ValueError(
            f"suffix_tokens={suffix_tokens} exceeds token dimension: "
            f"current={tuple(current.shape)} source={tuple(source.shape)}"
        )
    patched = current.clone()
    patched_suffix = current[:, -suffix_tokens:, ...].lerp(
        source[:, -suffix_tokens:, ...],
        alpha,
    )
    patched[:, -suffix_tokens:, ...] = patched_suffix
    return patched


class SteeringApplier(AbstractContextManager["SteeringApplier"]):
    def __init__(
        self,
        model: torch.nn.Module,
        spec: HookSpec,
        steering_vectors: dict[str, torch.Tensor],
        *,
        scale: float,
        suffix_tokens: int | None = None,
    ):
        self.model = model
        self.spec = spec
        self.steering_vectors = steering_vectors
        self.scale = float(scale)
        self.suffix_tokens = suffix_tokens
        self.handles: list[Any] = []

    def __enter__(self) -> "SteeringApplier":
        for name, module in self.model.named_modules():
            if not name or not _matches(name, self.spec):
                continue
            if name in self.steering_vectors:
                self.handles.append(module.register_forward_hook(self._hook(name)))
        if not self.handles:
            raise ValueError("No modules matched both the steering spec and vector payload.")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            current = _get_tensor(output, self.spec.output_index)
            vector = self.steering_vectors[name].to(device=current.device, dtype=current.dtype)
            while vector.ndim < current.ndim:
                vector = vector.unsqueeze(0)
            if (
                self.suffix_tokens is not None
                and current.ndim >= 2
                and current.shape[1] >= self.suffix_tokens
            ):
                # Add the steering vector only to the trailing audio tokens, leaving
                # leading text tokens untouched (merged single-stream blocks).
                steered = current.clone()
                steered[:, -self.suffix_tokens :, ...] = (
                    current[:, -self.suffix_tokens :, ...] + self.scale * vector
                )
            else:
                steered = current + self.scale * vector
            return _replace_tensor(output, self.spec.output_index, steered)

        return hook


def mean_activation_difference(
    positive_payload: dict[str, Any],
    negative_payload: dict[str, Any],
    *,
    reduce_dims: tuple[int, ...] = (0,),
) -> dict[str, torch.Tensor]:
    vectors: dict[str, torch.Tensor] = {}
    positive = positive_payload["activations"]
    negative = negative_payload["activations"]
    for name in sorted(set(positive) & set(negative)):
        pos_tensors = positive[name]
        neg_tensors = negative[name]
        count = min(len(pos_tensors), len(neg_tensors))
        if count == 0:
            continue
        diffs = []
        for index in range(count):
            pos = pos_tensors[index].float()
            neg = neg_tensors[index].float()
            if pos.shape != neg.shape:
                continue
            diff = pos - neg
            for dim in sorted(reduce_dims, reverse=True):
                if diff.ndim > dim:
                    diff = diff.mean(dim=dim)
            diffs.append(diff)
        if diffs:
            vectors[name] = torch.stack(diffs).mean(dim=0).to(dtype=torch.float16)
    return vectors
