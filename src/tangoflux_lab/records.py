from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_DURATION = 10.0
DEFAULT_STEPS = 25
DEFAULT_GUIDANCE_SCALE = 4.5
DEFAULT_SEED = 0


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            rows.append(row)
    return rows


def slugify(value: Any, *, fallback: str = "item") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or fallback


def _prompt_sides(row: dict[str, Any]) -> list[tuple[str, str]]:
    if "prompt" in row:
        return [(str(row.get("side", "single")), str(row["prompt"]))]

    aliases = [
        ("bright", "dark"),
        ("a", "b"),
        ("prompt_a", "prompt_b"),
        ("positive", "negative"),
        ("left", "right"),
    ]
    for left_key, right_key in aliases:
        if left_key in row and right_key in row:
            return [(left_key, str(row[left_key])), (right_key, str(row[right_key]))]

    raise ValueError(
        "Prompt row must contain either `prompt` or a contrastive pair such as `a`/`b`."
    )


def _coerce_duration(value: Any) -> float:
    duration = float(value)
    if not 1 <= duration <= 30:
        raise ValueError("TangoFlux duration must be between 1 and 30 seconds.")
    return duration


def _coerce_steps(value: Any) -> int:
    steps = int(value)
    if steps <= 0:
        raise ValueError("steps must be positive.")
    return steps


def expand_prompt_rows(
    rows: list[dict[str, Any]],
    *,
    samples_per_prompt: int = 1,
    default_duration: float = DEFAULT_DURATION,
    default_steps: int = DEFAULT_STEPS,
    default_guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    default_seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Expand prompt-pair JSONL rows into one generation record per side/sample."""

    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be at least 1.")

    expanded: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        pair_id = str(row.get("pair_id") or row.get("id") or f"row-{row_index:04d}")
        row_samples = int(row.get("samples", samples_per_prompt))
        base_seed = int(row.get("seed", default_seed + row_index * 1000))
        duration = _coerce_duration(row.get("duration", default_duration))
        steps = _coerce_steps(row.get("steps", default_steps))
        guidance_scale = float(row.get("guidance_scale", default_guidance_scale))
        metadata = dict(row.get("metadata", {}))
        if "concept" in row and "concept" not in metadata:
            metadata["concept"] = row["concept"]

        for sample_index in range(row_samples):
            seed = base_seed + sample_index
            for side, prompt in _prompt_sides(row):
                safe_pair = slugify(pair_id, fallback=f"row-{row_index:04d}")
                safe_side = slugify(side, fallback="side")
                job_id = f"{safe_pair}__{safe_side}__sample-{sample_index:03d}__seed-{seed}"
                expanded.append(
                    {
                        "job_id": job_id,
                        "pair_id": pair_id,
                        "side": side,
                        "sample_index": sample_index,
                        "prompt": prompt,
                        "duration": duration,
                        "steps": steps,
                        "guidance_scale": guidance_scale,
                        "seed": seed,
                        "metadata": metadata,
                    }
                )
    return expanded


def normalize_generation_record(record: dict[str, Any]) -> dict[str, Any]:
    if "prompt" not in record:
        raise ValueError("Generation record must contain `prompt`.")

    normalized = dict(record)
    normalized["prompt"] = str(normalized["prompt"])
    normalized["duration"] = _coerce_duration(normalized.get("duration", DEFAULT_DURATION))
    normalized["steps"] = _coerce_steps(normalized.get("steps", DEFAULT_STEPS))
    normalized["guidance_scale"] = float(
        normalized.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)
    )
    normalized["seed"] = int(normalized.get("seed", DEFAULT_SEED))
    normalized["job_id"] = slugify(
        normalized.get("job_id") or normalized.get("id") or normalized["prompt"][:80],
        fallback="generation",
    )
    normalized.setdefault("pair_id", "")
    normalized.setdefault("side", "single")
    normalized.setdefault("sample_index", 0)
    normalized.setdefault("metadata", {})
    return normalized
