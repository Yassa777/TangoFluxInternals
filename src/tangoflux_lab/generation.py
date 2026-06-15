from __future__ import annotations

import csv
import json
import time
from pathlib import PurePosixPath
from typing import Any

from tangoflux_lab.records import normalize_generation_record, slugify


SAMPLE_RATE = 44100


def run_prefix(label: str | None = None) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    if label:
        return f"{stamp}-{slugify(label)}"
    return stamp


def wav_output_path(output_root: str, prefix: str, record: dict[str, Any]) -> str:
    normalized = normalize_generation_record(record)
    pair = slugify(normalized.get("pair_id"), fallback="single")
    side = slugify(normalized.get("side"), fallback="single")
    filename = f"{normalized['job_id']}.wav"
    return str(PurePosixPath(output_root) / prefix / pair / side / filename)


def json_output_path(root: str, prefix: str, filename: str) -> str:
    return str(PurePosixPath(root) / prefix / filename)


def result_manifest_row(record: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_generation_record(record)
    metadata = dict(normalized.get("metadata", {}))
    return {
        "job_id": normalized["job_id"],
        "pair_id": normalized.get("pair_id", ""),
        "side": normalized.get("side", "single"),
        "concept": metadata.get("concept", ""),
        "sample_index": normalized.get("sample_index", 0),
        "seed": normalized["seed"],
        "duration": normalized["duration"],
        "steps": normalized["steps"],
        "guidance_scale": normalized["guidance_scale"],
        "prompt": normalized["prompt"],
        "wav_path": result.get("wav_path", ""),
        "sample_rate": result.get("sample_rate", SAMPLE_RATE),
        "elapsed_seconds": result.get("elapsed_seconds", ""),
        "model_name": result.get("model_name", ""),
        "status": result.get("status", "ok"),
        "error": result.get("error", ""),
    }


def write_manifest_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
