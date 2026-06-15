from __future__ import annotations

import math
from typing import Any


def summarize_bright_dark_centroids(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("label") in {"bright", "dark"}
        and row.get("spectral_centroid_mean_hz") is not None
    ]
    bright_values = [
        float(row["spectral_centroid_mean_hz"]) for row in valid_rows if row["label"] == "bright"
    ]
    dark_values = [
        float(row["spectral_centroid_mean_hz"]) for row in valid_rows if row["label"] == "dark"
    ]

    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in valid_rows:
        by_pair.setdefault(str(row["pair_id"]), {})[str(row["label"])] = row

    paired_rows = []
    for pair_id in sorted(by_pair):
        pair = by_pair[pair_id]
        if "bright" not in pair or "dark" not in pair:
            continue
        bright = float(pair["bright"]["spectral_centroid_mean_hz"])
        dark = float(pair["dark"]["spectral_centroid_mean_hz"])
        paired_rows.append(
            {
                "pair_id": pair_id,
                "bright_centroid_mean_hz": bright,
                "dark_centroid_mean_hz": dark,
                "difference_hz": bright - dark,
                "bright_gt_dark": bright > dark,
            }
        )

    differences = [row["difference_hz"] for row in paired_rows]
    return {
        "n_rows": len(rows),
        "n_valid_rows": len(valid_rows),
        "n_bright": len(bright_values),
        "n_dark": len(dark_values),
        "n_pairs": len(paired_rows),
        "bright_mean_centroid_hz": _mean(bright_values),
        "dark_mean_centroid_hz": _mean(dark_values),
        "mean_paired_difference_hz": _mean(differences),
        "median_paired_difference_hz": _median(differences),
        "pairs_bright_higher": sum(1 for row in paired_rows if row["bright_gt_dark"]),
        "paired_rows": paired_rows,
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[midpoint])
    return float((sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2)


def finite_or_none(value: float) -> float | None:
    if math.isfinite(value):
        return float(value)
    return None


def spectral_centroid_from_waveform(
    waveform: Any,
    sample_rate: int,
) -> dict[str, float | None]:
    import torch

    mono = waveform.mean(dim=0).float()
    if mono.numel() == 0:
        return {"mean_hz": None, "median_hz": None}

    n_fft = min(2048, max(256, int(2 ** max(8, (mono.numel() // 8).bit_length() - 1))))
    hop_length = max(128, n_fft // 4)
    window = torch.hann_window(n_fft, device=mono.device)
    spectrum = torch.stft(
        mono,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    ).abs()
    freqs = torch.linspace(0, sample_rate / 2, spectrum.shape[0], device=spectrum.device)
    denominator = spectrum.sum(dim=0).clamp_min(1e-12)
    centroid = (freqs[:, None] * spectrum).sum(dim=0) / denominator
    return {
        "mean_hz": finite_or_none(float(centroid.mean().item())),
        "median_hz": finite_or_none(float(centroid.median().item())),
    }


def summarize_layer_patch_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("patched_centroid_mean_hz") is not None
        and row.get("target_baseline_centroid_mean_hz") is not None
    ]
    by_site_direction: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_site: dict[str, list[dict[str, Any]]] = {}
    for row in valid_rows:
        site = str(row["site"])
        direction = str(row["direction"])
        by_site_direction.setdefault((site, direction), []).append(row)
        by_site.setdefault(site, []).append(row)

    site_direction_rows = []
    for (site, direction), group in sorted(by_site_direction.items()):
        deltas = [float(row["target_delta_centroid_hz"]) for row in group]
        signed_effects = [float(row["signed_brightness_effect_hz"]) for row in group]
        fractions = [
            float(row["source_shift_fraction"])
            for row in group
            if row.get("source_shift_fraction") is not None
        ]
        site_direction_rows.append(
            {
                "site": site,
                "direction": direction,
                "n": len(group),
                "mean_target_delta_hz": _mean(deltas),
                "median_target_delta_hz": _median(deltas),
                "mean_signed_brightness_effect_hz": _mean(signed_effects),
                "median_signed_brightness_effect_hz": _median(signed_effects),
                "mean_source_shift_fraction": _mean(fractions),
                "success_count": sum(1 for value in signed_effects if value > 0),
            }
        )

    site_rows = []
    for site, group in sorted(by_site.items()):
        signed_effects = [float(row["signed_brightness_effect_hz"]) for row in group]
        fractions = [
            float(row["source_shift_fraction"])
            for row in group
            if row.get("source_shift_fraction") is not None
        ]
        site_rows.append(
            {
                "site": site,
                "n": len(group),
                "mean_signed_brightness_effect_hz": _mean(signed_effects),
                "median_signed_brightness_effect_hz": _median(signed_effects),
                "mean_source_shift_fraction": _mean(fractions),
                "success_count": sum(1 for value in signed_effects if value > 0),
            }
        )
    site_rows.sort(key=lambda row: row["mean_signed_brightness_effect_hz"] or 0, reverse=True)

    return {
        "n_rows": len(rows),
        "n_valid_rows": len(valid_rows),
        "n_sites": len(by_site),
        "n_site_direction_rows": len(site_direction_rows),
        "mean_signed_brightness_effect_hz": _mean(
            [float(row["signed_brightness_effect_hz"]) for row in valid_rows]
        ),
        "median_signed_brightness_effect_hz": _median(
            [float(row["signed_brightness_effect_hz"]) for row in valid_rows]
        ),
        "success_count": sum(
            1 for row in valid_rows if float(row["signed_brightness_effect_hz"]) > 0
        ),
        "site_rows": site_rows,
        "site_direction_rows": site_direction_rows,
    }


def movement_row(row: dict[str, Any]) -> dict[str, Any]:
    patched = _float_or_none(row.get("patched_centroid_mean_hz"))
    source = _float_or_none(row.get("source_baseline_centroid_mean_hz"))
    target = _float_or_none(row.get("target_baseline_centroid_mean_hz"))
    if patched is None or source is None or target is None:
        return {
            **row,
            "distance_target_to_source_hz": None,
            "distance_patched_to_source_hz": None,
            "closer_to_source": None,
            "toward_source": None,
            "overshot_source": None,
        }

    distance_target_to_source = abs(source - target)
    distance_patched_to_source = abs(source - patched)
    fraction = _float_or_none(row.get("source_shift_fraction"))
    return {
        **row,
        "distance_target_to_source_hz": distance_target_to_source,
        "distance_patched_to_source_hz": distance_patched_to_source,
        "closer_to_source": distance_patched_to_source < distance_target_to_source,
        "toward_source": fraction is not None and fraction > 0,
        "overshot_source": fraction is not None and fraction > 1,
    }


def summarize_movement_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get("status") == "ok"]
    by_site: dict[str, list[dict[str, Any]]] = {}
    by_site_direction: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in valid_rows:
        site = str(row["site"])
        direction = str(row["direction"])
        by_site.setdefault(site, []).append(row)
        by_site_direction.setdefault((site, direction), []).append(row)

    site_rows = []
    for site, group in sorted(by_site.items()):
        signed = [_float_or_none(row.get("signed_brightness_effect_hz")) for row in group]
        signed = [value for value in signed if value is not None]
        fractions = [_float_or_none(row.get("source_shift_fraction")) for row in group]
        fractions = [value for value in fractions if value is not None]
        site_rows.append(
            {
                "site": site,
                "n": len(group),
                "mean_signed_brightness_effect_hz": _mean(signed),
                "median_signed_brightness_effect_hz": _median(signed),
                "mean_source_shift_fraction": _mean(fractions),
                "toward_source_count": _count_true(group, "toward_source"),
                "closer_to_source_count": _count_true(group, "closer_to_source"),
                "overshot_source_count": _count_true(group, "overshot_source"),
            }
        )
    site_rows.sort(key=lambda row: row["mean_signed_brightness_effect_hz"] or 0, reverse=True)

    site_direction_rows = []
    for (site, direction), group in sorted(by_site_direction.items()):
        signed = [_float_or_none(row.get("signed_brightness_effect_hz")) for row in group]
        signed = [value for value in signed if value is not None]
        fractions = [_float_or_none(row.get("source_shift_fraction")) for row in group]
        fractions = [value for value in fractions if value is not None]
        site_direction_rows.append(
            {
                "site": site,
                "direction": direction,
                "n": len(group),
                "mean_signed_brightness_effect_hz": _mean(signed),
                "median_signed_brightness_effect_hz": _median(signed),
                "mean_source_shift_fraction": _mean(fractions),
                "toward_source_count": _count_true(group, "toward_source"),
                "closer_to_source_count": _count_true(group, "closer_to_source"),
                "overshot_source_count": _count_true(group, "overshot_source"),
            }
        )

    signed_all = [_float_or_none(row.get("signed_brightness_effect_hz")) for row in valid_rows]
    signed_all = [value for value in signed_all if value is not None]
    fractions_all = [_float_or_none(row.get("source_shift_fraction")) for row in valid_rows]
    fractions_all = [value for value in fractions_all if value is not None]
    return {
        "n_rows": len(rows),
        "n_valid_rows": len(valid_rows),
        "n_sites": len(by_site),
        "mean_signed_brightness_effect_hz": _mean(signed_all),
        "median_signed_brightness_effect_hz": _median(signed_all),
        "mean_source_shift_fraction": _mean(fractions_all),
        "toward_source_count": _count_true(valid_rows, "toward_source"),
        "closer_to_source_count": _count_true(valid_rows, "closer_to_source"),
        "overshot_source_count": _count_true(valid_rows, "overshot_source"),
        "site_rows": site_rows,
        "site_direction_rows": site_direction_rows,
    }


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _count_true(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True or row.get(key) == "True")
