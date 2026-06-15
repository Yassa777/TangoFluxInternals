from __future__ import annotations

from collections import defaultdict
from typing import Any


DEFAULT_METRIC_DIRECTIONS: dict[str, int] = {
    "spectral_centroid_mean_hz": 1,
    "spectral_centroid_median_hz": 1,
    "rolloff_85_mean_hz": 1,
    "rolloff_95_mean_hz": 1,
    "high_to_low_db": 1,
    "high_to_total_ratio": 1,
    "onset_strength_max": 1,
    "onset_strength_p95": 1,
    "onset_strength_mean": 1,
    "transient_energy_fraction": 1,
    "transient_energy_per_onset_mean": 1,
    "onset_count": 1,
    "onset_rate_per_second": 1,
    "attack_time_ms_first": -1,
    "attack_time_ms_median": -1,
    "attack_time_ms_min": -1,
    "tail_energy_fraction_250ms": -1,
    "tail_energy_fraction_500ms": -1,
    "tail_energy_fraction_1000ms": -1,
    "late_energy_fraction_300ms": -1,
    "late_energy_fraction_700ms": -1,
    "direct_to_late_ratio": 1,
    "direct_to_late_db": 1,
    "decay_time_to_minus_20db_ms": -1,
    "decay_slope_db_per_second": -1,
    "reverb_proxy_score": -1,
}


def summarize_metric_rows(
    rows: list[dict[str, Any]],
    *,
    positive_label: str = "positive",
    negative_label: str = "negative",
    metric_directions: dict[str, int] | None = None,
) -> dict[str, Any]:
    directions = dict(DEFAULT_METRIC_DIRECTIONS)
    if metric_directions:
        directions.update(metric_directions)

    valid_rows = [row for row in rows if row.get("status") == "ok"]
    metrics = [metric for metric in directions if any(_float_or_none(row.get(metric)) is not None for row in valid_rows)]

    group_summaries: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for metric in metrics:
        group_summaries[metric] = {}
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in valid_rows:
            label = str(row.get("label") or row.get("side") or "")
            value = _float_or_none(row.get(metric))
            if value is not None:
                grouped[label].append(value)
        for label, values in sorted(grouped.items()):
            group_summaries[metric][label] = {
                "n": len(values),
                "mean": _mean(values),
                "median": _median(values),
            }

    pair_rows: list[dict[str, Any]] = []
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in valid_rows:
        pair_id = str(row.get("pair_id", ""))
        label = str(row.get("label") or row.get("side") or "")
        if pair_id and label:
            by_pair[pair_id][label] = row

    metric_pair_values: dict[str, list[float]] = {metric: [] for metric in metrics}
    metric_expected_counts: dict[str, int] = {metric: 0 for metric in metrics}
    for pair_id in sorted(by_pair):
        pair = by_pair[pair_id]
        if positive_label not in pair or negative_label not in pair:
            continue
        pair_summary: dict[str, Any] = {"pair_id": pair_id}
        for metric in metrics:
            positive = _float_or_none(pair[positive_label].get(metric))
            negative = _float_or_none(pair[negative_label].get(metric))
            if positive is None or negative is None:
                diff = None
                expected = None
            else:
                diff = positive - negative
                expected = diff * directions[metric] > 0
                metric_pair_values[metric].append(diff)
                metric_expected_counts[metric] += int(expected)
            pair_summary[f"{metric}_{positive_label}"] = positive
            pair_summary[f"{metric}_{negative_label}"] = negative
            pair_summary[f"{metric}_difference"] = diff
            pair_summary[f"{metric}_expected_direction"] = expected
        pair_rows.append(pair_summary)

    paired_summaries: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        values = metric_pair_values[metric]
        paired_summaries[metric] = {
            "direction": directions[metric],
            "n_pairs": len(values),
            "mean_paired_difference": _mean(values),
            "median_paired_difference": _median(values),
            "expected_direction_count": metric_expected_counts[metric],
            "expected_direction_fraction": metric_expected_counts[metric] / len(values)
            if values
            else None,
        }

    return {
        "n_rows": len(rows),
        "n_valid_rows": len(valid_rows),
        "positive_label": positive_label,
        "negative_label": negative_label,
        "metrics": metrics,
        "group_summaries": group_summaries,
        "paired_summaries": paired_summaries,
        "paired_rows": pair_rows,
    }


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    return float((sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0)
