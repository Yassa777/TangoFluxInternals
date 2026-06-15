"""Pair-grouped linear probes over captured DiT features.

Two probe families are fit independently at every DiT site:

* a logistic-regression *label* probe (percussive=1 / sustained=0), scored by
  cross-validated accuracy -> "decodability";
* a ridge-regression *metric* probe per audio target (onset, decay, ...), scored
  by cross-validated R^2 -> "predictability".

All cross-validation is grouped by ``pair_id`` so the two sides of a pair never
straddle the train/test boundary -- otherwise a probe can recognise a shared
seed/source fingerprint instead of generalising the concept.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


# Strong percussive/sustained separators that exist as columns in the audio
# metric CSVs. Each becomes one ridge "predictability" probe.
DEFAULT_TARGETS: tuple[str, ...] = (
    "onset_strength_max",
    "decay_time_to_minus_20db_ms",
    "tail_energy_fraction_500ms",
    "spectral_centroid_mean_hz",
    "high_to_low_db",
)

POSITIVE_SIDE = "positive"  # percussive


def load_feature_bundle(path: str) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def _kfold(n_splits: int, n_groups: int):
    from sklearn.model_selection import GroupKFold

    return GroupKFold(n_splits=max(2, min(n_splits, n_groups)))


def label_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    C: float = 1.0,
) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_groups = int(np.unique(groups).size)
    if n_groups < 2:
        return float("nan"), float("nan")
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    scores = cross_val_score(
        pipe, X, y, cv=_kfold(n_splits, n_groups), groups=groups, scoring="accuracy"
    )
    return float(scores.mean()), float(scores.std())


def metric_r2(
    X: np.ndarray,
    t: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    alpha: float = 1.0,
) -> tuple[float, float, int]:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    mask = np.isfinite(t)
    Xs, ts, gs = X[mask], t[mask], groups[mask]
    n_groups = int(np.unique(gs).size)
    if n_groups < 2 or ts.size < 4:
        return float("nan"), float("nan"), int(mask.sum())
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    scores = cross_val_score(
        pipe, Xs, ts, cv=_kfold(n_splits, n_groups), groups=gs, scoring="r2"
    )
    return float(scores.mean()), float(scores.std()), int(mask.sum())


def target_array(
    metrics_lookup: dict[tuple[str, str], float],
    pair_ids: Sequence[str],
    sides: Sequence[str],
    *,
    exclude_pairs: Iterable[str] = (),
) -> np.ndarray:
    exclude = {str(p) for p in exclude_pairs}
    out = np.full(len(pair_ids), np.nan, dtype=float)
    for i, (pair_id, side) in enumerate(zip(pair_ids, sides)):
        if str(pair_id) in exclude:
            continue
        value = metrics_lookup.get((str(pair_id), str(side)))
        if value is not None and np.isfinite(value):
            out[i] = float(value)
    return out


def build_probe_map(
    bundle: dict[str, Any],
    metrics_lookups: dict[str, dict[tuple[str, str], float]],
    *,
    targets: Sequence[str] = DEFAULT_TARGETS,
    cfg_row: str | int = "auto",
    exclude_pairs: Iterable[str] = (),
    n_splits: int = 5,
    C: float = 1.0,
    alpha: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    features = bundle["features"]  # [N, n_sites, batch, d_model]
    labels = bundle["labels"].astype(int)
    pair_ids = bundle["pair_ids"].astype(str)
    sides = bundle["sides"].astype(str)
    sites = bundle["sites"].astype(str)
    stacks = bundle["stacks"].astype(str)
    blocks = bundle["blocks"].astype(int)
    n_obs, n_sites, batch, _ = features.shape

    # Select the CFG batch row. "auto" picks the row whose label probes are most
    # decodable on average -- the conditional pass carries the prompt signal.
    def decode_map(row: int) -> list[float]:
        return [
            label_accuracy(features[:, j, row, :], labels, pair_ids, n_splits=n_splits, C=C)[0]
            for j in range(n_sites)
        ]

    if cfg_row == "auto":
        per_row = {row: decode_map(row) for row in range(batch)}
        chosen = max(per_row, key=lambda r: float(np.nanmean(per_row[r])))
    else:
        chosen = int(cfg_row)

    target_vectors = {
        name: target_array(metrics_lookups[name], pair_ids, sides, exclude_pairs=exclude_pairs)
        for name in targets
        if name in metrics_lookups
    }

    rows: list[dict[str, Any]] = []
    for j in range(n_sites):
        X = features[:, j, chosen, :]
        acc, acc_std = label_accuracy(X, labels, pair_ids, n_splits=n_splits, C=C)
        row: dict[str, Any] = {
            "site": str(sites[j]),
            "stack": str(stacks[j]),
            "block": int(blocks[j]),
            "decode_accuracy": acc,
            "decode_accuracy_std": acc_std,
            "n": int(n_obs),
        }
        for name, vec in target_vectors.items():
            r2, r2_std, n_valid = metric_r2(X, vec, pair_ids, n_splits=n_splits, alpha=alpha)
            row[f"r2_{name}"] = r2
            row[f"r2_{name}_std"] = r2_std
            row[f"n_{name}"] = n_valid
        rows.append(row)

    meta = {
        "cfg_row": int(chosen),
        "cfg_batch": int(batch),
        "n_observations": int(n_obs),
        "n_pairs": int(np.unique(pair_ids).size),
        "n_sites": int(n_sites),
        "targets": [name for name in targets if name in metrics_lookups],
        "excluded_pairs": sorted({str(p) for p in exclude_pairs}),
        "n_splits": int(max(2, min(n_splits, np.unique(pair_ids).size))),
    }
    return rows, meta


def summarize_probe_map(
    rows: list[dict[str, Any]],
    targets: Sequence[str] = DEFAULT_TARGETS,
) -> dict[str, Any]:
    def best(rows_: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
        valid = [r for r in rows_ if r.get(key) is not None and np.isfinite(r.get(key, np.nan))]
        if not valid:
            return None
        top = max(valid, key=lambda r: r[key])
        return {"site": top["site"], "stack": top["stack"], "block": top["block"], "value": top[key]}

    def stack_mean(rows_: list[dict[str, Any]], stack: str, key: str) -> float:
        vals = [
            r[key]
            for r in rows_
            if r["stack"] == stack and r.get(key) is not None and np.isfinite(r.get(key, np.nan))
        ]
        return float(np.mean(vals)) if vals else float("nan")

    summary: dict[str, Any] = {
        "best_decodability": best(rows, "decode_accuracy"),
        "decode_accuracy_dual_mean": stack_mean(rows, "dual", "decode_accuracy"),
        "decode_accuracy_single_mean": stack_mean(rows, "single", "decode_accuracy"),
        "best_predictability": {},
    }
    for name in targets:
        key = f"r2_{name}"
        if any(key in r for r in rows):
            summary["best_predictability"][name] = best(rows, key)
            summary[f"r2_{name}_dual_mean"] = stack_mean(rows, "dual", key)
            summary[f"r2_{name}_single_mean"] = stack_mean(rows, "single", key)
    return summary
