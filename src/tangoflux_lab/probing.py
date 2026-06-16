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

# Heavy-tailed, strictly-positive targets that are better modeled in log space.
# Linear ridge on raw milliseconds/fractions tends to give negative R^2 because a
# few long-decay outliers dominate the squared error.
LOG1P_TARGETS: frozenset[str] = frozenset(
    {"decay_time_to_minus_20db_ms", "tail_energy_fraction_500ms"}
)


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


def _make_estimator(alpha: float, estimator: str):
    """Linear or non-linear regressor for probing. 'ridge' is the linear default."""
    if estimator == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=alpha)
    if estimator == "rbf":
        from sklearn.kernel_ridge import KernelRidge

        return KernelRidge(alpha=alpha, kernel="rbf", gamma=None)
    if estimator == "gb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_depth=3, max_iter=200, random_state=0)
    if estimator == "mlp":
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(hidden_layer_sizes=(64,), alpha=alpha, max_iter=1000, random_state=0)
    raise ValueError(f"Unknown estimator: {estimator}")


def _reg_pipeline(alpha: float, pca: int | None, estimator: str = "ridge"):
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=pca, random_state=0))
    steps.append(_make_estimator(alpha, estimator))
    return make_pipeline(*steps)


def metric_r2(
    X: np.ndarray,
    t: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    alpha: float = 1.0,
    pca: int | None = None,
    estimator: str = "ridge",
) -> tuple[float, float, int]:
    from sklearn.model_selection import cross_val_score

    mask = np.isfinite(t)
    Xs, ts, gs = X[mask], t[mask], groups[mask]
    n_groups = int(np.unique(gs).size)
    if n_groups < 2 or ts.size < 4:
        return float("nan"), float("nan"), int(mask.sum())
    scores = cross_val_score(
        _reg_pipeline(alpha, pca, estimator), Xs, ts, cv=_kfold(n_splits, n_groups),
        groups=gs, scoring="r2",
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
    log_targets: Iterable[str] = LOG1P_TARGETS,
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

    log_set = {str(name) for name in log_targets}
    target_vectors: dict[str, np.ndarray] = {}
    for name in targets:
        if name not in metrics_lookups:
            continue
        vec = target_array(metrics_lookups[name], pair_ids, sides, exclude_pairs=exclude_pairs)
        if name in log_set:
            with np.errstate(invalid="ignore"):
                vec = np.where(np.isfinite(vec) & (vec > -1.0), np.log1p(vec), np.nan)
        target_vectors[name] = vec

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
        "log1p_targets": sorted(log_set & {t for t in targets if t in metrics_lookups}),
        "n_splits": int(max(2, min(n_splits, np.unique(pair_ids).size))),
    }
    return rows, meta


def _toward_source(patched: Any, target_base: Any, source_base: Any) -> float | None:
    """Fraction of the source-target gap covered by the patched value.

    0 = no movement from the target baseline, 1 = fully reached the source value,
    negative = moved away from the source.
    """
    if patched is None or target_base is None or source_base is None:
        return None
    denom = float(source_base) - float(target_base)
    if abs(denom) < 1e-9:
        return None
    return (float(patched) - float(target_base)) / denom


def summarize_intervention_rows(
    rows: list[dict[str, Any]],
    metric_names: Sequence[str],
    *,
    on_target: str = "onset_strength_max",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate patch/steer rows into per-site movement and a specificity summary."""
    from collections import defaultdict

    by_site_metric: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        site = str(row.get("site", ""))
        for metric in metric_names:
            frac = _toward_source(
                row.get(f"patched_{metric}"),
                row.get(f"target_baseline_{metric}"),
                row.get(f"source_baseline_{metric}"),
            )
            if frac is not None and np.isfinite(frac):
                by_site_metric[(site, metric)].append(frac)

    sites = sorted({site for site, _ in by_site_metric})
    site_rows: list[dict[str, Any]] = []
    specificity: dict[str, Any] = {}
    for site in sites:
        row: dict[str, Any] = {"site": site}
        means: dict[str, float] = {}
        for metric in metric_names:
            vals = np.asarray(by_site_metric.get((site, metric), []), dtype=float)
            if vals.size:
                row[f"{metric}_toward_source_mean"] = float(vals.mean())
                row[f"{metric}_frac_moved_toward"] = float((vals > 0).mean())
                row[f"{metric}_n"] = int(vals.size)
                means[metric] = float(vals.mean())
        site_rows.append(row)
        off = [v for m, v in means.items() if m != on_target]
        specificity[site] = {
            "on_target": on_target,
            "on_target_toward_source_mean": means.get(on_target),
            "off_target_toward_source_mean": float(np.mean(off)) if off else None,
            "specificity_gap": (
                float(means[on_target] - np.mean(off))
                if on_target in means and off
                else None
            ),
        }
    return site_rows, {"on_target": on_target, "by_site": specificity}


def _unit_ridge_direction(
    X: np.ndarray, y: np.ndarray, *, alpha: float = 1.0
) -> tuple[np.ndarray, Any]:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    mask = np.isfinite(y)
    scaler = StandardScaler().fit(X[mask])
    ridge = Ridge(alpha=alpha).fit(scaler.transform(X[mask]), y[mask])
    w = ridge.coef_
    norm = float(np.linalg.norm(w))
    return (w / norm if norm > 0 else w), scaler


def _ranks(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    return ranks


def _cv_projection_corr(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, n_splits: int = 5, alpha: float = 1.0,
    pca: int | None = None,
) -> tuple[float, float]:
    """Out-of-sample Pearson/Spearman of grouped cross-validated factor predictions.

    Uses cross_val_predict so the linearity/monotonicity is held-out, not in-sample
    (an in-sample 1024-dim ridge on a few dozen points trivially gives ~1.0).
    """
    from sklearn.model_selection import cross_val_predict

    mask = np.isfinite(y)
    Xs, ys, gs = X[mask], y[mask], groups[mask]
    n_groups = int(np.unique(gs).size)
    if n_groups < 2 or ys.size < 4 or ys.std() < 1e-12:
        return float("nan"), float("nan")
    pred = cross_val_predict(_reg_pipeline(alpha, pca), Xs, ys, cv=_kfold(n_splits, n_groups), groups=gs)
    if pred.std() < 1e-12:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(pred, ys)[0, 1])
    spearman = float(np.corrcoef(_ranks(pred), _ranks(ys))[0, 1])
    return pearson, spearman


def _split_half_cosine(X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, alpha: float = 1.0):
    """|cos| between factor directions fit on two source-disjoint halves (stability)."""
    mask = np.isfinite(y)
    Xs, ys, gs = X[mask], y[mask], groups[mask]
    uniq = np.unique(gs)
    if uniq.size < 2:
        return float("nan")
    half = uniq[: uniq.size // 2]
    m_a = np.isin(gs, half)
    if m_a.sum() < 2 or (~m_a).sum() < 2:
        return float("nan")
    w_a, _ = _unit_ridge_direction(Xs[m_a], ys[m_a], alpha=alpha)
    w_b, _ = _unit_ridge_direction(Xs[~m_a], ys[~m_a], alpha=alpha)
    return float(abs(np.dot(w_a, w_b)))


def _reduce(X: np.ndarray, pca: int | None) -> np.ndarray:
    """Standardize then PCA-reduce (full-data fit) for descriptive direction work."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    if pca and pca < min(Xs.shape):
        Xs = PCA(n_components=pca, random_state=0).fit_transform(Xs)
    return Xs


def build_geometry_map(
    bundle: dict[str, Any],
    factors: dict[str, str],
    *,
    cfg_row: int = 0,
    n_splits: int = 5,
    alpha: float = 1.0,
    pca_components: int | None = None,
    log_factors: Iterable[str] = (),
    nonlinear_estimator: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-layer representation-geometry map against physical ground-truth factors.

    For each site: linear encodability (grouped-CV R²) of each realized factor, the
    projection's linearity/monotonicity, and the cosine between factor directions
    (disentanglement). ``factors`` maps a factor name to its realized-metric column.
    """
    features = bundle["features"]
    sites = bundle["sites"].astype(str)
    stacks = bundle["stacks"].astype(str)
    blocks = bundle["blocks"].astype(int)
    sources = bundle["sources"].astype(str)
    realized_names = bundle["realized_names"].astype(str).tolist()
    realized = bundle["realized"]
    n_sites = features.shape[1]

    log_set = {str(name) for name in log_factors}
    factor_y: dict[str, np.ndarray] = {}
    for name, metric in factors.items():
        if metric not in realized_names:
            continue
        y = realized[:, realized_names.index(metric)].astype(float)
        if name in log_set:
            with np.errstate(invalid="ignore"):
                y = np.where(np.isfinite(y) & (y > -1.0), np.log1p(y), np.nan)
        factor_y[name] = y
    factor_list = list(factor_y)

    rows: list[dict[str, Any]] = []
    layer_directions: list[dict[str, np.ndarray]] = []
    for j in range(n_sites):
        X = features[:, j, cfg_row, :]
        X_red = _reduce(X, pca_components)  # shared basis for descriptive directions
        row: dict[str, Any] = {"site": str(sites[j]), "stack": str(stacks[j]), "block": int(blocks[j])}
        directions: dict[str, np.ndarray] = {}
        for name, y in factor_y.items():
            r2, r2_std, n_valid = metric_r2(X, y, sources, n_splits=n_splits, alpha=alpha, pca=pca_components)
            pearson, spearman = _cv_projection_corr(
                X, y, sources, n_splits=n_splits, alpha=alpha, pca=pca_components
            )
            w, _ = _unit_ridge_direction(X_red, y, alpha=alpha)
            row[f"r2_{name}"] = r2
            row[f"cv_pearson_{name}"] = pearson
            row[f"cv_spearman_{name}"] = spearman
            row[f"stability_cos_{name}"] = _split_half_cosine(X_red, y, sources, alpha=alpha)
            row[f"n_{name}"] = n_valid
            if nonlinear_estimator:
                r2_nl, _, _ = metric_r2(
                    X, y, sources, n_splits=n_splits, alpha=alpha, pca=pca_components,
                    estimator=nonlinear_estimator,
                )
                row[f"r2_nl_{name}"] = r2_nl
                row[f"nonlinearity_gain_{name}"] = (
                    float(r2_nl - r2) if np.isfinite(r2_nl) and np.isfinite(r2) else float("nan")
                )
            directions[name] = w
        layer_directions.append(directions)
        rows.append(row)

    def stack_mean(key: str, stack: str) -> float:
        vals = [
            r[key]
            for r in rows
            if r["stack"] == stack and r.get(key) is not None and np.isfinite(r.get(key, np.nan))
        ]
        return float(np.mean(vals)) if vals else float("nan")

    def best(key: str) -> dict[str, Any] | None:
        valid = [r for r in rows if r.get(key) is not None and np.isfinite(r.get(key, np.nan))]
        if not valid:
            return None
        top = max(valid, key=lambda r: r[key])
        return {"site": top["site"], "stack": top["stack"], "block": top["block"], "value": top[key]}

    d_model = int(features.shape[3])
    eff_dim = int(pca_components) if pca_components else d_model
    random_cos_baseline = float(np.sqrt(2.0 / (np.pi * eff_dim)))

    summary: dict[str, Any] = {
        "factors": factor_list,
        "d_model": d_model,
        "random_abs_cosine_baseline": random_cos_baseline,
        "by_factor": {},
    }
    def overall_mean(key: str) -> float:
        vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r.get(key, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    for name in factor_list:
        summary["by_factor"][name] = {
            "best_encodable": best(f"r2_{name}"),
            "r2_dual_mean": stack_mean(f"r2_{name}", "dual"),
            "r2_single_mean": stack_mean(f"r2_{name}", "single"),
            "cv_spearman_best": best(f"cv_spearman_{name}"),
            "stability_cos_mean": overall_mean(f"stability_cos_{name}"),
        }
    # Disentanglement is meaningful only where factor directions are stable (clear the
    # random baseline). Pick the analysis layer with the highest mean stability across
    # factors, then compare the direction-cosine matrix to the realized-factor correlation
    # matrix. A factor's geometry is trustworthy only if its stability > ~2x random.
    stability_by_factor = {
        name: overall_mean(f"stability_cos_{name}") for name in factor_list
    }
    stable_factors = [
        name for name in factor_list if stability_by_factor[name] > 2.0 * random_cos_baseline
    ]
    mean_stab_per_layer = [
        float(np.nanmean([rows[j].get(f"stability_cos_{f}", np.nan) for f in factor_list]))
        for j in range(n_sites)
    ]
    analysis_j = int(np.nanargmax(mean_stab_per_layer)) if mean_stab_per_layer else 0
    dirs = layer_directions[analysis_j]

    def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 4 or a[mask].std() < 1e-12 or b[mask].std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(_ranks(a[mask]), _ranks(b[mask]))[0, 1])

    cosine_matrix = {
        f1: {f2: float(abs(np.dot(dirs[f1], dirs[f2]))) for f2 in factor_list}
        for f1 in factor_list
    }
    realized_corr = {
        f1: {f2: spearman_corr(factor_y[f1], factor_y[f2]) for f2 in factor_list}
        for f1 in factor_list
    }
    summary["disentanglement"] = {
        "analysis_layer": {
            "site": rows[analysis_j]["site"],
            "stack": rows[analysis_j]["stack"],
            "block": rows[analysis_j]["block"],
        },
        "random_abs_cosine_baseline": random_cos_baseline,
        "stability_threshold": 2.0 * random_cos_baseline,
        "factor_stability": stability_by_factor,
        "stable_factors": stable_factors,
        "direction_cosine_matrix": cosine_matrix,
        "realized_factor_spearman_matrix": realized_corr,
    }
    return rows, summary


def summarize_commitment_rows(
    rows: list[dict[str, Any]],
    attributes: Sequence[str],
    num_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate windowed-patch rows into per-attribute commitment curves.

    For each site and attribute, builds the prefix sufficiency curve (patching
    steps ``[0, k)``) and the suffix curve (patching steps ``[k, T)``), then
    estimates a sufficiency commit step (smallest prefix k reaching half of full
    movement) and a point-of-no-return (largest suffix start still reaching half).
    """
    from collections import defaultdict

    acc: dict[tuple[str, tuple[int, int]], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    windows_by_site: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for row in rows:
        if row.get("status") != "ok":
            continue
        site = str(row["site"])
        window = (int(row["window_start"]), int(row["window_end"]))
        windows_by_site[site].add(window)
        for attribute in attributes:
            frac = _toward_source(
                row.get(f"patched_{attribute}"),
                row.get(f"target_baseline_{attribute}"),
                row.get(f"source_baseline_{attribute}"),
            )
            if frac is not None and np.isfinite(frac):
                acc[(site, window)][attribute].append(frac)

    movement: dict[tuple[str, str, tuple[int, int]], float] = {}
    window_rows: list[dict[str, Any]] = []
    for (site, window), per_attr in acc.items():
        out: dict[str, Any] = {"site": site, "window_start": window[0], "window_end": window[1]}
        for attribute in attributes:
            vals = per_attr.get(attribute, [])
            if vals:
                mean = float(np.mean(vals))
                out[f"{attribute}_toward_source_mean"] = mean
                out[f"{attribute}_n"] = len(vals)
                movement[(site, attribute, window)] = mean
        window_rows.append(out)
    window_rows.sort(key=lambda r: (r["site"], r["window_start"], r["window_end"]))

    def crosses(value: float, threshold: float, full: float) -> bool:
        return value >= threshold if full > 0 else value <= threshold

    summary: dict[str, Any] = {}
    full_window = (0, num_steps)
    for site in sorted(windows_by_site):
        wins = sorted(windows_by_site[site])
        site_summary: dict[str, Any] = {}
        for attribute in attributes:
            full = movement.get((site, attribute, full_window))
            prefix = sorted(
                (end, movement[(site, attribute, (0, end))])
                for start, end in wins
                if start == 0 and (site, attribute, (0, end)) in movement
            )
            suffix = sorted(
                (start, movement[(site, attribute, (start, num_steps))])
                for start, end in wins
                if end == num_steps and (site, attribute, (start, num_steps)) in movement
            )
            commit_step = None
            point_of_no_return = None
            if full is not None and abs(full) > 1e-9:
                threshold = 0.5 * full
                for end, value in prefix:
                    if crosses(value, threshold, full):
                        commit_step = end
                        break
                survivors = [start for start, value in suffix if crosses(value, threshold, full)]
                point_of_no_return = max(survivors) if survivors else None
            site_summary[attribute] = {
                "full_movement": full,
                "prefix_curve": prefix,
                "suffix_curve": suffix,
                "sufficiency_commit_step": commit_step,
                "point_of_no_return_step": point_of_no_return,
            }
        summary[site] = site_summary
    return window_rows, summary


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
