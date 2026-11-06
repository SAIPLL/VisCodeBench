"""
Data-similarity metric for T2V evaluation.

Entry point: ``calculate_similarity_score(pred_log_dir, gt_log_dir, ...)``
"""

import os
import json
import math

import numpy as np
import scipy.optimize
import scipy.spatial.distance
from joblib import Parallel, delayed
from tqdm import tqdm

from .metric_utils import (read_t2v_log_dir, summarize_fig_data, data_to_np_1d, data_to_np_4d,
                           find_min_max)


# ---------------------------------------------------------------------------
# Cost-matrix helpers
# ---------------------------------------------------------------------------

def create_dist_mat(seq1: list, seq2: list, compare) -> np.ndarray:
    """Build a pairwise distance/cost matrix between two sequences."""
    l1, l2 = len(seq1), len(seq2)
    mat = np.full((l1, l2), -1.0)
    for i in range(l1):
        for j in range(l2):
            mat[i, j] = compare(seq1[i], seq2[j])
            assert not math.isnan(mat[i, j]), \
                f"NaN in distance matrix at ({i}, {j})"
    return mat


def _pad_to_square(mat: np.ndarray) -> np.ndarray:
    """Pad a rectangular cost matrix to square with cost 1.0."""
    h, w = mat.shape
    if h == w:
        return mat
    n = max(h, w)
    padded = np.ones((n, n))
    padded[:h, :w] = mat
    return padded


def get_score(cost_mat: np.ndarray) -> float:
    """
    Solve the linear assignment problem on *cost_mat* and return a similarity
    score in [0, 1].

    Uses ``scipy.optimize.linear_sum_assignment`` (Hungarian / Jonker-Volgenant).
    """
    cost_mat = _pad_to_square(cost_mat)
    k = cost_mat.shape[0]
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_mat)
    avg_cost = cost_mat[row_ind, col_ind].sum() / k
    return 1.0 - min(1.0, avg_cost)


# ---------------------------------------------------------------------------
# Per-series comparators
# ---------------------------------------------------------------------------

def compare_scatter(pred_ds: dict, gt_ds: dict,
                    covar_epsilon: float = 1e-5,
                    debug: bool = False) -> float:
    """Compare two multi-dimensional scatter point clouds (Mahalanobis distance)."""
    pred_np = data_to_np_4d(pred_ds)
    gt_np   = data_to_np_4d(gt_ds)

    if gt_np.shape[0] == 1:
        if debug:
            print("\t\tWarning: single GT point — falling back to Euclidean distance.")
        cost_mat = np.minimum(
            1.0, scipy.spatial.distance.cdist(pred_np, gt_np, metric="euclidean")
        )
    else:
        V  = np.cov(gt_np.T) + np.eye(gt_np.shape[1]) * covar_epsilon
        VI = np.linalg.inv(V).T
        cost_mat = np.minimum(
            1.0, scipy.spatial.distance.cdist(pred_np, gt_np, metric="mahalanobis", VI=VI)
        )

    # Replace any residual NaNs with maximum cost
    cost_mat = np.where(np.isnan(cost_mat), 1.0, cost_mat)

    score = get_score(cost_mat)
    if debug:
        print(f"\t\tScatter: pred={len(pred_ds['x'])} pts, gt={len(gt_ds['x'])} pts → {score:.4f}")
    return score


def get_cont_recall_new(p_xs, p_ys, g_xs, g_ys, epsilon: float) -> float:
    """
    Parametric (arc-length) interpolation recall for continuous (line) series.

    Vectorised implementation. Both curves are expected in the reference axis frame
    (reference data in [0, 1]), so the point error is divided by the reference range,
    1, rather than by the point's distance from the origin.
    """
    p_ts = np.concatenate([[0], np.cumsum(np.hypot(np.diff(p_xs), np.diff(p_ys)))])
    g_ts = np.concatenate([[0], np.cumsum(np.hypot(np.diff(g_xs), np.diff(g_ys)))])

    if p_ts[-1] == 0: p_ts[-1] = epsilon
    if g_ts[-1] == 0: g_ts[-1] = epsilon
    p_ts = p_ts / p_ts[-1]
    g_ts = g_ts / g_ts[-1]

    n = len(g_ts)
    if n < 2:
        # single ground-truth point → no interval to integrate over.
        # (In practice single-point 'plot' series are reclassified as scatter
        # upstream, so this guard is just defensive.)
        return 0.0

    # Interval weights (trapezoidal): endpoints get a half interval, interior
    # points get the centred half-difference. Matches the original loop exactly.
    interval = np.empty(n)
    interval[0]    = (g_ts[1] - g_ts[0]) / 2
    interval[-1]   = (g_ts[-1] - g_ts[-2]) / 2
    interval[1:-1] = (g_ts[2:] - g_ts[:-2]) / 2
    interval = np.abs(interval)

    # Interpolate the predicted curve at every ground-truth parameter at once.
    px = np.interp(g_ts, p_ts, p_xs)
    py = np.interp(g_ts, p_ts, p_ys)
    error      = np.hypot(g_xs - px, g_ys - py)
    norm_error = np.minimum(1.0, error)

    total_score    = float(np.sum((1.0 - norm_error) * interval))
    total_interval = float(np.sum(interval))

    return min(1.0, total_score / total_interval) if total_interval > 0 else 0.0

def get_cont_recall(p_xs, p_ys, g_xs, g_ys, epsilon):
    """
    Use linear interpolation to estimate ground truth points on predicted curve to compute error
    Then compute recall/precision based on the normalized error.
    """
    
    total_score = 0
    total_interval = 0
    for i in range(g_xs.shape[0]):
        x = g_xs[i]
        if i == 0:
            interval = (g_xs[i+1] - x) / 2
        elif i == (g_xs.shape[0] - 1):
            interval = (x - g_xs[i-1]) / 2
        else:
            interval = (g_xs[i+1] - g_xs[i-1]) / 2

        y = g_ys[i]
        y_interp = np.interp(x, p_xs, p_ys)
        error = min(1, abs( (y - y_interp) / (abs(y) + epsilon)))
        total_score += (1 - error) * interval
        total_interval += interval
    assert np.isclose(total_interval, g_xs[-1] - g_xs[0])
    return min(1, total_score / total_interval) if total_interval > 0 else 0


def compare_continuous(pred_ds: dict, gt_ds: dict, debug: bool = False) -> float:
    """Compare two line-plot series using parametric recall/precision (F1)."""
    p_xs, p_ys = pred_ds["x"], pred_ds["y"]
    g_xs, g_ys = gt_ds["x"],  gt_ds["y"]

    # If either curve is vertical, swap axes so interpolation works on x
    if np.unique(p_xs).size == 1 or np.unique(g_xs).size == 1:
        p_xs, p_ys = p_ys, p_xs
        g_xs, g_ys = g_ys, g_xs

    epsilon = (g_ys.max() - g_ys.min()) / 100 or 1e-5
    recall    = get_cont_recall_new(p_xs, p_ys, g_xs, g_ys, epsilon)
    precision = get_cont_recall_new(g_xs, g_ys, p_xs, p_ys, epsilon)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    f1 = max(0.0, min(1.0, f1))

    if debug:
        print(f"\t\tContinuous: pred={len(p_xs)} pts, gt={len(g_xs)} pts "
              f"→ recall={recall:.4f}, precision={precision:.4f}, F1={f1:.4f}")
    return f1


def compare_discrete(pred_data: dict, gt_data: dict, debug: bool = False) -> float:
    """Compare 1-D discrete (pie) series using Mahalanobis / Euclidean distance."""
    pred_vals = data_to_np_1d(pred_data)
    gt_vals   = data_to_np_1d(gt_data)

    if len(gt_vals) == 1:
        cost_mat = np.minimum(
            1.0, scipy.spatial.distance.cdist(pred_vals, gt_vals, metric="euclidean")
        )
    else:
        VI = 1.0 / np.cov(gt_vals.T)
        cost_mat = np.minimum(
            1.0, scipy.spatial.distance.cdist(pred_vals, gt_vals,
                                               metric="mahalanobis", VI=VI)
        )

    score = get_score(cost_mat)
    if debug:
        print(f"\t\tDiscrete: pred={len(pred_vals)} pts, gt={len(gt_vals)} pts → {score:.4f}")
    return score


# ---------------------------------------------------------------------------
# Axis / figure comparators
# ---------------------------------------------------------------------------

def compare_data_series(pred_series: dict, gt_series: dict, debug: bool = False) -> float:
    """
    Compare two series dicts produced by ``read_t2v_log_dir``.

    Dispatches to the appropriate comparator based on ``plt_func``.
    """
    pred_func = pred_series["plt_func"]
    gt_func   = gt_series["plt_func"]
    pred_data = pred_series["data"]
    gt_data   = gt_series["data"]

    if pred_func == gt_func == "plot":
        return compare_continuous(pred_data, gt_data, debug=debug)
    if pred_func == gt_func == "pie":
        return compare_discrete(pred_data, gt_data, debug=debug)
    return compare_scatter(pred_data, gt_data, debug=debug)


_COORDS = ("x", "y", "z", "t")


def reference_axis_frame(gt_series_list: list) -> dict:
    """
    ``{coord: (offset, scale)}`` mapping the reference axis into [0, 1].

    The range is taken over all non-pie series in the reference axis. A constant
    reference coordinate (e.g. a flat line) has no range, so it is centred on 0 and
    scaled by its own magnitude (or 1 when it is 0): matching values map to 0, and a
    prediction that doubles the constant maps to 1.
    Coordinates absent from the reference axis get no entry and are left as is.
    """
    bounds = find_min_max([s for s in gt_series_list if s["plt_func"] != "pie"])
    frame = {}
    for coord, (lo, hi) in zip(_COORDS, bounds):
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        frame[coord] = (lo, hi - lo) if hi > lo else (lo, abs(lo) or 1.0)
    return frame


def to_reference_frame(series_list: list, frame: dict) -> list:
    """Copies of *series_list* expressed in *frame*; pie slices become proportions."""
    out = []
    for series in series_list:
        data = dict(series["data"])
        if series["plt_func"] == "pie":
            total = np.sum(data["x"])
            if total > 0:
                data["x"] = np.asarray(data["x"]) / total
        else:
            for coord, (offset, scale) in frame.items():
                if coord in data:
                    data[coord] = (np.asarray(data[coord], dtype=float) - offset) / scale
        out.append({**series, "data": data})
    return out


def compare_axes(pred_series_list: list, gt_series_list: list,
                 debug: bool = False, n_jobs: int = 1,
                 normalize: bool = True) -> float:
    """
    Compare two axes by solving the optimal series-assignment problem.

    With *normalize*, both axes are first mapped into the reference axis's frame
    (:func:`reference_axis_frame`), so scale, unit and offset errors in the
    prediction are scored instead of being normalised away.
    """
    if normalize:
        frame = reference_axis_frame(gt_series_list)
        pred_series_list = to_reference_frame(pred_series_list, frame)
        gt_series_list   = to_reference_frame(gt_series_list, frame)

    n_pred = len(pred_series_list)
    n_gt   = len(gt_series_list)

    results = Parallel(n_jobs=n_jobs)(
        delayed(compare_data_series)(pred_series_list[i], gt_series_list[j], debug=debug)
        for i in range(n_pred)
        for j in range(n_gt)
    )
    score_matrix = np.array(results).reshape(n_pred, n_gt)
    score_matrix = np.where(np.isnan(score_matrix), -1.0, score_matrix)

    score = get_score(1.0 - score_matrix)
    if debug:
        print(f"\tAxes: pred={n_pred} series, gt={n_gt} series")
        print(f"\tScore matrix:\n{score_matrix}")
        print(f"\tFinal: {score:.4f}\n")
    return score


def compare_figures(pred_fig_data: dict, gt_fig_data: dict,
                    debug: bool = False, n_jobs: int = 1,
                    normalize: bool = True) -> float:
    """
    Compare two full figures (dicts of axes) by optimal axis assignment.
    """
    pred_axes = list(pred_fig_data.values())
    gt_axes   = list(gt_fig_data.values())
    n_pred    = len(pred_axes)
    n_gt      = len(gt_axes)

    iterator = (
        delayed(compare_axes)(pred_axes[i], gt_axes[j], debug=debug, normalize=normalize)
        for i in range(n_pred)
        for j in range(n_gt)
    )
    results = Parallel(n_jobs=n_jobs)(
        tqdm(iterator, total=n_pred * n_gt,
             desc="Computing score matrix", disable=not debug)
    )
    score_matrix = np.array(results).reshape(n_pred, n_gt)
    score_matrix = np.where(np.isnan(score_matrix), -1.0, score_matrix)

    score = get_score(1.0 - score_matrix)
    if debug:
        print(f"Figures: pred={n_pred} axes, gt={n_gt} axes")
        print(f"Score matrix:\n{score_matrix}")
        print(f"Final: {score:.4f}\n")
    return score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_similarity_score(pred_log_dir: str,
                                gt_log_dir: str,
                                down_sampling: bool = True,
                                down_sampling_max_length: int = 10_000,
                                down_sampling_method: str = "random",
                                dedup: bool = False,
                                summarize_before: bool = False,
                                summarize_after: bool = False,
                                print_points: int = 0,
                                use_cache: bool = False,
                                save_results: bool = False,
                                verbose: bool = False,
                                reference_frame: bool = True) -> float:
    """
    Compute the data-similarity score between a predicted and a ground-truth
    T2V log directory.

    Parameters
    ----------
    pred_log_dir / gt_log_dir
        Paths to log directories produced by ``activate_t2v_logging()``.
    down_sampling
        Randomly subsample long series before comparison.
    down_sampling_max_length
        Maximum number of points per series after downsampling.
    down_sampling_method
        Downsampling method — only ``"random"`` is supported.
    dedup
        Remove duplicate points (not yet implemented).
    summarize_before / summarize_after
        Print a data summary before / after preprocessing.
    use_cache
        Return cached result from ``evaluation.json`` if it exists.
    save_results
        Write the result to ``evaluation.json`` in ``pred_log_dir``.
    verbose
        Print debug information during scoring.
    reference_frame
        Map each (prediction, reference) axis pair into the reference axis's frame
        before comparing (default). ``False`` normalises every log directory by its
        own ranges at load time, as before; the line-error divisor change in
        :func:`get_cont_recall_new` applies in both modes.

    Returns
    -------
    float  Similarity score in [0, 1].
    """
    eval_path = os.path.join(pred_log_dir, "evaluation.json")
    normalisation = "reference_frame" if reference_frame else "self"
    if use_cache and os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as fh:
            cached = json.load(fh)
        # Results written before reference-frame scoring carry no "normalisation" field.
        if cached.get("normalisation", "self") == normalisation:
            return cached["similarity_score"]

    _kwargs = dict(
        dedup=dedup,
        down_sampling=down_sampling,
        down_sampling_max_length=down_sampling_max_length,
        down_sampling_method=down_sampling_method,
        verbose=verbose,
        summarize_before=summarize_before,
        summarize_after=summarize_after,
        print_points=print_points,
        normalize=not reference_frame,
    )
    pred_fig_data = read_t2v_log_dir(pred_log_dir, **_kwargs)
    gt_fig_data   = read_t2v_log_dir(gt_log_dir,   **_kwargs)

    if verbose:
        gt_n   = sum(len(v) for v in gt_fig_data.values())
        pred_n = sum(len(v) for v in pred_fig_data.values())
        if gt_n * pred_n > 10:
            print(f"GT series: {gt_n}, Pred series: {pred_n} — may take a while …")

    similarity_score = compare_figures(pred_fig_data, gt_fig_data, debug=verbose,
                                       normalize=reference_frame)

    if save_results:
        result = {
            "pred_log_dir":    pred_log_dir,
            "pred_summary":    summarize_fig_data(pred_fig_data, data_type="Prediction", verbose=False),
            "gt_log_dir":      gt_log_dir,
            "gt_summary":      summarize_fig_data(gt_fig_data,   data_type="Ground Truth", verbose=False),
            "similarity_score": similarity_score,
            "normalisation":    normalisation,
        }
        with open(eval_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=4)

    return similarity_score
