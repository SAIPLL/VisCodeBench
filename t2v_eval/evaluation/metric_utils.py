"""
Utility functions for reading and preprocessing T2V log directories.

Figure-data schema
------------------
The central data structure passed between functions is a ``defaultdict`` keyed
by axis ID.  Each value is a list of series dicts::

    {
        "<axis_id>": [
            {
                "plt_func": "<plotting function name>",
                "data": {
                    "x": np.ndarray,          # required
                    "y": np.ndarray,          # optional
                    "z": np.ndarray,          # optional
                    "t": np.ndarray,          # optional
                },
                "name":         "<series name>",
                "plot_func_id": "<matplotlib axes ID>",
            },
            ...
        ],
        ...
    }

Example
-------
::

    {
        "ax_1": [
            {
                "plt_func": "scatter",
                "data": {
                    "x": [0.1, 0.2, 0.3],
                    "y": [0.4, 0.5, 0.6],
                    "z": [0.7, 0.8, 0.9],   # marker size
                },
                "name": "Series 1",
            },
            {
                "plt_func": "plot",
                "data": {
                    "x": [0.1, 0.2, 0.3],
                    "y": [0.4, 0.5, 0.6],
                },
                "name": "Series 2",
            },
        ],
        "ax_2": [...],
    }
"""

import copy
import json
import math
import os
from collections import defaultdict

import numpy as np


def data_to_np_1d(ds: dict) -> np.ndarray:
    """Reshape ds["x"] to a column vector (N, 1)."""
    return ds["x"].reshape(-1, 1)


def data_to_np_2d(ds: dict) -> np.ndarray:
    """Stack ds["x"] and ds["y"] into an (N, 2) array."""
    arr = np.zeros((len(ds["x"]), 2))
    arr[:, 0] = ds["x"]
    if "y" in ds:
        arr[:, 1] = ds["y"]
    return arr


def data_to_np_3d(ds: dict) -> np.ndarray:
    """Stack ds["x"], ds["y"], ds["z"] into an (N, 3) array."""
    arr = np.zeros((len(ds["x"]), 3))
    arr[:, 0] = ds["x"]
    if "y" in ds:
        arr[:, 1] = ds["y"]
    if "z" in ds:
        arr[:, 2] = ds["z"]
    return arr


def data_to_np_4d(ds: dict) -> np.ndarray:
    """Stack ds["x"], ds["y"], ds["z"], ds["t"] into an (N, 4) array."""
    arr = np.zeros((len(ds["x"]), 4))
    arr[:, 0] = ds["x"]
    if "y" in ds:
        arr[:, 1] = ds["y"]
    if "z" in ds:
        arr[:, 2] = ds["z"]
    if "t" in ds:
        arr[:, 3] = ds["t"]
    return arr


def is_valid_value(value) -> bool:
    """Return True if *value* is a finite, non-None number (not NaN, not inf)."""
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return False


def check_nan_values(ax_data: list) -> bool:
    """Return True if any coordinate in ax_data contains a non-finite value."""
    for series in ax_data:
        data = series["data"]
        for key in ("x", "y", "z", "t"):
            for val in data.get(key, []):
                if not is_valid_value(val):
                    return True
    return False

def dedup_data(data, epsilon=1e-5):
    print("Deduplication has not implemented yet")
    return data

def check_discrete_values(array) -> bool:
    """Return True if *array* contains at least one string value (discrete axis)."""
    return any(isinstance(v, str) for v in array)


## Maps each raw plt_func to a canonical group name.
## "plot" and "pie" series are never merged — kept separate to preserve curve
## identity and each pie's own proportions.
_PLT_FUNC_GROUP: dict[str, str] = {
    # keep separate (2d)
    "plot":             "plot",
    "fill_between":     "plot",
    "fill_betweenx":    "plot",
    "stem":             "plot",
    "stairs":           "plot",
    # keep separate (1d)
    "pie":              "pie",
    # merge — 2d
    "bar":              "bar",
    "barh":             "bar",
    "hist":             "bar",
    "errorbar":         "bar",
    "bxp":              "box",
    # merge — 3d [x, y, z]
    "scatter":          "scatter",
    "sns_plot_strips":  "scatter",
    "sns_plot_swarms":  "scatter",
    "bar3d":            "bar3d",
    "pcolormesh":       "matrix",
    "imshow":           "matrix",
    "hexbin":           "matrix",
    "hist2d":           "matrix",
    "contour":          "contour",
    "contourf":         "contour",
    "plot_surface3d":   "surface3d",
    "geo":              "geo",
    # merge — 4d [x, y, z, t]
    "scatter3d":        "scatter3d",
}

## Coordinate keys to concatenate when merging, keyed by canonical group name.
_MERGE_KEYS: dict[str, list[str]] = {
    "bar":       ["x", "y"],
    "box":       ["x", "y"],
    "scatter":   ["x", "y", "z"],
    "bar3d":     ["x", "y", "z"],
    "matrix":    ["x", "y", "z"],
    "contour":   ["x", "y", "z"],
    "surface3d": ["x", "y", "z"],
    "geo":       ["x", "y", "z"],
    "scatter3d": ["x", "y", "z", "t"],
}


def group_fig_data_by_plt_func(fig_data: dict) -> dict:
    """
    Normalise plt_func names and merge same-type series within each axis.

    ``plot`` and ``pie`` series are kept separate (preserves individual curve
    identity, and lets each pie be scored as its own proportions).  All other types are merged into a single accumulated series per axis so
    that scatter/bar/etc. distances are computed on the full point cloud.
    """
    new_fig_data = defaultdict(list)

    for axis_id, series_list in fig_data.items():
        # accumulated[group] = single merged series dict (or list of "plot" series)
        accumulated: dict[str, list] = defaultdict(list)

        for series in series_list:
            group = _PLT_FUNC_GROUP[series["plt_func"]]

            if group in ("plot", "pie"):
                series["plt_func"] = group
                accumulated[group].append(series)
                continue

            if group not in accumulated:
                acc = copy.deepcopy(series)
                acc["plt_func"]     = group
                acc["name"]         = group
                acc["plot_func_id"] = "merged_" + group
                accumulated[group] = [acc]
            else:
                acc_data = accumulated[group][0]["data"]
                for key in _MERGE_KEYS[group]:
                    acc_data[key] = np.concatenate((acc_data[key], series["data"][key]))

        new_fig_data[axis_id] = [s for series_list in accumulated.values() for s in series_list]

    return new_fig_data

def create_interpolation_mappings_by_colors(
    color_z_info, verbose: bool = False
) -> dict:
    """
    Build a colour → z-value mapping by interpolating missing z values from colour
    darkness.

    Parameters
    ----------
    color_z_info : iterable of (avg_rgba, z_value, r, g, b, a)
        Each element describes one colour/z pair.  ``avg_rgba`` is the mean of
        the RGBA channels and is used as the interpolation x-axis.  ``z_value``
        may be NaN / None for entries whose z is unknown.
    verbose : bool
        Print before/after tables when True.

    Returns
    -------
    dict
        ``{(r, g, b, a): z_value}`` mapping with all NaNs filled by linear
        interpolation along the ``avg_rgba`` axis.
    """
    # sort by colour darkness (avg_rgba), then by z value
    color_z_info = sorted(color_z_info, key=lambda item: (item[0], item[1]))

    # collect valid z values; if none exist, anchor the endpoints for interpolation
    valid_z = [item[1] for item in color_z_info if is_valid_value(item[1])]
    if not valid_z:
        color_z_info[0]  = (color_z_info[0][0],  1,                  *color_z_info[0][2:])
        color_z_info[-1] = (color_z_info[-1][0], len(color_z_info),   *color_z_info[-1][2:])

    if verbose:
        print("Before interpolation:")
        for item in color_z_info:
            print(item)
        print("-" * 20)

    # interpolate NaN z values linearly along the avg_rgba axis
    xs = np.array([item[0] for item in color_z_info])
    ys = np.array([item[1] for item in color_z_info], dtype=float)
    mask      = ~np.isnan(ys)
    ys_interp = np.interp(xs, xs[mask], ys[mask])

    mappings: dict = {}
    for i, item in enumerate(color_z_info):
        z = item[1]
        # white (avg_rgba == 0) with no z → place just below the minimum known z
        if item[0] == 0 and not is_valid_value(z):
            z = (min(valid_z) - 1) if valid_z else 0
        # all other missing z values → use interpolated value
        if not is_valid_value(z):
            z = ys_interp[i]
        color_z_info[i] = (item[0], z, *item[2:])
        mappings[tuple(item[2:6])] = z

    if verbose:
        print("After interpolation:")
        for item in color_z_info:
            print(item)
        print("-" * 20)

    return mappings


def handle_nan_vals_in_geo_plots(fig_data: dict, verbose: bool = False) -> dict:
    """
    Resolve NaN z-values in geo-plot series by interpolating from colour darkness.

    Geopandas produces two broad patterns:

    **Case 1** — ``df.plot(target_column="…")``
        The target column may be NaN for countries with no data.  Float/int
        columns are logged directly as z; string columns are converted to
        ordinal integers.  NaN entries need to be filled before comparison.

    **Case 2** — ``df_filtered.plot(color="red")``
        No target column; all z values are NaN.  Countries are distinguished
        only by their ``plot_func_id`` assigned by matplotlib at logging time.

    In both cases the fix is to map each unique RGBA colour to an interpolated
    z value via :func:`create_interpolation_mappings_by_colors`, then write
    that value back into every matching series.

    Parameters
    ----------
    fig_data : dict
        Figure data as returned by :func:`parse_t2v_log_dir`.
    verbose : bool
        Pass through to :func:`create_interpolation_mappings_by_colors`.

    Returns
    -------
    dict
        The same *fig_data* dict with NaN z-values filled in-place.
    """
    for _, series_list in fig_data.items():
        # Each geo series represents one country:
        #   data["x"] / data["y"] — border coordinates
        #   data["z"]             — magnitude (may be NaN)
        #   data["color"]         — RGBA tuple
        color_z_info: set = set()
        for series in series_list:
            if series["plt_func"] != "geo":
                continue
            color   = series["data"].get("color", (1, 1, 1, 1))
            z       = series["data"]["z"][0]
            avg_rgba = sum(color) / len(color)
            color_z_info.add((avg_rgba, z, *color))

        if not color_z_info:
            continue

        color_to_z = create_interpolation_mappings_by_colors(color_z_info, verbose=verbose)

        for series in series_list:
            if series["plt_func"] != "geo":
                continue
            color    = series["data"].get("color", (1, 1, 1, 1))
            mapped_z = color_to_z.get(tuple(color[:4]))
            if mapped_z is not None:
                series["data"]["z"] = np.full(len(series["data"]["x"]), mapped_z, dtype=float)
            series["data"].pop("color", None)
            assert (
                len(series["data"]["x"])
                == len(series["data"]["y"])
                == len(series["data"]["z"])
            ), "After handling NaN z-values, x, y and z must have the same length."

    return fig_data

def filter_out_invalid_values(x, y=None, z=None, t=None):
    """
    Remove rows that contain at least one invalid value (None, NaN, or ±inf).

    Parameters
    ----------
    x : list or scalar
        Required coordinate array (or single value).
    y, z, t : list or scalar, optional
        Additional coordinate arrays.  Only non-None arrays are checked.

    Returns
    -------
    list
        Filtered *x* when only *x* is supplied.
    tuple of lists
        ``(x,)``, ``(x, y)``, ``(x, y, z)``, or ``(x, y, z, t)`` — matching
        whichever arguments were passed — with all invalid rows removed.
    """
    assert x is not None, "x cannot be None"

    def _to_list(v):
        return v if isinstance(v, list) else [v]

    coords = [_to_list(x)]
    for arr in (y, z, t):
        if arr is not None:
            coords.append(_to_list(arr))

    valid_rows = [
        row for row in zip(*coords)
        if all(is_valid_value(v) for v in row)
    ]

    if not valid_rows:
        filtered = tuple([] for _ in coords)
    else:
        filtered = tuple(list(col) for col in zip(*valid_rows))

    return filtered[0] if len(coords) == 1 else filtered

def is_color_bar_log(log_data: dict) -> bool:
    """
    Return True if *log_data* represents a colour-bar series, not real data.

    A colour-bar ``pcolormesh`` log has exactly one data series whose ``V``
    matrix consists entirely of single-element rows.

    Parameters
    ----------
    log_data : dict
        A single JSON log entry as loaded from the T2V log directory.
    """
    if log_data.get("plt_func") != "pcolormesh":
        return False
    data_series = log_data.get("data_series", [])
    if len(data_series) != 1:
        return False
    V = data_series[0].get("V")
    if not V:
        return False
    return all(len(v) == 1 for v in V)

# Log-file name prefixes that are metadata, not data series
_RESERVED_LOG_PREFIXES = ("variables", "evaluation", "execution")

# Compacted log format — one record per line. Produced by
# t2v_eval.logging.compact_log_dir.
_SERIES_JSONL_NAME = "series.jsonl"


def _parse_series(plt_func: str, series: dict, json_file: str) -> dict | None:
    """
    Parse one data-series entry into a standardised coordinate dict.

    Parameters
    ----------
    plt_func : str
        The plotting function name recorded in the log.
    series : dict
        Raw series dict from the JSON log.
    json_file : str
        Source file path — used in assertion messages only.

    Returns
    -------
    dict or None
        ``{"x": ndarray, …}`` on success; ``None`` for series types that carry
        no plottable data (``axvline`` / ``axhline``).

    Raises
    ------
    ValueError
        If *plt_func* is not recognised.
    """
    if plt_func in ("axvline", "axhline"):
        return None

    if plt_func == "pie":
        x = filter_out_invalid_values(series.get("x", []))
        return {"x": np.array(x, dtype=float)}

    if plt_func in ("plot", "bar", "barh", "hist",
                    "fill_between", "fill_betweenx",
                    "errorbar", "stem", "stairs", "bxp"):
        x, y = filter_out_invalid_values(series.get("x"), series.get("y"))
        assert len(x) == len(y), \
            f"x and y must have the same length. Check {json_file}"
        return {"x": np.array(x, dtype=float), "y": np.array(y, dtype=float)}

    if plt_func in ("scatter", "sns_plot_strips", "sns_plot_swarms"):
        raw_x = series.get("x")
        z = series.get("s")
        if z is None:
            z = [5] * len(raw_x)
        x, y, z = filter_out_invalid_values(raw_x, series.get("y"), z)
        assert len(x) == len(y) == len(z), \
            f"x, y, and z must have the same length. Check {json_file}"
        return {
            "x": np.array(x, dtype=float),
            "y": np.array(y, dtype=float),
            "z": np.array(z, dtype=float),
        }

    if plt_func in ("pcolormesh", "imshow", "contour", "contourf",
                    "plot_surface3d", "hexbin", "hist2d", "bar3d"):
        x, y, z = filter_out_invalid_values(
            series.get("x"), series.get("y"), series.get("z"))
        assert len(x) == len(y) == len(z), \
            f"x, y, and z must have the same length. Check {json_file}"
        return {
            "x": np.array(x, dtype=float),
            "y": np.array(y, dtype=float),
            "z": np.array(z, dtype=float),
        }

    if plt_func == "scatter3d":
        raw_x = series.get("x")
        t = series.get("s")
        if t is None:
            t = [5] * len(raw_x)
        x, y, z, t = filter_out_invalid_values(
            raw_x, series.get("y"), series.get("z"), t)
        assert len(x) == len(y) == len(z) == len(t), \
            f"x, y, z, and t must have the same length. Check {json_file}"
        return {
            "x": np.array(x, dtype=float),
            "y": np.array(y, dtype=float),
            "z": np.array(z, dtype=float),
            "t": np.array(t, dtype=float),
        }

    if plt_func == "geo":
        # x/y — border coordinates; z — magnitude (may be NaN); color — RGBA tuple
        x     = series.get("x")
        y     = series.get("y")
        z_raw = series.get("z")
        color = series.get("color")
        assert len(x) == len(y) == len(z_raw), \
            f"x, y, and z must have the same length. Check {json_file}"
        # Reduce each country's border polygon to a single centroid point;
        # use the mean RGBA as the z proxy (resolved later by handle_nan_vals_in_geo_plots)
        return {
            "x":     np.array([sum(x) / len(x)]         if x     else [], dtype=float),
            "y":     np.array([sum(y) / len(y)]         if y     else [], dtype=float),
            "z":     np.array([sum(color) / len(color)] if color else [], dtype=float),
            "color": color,
        }

    raise ValueError(f"Unsupported plt_func: {plt_func!r}")


def _ingest_record(json_data: dict,
                   axis_id: str,
                   plot_func_id: str,
                   source_label: str,
                   fig_data: defaultdict,
                   verbose: bool) -> bool:
    """
    Add the parsed series of one log record into *fig_data*.

    Returns True iff this record contained at least one geo series.
    """
    if is_color_bar_log(json_data):
        return False

    is_geo = False
    plt_func = json_data["plt_func"]
    for series in json_data.get("data_series", []):
        data = _parse_series(plt_func, series, source_label)
        if data is None:
            continue

        # linestyle-less plot → reclassify as scatter
        actual_plt_func = plt_func
        if plt_func == "plot" and str(series.get("linestyle", "None")) == "None":
            actual_plt_func = "scatter"
            data["z"] = np.full(len(data["x"]), 5, dtype=float)
            if verbose:
                print(f"Warning: {source_label} has a plot with no linestyle, "
                      f"treating it as scatter plot.")

        if actual_plt_func == "geo":
            is_geo = True

        fig_data[axis_id].append({
            "plt_func":     actual_plt_func,
            "data":         data,
            "name":         series.get("name", ""),
            "plot_func_id": plot_func_id,
        })

    return is_geo


def parse_t2v_log_dir(log_dir: str, verbose: bool = False) -> dict:
    """
    Parse a T2V log directory and return per-axis series data.

    Only the **compacted** log format is supported: ``<log_dir>/series.jsonl``
    must exist, with one JSON record per line, each carrying ``axis_id`` and
    ``plot_func_id`` fields. This is the format produced by
    :func:`t2v_eval.logging.compact_log_dir`.

    The legacy "loose" layout (many ``<fig_id>_<ax_id>_<ts>.json`` files in
    the directory) is no longer accepted. Use ``compact_log_dir`` to convert
    pre-existing log directories before scoring.

    Parameters
    ----------
    log_dir : str
        Path to the log directory produced by ``compact_log_dir``.
    verbose : bool
        Print a warning when a linestyle-less ``plot`` series is re-classified
        as ``scatter``.

    Returns
    -------
    defaultdict
        ``{axis_id: [series_dict, …]}`` — see module docstring for the schema.

    Raises
    ------
    FileNotFoundError
        If ``<log_dir>/series.jsonl`` is missing.
    """
    fig_data: defaultdict = defaultdict(list)
    is_having_geo_plot = False

    jsonl_path = os.path.join(log_dir, _SERIES_JSONL_NAME)
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(
            f"series.jsonl not found in {log_dir!r}. Maybe the program did not produce any charts? Or maybe the log directory is in the old format? "
            f"This loader only supports the compacted JSONL layout — run "
            f"t2v_eval.logging.compact_log_dir() to convert legacy log "
            f"directories if you used the old logging format."
        )

    with open(jsonl_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            json_data = json.loads(line)
            axis_id      = json_data.get("axis_id", "")
            plot_func_id = json_data.get("plot_func_id", "")
            source_label = f"{jsonl_path}:{lineno}"
            if _ingest_record(json_data, axis_id, plot_func_id,
                              source_label, fig_data, verbose):
                is_having_geo_plot = True

    if is_having_geo_plot:
        fig_data = handle_nan_vals_in_geo_plots(fig_data, verbose=verbose)

    return fig_data

def find_min_max(series_list: list) -> tuple:
    """
    Return the (min, max) range of each coordinate across all series.

    Parameters
    ----------
    series_list : list
        Series dicts as produced by :func:`parse_t2v_log_dir`.

    Returns
    -------
    tuple
        ``(min_x, max_x), (min_y, max_y), (min_z, max_z), (min_t, max_t)``
        — finite sentinels (``inf`` / ``-inf``) are preserved when a
        coordinate is absent from every series.
    """
    INF, NINF = float("inf"), float("-inf")
    bounds = {key: [INF, NINF] for key in ("x", "y", "z", "t")}

    for series in series_list:
        for key, (lo, hi) in bounds.items():
            arr = series["data"].get(key, [])
            if len(arr):
                bounds[key][0] = min(lo, arr.min())
                bounds[key][1] = max(hi, arr.max())

    return (
        tuple(bounds["x"]),
        tuple(bounds["y"]),
        tuple(bounds["z"]),
        tuple(bounds["t"]),
    )


def normalize_fig_data(fig_data: dict, epsilon: float = 1e-5) -> dict:
    """
    Apply min-max normalisation [0, 1] to all coordinate arrays in-place.

    Pie series are normalised by proportion (divided by their total sum).
    All other series are scaled per-axis using the global min/max of that axis.
    When min == max the range is widened by *epsilon* to avoid division by zero.

    Parameters
    ----------
    fig_data : dict
        Figure data as returned by :func:`parse_t2v_log_dir`.
    epsilon : float
        Small value added to the range when all values on an axis are identical.

    Returns
    -------
    dict
        The same *fig_data* dict, modified in-place.
    """
    for _, series_list in fig_data.items():
        (min_x, max_x), (min_y, max_y), (min_z, max_z), (min_t, max_t) = find_min_max(series_list)

        # widen degenerate ranges
        if min_x == max_x: max_x += epsilon
        if min_y == max_y: max_y += epsilon
        if min_z == max_z: max_z += epsilon
        if min_t == max_t: max_t += epsilon

        def _scale(arr, lo, hi):
            """Normalise *arr* to [0, 1] only when the range is finite and non-zero."""
            if lo < hi and lo != float("inf") and hi != float("-inf"):
                return (arr - lo) / (hi - lo)
            return arr

        for series in series_list:
            data = series["data"]
            if series["plt_func"] == "pie":
                total = np.sum(data["x"])
                if total > 0:
                    data["x"] = data["x"] / total
                continue
            for key, lo, hi in (
                ("x", min_x, max_x),
                ("y", min_y, max_y),
                ("z", min_z, max_z),
                ("t", min_t, max_t),
            ):
                if key in data:
                    data[key] = _scale(data[key], lo, hi)

    return fig_data


def filter_fig_data(fig_data: dict, dedup: bool = False) -> dict:
    """
    Drop empty series and optionally deduplicate points within each series.

    A ``plot`` series reduced to a single point is reclassified as ``scatter``.

    Parameters
    ----------
    fig_data : dict
        Figure data as returned by :func:`parse_t2v_log_dir`.
    dedup : bool
        Remove duplicate coordinate rows when True.

    Returns
    -------
    dict
        The same *fig_data* dict with empty series removed.
    """
    for axis_id, series_list in fig_data.items():
        kept = []
        for series in series_list:
            data = dedup_data(series["data"]) if dedup else series["data"]
            if series["plt_func"] == "plot" and len(data["x"]) == 1:
                series["plt_func"] = "scatter"
            if len(data["x"]) > 0:
                series["data"] = data
                kept.append(series)
        fig_data[axis_id] = kept
    return fig_data


def downsample_fig_data(fig_data: dict,
                        max_length: int = 20_000,
                        method: str = "random") -> dict:
    """
    Randomly subsample series that exceed *max_length* points.

    Parameters
    ----------
    fig_data : dict
        Figure data as returned by :func:`parse_t2v_log_dir`.
    max_length : int
        Maximum number of points to retain per series.
    method : str
        Downsampling strategy — only ``"random"`` is supported.

    Returns
    -------
    dict
        The same *fig_data* dict, modified in-place.

    Raises
    ------
    ValueError
        If *method* is not ``"random"``.
    """
    if method != "random":
        raise ValueError(f"Unsupported downsampling method: {method!r}")

    for _, series_list in fig_data.items():
        for series in series_list:
            data = series["data"]
            n = len(data.get("x", []))
            if n > max_length:
                indices = np.sort(np.random.choice(n, size=max_length, replace=False))
                series["data"] = {key: np.take(arr, indices) for key, arr in data.items()}

    return fig_data


def summarize_fig_data(fig_data: dict,
                       data_type: str = "Prediction",
                       verbose: bool = True,
                       print_points: int = 0) -> str:
    """
    Build a human-readable summary of *fig_data* and optionally print it.

    Parameters
    ----------
    fig_data : dict
        Figure data as returned by :func:`parse_t2v_log_dir`.
    data_type : str
        Label used in the header line (e.g. ``"Prediction"`` or ``"Ground Truth"``).
    verbose : bool
        Print the summary to stdout when True.
    print_points : int
        Number of individual data points to include per series (0 = none).

    Returns
    -------
    str
        The full summary string.
    """
    lines = [f"{data_type.capitalize()} Data Series: `{len(fig_data)}` subplots"]

    for axis_id, data_series in fig_data.items():
        lines.append(f"\tAxis {axis_id}: `{len(data_series)}` data series")
        for series in data_series:
            data = series["data"]
            lines.append(
                f"\t\tSeries: `{series['name']}`, "
                f"#Points: `{len(data['x'])}`, "
                f"plt_func: `{series['plt_func']}`"
            )
            for i in range(min(print_points, len(data["x"]))):
                coords = {
                    k: f"{data[k][i]:.2f}" if k in data else None
                    for k in ("x", "y", "z", "t")
                }
                lines.append(
                    f"\t\t\tPoint {i}: "
                    f"(x={coords['x']}, y={coords['y']}, z={coords['z']}, t={coords['t']})"
                )

        (min_x, max_x), (min_y, max_y), (min_z, max_z), (min_t, max_t) = find_min_max(data_series)
        is_nan = check_nan_values(data_series)
        lines += [
            "\t\tMin/Max Values:",
            f"\t\t\tX: ({min_x}, {max_x})",
            f"\t\t\tY: ({min_y}, {max_y})",
            f"\t\t\tZ: ({min_z}, {max_z})",
            f"\t\t\tT: ({min_t}, {max_t})",
            f"\t\t\tContains NaN values: {is_nan}",
        ]

    message = "\n".join(lines) + "\n"
    if verbose:
        print(message)
    return message

def read_t2v_log_dir(
    log_dir: str,
    dedup: bool = False,
    down_sampling: bool = False,
    down_sampling_max_length: int = 20_000,
    down_sampling_method: str = "random",
    verbose: bool = True,
    summarize_before: bool = False,
    summarize_after: bool = False,
    print_points: int = 0,
) -> dict:
    """
    Parse and preprocess a T2V log directory.

    Chains: parsing → grouping → (optional) downsampling → filtering.  This is the
    main entry point for loading logged figure data.  Values are returned as logged;
    the data-similarity metric normalises each prediction/reference axis pair itself
    (see ``ddata_similarity.compare_axes``).

    Parameters
    ----------
    log_dir : str
        Path to the log directory produced by ``activate_t2v_logging()``.
    dedup : bool
        Remove duplicate coordinate rows from each series.
    down_sampling : bool
        Subsample series that exceed *down_sampling_max_length* points.
    down_sampling_max_length : int
        Maximum points per series after downsampling.
    down_sampling_method : str
        Downsampling strategy — only ``"random"`` is supported.
    verbose : bool
        Print warnings and (optionally) data summaries.
    summarize_before : bool
        Print a data summary after parsing, before preprocessing.
    summarize_after : bool
        Print a data summary after all preprocessing steps.
    print_points : int
        Number of individual data points to show per series in summaries.

    Returns
    -------
    dict
        Preprocessed figure data — see module-level schema string for details.
    """
    fig_data  = parse_t2v_log_dir(log_dir, verbose=verbose)
    data_type = "Predicted" if "pred/" in log_dir else "Ground Truth"

    if verbose and summarize_before:
        summarize_fig_data(fig_data, data_type=data_type,
                           verbose=verbose, print_points=print_points)

    fig_data = group_fig_data_by_plt_func(fig_data)

    if down_sampling:
        fig_data = downsample_fig_data(
            fig_data,
            max_length=down_sampling_max_length,
            method=down_sampling_method,
        )

    fig_data = filter_fig_data(fig_data, dedup=dedup)

    if verbose and summarize_after:
        summarize_fig_data(fig_data, data_type=data_type,
                           verbose=verbose, print_points=print_points)

    return fig_data


if __name__ == "__main__":
    log_dir = "<log_dir_path>"
    read_t2v_log_dir(log_dir=log_dir, verbose=True)
