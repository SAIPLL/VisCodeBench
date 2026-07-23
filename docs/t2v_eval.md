# t2v_eval — Evaluation Library

The `t2v_eval` package is the scoring engine behind the
[VisCodeBench leaderboard](https://www.codabench.org/competitions/16125/). It
logs the data that plotting libraries actually render, then compares predicted
visualisations against reference visualisations using data-centric metrics. The
logger works by monkey-patching common plotting APIs from matplotlib, seaborn,
pandas, and geopandas, appending each captured plotting call to a compact
`series.jsonl` file.

## Installation

```bash
pip install -e .
```

Requires Python 3.10 to 3.12. Core dependencies are listed in
`requirements.txt`.

## Logging a visualisation

Activate logging before running the visualisation code you want to capture:

```python
from t2v_eval.logging import activate_t2v_logging, setup_log_dir

LOG_DIR = "logs/example"
setup_log_dir(LOG_DIR)
activate_t2v_logging(LOG_DIR)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter([1, 2, 3], [4, 5, 6])
plt.show()
```

After execution, the log directory contains:

```text
logs/example/
├── series.jsonl
└── variables.json
```

`series.jsonl` is written directly while plotting. You do not need a separate
compaction step for current logs.

## Scoring predictions against references

Compare a predicted log directory with a reference log directory:

```python
from t2v_eval.evaluation import (
    calculate_similarity_score,
    calculate_chart_type_accuracy,
    calculate_accessibility_score,
)

similarity = calculate_similarity_score(
    pred_log_dir="logs/pred",
    gt_log_dir="logs/reference",
)

chart_type_ok = calculate_chart_type_accuracy(
    pred_log_dir="logs/pred",
    gt_log_dir="logs/reference",
)

print(similarity)
print(chart_type_ok)
```

`calculate_similarity_score` returns a float in `[0, 1]` (higher is better) based
on **Displayed Data Similarity**, a flexible, LLM-independent measure of how
closely the generated chart reproduces the reference data points.
`calculate_chart_type_accuracy` checks whether the predicted chart types cover
the reference chart types. `calculate_accessibility_score` computes image-level
**Accessibility** signals such as text contrast and spacing.

Useful `calculate_similarity_score` options include:

| Parameter | Default | Description |
| --- | --- | --- |
| `down_sampling` | `True` | Downsample long series before comparison. |
| `down_sampling_max_length` | `5000` | Maximum points per series after downsampling. |
| `use_cache` | `False` | Reuse a cached `evaluation.json` result if available. |
| `save_results` | `False` | Save the computed result to `evaluation.json`. |
| `verbose` | `False` | Print additional debugging information. |

## JSONL Format

Each line in `series.jsonl` is one JSON record for one plotting call:

```json
{
  "axis_id": "<fig_id>_<ax_id>",
  "plot_func_id": "<timestamp>",
  "t2v_eval_version": "0.2.2",
  "plt_func": "scatter",
  "args": [],
  "kargs": {
    "ax_col": 0,
    "ax_row": 0
  },
  "data_series": [
    {
      "x": [1, 2, 3],
      "y": [4, 5, 6]
    }
  ]
}
```

The exact `data_series` keys depend on the chart type. Scatter plots may include
marker sizes and colors, box plots include summary statistics, and gridded
charts include matrix-like values.

## Main APIs

### Logging

```python
from t2v_eval.logging import (
    activate_t2v_logging,
    setup_log_dir,
    reset_log_dir,
    iter_series_records,
)
```

| Function | Description |
| --- | --- |
| `setup_log_dir(log_dir)` | Set and reset the active log directory. |
| `activate_t2v_logging(log_dir)` | Enable monkey-patched logging for supported plotting libraries. |
| `reset_log_dir()` | Reset logging to the default `/tmp/t2vlog` directory. |
| `iter_series_records(log_dir)` | Stream records from `<log_dir>/series.jsonl`. |

### Evaluation

```python
from t2v_eval.evaluation import (
    calculate_similarity_score,
    calculate_chart_type_accuracy,
    calculate_accessibility_score,
)
```

| Function | Description |
| --- | --- |
| `calculate_similarity_score(pred_log_dir, gt_log_dir, ...)` | Compare displayed data series and return a score in `[0, 1]`. |
| `calculate_chart_type_accuracy(pred_log_dir, gt_log_dir, ...)` | Check whether predicted chart types cover reference chart types. |
| `calculate_accessibility_score(image_path, ...)` | Compute image-level accessibility signals such as contrast and text spacing. |

## Supported Plotting Libraries

| Library | Examples of captured charts |
| --- | --- |
| matplotlib | line, scatter, bar, histogram, pie, box, violin, errorbar, eventplot, image, contour, pcolormesh, hexbin, 3D scatter, 3D bar, surface |
| seaborn | relational, categorical, distribution, regression, FacetGrid, PairGrid, pairplot, relplot, catplot, displot, lmplot |
| pandas | DataFrame/Series plots including line, area, bar, scatter, box, hist, KDE, pie, hexbin |
| geopandas | GeoDataFrame plots such as choropleths and heatmaps |

## Legacy Logs

Older versions of `t2v_eval` wrote one loose JSON file per plotting call.
Current logging writes `series.jsonl` directly. To convert an old log directory:

```python
from t2v_eval.logging import compact_log_dir

compact_log_dir("logs/old_format")
```

## Tests

Run the logging test suite with:

```bash
pytest tests/t2v_logging/ -v
```

To run a smaller subset:

```bash
pytest tests/t2v_logging/test_matplotlib_pairwise.py -v
pytest tests/t2v_logging/test_seaborn_relational.py -v
pytest tests/t2v_logging/test_geopandas.py -v
```
