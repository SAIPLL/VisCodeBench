# Normalisation in Displayed Data Similarity

This note explains how **Displayed Data Similarity** (`calculate_similarity_score`)
normalises chart data, the limitation that follows from it, and why we decided
not to switch to "reference-frame" normalisation after evaluating it on real
predictions.

Related discussion: [issue #1](https://github.com/SAIPLL/VisCodeBench/issues/1)
and [pull request #2](https://github.com/SAIPLL/VisCodeBench/pull/2).

## Summary

- Each subplot (axes) is min-max normalised to `[0, 1]` **using its own range**,
  separately for the prediction and the reference.
- As a result, the score ignores a **uniform scale or offset error within an axis**,
  such as plotting a sum instead of a mean, or every value shifted by +50.
  This is a known limitation.
- Normalising the prediction with the **reference's** range (the
  "reference frame") would catch those errors. But on our data it penalises far
  more correct charts than it catches real errors. Many logged coordinates are
  layout positions or styling, not data values.
- **Decision:** keep per-axis self-normalisation and document the limitation.

## How normalisation works

`read_t2v_log_dir` (in `t2v_eval/evaluation/metric_utils.py`) calls
`normalize_fig_data` when it loads a log directory:

- For every axes, the min and max of each coordinate (`x`, `y`, `z`, `t`) are
  taken over all series in that axes, and each coordinate is scaled to `[0, 1]`.
  A constant coordinate has its range widened by a small epsilon.
- Pie series are divided by their own total, so they become proportions.
- This runs independently on the prediction and the reference, before the two are
  paired by `compare_figures` / `compare_axes`.

The score therefore compares the **shape** of the data in each axes: relative
positions within the axes, not absolute values.

## Known limitation: scale and offset errors

Because each side is normalised by its own range, any error that stretches or
shifts a whole axis disappears before comparison. With reference bar values
`[12, 15, 11, 18, 22, 20]`, all of these predictions score **1.000**:

| Prediction | Score |
|---|---|
| sum instead of mean (5 rows per group) | 1.000 |
| fractions where percentages were asked | 1.000 |
| every value +50 | 1.000 |
| line with every value doubled | 1.000 |

Errors that change the *relative* values within an axis, such as a wrong value
for one category, a wrong trend, or missing points, are still penalised.

## Alternative evaluated: reference-frame normalisation

[PR #2](https://github.com/SAIPLL/VisCodeBench/pull/2) proposed moving
normalisation into `compare_axes` and mapping each (prediction, reference) axes
pair into the **reference axes' frame**. Scale and offset errors then show up as
large distances. On the examples above, all four predictions drop to 0.000.

### Evaluation on real predictions

We scored 454 `arxiv-t2v-easy` predictions from one model run in both modes,
against the reference series. To find the cause of each change, we also
re-scored every sample with **one coordinate at a time** switched to the
reference frame.

| | Self frame (current) | Reference frame |
|---|---|---|
| Mean score | **0.86** | **0.73** |
| Samples whose score drops by more than 0.05 | | 124 (92 by more than 0.2; 13 from 1.000 to 0.000) |
| Samples whose score rises by more than 0.05 | | 12 |

Where the lost score comes from (the 124 drops):

| Coordinate causing the drop | Samples | Share of lost score | What the coordinate really is |
|---|---|---|---|
| `x` | 76 (bar 47, line 19, box 6, scatter 3) | 58% | **Positions**, not values |
| `z` | 34 (33 scatter) | 30% | **Marker size** `s`, a style setting |
| `y` | 13 | 10% | A mix of real errors and display-equivalent values |
| `t` | 1 | 2% | Marker size (scatter3d) |

About **88% of the lost score comes from correct charts**. Real scale or unit
errors, the case reference-frame scoring is meant to catch, appear in only a
handful of samples.

### Why the logged coordinates are not always data values

Coordinates in the logs play three different roles:

1. **Values**: bar heights, line y-values, numeric scatter x/y. Comparing these
   in the reference frame is correct.
2. **Positions**: the number only says where matplotlib drew the mark; the meaning
   is in the tick labels. Examples seen in the data:
   - bar positions `0..5` vs `1..6`, or different grouped-bar offsets;
   - category slots `0..5` labelled `16..512`, vs the real values `16..512` on a
     log2 axis;
   - dates as an hour index `0..8759` vs matplotlib date numbers;
   - radar angles with a different start or direction.
3. **Style**: scatter marker size is logged as `z` (`s=36` vs `s=65`), so
   bigger dots count as different data.

Some **values** are also logged differently while displaying the same thing:
fractions shown with a percent formatter vs values logged as `0..100`, or radar
values shifted by a constant with relabelled radial ticks.

Self-normalisation happens to tolerate most of these, because any linear
relabelling of positions or a constant marker size maps to the same `[0, 1]`
values. Reference-frame normalisation turns each of them into an error.
Mahalanobis distance (used for bar and scatter) then amplifies it. A reference
coordinate with zero variance, such as a constant marker size, gives every
predicted point the maximum cost. That is why many of these charts drop all the
way to 0.000.

### Where the reference frame does help

All 12 samples whose score **rose** come from `y`. In each, one side has an extra
series the other lacks (for example a horizontal reference line). Under
self-normalisation, that series changes the side's own `y` range and shifts the
matching series. The reference frame doesn't have this problem. Applying the
reference frame to `y` only gives a mean of 0.85 (19 drops, 14 rises), but about
half of those `y` drops are still display-equivalent (fractions vs percent).

### Reproduction

The following reproduces the two most common false penalties. It needs the
`normalisation` option from the PR #2 branch (`gt-frame-normalisation`), which is
not on `main`. Run it from the repository root on that branch.

```python
import sys; sys.path.insert(0, ".")
import numpy as np
from t2v_eval.evaluation.ddata_similarity import compare_figures


def series(plt_func, **data):  # one series, in the format read_t2v_log_dir returns
    return {"plt_func": plt_func, "name": plt_func, "plot_func_id": plt_func,
            "data": {k: np.asarray(v, dtype=float) for k, v in data.items()}}


def report(title, pred, ref):
    s = {m: compare_figures(pred, ref, normalisation=m) for m in ("self_frame", "reference_frame")}
    print(f"{title:50s} self_frame={s['self_frame']:.3f}  reference_frame={s['reference_frame']:.3f}")


# Case 1: x is a position. Reference puts K at slots 0..5 with tick labels 16..512;
# the prediction plots the real K (16..512) on a log2 axis. The charts look the same.
K = [16, 32, 64, 128, 256, 512]
mem = {"QT-CP": [0.1, 0.3, 0.8, 1.0, 2.6, 7.6], "QG-CP": [0.2, 0.5, 1.0, 2.3, 6.8, 22.0]}
report("Case 1: category slots vs real K on log2 axis",
       {"ax": [series("plot", x=K, y=y) for y in mem.values()]},
       {"ax": [series("plot", x=range(6), y=y) for y in mem.values()]})

# Case 2: scatter marker size is logged as z. Identical x/y; only s differs (36 vs 65).
rng = np.random.default_rng(0)
x, y = rng.uniform(500, 3700, 16), rng.uniform(62, 79, 16)
report("Case 2: identical scatter, marker size 36 vs 65",
       {"ax": [series("scatter", x=x, y=y, z=np.full(16, 65))]},
       {"ax": [series("scatter", x=x, y=y, z=np.full(16, 36))]})
```

Output:

```
Case 1: category slots vs real K on log2 axis      self_frame=0.909  reference_frame=0.000
Case 2: identical scatter, marker size 36 vs 65    self_frame=1.000  reference_frame=0.000
```

Case 1 is not perfect under self-normalisation either (0.909). Slots `0..5` and
values `16..512` are not linearly related, so min-max scaling cannot line them up
exactly.

## Decision

We keep per-axis self-normalisation. Scale and offset errors are rare in our
datasets, while position and style coordinates are common. Switching to the
reference frame would lower the scores of many correct charts to catch a few
real errors.

## What a better approach would need

Scoring absolute values correctly would require knowing the **role of each
coordinate** in each axes, rather than treating every logged number as data:

- compare positional axes (bars, boxes, categories, radar angles) by **tick label**,
  not by coordinate;
- drop marker size and colour unless the chart maps a variable to them
  (bubble charts, colour scales);
- convert dates to one common unit;
- use the axis **scale and formatter** (log scale, percent formatter, offset text)
  to recover the displayed values.

The current logs record raw coordinates and only part of this axis information,
so this would also need changes to the logger.

## Other known issue

- **Line error weighting.** For line charts, `get_cont_recall_new` divides each
  point's error by `hypot(g_x, g_y)`, the point's distance from the bottom-left
  corner of the normalised plot. The same absolute error therefore costs more near
  the bottom-left than near the top-right. For example, a line shifted up by 0.05
  scores about 0.83 near `y = 0.05` and about 0.96 near `y = 0.95`. Dividing by a
  constant would remove this, but it would change every line score, so it is left
  for a future metric version.
