"""
Tests for the axis-frame normalisation in the Displayed Data Similarity metric.

The metric maps each (prediction, reference) axis pair into a frame before scoring:

- ``"reference_frame"`` (default) uses the reference axis's range for both axes, so
  a prediction with the wrong scale, unit or offset is penalised;
- ``"self_frame"`` maps each axis into its own range, which ignores scale and offset.

Run with ``pytest tests/evaluation -v``.
"""

import numpy as np
import pytest

from t2v_eval.evaluation.ddata_similarity import (
    axis_frame,
    calculate_similarity_score,
    compare_axes,
    compare_figures,
    to_frame,
)
from t2v_eval.evaluation.metric_utils import group_fig_data_by_plt_func

REF = [12.0, 15.0, 11.0, 18.0, 22.0, 20.0]
XS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def _series(plt_func, x, y=None, name="s"):
    data = {"x": np.asarray(x, dtype=float)}
    if y is not None:
        data["y"] = np.asarray(y, dtype=float)
    return {"plt_func": plt_func, "name": name, "data": data}


def _bar(ys):
    return [_series("bar", XS, ys)]


def _line(ys):
    return [_series("plot", XS, ys)]


# ---------------------------------------------------------------------------
# axis_frame / to_frame
# ---------------------------------------------------------------------------

def test_axis_frame_maps_reference_into_unit_range():
    frame = axis_frame(_bar(REF))
    assert frame["x"] == (0.0, 5.0)
    assert frame["y"] == (11.0, 11.0)
    out = to_frame(_bar(REF), frame)[0]["data"]
    assert out["y"].min() == pytest.approx(0.0)
    assert out["y"].max() == pytest.approx(1.0)


def test_axis_frame_skips_absent_coordinates():
    frame = axis_frame(_bar(REF))
    assert "z" not in frame and "t" not in frame


def test_axis_frame_constant_coordinate():
    # A flat line has no range: centred on its value, scaled by its magnitude.
    frame = axis_frame(_line([5.0] * 6))
    assert frame["y"] == (5.0, 5.0)
    zero = axis_frame(_line([0.0] * 6))
    assert zero["y"] == (0.0, 1.0)


def test_axis_frame_ignores_pie_series():
    series = _bar(REF) + [_series("pie", [100.0, 300.0], name="pie")]
    assert axis_frame(series)["x"] == (0.0, 5.0)


def test_to_frame_maps_pie_to_own_proportions_and_copies():
    pie = [_series("pie", [1.0, 3.0])]
    out = to_frame(pie, {"x": (0.0, 10.0)})
    np.testing.assert_allclose(out[0]["data"]["x"], [0.25, 0.75])
    np.testing.assert_allclose(pie[0]["data"]["x"], [1.0, 3.0])   # input untouched


# ---------------------------------------------------------------------------
# compare_axes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("normalisation", ["reference_frame", "self_frame"])
def test_identical_axes_score_one(normalisation):
    assert compare_axes(_bar(REF), _bar(REF), normalisation=normalisation) == pytest.approx(1.0)
    assert compare_axes(_line(REF), _line(REF), normalisation=normalisation) == pytest.approx(1.0)


@pytest.mark.parametrize("name, pred", [
    ("sum instead of mean", [v * 6 for v in REF]),
    ("fraction instead of percent", [v / 100 for v in REF]),
    ("offset by 50", [v + 50 for v in REF]),
])
def test_scale_and_offset_errors(name, pred):
    # Penalised in the reference frame, ignored in each axis's own frame.
    assert compare_axes(_bar(pred), _bar(REF)) == pytest.approx(0.0, abs=1e-9), name
    assert compare_axes(_bar(pred), _bar(REF), normalisation="self_frame") == pytest.approx(1.0), name


def test_doubled_line():
    pred = [2 * v for v in REF]
    assert compare_axes(_line(pred), _line(REF)) == pytest.approx(0.0, abs=1e-9)
    assert compare_axes(_line(pred), _line(REF), normalisation="self_frame") == pytest.approx(1.0)


def test_small_noise_keeps_a_high_score():
    rng = np.random.default_rng(0)
    pred = list(np.asarray(REF) + rng.uniform(-0.2, 0.2, len(REF)))
    score = compare_axes(_line(pred), _line(REF))
    assert 0.9 < score < 1.0


def test_flat_reference_line():
    flat = [5.0] * 6
    assert compare_axes(_line(flat), _line(flat)) == pytest.approx(1.0)
    assert compare_axes(_line([10.0] * 6), _line(flat)) == pytest.approx(0.0, abs=1e-9)


def test_compare_axes_does_not_modify_inputs():
    pred, gt = _bar([v * 6 for v in REF]), _bar(REF)
    compare_axes(pred, gt)
    np.testing.assert_allclose(gt[0]["data"]["y"], REF)
    np.testing.assert_allclose(pred[0]["data"]["y"], [v * 6 for v in REF])


def test_invalid_normalisation_raises():
    with pytest.raises(ValueError, match="normalisation"):
        compare_axes(_bar(REF), _bar(REF), normalisation="self")
    with pytest.raises(ValueError, match="normalisation"):
        compare_figures({0: _bar(REF)}, {0: _bar(REF)}, normalisation="none")


# ---------------------------------------------------------------------------
# Grouping: pies are kept as their own series
# ---------------------------------------------------------------------------

def test_pie_series_are_not_merged():
    fig_data = {0: [_series("pie", [1.0, 1.0], name="a"), _series("pie", [1.0, 3.0], name="b")]}
    grouped = group_fig_data_by_plt_func(fig_data)
    assert [s["name"] for s in grouped[0]] == ["a", "b"]


def test_other_series_are_still_merged():
    fig_data = {0: [_series("bar", [0, 1], [1, 2], name="a"), _series("barh", [2, 3], [3, 4], name="b")]}
    grouped = group_fig_data_by_plt_func(fig_data)
    assert len(grouped[0]) == 1
    np.testing.assert_allclose(grouped[0][0]["data"]["x"], [0, 1, 2, 3])


# ---------------------------------------------------------------------------
# End to end: logged figures through calculate_similarity_score
# ---------------------------------------------------------------------------

def _bars(ys):
    return lambda ax: ax.bar(XS, ys)


def _plot(ys):
    return lambda ax: ax.plot(XS, ys)


@pytest.mark.parametrize("draw_ref, draw_pred, expected", [
    (_bars(REF), _bars(REF), 1.0),
    (_bars(REF), _bars([v * 6 for v in REF]), 0.0),
    (_plot(REF), _plot(REF), 1.0),
    (_plot(REF), _plot([2 * v for v in REF]), 0.0),
])
def test_calculate_similarity_score_end_to_end(log_figure, draw_ref, draw_pred, expected):
    gt_dir = log_figure("gt", draw_ref)
    pred_dir = log_figure("pred", draw_pred)
    score = calculate_similarity_score(pred_dir, gt_dir)
    assert score == pytest.approx(expected, abs=1e-6)


def test_calculate_similarity_score_self_frame_ignores_scale(log_figure):
    gt_dir = log_figure("gt", _bars(REF))
    pred_dir = log_figure("pred", _bars([v * 6 for v in REF]))
    assert calculate_similarity_score(pred_dir, gt_dir, normalisation="self_frame") == pytest.approx(1.0)


def test_cache_is_keyed_by_normalisation(log_figure):
    gt_dir = log_figure("gt", _bars(REF))
    pred_dir = log_figure("pred", _bars([v * 6 for v in REF]))
    self_score = calculate_similarity_score(pred_dir, gt_dir, normalisation="self_frame",
                                            save_results=True)
    ref_score = calculate_similarity_score(pred_dir, gt_dir, use_cache=True)
    assert self_score == pytest.approx(1.0)
    assert ref_score == pytest.approx(0.0, abs=1e-6)   # not served from the self_frame cache


def test_calculate_similarity_score_rejects_unknown_normalisation(log_figure):
    gt_dir = log_figure("gt", _bars(REF))
    with pytest.raises(ValueError, match="normalisation"):
        calculate_similarity_score(gt_dir, gt_dir, normalisation="reference")
