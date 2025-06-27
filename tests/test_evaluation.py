import numpy as np
import pytest
from pathlib import Path

from brfss_diabetes.evaluation import find_optimal_threshold, plot_classification_report

# Sample binary classification data
y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1])
y_probs = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.3, 0.4, 0.85, 0.05, 0.95])
y_pred = (y_probs >= 0.5).astype(int)


def test_find_optimal_threshold_valid():
    threshold = find_optimal_threshold(y_true, y_probs, beta=1.0)
    assert isinstance(threshold, float)
    assert 0 <= threshold <= 1


def test_find_optimal_threshold_invalid_beta():
    with pytest.raises(ValueError):
        find_optimal_threshold(y_true, y_probs, beta=0)


def test_find_optimal_threshold_mismatched_shapes():
    with pytest.raises(ValueError):
        find_optimal_threshold(y_true, y_probs[:-1])


def test_find_optimal_threshold_wrong_dtype():
    with pytest.raises(ValueError):
        find_optimal_threshold(y_true.astype(float), y_probs)


def test_find_optimal_threshold_non_float_probs():
    y_true = np.array([0, 1, 0, 1])
    y_probs = np.array([0, 1, 1, 0])  # Integers, not floats

    with pytest.raises(ValueError, match="must be an array of floats"):
        find_optimal_threshold(y_true, y_probs, beta=1.0)


def test_find_optimal_threshold_invalid_probs_gt_one():
    y_true = np.array([0, 1, 0, 1])
    y_probs = np.array([0.2, 0.8, 0.5, 1.2])  # Invalid (> 1)
    with pytest.raises(ValueError, match="y_probs.*outside"):
        find_optimal_threshold(y_true, y_probs)


def test_find_optimal_threshold_invalid_probs_lt_zero():
    y_true = np.array([0, 1, 1, 0])
    y_probs = np.array([0.3, -0.1, 0.6, 0.4])  # Invalid (< 0)
    with pytest.raises(ValueError, match="y_probs.*outside"):
        find_optimal_threshold(y_true, y_probs)


def test_plot_classification_report_valid(tmp_path):
    save_path = tmp_path / "test_classification_report.png"
    plot_classification_report(y_true, y_pred, title="Test Report", save_path=save_path)
    assert save_path.exists()


def test_plot_classification_report_mismatched_shapes():
    with pytest.raises(ValueError):
        plot_classification_report(y_true, y_pred[:-1])


def test_plot_classification_report_wrong_dtype():
    with pytest.raises(ValueError):
        plot_classification_report(y_true.astype(float), y_pred)


def test_plot_classification_report_bad_savepath():
    with pytest.raises(ValueError):
        plot_classification_report(y_true, y_pred, save_path=12345)


def test_plot_classification_report_no_save(monkeypatch):
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])

    # Patch plt.show so it doesn't actually render a popup window
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    # This should run silently and hit the `else: plt.show()` block
    plot_classification_report(y_true, y_pred, save_path=None)
