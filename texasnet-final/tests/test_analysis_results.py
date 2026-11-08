"""Regression checks for the notebook-to-dashboard result reporting layer."""
from pathlib import Path
import pandas as pd
import pytest

from roadlab.analysis_results import summarize_accuracy, load_analysis


def test_accuracy_is_calculated_not_hardcoded():
    samples = pd.DataFrame([
        dict(algorithm='BFS', case_id=1, repetition=1, found=True,
             path_valid=True, optimal=True, returned_hops=4, reference_hops=4),
        dict(algorithm='BFS', case_id=2, repetition=1, found=True,
             path_valid=True, optimal=False, returned_hops=5, reference_hops=4),
    ])
    outcome = summarize_accuracy(samples).iloc[0]
    assert outcome['Executions'] == 2
    assert outcome['Optimal paths'] == 1
    assert outcome['Optimal-path accuracy (%)'] == 50.0


def test_invalid_boolean_rejected():
    samples = pd.DataFrame([
        dict(algorithm='BFS', case_id=1, repetition=1, found='unknown',
             path_valid=True, optimal=True, returned_hops=4, reference_hops=4),
    ])
    with pytest.raises(ValueError):
        summarize_accuracy(samples)


def test_missing_csvs_reported(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_analysis(tmp_path)
