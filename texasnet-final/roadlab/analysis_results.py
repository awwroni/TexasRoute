"""Read and validate the final standalone Colab experiment exports.

No metrics are hardcoded. Accuracy is recomputed from every recorded query
execution; the notebook's independent SciPy references validate route cost.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED_FILES = {
    'trials': 'all_162_timing_trials.csv',
    'routes': 'per_route_algorithm_results.csv',
    'pairs': 'paired_BFS_vs_Astar.csv',
    'environment': 'environment_and_runtime.csv',
    'distance': 'route_distance_summary.csv',
    'special': 'special_case_correctness.csv',
    'dataset': 'dataset_profile.csv',
    'memory': 'working_memory_summary.csv',
    'unique_memory': 'rss_profile_sample.csv',
    'statistics': 'statistical_summary.csv',
}


def _boolean_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Enforce explicit boolean CSV data: no truthiness of nonempty strings."""
    if name not in frame:
        raise ValueError(f'Missing benchmark column: {name}')
    normalized = frame[name].astype(str).str.strip().str.lower()
    if not normalized.isin(['true', 'false', '1', '0']).all():
        raise ValueError(f'Benchmark column {name} contains non-boolean values')
    return normalized.isin(['true', '1'])


def summarize_accuracy(trials: pd.DataFrame) -> pd.DataFrame:
    """Accuracy = valid, reachable optimal results / evaluated reachable trials.

    Disconnected-case handling is checked separately because a no-path answer
    should not be counted as a failed reachable shortest-path query.
    """
    required = {'algorithm', 'case_id', 'repetition', 'found', 'path_valid',
                'optimal', 'returned_hops', 'reference_hops'}
    missing = required - set(trials.columns)
    if missing:
        raise ValueError(f'Missing trial columns: {sorted(missing)}')

    results = trials.copy()
    results['_found'] = _boolean_column(results, 'found')
    results['_valid'] = _boolean_column(results, 'path_valid')
    results['_optimal'] = _boolean_column(results, 'optimal')
    results['_hops_match'] = pd.to_numeric(results['returned_hops'], errors='coerce').eq(
        pd.to_numeric(results['reference_hops'], errors='coerce'))
    if results.duplicated(subset=['algorithm', 'case_id', 'repetition']).any():
        raise ValueError('Duplicate algorithm/case/repetition rows in benchmark')
    results['_success'] = (results['_found'] & results['_valid'] &
                           results['_optimal'] & results['_hops_match'])
    records = []
    for algorithm, group in results.groupby('algorithm', sort=True):
        executions = len(group)
        records.append({
            'Algorithm': algorithm,
            'Queries': int(group['case_id'].nunique()),
            'Executions': executions,
            'Valid routes': int((group['_found'] & group['_valid']).sum()),
            'Optimal paths': int(group['_success'].sum()),
            'Optimal-path accuracy (%)': 100.0 * group['_success'].sum() / executions,
            'Valid-route rate (%)': 100.0 * (group['_found'] & group['_valid']).sum() / executions,
        })
    return pd.DataFrame(records)


def load_analysis(directory: Path) -> dict:
    """Validate consistency of notebook outputs before displaying them."""
    directory = Path(directory)
    missing = [v for v in REQUIRED_FILES.values() if not (directory / v).is_file()]
    if missing:
        raise FileNotFoundError('Missing final Colab exports: ' + ', '.join(missing))
    frames = {name: pd.read_csv(directory / filename)
              for name, filename in REQUIRED_FILES.items()}
    accuracy = summarize_accuracy(frames['trials'])
    pairs = frames['pairs']
    routes = frames['routes']
    trials = frames['trials']
    if pairs.empty or routes.empty or trials.empty:
        raise ValueError('Final Colab results contain empty benchmark tables')
    if pairs['case_id'].duplicated().any():
        raise ValueError('Paired comparison contains duplicate route IDs')
    if set(pairs['case_id']) != set(trials['case_id']) or set(pairs['case_id']) != set(routes['case_id']):
        raise ValueError('Trial, route and paired-comparison case IDs disagree')
    if set(accuracy['Algorithm']) != set(routes['algorithm']):
        raise ValueError('Trial and route algorithm labels disagree')
    if (trials.groupby(['case_id', 'algorithm']).size().nunique() != 1):
        raise ValueError('Trials do not have equal repetitions per route and algorithm')
    if not _boolean_column(frames['special'], 'correct').all():
        raise ValueError('A recorded special-case correctness test has failed')
    for name in ('speedup_BFS_over_Astar', 'expanded_reduction_pct'):
        if not pd.to_numeric(pairs[name], errors='coerce').notna().all():
            raise ValueError(f'Paired benchmark contains an invalid {name}')
    frames['accuracy'] = accuracy
    return frames
