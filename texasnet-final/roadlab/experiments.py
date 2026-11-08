"""Reproducible route sampling, correctness checks and benchmark tables."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import statistics

import numpy as np
from scipy.sparse.csgraph import shortest_path

from .graph import RoadGraph
from .landmarks import Landmarks
from .search import bfs, astar, validate_path


def representative_pairs(graph: RoadGraph, seed: int = 42, anchors: int = 3) -> list[dict]:
    """Find reachable targets at near/mid/far *actual hop distances*.

    Source nodes are random non-landmarks to avoid artificially ideal A* results.
    Targets are deterministic for fixed graph and seed. Uses SciPy's C graph
    search for sampling; none of these distances are fed into the tested methods.
    """
    rng = np.random.default_rng(seed)
    giant = np.flatnonzero(graph.component == graph.largest_component)
    sources = rng.choice(giant, size=anchors, replace=False)
    csr = graph.as_csr()
    bins = [(10, 30, 'Local'), (100, 200, 'Regional'), (350, 650, 'Long-range')]
    output = []
    for src in sources:
        d = shortest_path(csr, directed=False, unweighted=True,
                          indices=int(src), return_predecessors=False)
        for low, high, label in bins:
            candidates = np.flatnonzero((d >= low) & (d <= high))
            if candidates.size == 0:
                # Some sources cannot reach the requested range: find closest
                # achievable positive hop distance instead of fabricating one.
                reachable = np.flatnonzero(np.isfinite(d) & (d > 0))
                target_dist = (low + high) // 2
                gap = np.abs(d[reachable] - target_dist)
                candidates = reachable[gap == gap.min()]
            target = int(rng.choice(candidates))
            output.append({'category': label, 'source': graph.external_id(int(src)),
                           'target': graph.external_id(target),
                           'reference_hops': int(d[target])})
    return output


def run_benchmark(graph: RoadGraph, landmarks: Landmarks,
                  pairs: list[dict], repeats: int = 1, include_ablation: bool = True) -> list[dict]:
    if repeats < 1:
        raise ValueError('repeats must be >= 1')
    rows = []
    for i, pair in enumerate(pairs, 1):
        src, dst = int(pair['source']), int(pair['target'])
        routines = [('BFS', lambda: bfs(graph, src, dst)),
                    ('A* (ALT)', lambda: astar(graph, src, dst, landmarks, True))]
        if include_ablation:
            routines.append(('A* (h=0)', lambda: astar(graph, src, dst, landmarks, False)))
        outcomes = {}
        for name, fn in routines:
            attempts = [fn() for _ in range(repeats)]
            result = attempts[0]
            if any(not validate_path(graph, r) for r in attempts):
                raise AssertionError(f'Invalid path from {name} for pair {i}')
            if any(r.hops != result.hops for r in attempts):
                raise AssertionError(f'Non-deterministic path cost for pair {i}')
            outcomes[name] = result
            rows.append({'case': i, 'category': pair.get('category', 'Unspecified'),
                         'source': src, 'target': dst,
                         'algorithm': name, 'found': result.found,
                         'hops': result.hops, 'expanded': result.expanded,
                         'discovered': result.discovered,
                         'peak_frontier': result.peak_frontier,
                         'median_ms': round(statistics.median(r.elapsed_ms for r in attempts), 3),
                         'search_array_mb': round(result.search_array_mb, 3),
                         'heuristic_at_source': result.heuristic_at_source})
        if len({v.hops for v in outcomes.values()}) != 1:
            raise AssertionError(f'BFS and A* disagree on optimal cost for pair {i}')
        if pair.get('reference_hops') is not None and outcomes['BFS'].hops != pair['reference_hops']:
            raise AssertionError(f'SciPy reference shortest path disagrees for pair {i}')
        print(f"Case {i}/{len(pairs)} ({pair.get('category','')}) {src}->{dst}: "
              f"{outcomes['BFS'].hops} hops; "
              + '; '.join(f'{k}={v.expanded:,} expanded' for k,v in outcomes.items()), flush=True)
    return rows


def write_benchmark(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_pairs(pairs: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pairs, indent=2), encoding='utf-8')
