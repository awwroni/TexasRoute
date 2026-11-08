"""Real-network regression checks, independent of any visualization or report."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .graph import RoadGraph
from .landmarks import Landmarks
from .search import bfs, astar, validate_path
from .cli import DEFAULT_GRAPH, DEFAULT_LANDMARKS, DEFAULT_CASES


def verify(seed: int = 17, edge_samples: int = 6000) -> None:
    g = RoadGraph.load(DEFAULT_GRAPH)
    landmarks = Landmarks.load(DEFAULT_LANDMARKS, g)
    assert g.n == 1_379_917, 'Unexpected number of roadNet-TX node IDs'
    assert g.m == 1_921_660, 'Undirected edge normalization mismatch'
    assert g.largest_size == 1_351_137, 'Largest connected component mismatch'
    assert len(g.component_sizes) == 424, 'Unexpected component count for bundled input'
    rng = np.random.default_rng(seed)
    edges = rng.integers(0, len(g.indices), size=edge_samples)
    u = np.searchsorted(g.indptr, edges, side='right') - 1
    v = g.indices[edges]
    for a, b in zip(u, v):
        assert a in g.neighbors(int(b)), 'Input normalization lost edge symmetry'
    cases = json.loads(Path(DEFAULT_CASES).read_text())
    for i, case in enumerate(cases, 1):
        s, t = case['source'], case['target']
        r_b, r_a = bfs(g, s, t), astar(g, s, t, landmarks)
        assert r_b.hops == r_a.hops == case['reference_hops']
        assert validate_path(g, r_b) and validate_path(g, r_a)
        ht = landmarks.heuristic(g.internal_id(t))
        assert int(ht[g.internal_id(s)]) <= r_b.hops
        assert np.all(ht[u] <= ht[v] + 1), 'Heuristic violated edge consistency'
        assert np.all(ht[v] <= ht[u] + 1), 'Heuristic violated reverse-edge consistency'
        print(f'PASS {i}/{len(cases)}: {r_b.hops} hops; '
              f'BFS {r_b.expanded:,}, A* {r_a.expanded:,} settled vertices')
    print(f'VALIDATED {len(cases)} independent reference distances and '
          f'{edge_samples} graph edges, with heuristic consistency checks.')


if __name__ == '__main__':
    verify()
