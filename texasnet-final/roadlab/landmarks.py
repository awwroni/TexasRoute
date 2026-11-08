"""ALT: A* using Landmarks and the Triangle inequality, in unit edge costs."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from scipy.sparse.csgraph import shortest_path
from .graph import RoadGraph


@dataclass(slots=True)
class Landmarks:
    nodes: np.ndarray       # internal node indices
    distances: np.ndarray   # shape (k, n); unreachable = dtype max

    @property
    def sentinel(self) -> int:
        return int(np.iinfo(self.distances.dtype).max)

    def estimate(self, node: int, target: int) -> int:
        """Compute ALT lower bound for a single node without scanning the graph."""
        best = 0
        for row in self.distances:
            a, b = int(row[node]), int(row[target])
            if a != self.sentinel and b != self.sentinel:
                best = max(best, abs(a - b))
        return best

    def heuristic(self, target: int) -> np.ndarray:
        """h(v)=max_L |d(L,t)-d(L,v)| for reachable landmarks; otherwise 0.

        The triangle inequality proves admissibility and consistency on
        unweighted, undirected graphs. Vectorized once per query for speed.
        """
        h = np.zeros(self.distances.shape[1], dtype=np.int32)
        for row in self.distances:
            dt = int(row[target])
            if dt == self.sentinel:
                continue
            reachable = row != self.sentinel
            # cast BEFORE subtracting to avoid unsigned integer wraparound
            diff = np.abs(row[reachable].astype(np.int32) - dt)
            h[reachable] = np.maximum(h[reachable], diff)
        return h

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, nodes=self.nodes, distances=self.distances)

    @classmethod
    def load(cls, path: str | Path, graph: RoadGraph | None = None) -> 'Landmarks':
        with np.load(path, allow_pickle=False) as z:
            result = cls(z['nodes'].astype(np.int32), z['distances'])
        if graph is not None and result.distances.shape[1] != graph.n:
            raise ValueError('Landmark cache does not match this graph')
        return result


def build_landmarks(graph: RoadGraph, count: int = 8) -> Landmarks:
    """Deterministic farthest-first landmarks in the largest connected component.

    A greedy max-min policy distributes landmarks across distant network regions.
    Other components automatically use h=0; no invalid cross-component metric.
    """
    if not 1 <= count <= graph.largest_size:
        raise ValueError('Landmark count outside supported range')
    t0 = time.perf_counter()
    csr = graph.as_csr()
    in_giant = graph.component == graph.largest_component
    giant_nodes = np.flatnonzero(in_giant)
    degrees = np.diff(graph.indptr)
    start = int(giant_nodes[np.argmax(degrees[giant_nodes])])
    min_distance = np.full(graph.n, np.inf, dtype=np.float64)
    chosen = []
    rows = []
    node = start
    for i in range(count):
        chosen.append(node)
        d = shortest_path(csr, directed=False, unweighted=True,
                          indices=node, return_predecessors=False)
        if not np.isfinite(d[in_giant]).all():
            raise RuntimeError('Unexpected disconnected giant component')
        rows.append(d)
        min_distance = np.minimum(min_distance, d)
        min_distance[~in_giant] = -1
        min_distance[np.array(chosen, dtype=np.int32)] = -1
        if i != count - 1:
            node = int(np.argmax(min_distance))
        print(f"Landmark {i+1}/{count}: SNAP ID {graph.external_id(chosen[-1])}; "
              f"max distance {int(np.max(d[in_giant]))}; "
              f"elapsed {time.perf_counter()-t0:.1f}s")

    max_finite = max(float(np.max(row[np.isfinite(row)])) for row in rows)
    dtype = np.uint16 if max_finite < np.iinfo(np.uint16).max else np.uint32
    sentinel = np.iinfo(dtype).max
    matrix = np.full((count, graph.n), sentinel, dtype=dtype)
    for i, d in enumerate(rows):
        valid = np.isfinite(d)
        matrix[i, valid] = d[valid].astype(dtype)
    return Landmarks(np.array(chosen, dtype=np.int32), matrix)
