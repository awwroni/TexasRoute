"""Memory-efficient, undirected, unweighted graph built from SNAP roadNet-TX."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip
import time

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(slots=True)
class RoadGraph:
    ids: np.ndarray             # Internal index -> original SNAP node ID
    indptr: np.ndarray          # CSR row offsets
    indices: np.ndarray         # CSR neighbor indices
    component: np.ndarray       # Component label per internal vertex
    component_sizes: np.ndarray
    largest_component: int

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def m(self) -> int:
        return len(self.indices) // 2

    @property
    def largest_size(self) -> int:
        return int(self.component_sizes[self.largest_component])

    def degree(self, v: int) -> int:
        return int(self.indptr[v + 1] - self.indptr[v])

    def neighbors(self, v: int) -> np.ndarray:
        return self.indices[self.indptr[v]:self.indptr[v + 1]]

    def internal_id(self, external: int) -> int:
        pos = int(np.searchsorted(self.ids, external))
        if pos >= self.n or int(self.ids[pos]) != external:
            raise ValueError(f"Unknown SNAP node ID: {external}")
        return pos

    def external_id(self, internal: int) -> int:
        return int(self.ids[internal])

    def same_component(self, source: int, target: int) -> bool:
        return bool(self.component[source] == self.component[target])

    def as_csr(self) -> csr_matrix:
        return csr_matrix((np.ones(len(self.indices), dtype=np.uint8),
                           self.indices, self.indptr), shape=(self.n, self.n))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ids=self.ids, indptr=self.indptr,
                            indices=self.indices, component=self.component,
                            component_sizes=self.component_sizes,
                            largest_component=np.array(self.largest_component))

    @classmethod
    def load(cls, path: str | Path) -> 'RoadGraph':
        with np.load(path, allow_pickle=False) as z:
            return cls(*(z[k] for k in ("ids", "indptr", "indices", "component", "component_sizes")),
                       int(z["largest_component"]))


def from_edge_pairs(pairs: np.ndarray) -> RoadGraph:
    """Remove loops/duplicates; symmetrize even when input has directed arcs.

    Original external IDs are preserved, even if noncontiguous (as in roadNet-TX).
    """
    pairs = np.asarray(pairs, dtype=np.int32)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
        raise ValueError("Expected a nonempty two-column edge list")
    ids = np.unique(pairs)
    src = np.searchsorted(ids, pairs[:, 0]).astype(np.int32)
    dst = np.searchsorted(ids, pairs[:, 1]).astype(np.int32)
    mask = src != dst
    adjacency = coo_matrix((np.ones(int(mask.sum()), dtype=np.uint8),
                            (src[mask], dst[mask])), shape=(len(ids), len(ids))).tocsr()
    adjacency.sum_duplicates()
    adjacency.data[:] = 1   # duplicates in input are NOT road weights
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    _, labels = connected_components(adjacency, directed=False)
    sizes = np.bincount(labels).astype(np.int32)
    return RoadGraph(ids=ids.astype(np.int32),
                     indptr=adjacency.indptr.astype(np.int32),
                     indices=adjacency.indices.astype(np.int32),
                     component=labels.astype(np.int32), component_sizes=sizes,
                     largest_component=int(np.argmax(sizes)))


def from_snap_gzip(path: str | Path) -> RoadGraph:
    start = time.perf_counter()
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        pairs = np.loadtxt(f, comments='#', dtype=np.int32)
    graph = from_edge_pairs(pairs)
    print(f"Graph prepared: {graph.n:,} vertices, {graph.m:,} undirected edges, "
          f"{len(graph.component_sizes):,} components in {time.perf_counter()-start:.1f}s")
    return graph
