"""Exact shortest-hop route search: uninformed BFS and informed ALT A*."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import time
from typing import Literal

import numpy as np
from .graph import RoadGraph
from .landmarks import Landmarks
from .trace import TraceRecorder


@dataclass(slots=True)
class SearchResult:
    algorithm: str
    source: int           # original SNAP IDs
    target: int
    found: bool
    path: list[int]      # original SNAP IDs
    hops: int | None
    expanded: int        # vertices settled incl. target, if reached
    discovered: int      # unique vertices first reached
    peak_frontier: int   # peak queue size (heap includes stale entries)
    elapsed_ms: float
    search_array_mb: float  # NumPy search-state bytes only, excluding queue/heap
    heuristic_at_source: int | None = None
    message: str = ''

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _route(parent: np.ndarray, source: int, target: int, ids: np.ndarray) -> list[int]:
    chain = [target]
    while chain[-1] != source:
        prev = int(parent[chain[-1]])
        if prev == -1:
            raise AssertionError('Broken predecessor chain')
        chain.append(prev)
    chain.reverse()
    return [int(ids[node]) for node in chain]


def _early_result(name: str, source: int, target: int, reason: str, ms: float, found=False) -> SearchResult:
    path = [source] if found else []
    return SearchResult(name, source, target, found, path, 0 if found else None,
                        int(found), int(found), int(found), ms, 0.0,
                        0 if name.startswith('A*') and found else None, reason)


def bfs(graph: RoadGraph, source_id: int, target_id: int,
        trace: TraceRecorder | None = None) -> SearchResult:
    t0 = time.perf_counter()
    s, t = graph.internal_id(source_id), graph.internal_id(target_id)
    if trace is not None and trace.source != s:
        raise ValueError("Trace recorder source does not match query")
    if s == t:
        if trace is not None:
            trace.settle(s)
            trace.end_step(1)
        return _early_result('BFS', source_id, target_id, 'Source equals target', (time.perf_counter()-t0)*1000, True)
    if not graph.same_component(s, t):
        return _early_result('BFS', source_id, target_id, 'Different connected components: no route', (time.perf_counter()-t0)*1000)

    parent = np.full(graph.n, -1, dtype=np.int32)
    parent[s] = s
    queue = deque([s])
    expanded, discovered, peak = 0, 1, 1
    found = False
    while queue:
        node = queue.popleft()
        expanded += 1
        if trace is not None:
            trace.settle(node)
        if node == t:
            found = True
            if trace is not None:
                trace.end_step(discovered)
            break
        for neighbor in graph.neighbors(node):
            u = int(neighbor)
            if parent[u] == -1:
                parent[u] = node
                discovered += 1
                queue.append(u)
                if trace is not None:
                    trace.discover(u)
        peak = max(peak, len(queue))
        if trace is not None:
            trace.end_step(discovered)

    path = _route(parent, s, t, graph.ids) if found else []
    return SearchResult('BFS', source_id, target_id, found, path,
                        len(path)-1 if found else None, expanded, discovered, peak,
                        (time.perf_counter()-t0)*1000, parent.nbytes/1e6)


def astar(graph: RoadGraph, source_id: int, target_id: int,
          landmarks: Landmarks | None, use_heuristic: bool = True,
          trace: TraceRecorder | None = None) -> SearchResult:
    """A* with consistent ALT h; setting use_heuristic=False is a Dijkstra ablation."""
    t0 = time.perf_counter()
    s, t = graph.internal_id(source_id), graph.internal_id(target_id)
    name = 'A* (ALT)' if use_heuristic else 'A* (h=0)'
    if trace is not None and trace.source != s:
        raise ValueError("Trace recorder source does not match query")
    if s == t:
        if trace is not None:
            trace.settle(s)
            trace.end_step(1)
        result = _early_result(name, source_id, target_id, 'Source equals target', (time.perf_counter()-t0)*1000, True)
        return result
    if not graph.same_component(s, t):
        return _early_result(name, source_id, target_id, 'Different connected components: no route', (time.perf_counter()-t0)*1000)
    if use_heuristic and landmarks is None:
        raise ValueError('ALT landmark cache required for heuristic A*')
    # For short routes, generating h(v) for all 1.38M nodes is wasteful.
    # Use on-demand estimates when the start lower bound is small; otherwise
    # use a vectorized full potential to keep larger searches inexpensive.
    if use_heuristic:
        start_lower_bound = landmarks.estimate(s, t)
        if start_lower_bound <= 40:
            h_vector = None
            h_at = lambda v: landmarks.estimate(v, t)
        else:
            h_vector = landmarks.heuristic(t)
            h_at = lambda v: int(h_vector[v])
    else:
        start_lower_bound = 0
        h_vector = None
        h_at = lambda v: 0
    best = np.full(graph.n, np.iinfo(np.int32).max, dtype=np.int32)
    parent = np.full(graph.n, -1, dtype=np.int32)
    closed = np.zeros(graph.n, dtype=np.bool_)
    best[s] = 0
    # Larger g wins on f-ties: approaches the goal faster with consistent h.
    heap: list[tuple[int, int, int]] = [(start_lower_bound, 0, s)]
    expanded, discovered, peak = 0, 1, 1
    found = False
    while heap:
        f, neg_g, node = heapq.heappop(heap)
        g = -neg_g
        if closed[node] or g != best[node]:
            continue
        closed[node] = True
        expanded += 1
        if trace is not None:
            trace.settle(node)
        if node == t:
            found = True
            if trace is not None:
                trace.end_step(discovered)
            break
        for neighbor in graph.neighbors(node):
            u = int(neighbor)
            trial = g + 1
            if not closed[u] and trial < best[u]:
                if best[u] == np.iinfo(np.int32).max:
                    discovered += 1
                    if trace is not None:
                        trace.discover(u)
                best[u] = trial
                parent[u] = node
                heapq.heappush(heap, (trial + h_at(u), -trial, u))
        peak = max(peak, len(heap))
        if trace is not None:
            trace.end_step(discovered)

    path = _route(parent, s, t, graph.ids) if found else []
    return SearchResult(name, source_id, target_id, found, path,
                        len(path)-1 if found else None, expanded, discovered, peak,
                        (time.perf_counter()-t0)*1000,
                        (best.nbytes + parent.nbytes + closed.nbytes + (h_vector.nbytes if h_vector is not None else 0))/1e6,
                        start_lower_bound)


def validate_path(graph: RoadGraph, result: SearchResult) -> bool:
    """Independently verify all consecutive edges, endpoints and hop count."""
    if not result.found:
        return not result.path and result.hops is None
    if not result.path or result.path[0] != result.source or result.path[-1] != result.target:
        return False
    if result.hops != len(result.path) - 1:
        return False
    for a, b in zip(result.path, result.path[1:]):
        u, v = graph.internal_id(a), graph.internal_id(b)
        if v not in graph.neighbors(u):
            return False
    return True
