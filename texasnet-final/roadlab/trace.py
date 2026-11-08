"""Compact, *lossless* search-event recording for interactive replay.

Recorded steps are actual node settlements, not invented animation states.
Dense int32 timestamps avoid holding millions of Python objects.  The UI
can replay any event budget without re-running either search algorithm.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class SearchTrace:
    """Timestamps are 0-based before the first settlement, 1..expanded after.

    discovered_at == -1: never discovered; settled_at == -1: never settled.
    A* can improve the cost of a discovered node; this does not rediscover it.
    A* frontier here means discovered but not settled (ignoring stale heap entries).
    """
    source: int  # internal node index
    discovered_at: np.ndarray
    settled_at: np.ndarray
    discovered_prefix: np.ndarray  # cumulative distinct discovered after step k

    @property
    def expanded(self) -> int:
        return len(self.discovered_prefix) - 1

    def stats(self, budget: int) -> tuple[int, int, int]:
        k = max(0, min(int(budget), self.expanded))
        discovered = int(self.discovered_prefix[k])
        return k, discovered, discovered - k

    def status(self, nodes: np.ndarray, budget: int) -> np.ndarray:
        """0 unseen; 1 in frontier; 2 settled. Nodes are internal IDs."""
        k = max(0, min(int(budget), self.expanded))
        discovered = self.discovered_at[nodes]
        settled = self.settled_at[nodes]
        result = np.zeros(len(nodes), dtype=np.uint8)
        result[(discovered >= 0) & (discovered <= k)] = 1
        result[(settled >= 0) & (settled <= k)] = 2
        return result


class TraceRecorder:
    """Optional hooks inserted into the production BFS and A* inner loops."""

    def __init__(self, n: int, source: int):
        if not 0 <= source < n:
            raise ValueError('Invalid internal source node')
        self.source = int(source)
        self.discovered_at = np.full(n, -1, dtype=np.int32)
        self.settled_at = np.full(n, -1, dtype=np.int32)
        self.discovered_at[source] = 0
        self.step = 0
        self._discovered_prefix = array('i', [1])

    def settle(self, node: int) -> None:
        self.step += 1
        if self.settled_at[node] != -1:
            raise AssertionError('Node settled more than once')
        self.settled_at[node] = self.step

    def discover(self, node: int) -> None:
        if self.discovered_at[node] != -1:
            raise AssertionError('Node marked discovered twice')
        self.discovered_at[node] = self.step

    def end_step(self, discovered_count: int) -> None:
        if len(self._discovered_prefix) != self.step:
            raise AssertionError('Recorder missing a settlement')
        self._discovered_prefix.append(discovered_count)

    def snapshot(self) -> SearchTrace:
        if len(self._discovered_prefix) != self.step + 1:
            raise AssertionError('Incomplete recorder progress data')
        return SearchTrace(self.source, self.discovered_at, self.settled_at,
                           np.frombuffer(self._discovered_prefix, dtype=np.int32).copy())
