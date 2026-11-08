import numpy as np
import pytest

from roadlab import from_edge_pairs, build_landmarks, bfs, astar, validate_path


@pytest.fixture(scope='module')
def toy():
    # Non-contiguous external IDs; a cycle, shortcuts and an isolated component.
    pairs = np.array([[10, 20], [20, 10], [20, 30], [30, 40], [40, 50],
                      [50, 60], [60, 10], [20, 50], [30, 50], [20, 20],
                      [900, 901], [901, 900], [900, 901]], dtype=np.int32)
    graph = from_edge_pairs(pairs)
    landmarks = build_landmarks(graph, 3)
    return graph, landmarks


def test_preprocessing(toy):
    graph, _ = toy
    assert graph.n == 8
    assert graph.m == 9  # eight in the giant component, one in the small component
    assert graph.largest_size == 6
    assert graph.degree(graph.internal_id(20)) == 3


def test_routes_and_unreachable(toy):
    graph, landmarks = toy
    for s in [10, 20, 30, 40, 50, 60, 900, 901]:
        for t in [10, 20, 30, 40, 50, 60, 900, 901]:
            result_b = bfs(graph, s, t)
            result_a = astar(graph, s, t, landmarks)
            result_0 = astar(graph, s, t, landmarks, False)
            assert result_b.hops == result_a.hops == result_0.hops
            assert validate_path(graph, result_b)
            assert validate_path(graph, result_a)
            assert validate_path(graph, result_0)
            if not graph.same_component(graph.internal_id(s), graph.internal_id(t)):
                assert not result_b.found


def test_heuristic_admissible_consistent(toy):
    graph, landmarks = toy
    for target_id in graph.ids:
        t = graph.internal_id(int(target_id))
        h = landmarks.heuristic(t)
        assert h[t] == 0
        for u in range(graph.n):
            for v in graph.neighbors(u):
                assert h[u] <= 1 + h[v]
            if graph.same_component(u, t):
                exact = bfs(graph, graph.external_id(u), graph.external_id(t)).hops
                assert int(h[u]) <= exact


def test_original_id_validation(toy):
    graph, landmarks = toy
    with pytest.raises(ValueError, match='Unknown SNAP'):
        bfs(graph, 99, 20)
    assert astar(graph, 900, 901, landmarks).hops == 1
    assert astar(graph, 900, 901, landmarks).heuristic_at_source == 0
