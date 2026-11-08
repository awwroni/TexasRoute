"""Search-event playback must exactly match the production algorithms."""
import numpy as np
import pytest

from roadlab import from_edge_pairs, build_landmarks, bfs, astar, validate_path
from roadlab.trace import TraceRecorder
from roadlab.live import (FRAMES, build_live_scene, local_projection,
                          search_frame)


@pytest.fixture(scope='module')
def toy():
    edges = np.array([[10,20],[20,30],[30,40],[40,50],[10,60],
                      [60,70],[70,50],[20,60],[20,70],
                      [300,301],[301,302],[302,303]],dtype=np.int32)
    graph = from_edge_pairs(edges)
    landmarks = build_landmarks(graph, 3)
    return graph, landmarks


def test_real_bfs_and_astar_replay(toy):
    graph, landmarks = toy
    start = graph.internal_id(10)
    b_record = TraceRecorder(graph.n, start)
    a_record = TraceRecorder(graph.n, start)
    b = bfs(graph, 10, 50, trace=b_record)
    a = astar(graph, 10, 50, landmarks, trace=a_record)
    assert validate_path(graph,b) and validate_path(graph,a)
    assert b.hops == a.hops == 3
    for result, recorder in ((b,b_record),(a,a_record)):
        trace = recorder.snapshot()
        assert trace.expanded == result.expanded
        assert trace.stats(trace.expanded) == (result.expanded, result.discovered,
                                               result.discovered - result.expanded)
        assert trace.stats(0) == (0, 1, 1)
        assert trace.settled_at[start] == 1
        assert trace.discovered_at[start] == 0
        assert np.all(np.diff(trace.discovered_prefix) >= 0)
        assert np.count_nonzero(trace.settled_at >= 0) == trace.expanded
        assert np.count_nonzero(trace.discovered_at >= 0) == result.discovered
        assert np.all(trace.settled_at[trace.settled_at != -1] >= 1)
        assert trace.status(np.array([start],dtype=np.int32), 0).tolist() == [1]
        assert trace.status(np.array([start],dtype=np.int32), 1).tolist() == [2]

    # This tiny graph uses the query-local projection, exercising shared frames.
    coords = local_projection(graph,start,graph.internal_id(50))
    scene = build_live_scene(graph,coords,b_record.snapshot(),a_record.snapshot(),
                             b,a,'local')
    assert scene.bfs_heatmaps[-1].sum() == b.expanded
    assert scene.astar_heatmaps[-1].sum() == a.expanded
    assert scene.budgets[0] == 0
    assert scene.budgets[-1] == max(b.expanded,a.expanded)
    f0 = search_frame(scene,b_record.snapshot(),b,'BFS',0)
    f1 = search_frame(scene,a_record.snapshot(),a,'A* · ALT',FRAMES)
    assert not any(t.name == 'Final verified path' for t in f0.data)
    assert any(t.name == 'Final verified path' for t in f1.data)


def test_outside_landmark_component(toy):
    graph, landmarks = toy
    s, t = 300, 303
    record_b = TraceRecorder(graph.n, graph.internal_id(s))
    record_a = TraceRecorder(graph.n, graph.internal_id(s))
    b = bfs(graph,s,t,trace=record_b)
    a = astar(graph,s,t,landmarks,trace=record_a)
    assert b.hops == a.hops == 3
    coords = local_projection(graph,graph.internal_id(s),graph.internal_id(t))
    scene = build_live_scene(graph,coords,record_b.snapshot(),record_a.snapshot(),
                             b,a,'local')
    assert scene.bfs_heatmaps[-1].sum() == b.expanded
    assert scene.astar_heatmaps[-1].sum() == a.expanded


def test_edge_cases_and_invalid_recorder(toy):
    graph, landmarks = toy
    wrong = TraceRecorder(graph.n,graph.internal_id(20))
    with pytest.raises(ValueError,match='source'):
        bfs(graph,10,50,trace=wrong)
    with pytest.raises(ValueError,match='source'):
        astar(graph,10,50,landmarks,trace=wrong)
    one = TraceRecorder(graph.n,graph.internal_id(20))
    trivial = bfs(graph,20,20,trace=one)
    assert trivial.found and trivial.hops == 0
    assert one.snapshot().stats(1) == (1,1,0)
