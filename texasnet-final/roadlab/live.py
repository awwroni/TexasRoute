"""Real BFS / ALT A* event playback in a topology-derived, non-GPS view.

No fictitious Texas coordinates: a PCA of exact graph-distance-to-landmark
vectors embeds the giant component in a shared 2-D topological space.
For other components, a query-local graph-distance embedding is used.
All heatmap cells and node statuses derive from *actual* search events.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import numpy as np
import plotly.graph_objects as go

from .graph import RoadGraph
from .landmarks import Landmarks
from .trace import SearchTrace
from .search import SearchResult

FRAMES = 120
WIDTH, HEIGHT = 64, 48


def landmark_projection(graph: RoadGraph, landmarks: Landmarks) -> np.ndarray:
    """Memory-bounded graph-distance PCA; ~11 MB final positions on Texas network."""
    if landmarks.distances.shape[1] != graph.n or len(landmarks.nodes) < 3:
        raise ValueError('Need at least 3 consistent landmarks for projection')
    sentinel = landmarks.sentinel
    valid = landmarks.distances[0] != sentinel
    ids = np.flatnonzero(valid)
    sample = np.random.default_rng(2026).choice(ids, size=min(14000, len(ids)), replace=False)
    d = landmarks.distances[:, sample].T.astype(np.float32)
    center = d.mean(axis=0)
    _, _, basis = np.linalg.svd(d - center, full_matrices=False)
    components = basis[:2].T
    coordinates = np.full((graph.n, 2), np.nan, dtype=np.float32)
    for chunk in np.array_split(ids, max(1, (len(ids) + 79999) // 80000)):
        block = landmarks.distances[:, chunk].T.astype(np.float32)
        coordinates[chunk] = (block - center) @ components
    return coordinates


def _bfs_distances(graph: RoadGraph, source: int) -> np.ndarray:
    d = np.full(graph.n, -1, dtype=np.int32)
    q = deque([source]); d[source] = 0
    while q:
        v = q.popleft()
        for neighbor in graph.neighbors(v):
            u = int(neighbor)
            if d[u] == -1:
                d[u] = d[v] + 1
                q.append(u)
    return d


def local_projection(graph: RoadGraph, source: int, target: int) -> np.ndarray:
    """Same distance-based principle for components without stored landmarks."""
    d1 = _bfs_distances(graph, source)
    d2 = _bfs_distances(graph, target)
    comp = np.flatnonzero(d1 >= 0)
    far1 = int(comp[np.argmax(d1[comp])])
    far2 = int(comp[np.argmax(d2[comp])])
    d3 = _bfs_distances(graph, far1)
    d4 = _bfs_distances(graph, far2)
    features = np.stack([d1[comp], d2[comp], d3[comp], d4[comp]], axis=1).astype(np.float32)
    features -= features.mean(axis=0)
    _, _, directions = np.linalg.svd(features, full_matrices=False)
    coords = np.full((graph.n, 2), np.nan, dtype=np.float32)
    coords[comp] = features @ directions[:2].T
    return coords


@dataclass(slots=True)
class LiveScene:
    nodes: np.ndarray               # original INTERNAL indices, common to both views
    xy: np.ndarray                  # positions of those nodes
    node_ids: np.ndarray            # original SNAP IDs, for tooltips
    source_xy: np.ndarray
    target_xy: np.ndarray
    bfs_path_xy: np.ndarray         # actual BFS path edges
    astar_path_xy: np.ndarray       # actual ALT A* path edges
    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    bfs_heatmaps: np.ndarray        # (FRAMES+1, HEIGHT, WIDTH), cumulative settled node density
    astar_heatmaps: np.ndarray
    max_expanded: int
    budgets: np.ndarray             # early events get more animation frames
    source_id: int
    target_id: int
    projection_name: str


def _settled_heatmaps(trace: SearchTrace, binmap: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    """Aggregate ALL settled nodes; not restricted to plotted marker sample."""
    nodes = np.flatnonzero(trace.settled_at >= 1)
    step = trace.settled_at[nodes]
    frame = np.searchsorted(budgets, step, side="left")
    valid = binmap[nodes] >= 0
    merged = frame[valid] * (WIDTH * HEIGHT) + binmap[nodes[valid]]
    hist = np.bincount(merged, minlength=(FRAMES + 1) * WIDTH * HEIGHT)
    hist = hist.reshape(FRAMES + 1, HEIGHT, WIDTH)
    return np.cumsum(hist, axis=0, dtype=np.int32)


def build_live_scene(graph: RoadGraph, coords: np.ndarray,
                     bfs_trace: SearchTrace, astar_trace: SearchTrace,
                     bfs_result: SearchResult, astar_result: SearchResult,
                     projection_name: str) -> LiveScene:
    if not (bfs_result.found and astar_result.found and bfs_result.hops == astar_result.hops):
        raise ValueError('Searches must both find paths of equal optimal cost')
    s = graph.internal_id(bfs_result.source)
    t = graph.internal_id(bfs_result.target)
    sample_pool = np.flatnonzero((bfs_trace.discovered_at >= 0) |
                                 (astar_trace.discovered_at >= 0))
    if not len(sample_pool):
        raise ValueError('No search events recorded')
    if not np.isfinite(coords[s]).all() or not np.isfinite(coords[t]).all():
        raise ValueError('No valid topological projection for query component')
    rng = np.random.default_rng(104729 + s % 10000 + t % 10000)
    # Stratify in both actual searches so the smaller, goal-directed A* wave is
    # not drowned out by the much larger BFS wave.
    b_pool = np.flatnonzero(bfs_trace.discovered_at >= 0)
    a_pool = np.flatnonzero(astar_trace.discovered_at >= 0)
    def take(pool, n):
        return rng.choice(pool, size=min(n, len(pool)), replace=False)
    b_path = np.array([graph.internal_id(int(v)) for v in bfs_result.path], dtype=np.int32)
    a_path = np.array([graph.internal_id(int(v)) for v in astar_result.path], dtype=np.int32)
    visible = np.unique(np.concatenate([take(b_pool, 1850), take(a_pool, 1850), b_path, a_path,
                                         np.array([s,t], dtype=np.int32)]))
    visible = visible[np.isfinite(coords[visible]).all(axis=1)]
    # Both sides use precisely the same projection, viewport, markers, and sample.
    active_xy = coords[sample_pool]
    lo = np.min(active_xy, axis=0)
    hi = np.max(active_xy, axis=0)
    span = np.maximum(hi-lo, 18)
    padding = span * 0.09
    lo -= padding; hi += padding
    x_bounds = (float(lo[0]), float(hi[0]))
    y_bounds = (float(lo[1]), float(hi[1]))
    # Common cell mapping for every node. Isolated smaller components are NaN.
    binmap = np.full(graph.n, -1, dtype=np.int32)
    active = np.flatnonzero(np.isfinite(coords).all(axis=1))
    xx = np.clip(((coords[active,0]-lo[0])/(hi[0]-lo[0]) * WIDTH).astype(np.int32),0,WIDTH-1)
    yy = np.clip(((coords[active,1]-lo[1])/(hi[1]-lo[1]) * HEIGHT).astype(np.int32),0,HEIGHT-1)
    binmap[active] = yy*WIDTH + xx
    common_max = max(bfs_trace.expanded, astar_trace.expanded, 1)
    budgets = np.floor(common_max * (np.arange(FRAMES + 1) / FRAMES)**2).astype(np.int32)
    return LiveScene(visible, coords[visible], graph.ids[visible],
                     coords[s].copy(), coords[t].copy(),
                     coords[b_path], coords[a_path], x_bounds, y_bounds,
                     _settled_heatmaps(bfs_trace, binmap, budgets),
                     _settled_heatmaps(astar_trace, binmap, budgets),
                     common_max, budgets, bfs_result.source, bfs_result.target, projection_name)


def search_frame(scene: LiveScene, trace: SearchTrace, result: SearchResult,
                 algorithm: str, frame: int) -> go.Figure:
    """Same event budget is applied to both plots at each frame."""
    frame = int(np.clip(frame, 0, FRAMES))
    budget = int(scene.budgets[frame])
    settled, discovered, frontier = trace.stats(budget)
    statuses = trace.status(scene.nodes, budget)
    heat = scene.bfs_heatmaps if algorithm == 'BFS' else scene.astar_heatmaps
    density = heat[frame].astype(np.float32)
    density[density == 0] = np.nan
    density = np.log1p(density)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=density, x=np.linspace(*scene.x_bounds, WIDTH),
        y=np.linspace(*scene.y_bounds, HEIGHT),
        colorscale=[[0,'#102A40'],[0.22,'#155168'],[0.62,'#198F9A'],[1,'#41F1D2']],
        opacity=0.56, zsmooth='best', showscale=False,
        hoverinfo='skip', name='All settled nodes (density)'))
    status_classes = [
        (0, 'Not yet discovered · sampled', '#52617D', 3.5, 0.28),
        (2, 'Settled · sampled', '#37CDBF', 4.8, 0.86),
        (1, 'Frontier · sampled', '#FFC56D', 6.0, 0.97),
    ]
    for status, name, color, size, opacity in status_classes:
        select = statuses == status
        if not np.any(select):
            continue
        points = scene.xy[select]
        nodes = scene.node_ids[select]
        fig.add_trace(go.Scattergl(
            x=points[:,0],y=points[:,1],mode='markers',name=name,
            customdata=nodes,
            marker=dict(color=color,size=size,opacity=opacity),
            hovertemplate='SNAP %{customdata}<extra>'+name+'</extra>'))
    if settled >= trace.expanded and result.found:
        fig.add_trace(go.Scattergl(
            x=(scene.bfs_path_xy if algorithm == 'BFS' else scene.astar_path_xy)[:,0],
            y=(scene.bfs_path_xy if algorithm == 'BFS' else scene.astar_path_xy)[:,1],mode='lines',
            name='Final verified path',
            line=dict(color='#F1EC76',width=4),hoverinfo='skip'))
    for label, point, color, node_id in [
        ('S',scene.source_xy,'#FFFFFF',scene.source_id),
        ('T',scene.target_xy,'#FF7E98',scene.target_id),
    ]:
        fig.add_trace(go.Scatter(
            x=[point[0]],y=[point[1]],mode='markers+text',
            marker=dict(size=17,color=color,line=dict(color='#071525',width=2)),
            text=[label],textposition='top center',textfont=dict(size=14,color=color),
            name=f'{"Source" if label=="S" else "Target"} · SNAP {node_id}',
            hovertemplate=f'SNAP {node_id}<extra></extra>'))
    is_done = settled >= trace.expanded
    fig.update_layout(
        title=dict(text=f'{algorithm} <span style="color:#97A9C0;font-size:12px">'
                        f'· {settled:,} settled · {frontier:,} frontier'
                        f'{" · GOAL REACHED" if is_done else ""}</span>',x=0.02,font=dict(size=18)),
        template='plotly_dark',height=455,
        paper_bgcolor='#0C1728',plot_bgcolor='#0C1728',
        margin=dict(l=6,r=6,t=57,b=52),
        xaxis=dict(range=list(scene.x_bounds),visible=False,showgrid=False),
        yaxis=dict(range=list(scene.y_bounds),visible=False,showgrid=False),
        legend=dict(orientation='h',y=-0.06,x=0,font=dict(size=9),itemsizing='constant'),
        hoverlabel=dict(bgcolor='#19283D'),
        uirevision='live-scene')
    return fig
