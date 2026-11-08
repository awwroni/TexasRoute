"""Faithful topology-only route schematic: never claims geospatial coordinates."""
from __future__ import annotations

import plotly.graph_objects as go
from .graph import RoadGraph


def route_schematic(graph: RoadGraph, path: list[int], offset: int = 0,
                    window: int = 20, max_branches: int = 2) -> go.Figure:
    """Draw actual subgraph edges near a short consecutive path segment.

    Coordinates are schematic positions, NOT geographic longitude/latitude.
    Only graph edges that really exist are drawn.
    """
    if not path:
        raise ValueError('Cannot draw an empty path')
    offset = max(0, min(offset, max(0, len(path) - 2)))
    section = path[offset:offset + window]
    main_nodes = set(section)
    positions: dict[int, tuple[float, float]] = {node: (float(i), 0.0)
                                                    for i, node in enumerate(section)}
    for i, external in enumerate(section):
        neighbors = [graph.external_id(int(n)) for n in graph.neighbors(graph.internal_id(external))
                     if graph.external_id(int(n)) not in main_nodes]
        for j, neighbor in enumerate(neighbors[:max_branches]):
            if neighbor not in positions:
                # Deterministic offsets prevent duplicate nodes appearing twice.
                positions[neighbor] = (i + (0.23 if j % 2 else -0.23),
                                       (1.15 if j % 2 == 0 else -1.15))

    route_edges = {tuple(sorted((u, v))) for u, v in zip(section, section[1:])}
    background_x, background_y, route_x, route_y = [], [], [], []
    for a in positions:
        ia = graph.internal_id(a)
        for internal_b in graph.neighbors(ia):
            b = graph.external_id(int(internal_b))
            if b not in positions or a >= b:
                continue
            if tuple(sorted((a, b))) in route_edges:
                route_x += [positions[a][0], positions[b][0], None]
                route_y += [positions[a][1], positions[b][1], None]
            else:
                background_x += [positions[a][0], positions[b][0], None]
                background_y += [positions[a][1], positions[b][1], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=background_x, y=background_y, mode='lines',
                             line={'color': '#53647C', 'width': 1.15},
                             hoverinfo='skip', name='Adjacent roads'))
    fig.add_trace(go.Scatter(x=route_x, y=route_y, mode='lines',
                             line={'color': '#21D6C5', 'width': 4},
                             hoverinfo='skip', name='Selected shortest path'))
    for name, nodes, color, size in [
        ('Neighboring nodes', [n for n in positions if n not in main_nodes], '#677C9A', 7),
        ('Route nodes', [n for n in section if n not in (path[0], path[-1])], '#2ED4C6', 12),
        ('Source', [path[0]] if path[0] in positions else [], '#FFD479', 17),
        ('Destination', [path[-1]] if path[-1] in positions else [], '#FF899A', 17),
    ]:
        fig.add_trace(go.Scatter(x=[positions[n][0] for n in nodes],
                                 y=[positions[n][1] for n in nodes],
                                 mode='markers', name=name,
                                 marker={'size': size, 'color': color,
                                         'line': {'color': '#0B1322', 'width': 1.5}},
                                 text=[f'SNAP {n} · degree {graph.degree(graph.internal_id(n))}'
                                       for n in nodes], hovertemplate='%{text}<extra></extra>'))
    fig.update_layout(template='plotly_dark', height=340, margin=dict(l=10, r=10, t=12, b=10),
                      paper_bgcolor='#0B1322', plot_bgcolor='#0B1322',
                      xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False, 'title': ''},
                      yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False,
                             'range': [-2.0, 2.0], 'title': ''},
                      legend={'orientation': 'h', 'y': -0.08, 'font': {'size': 10}},
                      hoverlabel={'bgcolor': '#19283D'})
    return fig
