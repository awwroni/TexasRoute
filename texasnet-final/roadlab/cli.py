"""Commands: prepare, route, sample, benchmark, inspect."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .graph import RoadGraph, from_snap_gzip
from .landmarks import Landmarks, build_landmarks
from .search import bfs, astar, validate_path
from .experiments import representative_pairs, run_benchmark, write_pairs, write_benchmark

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / 'data' / 'roadNet-TX.txt.gz'
DEFAULT_GRAPH = ROOT / 'data' / 'graph.npz'
DEFAULT_LANDMARKS = ROOT / 'data' / 'landmarks.npz'
DEFAULT_CASES = ROOT / 'outputs' / 'sample_cases.json'
DEFAULT_BENCHMARK = ROOT / 'outputs' / 'benchmark.csv'


def main():
    parser = argparse.ArgumentParser(description='Texas RoadNet: BFS and landmark A* search')
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Build CSR graph and ALT landmark index')
    prep.add_argument('--data', type=Path, default=DEFAULT_DATA)
    prep.add_argument('--graph-cache', type=Path, default=DEFAULT_GRAPH)
    prep.add_argument('--landmark-cache', type=Path, default=DEFAULT_LANDMARKS)
    prep.add_argument('--landmarks', type=int, default=8)
    prep.add_argument('--force', action='store_true')
    route = sub.add_parser('route', help='Run BFS and A* on original SNAP node IDs')
    route.add_argument('source', type=int)
    route.add_argument('target', type=int)
    route.add_argument('--algorithm', choices=['bfs', 'astar', 'zero', 'both'], default='both')
    route.add_argument('--show-path', action='store_true')
    sample = sub.add_parser('sample', help='Generate reproducible near/mid/far route cases')
    sample.add_argument('--seed', type=int, default=42)
    sample.add_argument('--anchors', type=int, default=3)
    sample.add_argument('--out', type=Path, default=DEFAULT_CASES)
    benchmark = sub.add_parser('benchmark', help='Validate algorithms and export benchmark CSV')
    benchmark.add_argument('--cases', type=Path, default=DEFAULT_CASES)
    benchmark.add_argument('--out', type=Path, default=DEFAULT_BENCHMARK)
    benchmark.add_argument('--repeats', type=int, default=1)
    benchmark.add_argument('--no-ablation', action='store_true')
    sub.add_parser('inspect', help='Show basic network properties')
    args = parser.parse_args()

    if args.command == 'prepare':
        if args.force or not args.graph_cache.is_file():
            graph = from_snap_gzip(args.data)
            graph.save(args.graph_cache)
            print('Saved graph:', args.graph_cache)
        else:
            graph = RoadGraph.load(args.graph_cache)
            print('Using existing graph index:', args.graph_cache)
        if args.force or not args.landmark_cache.is_file():
            landmark_data = build_landmarks(graph, args.landmarks)
            landmark_data.save(args.landmark_cache)
            print('Saved ALT landmarks:', args.landmark_cache)
        else:
            existing = Landmarks.load(args.landmark_cache, graph)
            print(f'Using existing {len(existing.nodes)} landmarks:', args.landmark_cache)
        return

    if not DEFAULT_GRAPH.is_file():
        parser.error('No graph index. First run: python -m roadlab.cli prepare')
    graph = RoadGraph.load(DEFAULT_GRAPH)
    if args.command == 'inspect':
        print(json.dumps({'nodes': graph.n, 'undirected_edges': graph.m,
                          'connected_components': len(graph.component_sizes),
                          'largest_component_nodes': graph.largest_size,
                          'largest_component_percent': round(100*graph.largest_size/graph.n, 3)}, indent=2))
        return
    if args.command == 'sample':
        pairs = representative_pairs(graph, seed=args.seed, anchors=args.anchors)
        write_pairs(pairs, args.out)
        print('Saved route cases:', args.out)
        return
    if not DEFAULT_LANDMARKS.is_file():
        parser.error('No landmark index. First run: python -m roadlab.cli prepare')
    landmarks = Landmarks.load(DEFAULT_LANDMARKS, graph)
    if args.command == 'route':
        selected = ([args.algorithm] if args.algorithm != 'both' else ['bfs', 'astar'])
        for alg in selected:
            result = bfs(graph, args.source, args.target) if alg == 'bfs' else astar(graph, args.source, args.target, landmarks, alg != 'zero')
            assert validate_path(graph, result)
            data = result.to_dict()
            if not args.show_path:
                data.pop('path')
            print(json.dumps(data, indent=2))
    elif args.command == 'benchmark':
        pairs = json.loads(args.cases.read_text(encoding='utf-8'))
        rows = run_benchmark(graph, landmarks, pairs, args.repeats, not args.no_ablation)
        write_benchmark(rows, args.out)
        print('Saved benchmark:', args.out)


if __name__ == '__main__':
    main()
