# Texas RoadNet / Route Intelligence

**BFS vs landmark-guided A\* on the entire 1.38-million-node Stanford SNAP Texas road network.**

A reproducible, interactive classical-AI search project with a working CLI, an optimized CSR graph, eight precomputed ALT landmarks, exact route verification, a topology-only route viewer, connected-component analysis, algorithm ablation, and benchmark exports, plus synchronized live BFS/A* search-event animation. The original dataset and generated indexes are included; the AI project report has intentionally **not** been generated.

## Launch

Requires **Python 3.10 or newer**. From the extracted `texas-roadnet-ai` directory:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The browser should open the Streamlit interface (typically `http://localhost:8501`). Indexes and nine benchmark cases are already included in `data/` and `outputs/` so **no first-launch preprocessing** is needed.

If an indexed file is missing or you want to rebuild it from the original `roadNet-TX.txt.gz`:

```bash
python -m roadlab.cli prepare --force --landmarks 8
```

### Demonstrate a route from the command line

```bash
python -m roadlab.cli inspect
python -m roadlab.cli route 918454 990282 --algorithm both
python -m roadlab.cli route 918454 990282 --algorithm astar --show-path
```

These IDs are **original SNAP node IDs** (which are not contiguous). The second route has **426 hops** in the bundled graph. All reported paths are validated edge by edge.

## Explore the interface

- **Live Search (new):** Record BFS and ALT A* on your chosen original SNAP node IDs, then play, pause, restart, scrub or change the playback speed. Two synchronized network-projection panels show *actual* explored regions (density heatmaps), discovered-but-unsettled frontier nodes (orange), settled nodes (teal), and the real validated shortest path (yellow) after the algorithm finishes. The same topology-derived node positions and **same node-expansion budget** are used for both algorithms. Exact settlement/discovery/frontier counters update with the animation. More frames are allocated to early exploration, so A* does not complete instantly in long BFS runs. A saved offline browser demo is included at `outputs/live_search_demo.html`.
- **Route Explorer:** select a representative local/regional/long-distance pair or enter any valid IDs. Compare BFS and A\* on actual road edges; inspect hop count, settled/discovered vertices, frontier size, elapsed time, and exact path. Pan through a *schematic*, not a fabricated geographic map, and export the path CSV.
- **Experiments:** view the **final notebook’s 18-route / 162-execution** benchmark, dynamically calculated optimal-path accuracy, observed runtimes, explored nodes, zero-heuristic control, memory evidence, and downloadable results. The older 9-route data is retained in an explicitly labeled legacy expander.
- **Network Structure:** see the graph's components, degree distribution, and the consequences of disconnected pieces.
- **Algorithm Design:** read the exact state-space representation, the ALT heuristic and why it guarantees an optimal shortest-hop route.

## Demonstrate the new Live Search tab

1. Launch the app, select **▶️ Live Search**, choose a reference route (or enter two original SNAP node IDs), and press **Record both searches**.
2. Click **▶ Play** to animate genuine recorded search events, **Ⅱ Pause** to freeze, **↺ Restart** to reset, **⏭ Finish** to reveal both verified paths, or drag the timeline to inspect any frame.
3. Compare **settled vertices**, **discovered vertices**, and **frontier vertices** under the **same expansion budget**. The yellow path appears independently for each algorithm only after that algorithm reaches the goal.
4. For a presentation that does not depend on Streamlit, open `outputs/live_search_demo.html` in any modern desktop browser. This is a fully self-contained HTML with a bundled rendering engine and recorded **real** BFS/A* event data for reference route 918454 → 990282 (426 hops); it works offline. To regenerate for any valid pair:

```bash
python scripts/export_live_demo.py --source 918454 --target 990282
```

### Interpretation and honesty

- **Not GPS:** The SNAP dataset supplies no latitude/longitude. The visualization uses PCA of **exact shortest-hop distances from landmarks** to construct a *topology-derived 2-D projection*; nodes near one another in a plot are **not necessarily geographically close**. Pairs in smaller components use a local graph-distance PCA instead.
- **Not a fake animated search:** The existing production BFS/A* loops record every node's first discovery and exact settlement order in compact numeric arrays. After both searches complete, the browser replays those factual events, so playback is deterministic and supports arbitrary scrubbing. It does **not** pretend to perform live incremental computation.
- **All-node density, sampled dots:** The density field counts **every** settled vertex. Only the marker layer plots a reproducible common sample (plus all verified path nodes) to keep browser performance reasonable on ~1.38 million vertices.
- **Fair timeline:** Both panels have the same graph coordinates, domain, and expansion budget. If A* terminates after fewer settlements, its state freezes while BFS continues. Animation-time seconds are **not** the algorithms' measured wall-clock runtimes (see the Experiments tab for timings).
- **Two valid optimal paths:** BFS and A* can return different routes of the same minimum hop count; each panel reveals its **own** route, not a shared invented path.
- **Memory:** For one full query, the recorder maintains two `int32` timestamp arrays per algorithm; per-frame density maps are preaggregated for responsive controls. Changing nodes reruns actual search once, not once per animation frame.

## Why the A\* heuristic is genuinely meaningful

The data contains edges and node identifiers, **not** coordinates, road distances, traffic speeds, or city names. Thus a Euclidean, Manhattan, or travel-time heuristic would be invented data.

We use **A\* + Landmarks + Triangle inequality (ALT)** instead. Choose eight farthest-first landmarks in the largest connected component and precompute the exact unweighted shortest-path distance `d(L, v)` from each landmark `L` to each reachable node `v`. For destination `t`:

\[
h(v,t)=\max_{L\in\mathcal{L}} |d(L,t)-d(L,v)|.
\]

By the triangle inequality, this is always **admissible** (`h(v,t) <= d(v,t)`) and **consistent** (`h(u,t) <= 1+h(v,t)` for every edge `u-v`). Therefore A\* returns an optimal path measured in number of edges. Nodes in non-giant components safely use `h=0`. The query first checks connected-component labels so impossible routes end immediately.

Optimization: queries with initial ALT bound of at most 40 hops evaluate potentials lazily; longer queries vectorize heuristic computation once. Both paths use the same mathematical heuristic. The 40-hop cutoff is a performance heuristic, **not** an optimality restriction.

## Benchmark and correctness

```bash
python -m pytest tests -q
python -m roadlab.validation
python -m roadlab.cli sample --seed 42 --anchors 3
python -m roadlab.cli benchmark --repeats 1
```

Use `--repeats 3` for median timing measurements on your presentation machine; benchmark time depends on hardware. `sample` regenerates the three distance classes using an **independent SciPy C-backed shortest-path search** from each source. These distances are used only to select/verify test pairs, not to make BFS or A\* cheat.

`benchmark` checks every returned route is a real sequence of adjacent nodes; all three methods must agree on the same minimum hop count. `validation` additionally checks sampled symmetry, heuristic consistency on actual road edges, and source-to-target lower bounds. `pytest` verifies disconnected graphs, duplicate reciprocal arcs, noncontiguous IDs, optimality, consistency, invalid identifiers, and the fidelity of search-event replay on synthetic networks (including paths outside the landmark-covered component).

The included `outputs/benchmark.csv` is a **legacy nine-route single-run snapshot**, not the final experiment. The Experiments tab now uses `outputs/colab_results/*.csv` from the **final, executed Colab notebook**, with 18 routes and three repetitions per method. See [UPDATE_EXPERIMENTS.md](UPDATE_EXPERIMENTS.md) for exact synchronization steps. Old and new results are displayed separately to avoid misleading comparisons.

### Verified bundled sample results (seed 42, 9 pairs)

| Distance class | Example SNAP IDs | Optimal hops | BFS settled | ALT A\* settled |
| --- | --- | ---: | ---: | ---: |
| Local | 918454 → 918849 | 28 | 1,610 | 110 |
| Regional | 918454 → 383065 | 183 | 224,820 | 50,460 |
| Long-range | 918454 → 990282 | 426 | 640,484 | 41,490 |
| Local | 121954 → 121541 | 16 | 570 | 17 |
| Regional | 121954 → 57844 | 131 | 100,639 | 15,707 |
| Long-range | 121954 → 724778 | 413 | 605,290 | 21,241 |
| Local | 1081288 → 1091670 | 24 | 1,830 | 404 |
| Regional | 1081288 → 894840 | 179 | 238,115 | 180 |
| Long-range | 1081288 → 1037922 | 458 | 773,133 | 24,130 |

The median reduction in nodes settled across these nine queries is **93.5%**. The selection is not a random population-wide performance estimate, and reductions in exploration do not guarantee faster elapsed times for every route: heap management and heuristic setup have overhead. The A\* `h=0` control demonstrates that the landmark guidance, not simply using a heap, drives pruning.

## Graph engineering

The supplied `.gz` contains **3,843,320 directed adjacency entries** (often both directions of each road). SNAP states there are **1,921,660 undirected edges**. The loader maps distinct external node identifiers to dense integer indices, removes loops/duplicates, restores symmetry and stores CSR `indptr`/`indices` arrays with 32-bit indices. That is substantially more memory-efficient than creating 1.38 million Python `networkx` node objects. We use SciPy only for preprocessing/landmark computation/reference sampling; **the evaluated BFS and A\* are manually implemented in `roadlab/search.py`**.

### Complexity

For `V` nodes and `E` undirected roads: BFS has `O(V+E)` search time and `O(V)` auxiliary space. A\* has `O((V+E) log V)` worst-case heap work plus heuristic evaluation; both may explore most of the giant component in pathological queries. Landmark precomputation runs one unweighted shortest-path search per landmark, `O(k(V+E))` time and `O(kV)` stored distances for `k=8`. It is **offline** preprocessing, not included in individual query wall-clock timings. `search_array_mb` counts NumPy search-state arrays only, and **is explicitly not process peak RAM**, because Python heap and queue object memory are excluded.

## Project tree

```text
texas-roadnet-ai/
├── app.py                       Streamlit interactive demonstrator
├── roadlab/
│   ├── graph.py                SNAP parser, CSR, components, cache
│   ├── landmarks.py            Farthest-first landmark selection and ALT h
│   ├── search.py               Handwritten BFS, A*, path validation
│   ├── experiments.py          Seeded cases, independent reference, CSVs
│   ├── visualization.py        Faithful, local topology route schematic
│   ├── validation.py           Full-dataset regression suite
│   └── cli.py                  prepare, inspect, route, sample, benchmark
├── data/
│   ├── roadNet-TX.txt.gz       Original user-supplied SNAP dataset
│   ├── graph.npz               Precomputed CSR and component index
│   └── landmarks.npz           Precomputed eight-landmark index
├── outputs/
│   ├── sample_cases.json       Nine seeded reference queries
│   └── benchmark.csv           Baseline reference CSV
├── tests/test_search.py
├── .streamlit/config.toml
└── requirements.txt
```

## Scope and limitations (defend these in the lab viva)

- **Road routing here means fewest road segments (hops)**, not minimum miles or travel time. Such claims would require data not provided by SNAP.
- **No geographic map is possible from this edge list alone.** The interactive plot deliberately draws a neighborhood in abstract topological space, with real links but schematic positions.
- **The graph includes 424 connected components.** A cross-component query cannot return a valid path; that is correctly handled rather than being treated as an algorithm failure.
- **A\* does not always win on elapsed time**, especially for very short routes or an unlucky heuristic. Compare both exploration counts and clock time; preserve negative results.
- Rebuilding/retraining is not required: there is **no machine-learning model and no training/prediction**. This project studies **classical informed vs uninformed state-space search** as required by an AI lab.

## Data provenance

Stanford Network Analysis Project (SNAP), [Texas Road Network](https://snap.stanford.edu/data/roadNet-TX.html). SNAP describes undirected roads between intersections/endpoints, with **1,379,917 nodes**, **1,921,660 undirected edges**, and **1,351,137 nodes in the largest connected component**. Project implementation does not modify the original compressed dataset.
