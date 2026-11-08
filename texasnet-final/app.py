"""Research-grade interactive graph search demonstrator (topology, not GPS)."""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from roadlab import RoadGraph, Landmarks, bfs, astar, validate_path
from roadlab.cli import DEFAULT_GRAPH, DEFAULT_LANDMARKS, DEFAULT_CASES, DEFAULT_BENCHMARK
from roadlab.visualization import route_schematic
from roadlab.trace import TraceRecorder
from roadlab.analysis_results import load_analysis, summarize_accuracy
from roadlab.live import (FRAMES, landmark_projection, local_projection,
                          build_live_scene, search_frame)

st.set_page_config(page_title='Texas RoadNet | Search Lab', page_icon='🛣️',
                   layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container {padding-top:1.45rem; padding-bottom:2.2rem; max-width:1350px}
.hero {padding:20px 25px 18px;border:1px solid #2D3A52;border-radius:16px;
background:linear-gradient(105deg,#13243A 0%,#0E192B 75%);margin-bottom:16px}
.hero small{letter-spacing:0.16em;color:#42D5C4;font-weight:700;font-size:0.72rem}
.hero h1{font-size:2.0rem;letter-spacing:-.04em;color:#F4F8FF;margin:6px 0}
.hero p{color:#B6C5DB;margin:0;font-size:0.93rem}
.pill{font-size:0.74rem;background:#183E40;color:#8DF0E6;border-radius:30px;
 padding:5px 10px;display:inline-block;margin:12px 6px 0 0}
[data-testid="stMetric"]{border:1px solid #2A3B53;background:#111E30;
border-radius:12px;padding:12px 14px}
[data-testid="stMetricLabel"]{color:#AFC0D9}
.sidebar-note{font-size:0.81rem;color:#A7B9CD}
</style>''', unsafe_allow_html=True)

@st.cache_resource(show_spinner='Loading network indexes into memory…')
def load_indexes():
    return RoadGraph.load(DEFAULT_GRAPH), Landmarks.load(DEFAULT_LANDMARKS)

if not DEFAULT_GRAPH.is_file() or not DEFAULT_LANDMARKS.is_file():
    st.error('Network indexes missing. Run `python -m roadlab.cli prepare` in the project directory, then reload.')
    st.stop()
graph, landmarks = load_indexes()

@st.cache_data
def load_cases():
    if DEFAULT_CASES.is_file():
        return json.loads(DEFAULT_CASES.read_text(encoding='utf-8'))
    return []

@st.cache_data
def load_benchmark():
    return pd.read_csv(DEFAULT_BENCHMARK) if DEFAULT_BENCHMARK.is_file() else pd.DataFrame()

COLAB_RESULTS_DIR = Path(__file__).resolve().parent / 'outputs' / 'colab_results'
cases = load_cases()

st.markdown('''<div class="hero"><small>AI SEARCH LAB / STANFORD SNAP DATASET</small>
<h1>Texas RoadNet <span style="color:#36DDC9">/</span> Route Intelligence</h1>
<p>Exact shortest-hop search on a 1.38-million-node road network · BFS vs landmark-guided A*</p>
<span class="pill">GRAPH SEARCH</span><span class="pill">ALT HEURISTICS</span>
<span class="pill">REPRODUCIBLE EXPERIMENTS</span></div>''', unsafe_allow_html=True)

st.sidebar.markdown('### Network index')
st.sidebar.metric('Nodes', f'{graph.n:,}')
st.sidebar.metric('Undirected edges', f'{graph.m:,}')
st.sidebar.metric('Connected components', f'{len(graph.component_sizes):,}')
st.sidebar.metric('Largest component', f'{graph.largest_size / graph.n:.2%}')
st.sidebar.caption(f'{len(landmarks.nodes)} farthest-first landmarks · CSR adjacency')
st.sidebar.divider()

st.sidebar.info('Each road segment costs **1 hop**. The dataset has no GPS coordinates, road lengths, speed limits, or city labels. Visualizations are topological schematics, not maps.')
st.sidebar.markdown('[Dataset documentation](https://snap.stanford.edu/data/roadNet-TX.html)')

route_tab, live_tab, benchmark_tab, graph_tab, method_tab = st.tabs([
    '🧭 Route Explorer', '▶️ Live Search', '📊 Experiments',
    '🕸️ Network Structure', '📐 Algorithm Design'])

with route_tab:
    st.markdown('#### Query the actual road network')
    if cases:
        names = ['Custom node IDs'] + [f"Case {i+1} · {c['category']} · {c['reference_hops']} hops"
                                        for i, c in enumerate(cases)]
        preset = st.selectbox('Example route (or enter your own SNAP IDs)', names, index=2)
        chosen = cases[preset_idx] if (preset_idx := names.index(preset) - 1) >= 0 else None
    else:
        chosen = None
    with st.form('search_form'):
        ca, cb, cc = st.columns([1, 1, 1])
        with ca:
            source_id = st.number_input('Source node ID', min_value=0,
                max_value=int(graph.ids[-1]), value=int(chosen['source']) if chosen else int(graph.ids[0]), step=1)
        with cb:
            target_id = st.number_input('Target node ID', min_value=0,
                max_value=int(graph.ids[-1]), value=int(chosen['target']) if chosen else int(graph.ids[-1]), step=1)
        with cc:
            mode = st.selectbox('Search mode', ['Compare both', 'BFS', 'A* (ALT)', 'A* (h=0)'])
        submit = st.form_submit_button('Run route search', type='primary', use_container_width=True)
    if submit:
        try:
            s, t = int(source_id), int(target_id)
            a, b = graph.internal_id(s), graph.internal_id(t)
            requested = ['BFS', 'A* (ALT)'] if mode == 'Compare both' else [mode]
            with st.spinner('Searching graph and verifying the returned paths…'):
                results = [bfs(graph, s, t) if name == 'BFS' else
                           astar(graph, s, t, landmarks, name != 'A* (h=0)')
                           for name in requested]
                if not all(validate_path(graph, r) for r in results):
                    raise AssertionError('An edge of a returned path failed validation')
                if len(results) > 1 and len({r.hops for r in results}) != 1:
                    raise AssertionError('Algorithm shortest-path costs differ')
            st.session_state['search_results'] = results
            st.session_state['search_context'] = (s, t, graph.same_component(a, b))
        except (ValueError, AssertionError) as exc:
            st.error(str(exc))
    if 'search_results' in st.session_state:
        results = st.session_state['search_results']
        s, t, reachable = st.session_state['search_context']
        st.markdown(f'**Latest validated query:** `{s}` → `{t}`')
        if not reachable:
            st.warning('These nodes are in different connected components, so no road path exists between them.')
        elif results[0].found:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Shortest path', f'{results[0].hops:,} hops')
            c2.metric('BFS nodes settled', f'{next((r.expanded for r in results if r.algorithm == "BFS"), 0):,}'
                      if any(r.algorithm=='BFS' for r in results) else '—')
            c3.metric('ALT nodes settled', f'{next((r.expanded for r in results if r.algorithm == "A* (ALT)"), 0):,}'
                      if any(r.algorithm=='A* (ALT)' for r in results) else '—')
            c4.metric('Path integrity', 'VERIFIED')
            records = [{'Algorithm':r.algorithm, 'Route hops':r.hops,
                        'Nodes settled':r.expanded, 'Unique discovered':r.discovered,
                        'Peak frontier':r.peak_frontier, 'Runtime (ms)':round(r.elapsed_ms, 2),
                        'Array MB (lower bound)':round(r.search_array_mb, 2)} for r in results]
            st.dataframe(pd.DataFrame(records), hide_index=True, use_container_width=True)
            if len(results) > 1:
                cl, cr = st.columns(2)
                with cl:
                    fig = px.bar(pd.DataFrame(records), x='Algorithm', y='Nodes settled', color='Algorithm',
                                 title='Search effort · lower is better', text_auto=',',
                                 color_discrete_sequence=['#8399B7','#2DD5C5'])
                    fig.update_layout(showlegend=False, template='plotly_dark', height=330,
                                      margin=dict(l=10,r=10,t=55,b=25))
                    st.plotly_chart(fig, use_container_width=True)
                with cr:
                    fig = px.bar(pd.DataFrame(records), x='Algorithm', y='Runtime (ms)', color='Algorithm',
                                 title='Single-query wall-clock time · lower is better', text_auto='.1f',
                                 color_discrete_sequence=['#8399B7','#2DD5C5'])
                    fig.update_layout(showlegend=False, template='plotly_dark', height=330,
                                      margin=dict(l=10,r=10,t=55,b=25))
                    st.plotly_chart(fig, use_container_width=True)
            if any(r.algorithm == 'A* (ALT)' for r in results):
                alt = next(r for r in results if r.algorithm == 'A* (ALT)')
                st.caption(f'ALT initial lower bound h(source) = {alt.heuristic_at_source} hops; '
                           f'exact shortest path = {alt.hops} hops. '
                           'The lower bound never exceeds the true distance.')
            st.markdown('#### Shortest path · actual road edges')
            path = results[0].path
            max_offset = max(0, len(path) - 2)
            if max_offset:
                offset = st.slider('View a segment of the route', min_value=0,
                                   max_value=max_offset, value=0, step=1,
                                   help='Each section shows up to 20 consecutive path vertices plus their direct neighbors.')
            else:
                offset = 0
                st.caption('This short route fits in a single view.')
            st.plotly_chart(route_schematic(graph, path, offset), use_container_width=True)
            st.caption('Schematic layout only. Horizontal position shows sequence along the path, '
                       'not latitude, longitude, or physical road length. The drawn edges are real dataset connections.')
            with st.expander('Inspect every path node / export result'):
                path_data = pd.DataFrame({'Step':np.arange(len(path)), 'SNAP node ID':path,
                                          'Degree':[graph.degree(graph.internal_id(n)) for n in path]})
                st.dataframe(path_data, hide_index=True, use_container_width=True, height=260)
                st.download_button('Download this route as CSV',
                                   data=path_data.to_csv(index=False).encode('utf-8'),
                                   file_name=f'route_{s}_{t}.csv', mime='text/csv')
        else:
            st.warning('No route could be found.')
    else:
        st.caption('Select a representative pair and run the algorithms to compare exact paths and their exploration cost.')

@st.cache_resource(show_spinner='Building topology-derived network projection…')
def giant_projection():
    # One projection shared across queries, not separately fitted to algorithms.
    return landmark_projection(graph, landmarks)


def on_live_preset_change():
    chosen_case = cases[example_names.index(st.session_state['live_choice'])]
    st.session_state['live_source'] = int(chosen_case['source'])
    st.session_state['live_target'] = int(chosen_case['target'])


with live_tab:
    st.markdown('#### Live search · actual BFS and A* exploration')
    st.caption('Watch **real node-settlement events** replayed after both searches run. '
               'The charts share the same nodes, axes and expansion budget—not an artificial animation of search.')

    example_names = [f"Case {i+1} · {case['category']} · {case['reference_hops']} hops"
                     for i, case in enumerate(cases)]
    if cases:
        choice = st.selectbox('Reference route', example_names,
                              index=min(2, len(cases)-1), key='live_choice',
                              on_change=lambda: on_live_preset_change())
        selected_case = cases[example_names.index(choice)]
        initial_s, initial_t = int(selected_case['source']), int(selected_case['target'])
    else:
        initial_s, initial_t = int(graph.ids[0]), int(graph.ids[-1])

    with st.form('live_form'):
        input1, input2, action = st.columns([1, 1, 1])
        with input1:
            live_s = st.number_input('Start SNAP node', 0, int(graph.ids[-1]),
                                     value=initial_s, step=1, key='live_source')
        with input2:
            live_t = st.number_input('Destination SNAP node', 0, int(graph.ids[-1]),
                                     value=initial_t, step=1, key='live_target')
        with action:
            st.write('**Build a search replay**')
            start_live = st.form_submit_button('Record both searches', type='primary',
                                                use_container_width=True)

    if start_live:
        for item in ('live_bundle', 'live_issue'):
            st.session_state.pop(item, None)
        st.session_state['live_playing'] = False
        st.session_state['live_frame'] = 0
        try:
            s_id, t_id = int(live_s), int(live_t)
            src, dst = graph.internal_id(s_id), graph.internal_id(t_id)
            if src == dst:
                st.session_state['live_issue'] = 'Start and destination are identical. Pick distinct nodes to see exploration.'
            elif not graph.same_component(src, dst):
                st.session_state['live_issue'] = 'Different connected components: no route exists. Choose a reachable pair.'
            else:
                with st.spinner('Running production BFS and ALT A*, capturing actual search events…'):
                    rec_b = TraceRecorder(graph.n, src)
                    b = bfs(graph, s_id, t_id, trace=rec_b)
                    rec_a = TraceRecorder(graph.n, src)
                    a = astar(graph, s_id, t_id, landmarks, trace=rec_a)
                    if not (b.found and a.found and b.hops == a.hops and
                            validate_path(graph, b) and validate_path(graph, a)):
                        raise AssertionError('Trace creation failed shortest-path verification')
                    tb, ta = rec_b.snapshot(), rec_a.snapshot()
                    if (tb.expanded != b.expanded or ta.expanded != a.expanded or
                        int(tb.discovered_prefix[-1]) != b.discovered or
                        int(ta.discovered_prefix[-1]) != a.discovered):
                        raise AssertionError('Recorded event counts do not match search results')
                with st.spinner('Preparing the common road-network projection and animation frames…'):
                    in_giant = int(graph.component[src]) == graph.largest_component
                    projection = giant_projection() if in_giant else local_projection(graph,src,dst)
                    scene = build_live_scene(graph, projection, tb, ta, b, a,
                        'ALT landmark-distance PCA' if in_giant else 'component-local graph-distance PCA')
                    st.session_state['live_bundle'] = (scene, tb, ta, b, a)
        except (ValueError, AssertionError, MemoryError) as exc:
            st.session_state['live_issue'] = str(exc)

    if 'live_issue' in st.session_state:
        st.warning(st.session_state['live_issue'])

    if 'live_bundle' not in st.session_state:
        st.info('Select an example or enter a connected pair of SNAP nodes, then click **Record both searches**. '
                'BFS and A* search the full graph; only plotting is sampled for responsiveness.')
    else:
        scene, tb, ta, b, a = st.session_state['live_bundle']
        if 'live_frame' not in st.session_state:
            st.session_state['live_frame'] = 0
        if 'live_playing' not in st.session_state:
            st.session_state['live_playing'] = False
        if 'live_speed' not in st.session_state:
            st.session_state['live_speed'] = 6
        st.success(f'Validated: **{b.source:,} → {b.target:,}** · '
                   f'**{b.hops:,} hops** · both algorithms return optimal routes.')

        k1, k2, k3, k4 = st.columns(4)
        k1.metric('BFS total settlements', f'{b.expanded:,}')
        k2.metric('A* total settlements', f'{a.expanded:,}')
        k3.metric('A* exploration savings', f'{(1-a.expanded/b.expanded):.1%}')
        k4.metric('Optimal path agreement', f'{b.hops} = {a.hops} hops')

        @st.fragment(run_every='350ms')
        def live_player():
            # Time-based playback continues while this tab is open. No background
            # worker or re-running million-node searches for each visual frame.
            st.markdown('##### Event playback')
            ctl0, ctl1, ctl2, ctl3, ctl4 = st.columns([1, 1, 1, 1.35, 1.35])
            with ctl0:
                if st.button('↺ Restart', use_container_width=True):
                    st.session_state['live_playing'] = False
                    st.session_state['live_frame'] = 0
            with ctl1:
                if st.button('▶ Play' if not st.session_state['live_playing'] else 'Ⅱ Pause',
                             type='primary', use_container_width=True):
                    if st.session_state['live_playing']:
                        st.session_state['live_playing'] = False
                    else:
                        if st.session_state['live_frame'] >= FRAMES:
                            st.session_state['live_frame'] = 0
                        st.session_state['live_playing'] = True
                        st.session_state['live_started'] = time.monotonic()
                        st.session_state['live_start_frame'] = st.session_state['live_frame']
            with ctl2:
                if st.button('⏭ Finish', use_container_width=True):
                    st.session_state['live_playing'] = False
                    st.session_state['live_frame'] = FRAMES
            with ctl3:
                st.select_slider('Playback speed (frames/s)', options=[1, 3, 6, 12, 24],
                                 key='live_speed')
            with ctl4:
                st.caption('**Playback is a replay**, with more frames allocated to early exploration. '
                           'Both algorithms receive the same node-expansion budget.')

            if st.session_state['live_playing']:
                elapsed = time.monotonic() - st.session_state['live_started']
                tick = int(st.session_state['live_start_frame'] + elapsed*st.session_state['live_speed'])
                st.session_state['live_frame'] = min(FRAMES, tick)
                if tick >= FRAMES:
                    st.session_state['live_playing'] = False

            def pause_on_scrub():
                st.session_state['live_playing'] = False

            frame = st.slider('Search event timeline', 0, FRAMES,
                              key='live_frame', on_change=pause_on_scrub)
            budget = int(scene.budgets[frame])
            settled_b, discovered_b, frontier_b = tb.stats(budget)
            settled_a, discovered_a, frontier_a = ta.stats(budget)
            st.caption(f'**Same search budget: {budget:,} node settlements per algorithm** '
                       '(or until the destination is reached). '
                       'This PCA topology view is **not** a geographic Texas map.')
            left, right = st.columns(2, gap='small')
            with left:
                st.plotly_chart(search_frame(scene,tb,b,'BFS',frame),
                                use_container_width=True, config={'displayModeBar': False})
                m1, m2, m3 = st.columns(3)
                m1.metric('Settled', f'{settled_b:,}')
                m2.metric('Discovered', f'{discovered_b:,}')
                m3.metric('Frontier', f'{frontier_b:,}')
            with right:
                st.plotly_chart(search_frame(scene,ta,a,'A* · ALT',frame),
                                use_container_width=True, config={'displayModeBar': False})
                m1, m2, m3 = st.columns(3)
                m1.metric('Settled', f'{settled_a:,}')
                m2.metric('Discovered', f'{discovered_a:,}')
                m3.metric('Frontier', f'{frontier_a:,}')
            if settled_a == ta.expanded and settled_b < tb.expanded:
                st.info(f'**A* has already reached the goal** after {ta.expanded:,} settlements. '
                        'With the same event budget, BFS has not reached it yet.')
            elif settled_a == ta.expanded and settled_b == tb.expanded:
                st.success('Both searches have reached the goal. The **yellow line** connects '
                           'verified real road edges of the optimal path.')

        live_player()
        with st.expander('Visualization integrity and interpretation', expanded=False):
            st.markdown(f"""- **Search events are real.** BFS and A* use the same production
                implementations as Route Explorer; each vertex settlement and first discovery
                is recorded without approximating the algorithm.
            - **Topological projection:** {scene.projection_name}. It uses shortest-hop
                graph distances, not longitude/latitude. Shapes and physical distances are **not** geographic.
            - **Density layer:** counts **all** settled vertices in each displayed region.
                Small points are a deterministic, common sample of discovered vertices.
                Orange means discovered but not yet settled; teal means settled;
                gray means not yet discovered. Source S and destination T are always pinned.
            - **True comparison:** both charts show **the same actual vertex positions** and
                use the same node-settlement budget. A* freezes at its real completion point.
            - **Path:** the yellow polyline is revealed only after each algorithm reaches the
                goal and follows a validated sequence of original graph edges.
            - **Not streaming computation:** the original searches complete before playback.
                Replaying event timestamps is deterministic and allows scrubbing / pausing
                without computing millions of nodes again.""")

with benchmark_tab:
    st.markdown('#### Colab-verified experiment results')
    st.caption('Recorded results from the **final standalone Colab notebook**. '
               'These are measured historical experiments, not new searches triggered by opening this tab. '
               'Accuracy is calculated from individual validated search trials, not hardcoded.')
    try:
        analysis = load_analysis(COLAB_RESULTS_DIR)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        st.warning(f'Final Colab benchmark CSVs are not ready: {exc}')
        st.info('Copy the notebook-exported CSV files into `outputs/colab_results/` '
                'and refresh the app. The old 9-route reference is still available below.')
        analysis = None

    if analysis is not None:
        trials = analysis['trials']
        routes = analysis['routes']
        paired = analysis['pairs']
        accuracy = analysis['accuracy']
        environment = analysis['environment'].iloc[0]
        statistics = analysis['statistics'].iloc[0]
        dataset = analysis['dataset'].iloc[0]
        acc_index = accuracy.set_index('Algorithm')
        bfs_acc = float(acc_index.loc['BFS', 'Optimal-path accuracy (%)'])
        alt_acc = float(acc_index.loc['A* (ALT)', 'Optimal-path accuracy (%)'])
        median_speedup = float(paired['speedup_BFS_over_Astar'].median())
        median_reduction = float(paired['expanded_reduction_pct'].median())

        a, b, c, d, e = st.columns(5)
        a.metric('BFS optimal-path accuracy', f'{bfs_acc:.1f}%')
        b.metric('ALT A* optimal-path accuracy', f'{alt_acc:.1f}%')
        c.metric('Reachable test routes', f'{len(paired)}')
        d.metric('Median A* speedup', f'{median_speedup:.2f}×')
        e.metric('Median nodes saved by A*', f'{median_reduction:.2f}%')
        st.caption('Accuracy = valid, independently shortest-hop-optimal route executions / '
                   'all evaluated reachable route executions. The denominator is 54 runs per algorithm '
                   '(18 routes × 3 repetitions), not all 1.38 million nodes. '
                   'Different shortest paths may have equal optimal hop counts.')

        st.markdown('##### Verification and experimental protocol')
        left, right = st.columns([1.3, 1])
        with left:
            st.dataframe(accuracy, hide_index=True, use_container_width=True)
        with right:
            st.metric('Recorded search executions', f'{len(trials):,}')
            st.metric('Total notebook execution', f"{float(environment['total_notebook_seconds']):.2f} s")
            st.metric('Search benchmark wall time', f"{float(environment['benchmark_wall_sec']):.2f} s")
        st.caption('The notebook uses a separate SciPy shortest-path reference for route selection '
                   'and optimal-cost checking. BFS, ALT A* and A* with h=0 are each measured '
                   'three times per route. These times were recorded on the notebook execution machine; '
                   'your own machine may differ. Preprocessing time is excluded from query timing.')

        categories = sorted(paired['category'].dropna().unique().tolist())
        selected = st.multiselect('Filter route distance categories', categories,
                                  default=categories, key='colab_distance_filter')
        filtered_routes = routes[routes['category'].isin(selected)].copy()
        filtered_paired = paired[paired['category'].isin(selected)].copy()
        filtered_trials = trials[trials['category'].isin(selected)].copy()

        if filtered_routes.empty:
            st.info('Select at least one route distance category to see the comparison charts.')
        else:
            st.markdown('##### Execution cost: same routes, three search strategies')
            algorithms = ['BFS', 'A* (ALT)', 'A* (h=0)']
            filtered_routes['case_label'] = 'Route ' + filtered_routes['case_id'].astype(str)
            metric_choice = st.radio('Performance metric',
                ['Execution time (ms)', 'Nodes expanded', 'Nodes discovered', 'Peak frontier'],
                horizontal=True, key='colab_chart_metric')
            metric_columns = {
                'Execution time (ms)': 'median_ms',
                'Nodes expanded': 'expanded',
                'Nodes discovered': 'discovered',
                'Peak frontier': 'peak_frontier',
            }
            y_field = metric_columns[metric_choice]
            fig = px.bar(filtered_routes, x='case_label', y=y_field, color='algorithm',
                         barmode='group', facet_col='category', facet_col_wrap=3,
                         category_orders={'algorithm':algorithms},
                         labels={'case_label':'Test route', y_field:metric_choice,
                                 'algorithm':'Search method'},
                         title=metric_choice + ' by route · lower is better',
                         color_discrete_map={'BFS':'#8AA1BF','A* (ALT)':'#2EDAC5',
                                             'A* (h=0)':'#F7BF78'})
            fig.update_xaxes(type='category')
            fig.update_layout(template='plotly_dark', height=470,
                              margin=dict(l=10,r=10,t=65,b=20))
            st.plotly_chart(fig, use_container_width=True)

            lcol, rcol = st.columns(2)
            with lcol:
                fig = px.scatter(filtered_paired, x='median_ms_BFS', y='median_ms_A* (ALT)',
                                 color='category', log_x=True, log_y=True,
                                 hover_data=['case_id', 'reference_hops', 'source_snap', 'target_snap'],
                                 labels={'median_ms_BFS':'BFS median (ms)',
                                         'median_ms_A* (ALT)':'ALT A* median (ms)'},
                                 title='Paired runtime · points below diagonal favor A*')
                values = pd.concat([filtered_paired['median_ms_BFS'],
                                    filtered_paired['median_ms_A* (ALT)']])
                low = max(float(values.min()) * .8, 0.001)
                high = float(values.max()) * 1.3
                fig.add_shape(type='line', x0=low, y0=low, x1=high, y1=high,
                              line=dict(color='#B7C0CF', dash='dash', width=1))
                fig.update_layout(template='plotly_dark', height=410)
                st.plotly_chart(fig, use_container_width=True)
            with rcol:
                fig = px.scatter(filtered_paired, x='expanded_BFS', y='expanded_A* (ALT)',
                                 color='category', log_x=True, log_y=True,
                                 hover_data=['case_id', 'reference_hops'],
                                 labels={'expanded_BFS':'BFS nodes expanded',
                                         'expanded_A* (ALT)':'ALT A* nodes expanded'},
                                 title='Paired search effort · below diagonal favors A*')
                values = pd.concat([filtered_paired['expanded_BFS'],
                                    filtered_paired['expanded_A* (ALT)']])
                low = max(float(values.min()) * .8, 1)
                high = float(values.max()) * 1.3
                fig.add_shape(type='line', x0=low, y0=low, x1=high, y1=high,
                              line=dict(color='#B7C0CF', dash='dash', width=1))
                fig.update_layout(template='plotly_dark', height=410)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown('##### Breakdown for the selected distance classes')
            class_summary = analysis['distance']
            st.dataframe(class_summary[class_summary['category'].isin(selected)],
                         hide_index=True, use_container_width=True)
            st.caption('Each query has a BFS, ALT A* and zero-heuristic A* result. '
                       'The h=0 control distinguishes priority-queue overhead from heuristic benefit. '
                       'Speedup is defined as BFS median milliseconds / A* median milliseconds; '
                       'values below 1 mean BFS was faster.')
            with st.expander('Correctness rates for selected routes'):
                st.dataframe(summarize_accuracy(filtered_trials), hide_index=True,
                             use_container_width=True)

        st.markdown('##### Resource use, preprocessing and experiment limits')
        m1, m2, m3 = st.columns(3)
        m1.metric('Graph construction', f"{float(environment['graph_build_sec']):.2f} s")
        m2.metric('Landmark precomputation', f"{float(environment['landmark_build_sec']):.2f} s")
        m3.metric('A* faster on routes',
                  f"{int((paired['speedup_BFS_over_Astar'] > 1).sum())}/{len(paired)}")
        st.dataframe(analysis['memory'], hide_index=True, use_container_width=True)
        with st.expander('Measured process-memory samples and special-case tests'):
            st.caption('The search-array column is an algorithmic allocation estimate, '
                       'not total application RAM. The isolated unique-memory samples '
                       'are supplementary machine-dependent measurements.')
            st.dataframe(analysis['unique_memory'], hide_index=True, use_container_width=True)
            st.dataframe(analysis['special'], hide_index=True, use_container_width=True)
        with st.expander('Dataset, confidence interval and limitations'):
            st.write(f"**Network:** {int(dataset['nodes']):,} nodes, "
                     f"{int(dataset['undirected_edges']):,} undirected edges; "
                     f"{int(dataset['components']):,} components.")
            st.write(f"**Paired speedup bootstrap 95% CI:** "
                     f"{float(statistics['speedup_cluster_bootstrap_95pct_low']):.2f}× to "
                     f"{float(statistics['speedup_cluster_bootstrap_95pct_high']):.2f}× "
                     '(from the notebook protocol).')
            st.info('100% accuracy is observed on the evaluated reachable routes, '
                    'not a guaranteed global success rate for every possible node pair. '
                    'These are unweighted shortest-hop paths, not travel-time or GPS routes. '
                    'Different equally short paths are equally correct. '
                    'The dataset provides neither driving speeds nor geographic coordinates.')

        l, r = st.columns(2)
        with l:
            st.download_button('Download all raw timing trials (CSV)',
                data=trials.to_csv(index=False).encode('utf-8'),
                file_name='texas_roadnet_final_timing_trials.csv', mime='text/csv')
        with r:
            st.download_button('Download paired BFS vs A* comparison (CSV)',
                data=paired.to_csv(index=False).encode('utf-8'),
                file_name='texas_roadnet_final_paired_comparison.csv', mime='text/csv')



with graph_tab:
    st.markdown('#### What this network actually contains')
    ncomp = len(graph.component_sizes)
    d = np.diff(graph.indptr)
    x, y, z = st.columns(3)
    x.metric('Connected components', f'{ncomp:,}')
    y.metric('Outside the giant component', f'{graph.n-graph.largest_size:,}')
    z.metric('Mean degree', f'{d.mean():.2f}')
    left, right = st.columns([1.4,1])
    with left:
        degree_hist = np.bincount(d)
        degree_df = pd.DataFrame({'Degree':np.arange(len(degree_hist)), 'Nodes':degree_hist})
        fig = px.bar(degree_df[degree_df.Nodes>0], x='Degree', y='Nodes',
                     title='Observed degree distribution', log_y=True,
                     color_discrete_sequence=['#2EDAC5'])
        fig.update_layout(template='plotly_dark', height=370, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        size_count = pd.Series(graph.component_sizes).value_counts().sort_index()
        comp_df = pd.DataFrame({'Component size':size_count.index, 'Number of components':size_count.values})
        fig = px.scatter(comp_df, x='Component size', y='Number of components',
                         log_x=True, log_y=True, size='Number of components',
                         title='Distribution of component sizes',
                         color_discrete_sequence=['#F1BF80'])
        fig.update_layout(template='plotly_dark', height=370, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('**Why this matters:** Some pairs are physically disconnected within this dataset. '
                'The application checks component membership before running either algorithm, '
                'so unreachable queries terminate immediately rather than exhaustively searching millions of vertices.')

with method_tab:
    st.markdown('#### Design choices and why they are defensible')
    st.markdown(r'''
**Problem.** Find a minimum-number-of-edges path from intersection/node $s$ to node $t$.
All edges are undirected and cost exactly one; node IDs are identifiers, not geographical coordinates.

**Representation.** Raw SNAP node IDs can have gaps. They are mapped to contiguous 32-bit
integer indices and stored using compressed sparse row (CSR) arrays. Reciprocal road entries
and duplicate lines are reduced to a single undirected edge.

**BFS (uninformed).** Uses a FIFO queue, tracks predecessors on first discovery,
and reaches each vertex at minimum depth. For an unweighted graph,
its solution is complete and shortest-hop optimal within any connected component.

**A* with ALT (informed).** Select eight farthest-first landmark nodes in the giant component.
Precompute exact shortest-hop distances from each landmark $L$ using graph search. The search heuristic is

$$h(v,t)=\max_{L\in\mathcal{L}}\left|d(L,t)-d(L,v)\right|.$$

On an undirected graph, the triangle inequality makes this a valid lower bound.
It is also consistent: $h(u,t)\le 1+h(v,t)$ for every edge $(u,v)$.
Consequently, A* returns an optimal shortest-hop path with no need to reopen settled nodes.
Vertices in smaller components receive zero heuristic (safe but uninformed).

**Optimization.** For small source lower bounds, calculate ALT values only when vertices are visited;
for longer routes, compute the complete vectorized heuristic once. Both use the exact same formula.

**Evaluation.** Compare optimal path depth, settled vertices, unique discovered vertices,
maximum frontier size and elapsed wall-clock time. The optional A* ($h=0$) control isolates the
contribution of the heuristic. All recorded results independently validate consecutive edges
and cross-check path length against BFS and SciPy.
''')
    st.warning('Important limitation: This is shortest-**hop** routing, not the fastest or physically shortest '
               'driving route. The dataset does not support those claims without external road weights or coordinates.')
