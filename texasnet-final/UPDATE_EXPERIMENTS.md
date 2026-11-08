# How to synchronize Streamlit with the final Colab notebook

This update requires **no edits to BFS, A*, the route explorer, or the Live Search tab**.
The Experiments tab uses the final Colab export instead of the old nine-route file.

## Ready-to-run updated Streamlit project

1. Extract the updated project ZIP.
2. Open a terminal inside `texas-roadnet-ai`.
3. Install dependencies: `python -m pip install -r requirements.txt`.
4. Run: `python -m streamlit run app.py`.
5. Open **📊 Experiments**. It initially displays the bundled 18-route, 162-execution benchmark.

The experiment snapshot is stored in `outputs/colab_results/*.csv`. Every accuracy percentage
is **calculated from `all_162_timing_trials.csv`**, where each recorded execution has `found`,
`path_valid`, `optimal`, and reference/returned hop counts. The older benchmark remains under
**Legacy nine-route benchmark** and is not mixed with the new data.

## Updating results after rerunning Colab

1. In the notebook, rerun the experiment and export its results ZIP/CSV files.
2. From the project directory, run:

   ```bash
   python scripts/import_colab_results.py /path/to/your/results.zip
   ```

   You can also pass a directory containing the CSV exports instead of a ZIP.
3. Refresh the Streamlit page. The importer checks the CSVs before replacing old results.

## Exactly what was updated

- `app.py`: Replaced the old Experiments section with Colab result loading, independently
  validated accuracy, search runtime, explored/discovered nodes, frontier comparisons,
  route-distance breakdown, confidence interval, resource-use summary and CSV downloads.
- `roadlab/analysis_results.py`: Strict CSV validation and dynamic accuracy calculation.
- `outputs/colab_results/`: Verified final Colab export CSVs and figures.
- `scripts/import_colab_results.py`: Repeatable future import from Colab result ZIP/directory.
- `tests/test_analysis_results.py`: Tests that accuracy changes with the recorded data.

## Metric interpretation

**Optimal-path accuracy** = number of *found, valid, independently optimal*
reachable-path executions / number of evaluated reachable-path executions. For the
bundled benchmark, BFS and ALT A* each achieve 54/54 (100%).
This is **not** a claim about all possible Texas-road-network routes or a classification score.

The benchmark contains 18 distinct reachable routes with 3 runs of BFS, ALT A* and
zero-heuristic A* each (162 runs in total). The bundled median paired speedup is about
2.12x, and the median node-expansion reduction is about 93.44%. These are results
from the original notebook environment, not live timings from Streamlit.

The notebook also tests disconnected and identical-source routes separately;
they are not counted as failures in the accuracy denominator for reachable paths.

**Do not present this as GPS or driving-time routing.** The original SNAP dataset only
supports a graph of undirected, unit-hop road connections.
