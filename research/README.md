# Research data and code

The data, notebooks and analysis scripts behind the BeeMonitor preprint
([bioRxiv](https://www.biorxiv.org/content/10.64898/2026.07.10.737879v1)). Field data were recorded with
BeeMonitor units at bee hotels in 2024; each site has a short name (for example `mendels`, `natalies`).

| Path | What it is |
|---|---|
| [`data_and_code/`](data_and_code/) | The final data and notebooks for the paper: hardware evaluation, software evaluation (detection, tracking and event accuracy against manual annotation) and foraging versus nesting. `research_data/` holds the CSVs the notebooks read |
| `data_and_code.zip` | The same folder as one download |
| [`analysis/`](analysis/) | Scripts that produced the data: batch processing on a GPU cluster (`batch_process.py`, `run_analysis.slurm`), extraction and aggregation (`extract_data.py`, `load.py`), plots, and the notebooks used to tune and test the analysis engine (`engine_*.ipynb`) |
| [`working_data/`](working_data/) | Intermediate tables from earlier analyses: per-release foraging trips and nesting data |
| [`natalies_pull/`](natalies_pull/) | The raw per-clip event tables for the `natalies` site (`natalies_events_raw.zip`, 1,885 CSVs) and the scripts that pulled and split them |

The notebooks were run from the repository root with the `beemonitor` package installed
([`src/README.md`](../src/README.md)). Some cells point at the raw videos, which are not in the repository
because of their size.

To cite the data or code, use the citation in the [main README](../README.md#citation).
