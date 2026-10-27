# Pancreatic Cancer Literature Embeddings Analysis

**DISCLAIMER**: My research has moved to a private repository and updates will be made soon :)

## Overview

Hello! My name is Margaret Galvez and this is my project for my ICS Honors Thesis. 
Thank you to my advisor, Professor Alberto Krone-Martins for all your guidance and here is a guide on this project.

This project explores **pancreatic cancer research** using natural language embeddings and clustering to uncover patterns, relationships, and potential gaps in the present literature.
Instead of manually reviewing thousands of papers, this pipeline:

- Collects research papers from PubMed using a specially curated query
- Converts them into semantic embeddings (using allenai/specter)
- Visualizes them in 2D space
- Identifies clusters of related research
- Provides descriptive cluster labels and cluster summaries

The goal is to analyze and interpret these embeddings and see what could possibly be improved in current pancreatic cancer literature.

### 1) Create a dataset 
- Input a desired time period and term to search through PubMed database
Run:

```bash
python generate_dataset.py --term "pancreatic cancer" --retmax 5000 --out pubmed_papers.csv
```

### 2) Pick an embedding model (recommended defaults)
For scientific-paper similarity on **titles + abstracts**, current default used is:
- **`allenai/specter`** (good semantic clustering of papers; works cleanly with SentenceTransformers)

### 3) Embed + visualize + cluster

```bash
python embed_viz.py --in runs/<timestamp>/pubmed_papers.csv --model sentence-transformers/allenai-specter
```

A new dataset creates a timestamped run folder. Visualization automatically continues that folder if it has not yet had a visualization stage. When the input CSV is outside a run folder, or its run already has a visualization attempt, visualization creates a new run and copies the input CSV into it.

```text
runs/
  2026-10-27_12-30-00_123456_a1b2c3/
    pubmed_papers.csv
    embedded.csv
    embeddings.npy
    plot.html
    config.json
```

A dataset-only run contains the papers CSV and `config.json`; the visualization stage adds the other three artifacts. Existing datasets and outputs are not moved. Output flags (`--out`, `--embeddings-path`, `--embeddings-csv`, `--plot-path`) now specify filenames inside the run folder; directory components are ignored.

`config.json` records creation time, artifact filenames, and each stage's input settings, start/end timestamps, and status. NCBI email and API keys are excluded. Failed runs retain their configuration and any partial artifacts. GUI force stops are marked `interrupted`; externally killing a CLI process may leave its status as `running`.

Use `--runs-dir experiments` to change the parent folder. The visualization `--outdir` option is an alias for this parent folder. Use `--run-dir runs/<timestamp>` to explicitly continue a run (each stage can execute only once per folder).

## Install

```bash
pip install -r requirements.txt
```


## Desktop GUI

An additional GUI feature was made to make the process of creating these plots more user-friendly. This GUI was created with Python Tkinter for simplicity.

```bash
python gui.py
```

<img width="550" height="600" alt="image" src="https://github.com/user-attachments/assets/d55bd238-2891-4069-a807-8778e837e6e5" />

Run `python gui.py` using the same Python environment as the scripts (on Windows, `.\.venv\Scripts\python.exe gui.py`). Tkinter is required; it is included in standard Windows Python installations.

GUI Options:
- **Generate dataset:** enter a CSV filename, PubMed query, start/end dates (`YYYY/MM/DD`), and maximum paper count. The `NCBI_EMAIL` and optionally `NCBI_API_KEY` variables are expected to be in a `.env`. A successful run fills the input on the visualization tab.
  - Every dataset starts a fresh timestamped folder; existing files within a run are protected. 
- **Embed and visualize:** select the input dataset CSV and output filenames for the embeddings `.npy`, interactive plot `.html`, and embedded `.csv`. Defaults are 100 global terms removed, `sentence-transformers/allenai-specter`, and GMM clustering. Click **Run**, then **Open completed plot** when finished.

The GUI saves all artifacts under the repository?s `runs/` folder. Successful dataset generation fills the visualization input automatically. Runs execute in the background and display script output in the log. The GUI uses the existing GMM configuration (20 clusters), so use a dataset with at least 20 usable papers for GMM.

The visualization CLI also accepts `--embeddings-path`, `--plot-path`, `--embeddings-csv`, and `--global-terms-to-remove`. All output filenames are placed inside the timestamped run folder.

Offline run-folder checks: `python -m unittest discover -s tests -v`.
