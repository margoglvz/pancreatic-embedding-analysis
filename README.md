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
python pubmed_collect.py --term "pancreatic cancer" --retmax 5000 --out data/pubmed_pancreatic_cancer.csv
```

### 2) Pick an embedding model (recommended defaults)
For scientific-paper similarity on **titles + abstracts**, current default used is:
- **`allenai/specter`** (good semantic clustering of papers; works cleanly with SentenceTransformers)

### 3) Embed + visualize + cluster

```bash
python embed_viz.py --in data/pubmed_pancreatic_cancer.csv --model allenai/specter --outdir outputs
```

This writes:
- `outputs/embedded_2d.csv` (PMID/title/abstract + 2D coords + cluster label)
- `outputs/embeddings.npy` (vector embeddings)
- `outputs/plot.html` (interactive plot)

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
- **Generate dataset:** enter a new CSV path, PubMed query, start/end dates (`YYYY/MM/DD`), and maximum paper count. The `NCBI_EMAIL` and optionally `NCBI_API_KEY` variables are expected to be in a `.env`. A successful run fills the input on the visualization tab.
  - Error Handling: Existing output paths are rejected. 
- **Embed and visualize:** select the input dataset CSV and new output paths for the embeddings `.npy`, interactive plot `.html`, and embedded `.csv`. Defaults are 100 global terms removed, `sentence-transformers/allenai-specter`, and GMM clustering. Click **Run**, then **Open completed plot** when finished.

Relative paths are resolved from the repository folder. Runs execute in the background and display script output in the log. The GUI uses the existing GMM configuration (20 clusters), so use a dataset with at least 20 usable papers for GMM.

The visualization CLI also accepts `--embeddings-path`, `--plot-path`, `--embeddings-csv`, and `--global-terms-to-remove`. Relative CLI output paths are resolved under `--outdir`.
