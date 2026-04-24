# Pancreatic Cancer Literature Embeddings Analysis

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
- **Currently querying 5,000 abstracts** with key and MeSH words to make sure the embedding space accurately represents pancreatic cancer research

Run:

```bash
python pubmed_collect.py --term "pancreatic cancer" --retmax 5000 --out data/pubmed_pancreatic_cancer.csv
```

### 2) Pick an embedding model (recommended defaults)
For scientific-paper similarity on **titles + abstracts**, current default used is:
- **`allenai/specter`** (good semantic clustering of papers; works cleanly with SentenceTransformers)
- 
### 3) Embed + visualize + cluster

```bash
python embed_viz.py --in data/pubmed_pancreatic_cancer.csv --model allenai/specter --outdir outputs
```

This writes:
- `outputs/embedded_2d.csv` (PMID/title/abstract + 2D coords + cluster label)
- `outputs/embeddings.npy` (vector embeddings)
- `outputs/plot.html` (interactive plot)

This repo does a first pass by adding `bridge_keywords` for papers that lie near the “between clusters” region. By clicking on 'plot.html' it will open an interactive map of embeddings.

## Install

```bash
pip install -r requirements.txt
```

