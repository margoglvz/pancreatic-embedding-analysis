# Pancreatic Embedding Analysis

Pipeline for **collecting PubMed abstracts**, **embedding them**, then **visualizing + cluster** to explore research “gaps”.

### 1) Create a dataset (how much data?)
- **Start with 2,000–10,000 abstracts** for a single query (e.g., `"pancreatic cancer"`)

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

