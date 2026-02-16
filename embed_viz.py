"""
Embed abstracts, reduce to 2D, cluster, and export an interactive plot.

Usage:
  python embed_viz.py --in data/pubmed_pancreatic_cancer.csv --text-col abstract --outdir outputs

Default embedding model:
  - "allenai/specter" (great for scientific paper similarity; SentenceTransformers-compatible)
Alternatives:
  - "pritamdeka/S-PubMedBert-MS-MARCO" (biomed-tuned sentence embeddings)
  - "sentence-transformers/all-MiniLM-L6-v2" (fast baseline)

Note on SPECTER2:
  - "allenai/specter2" is not a SentenceTransformers-packaged model, and may require a custom
    Transformers loading pipeline. If you want SPECTER2 specifically, start with "allenai/specter"
    first to validate your pipeline.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import umap
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import hdbscan
except Exception:  # pragma: no cover
    hdbscan = None


@dataclass
class BridgeResult:
    idx: int
    cluster_a: int
    cluster_b: int
    score: float


def embed_texts(texts: List[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def reduce_umap(emb: np.ndarray, n_neighbors: int = 20, min_dist: float = 0.05, random_state: int = 42) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(emb)


def cluster_hdbscan(emb: np.ndarray, min_cluster_size: int = 25) -> np.ndarray:
    if hdbscan is None:
        raise RuntimeError("hdbscan is not installed. Install it or use --cluster none.")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    return clusterer.fit_predict(emb)


def pick_two_largest_clusters(labels: np.ndarray) -> Optional[tuple[int, int]]:
    vals, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(vals) < 2:
        return None
    order = np.argsort(-counts)
    return int(vals[order[0]]), int(vals[order[1]])


def bridge_score(emb: np.ndarray, labels: np.ndarray, a: int, b: int) -> BridgeResult:
    # A simple, practical "bridge" heuristic:
    # pick the paper that is simultaneously similar to BOTH cluster centroids.
    idx_a = np.where(labels == a)[0]
    idx_b = np.where(labels == b)[0]
    ca = emb[idx_a].mean(axis=0, keepdims=True)
    cb = emb[idx_b].mean(axis=0, keepdims=True)
    ca /= np.linalg.norm(ca) + 1e-12
    cb /= np.linalg.norm(cb) + 1e-12
    s_to_a = (emb @ ca.T).ravel()
    s_to_b = (emb @ cb.T).ravel()
    score = np.minimum(s_to_a, s_to_b)  # must be good to both
    i = int(np.argmax(score))
    return BridgeResult(idx=i, cluster_a=a, cluster_b=b, score=float(score[i]))


def keywords_for_texts(texts: List[str], top_n: int = 8) -> List[List[str]]:
    kw = KeyBERT(model="all-MiniLM-L6-v2")  # lightweight just for keywords
    out: List[List[str]] = []
    for t in texts:
        pairs = kw.extract_keywords(t, top_n=top_n, stop_words="english", use_mmr=True, diversity=0.6)
        out.append([p[0] for p in pairs])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="data/pubmed_pancreatic_cancer.csv")
    p.add_argument("--text-col", default="abstract")
    p.add_argument("--model", default="allenai/specter")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cluster", choices=["hdbscan", "none"], default="hdbscan")
    p.add_argument("--min-cluster-size", type=int, default=25)
    p.add_argument("--outdir", default="outputs")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.inp)
    df = df.dropna(subset=[args.text_col]).reset_index(drop=True)
    texts = df[args.text_col].astype(str).tolist()

    # Check if embeddings already exist (cache)
    emb_path = os.path.join(args.outdir, "embeddings.npy")
    if os.path.exists(emb_path):
        print(f"Loading cached embeddings from {emb_path}")
        emb = np.load(emb_path)
        if len(emb) != len(texts):
            print(f"Warning: cached embeddings ({len(emb)} rows) don't match data ({len(texts)} rows). Recomputing...")
            emb = embed_texts(texts, args.model, batch_size=args.batch_size)
            np.save(emb_path, emb)
    else:
        print("Computing embeddings (this may take a while)...")
        emb = embed_texts(texts, args.model, batch_size=args.batch_size)
        np.save(emb_path, emb)
    
    xy = reduce_umap(emb)

    if args.cluster == "hdbscan":
        labels = cluster_hdbscan(emb, min_cluster_size=args.min_cluster_size)
    else:
        labels = np.full(len(df), -1, dtype=int)

    df["x"] = xy[:, 0]
    df["y"] = xy[:, 1]
    df["cluster"] = labels

    # Bridge paper between two largest clusters
    bridge = None
    two = pick_two_largest_clusters(labels)
    if two is not None:
        a, b = two
        bridge = bridge_score(emb, labels, a, b)
        df["is_bridge"] = False
        df.loc[bridge.idx, "is_bridge"] = True
    else:
        df["is_bridge"] = False

    # Keywords for (a) bridge and (b) a few along the centroid-to-centroid line for interpretability
    df["bridge_keywords"] = ""  # Always create the column, even if empty
    if bridge is not None:
        a, b = bridge.cluster_a, bridge.cluster_b
        idx_a = np.where(labels == a)[0]
        idx_b = np.where(labels == b)[0]
        ca = emb[idx_a].mean(axis=0, keepdims=True)
        cb = emb[idx_b].mean(axis=0, keepdims=True)
        ca /= np.linalg.norm(ca) + 1e-12
        cb /= np.linalg.norm(cb) + 1e-12

        # Papers near the "bridge direction": close to both centroids and close to the segment in embedding space
        # We approximate segment proximity by similarity balance.
        s_to_a = cosine_similarity(emb, ca).ravel()
        s_to_b = cosine_similarity(emb, cb).ravel()
        balance = 1.0 - np.abs(s_to_a - s_to_b)
        candidate = np.argsort(-balance)[:12]

        kw_texts = [texts[i] for i in candidate]
        kws = keywords_for_texts(kw_texts, top_n=6)
        for i, kk in zip(candidate, kws):
            df.loc[i, "bridge_keywords"] = ", ".join(kk)

    # Save artifacts
    out_csv = os.path.join(args.outdir, "embedded_2d.csv")
    df.to_csv(out_csv, index=False)
    np.save(os.path.join(args.outdir, "embeddings.npy"), emb)

    # Plot
    hover = ["pmid", "year", "journal", "title", "cluster", "is_bridge", "bridge_keywords"]
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=df["cluster"].astype(str),
        hover_data=hover,
        title=f"PubMed abstracts: {args.model}",
        opacity=0.75,
    )
    # emphasize bridge
    if df["is_bridge"].any():
        bridge_df = df[df["is_bridge"]]
        fig.add_scatter(
            x=bridge_df["x"],
            y=bridge_df["y"],
            mode="markers",
            marker=dict(size=14, symbol="x", color="black"),
            name="bridge",
            text=bridge_df["title"],
        )
    out_html = os.path.join(args.outdir, "plot.html")
    fig.write_html(out_html, include_plotlyjs="cdn")

    print(f"Wrote:\n- {out_csv}\n- {out_html}\n- {os.path.join(args.outdir, 'embeddings.npy')}")


if __name__ == "__main__":
    main()

