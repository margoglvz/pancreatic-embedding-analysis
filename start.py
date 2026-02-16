"""
Quick entrypoint.

Recommended workflow:
  1) Collect abstracts:
     python pubmed_collect.py --term "pancreatic cancer" --retmax 5000 --out data/pubmed_pancreatic_cancer.csv

  2) Embed + visualize:
     python embed_viz.py --in data/pubmed_pancreatic_cancer.csv --model allenai/specter2 --outdir outputs

This file is kept small on purpose; see README.md for details.
"""

if __name__ == "__main__":
    print(__doc__)