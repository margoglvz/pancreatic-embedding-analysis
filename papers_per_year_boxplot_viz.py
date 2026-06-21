from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def count_papers_per_year(df):
    years_and_papers = defaultdict(int)
    for paper in df.itertuples(index=True):
        year = paper.year

        years_and_papers[year] += 1

    return years_and_papers

def create_bar_plot(years_and_papers):
    fig, ax = plt.subplots()
    ax.bar(list(years_and_papers.keys()), list(years_and_papers.values()), color='purple')

    ax.set_xlabel('Years')
    ax.set_ylabel('Frequency')
    ax.set_title('Pancreatic Cancer Research Papers per Year')

    plt.show()


def main():
    df = pd.read_csv("data/pubmed_only_pancreatic_cancer.csv")
    df = df.dropna(subset=["title", "abstract", "year"], how="all").reset_index(drop=True)
    print(df.head())
    years_and_papers = count_papers_per_year(df)
    create_bar_plot(years_and_papers)


main()