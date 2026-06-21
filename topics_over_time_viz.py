import pandas as pd
from bertopic import BERTopic

# Prepare data
df = pd.read_csv("data/pubmed_only_pancreatic_cancer.csv")
df = df.dropna(subset=["title", "abstract", "year"]).reset_index(drop=True)
year = df.year.to_list()
docs = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).tolist()

# Create topics over time
model = BERTopic(verbose=True)
topics, probs = model.fit_transform(docs)
topics_over_time = model.topics_over_time(docs, year)
topics_over_time.to_csv("topics_over_time.csv", index=False)

topic_info = model.get_topic_info()

top_topics = (
    topic_info[topic_info.Topic != -1]
    .head(10)["Topic"]
    .tolist()
)

fig = model.visualize_topics_over_time(
    topics_over_time,
    topics=top_topics
)

fig.show()
fig.write_html("plots/topics_over_time.html")
