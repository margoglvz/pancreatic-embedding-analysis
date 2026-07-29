import pandas as pd
from bertopic import BERTopic
from pathlib import Path
import plotly.express as px

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


# ----- Animated Topic Frequency Plot -----

# Keep only the selected top topics
animated_df = topics_over_time[
    topics_over_time["Topic"].isin(top_topics)
].copy()

# Convert years to integers, if needed
animated_df["Timestamp"] = (
    pd.to_numeric(animated_df["Timestamp"], errors="coerce")
    .astype("Int64")
)

animated_df = animated_df.dropna(subset=["Timestamp"])
animated_df["Timestamp"] = animated_df["Timestamp"].astype(int)

# Create shorter, readable topic labels
topic_names = {
    0: "Robotic surgery",
    1: "Anticancer extracts",
    2: "Gut microbiome",
    3: "Neoadjuvant therapy",
    4: "Deep-learning segmentation",
    5: "Nanoparticle drug delivery",
    6: "Long noncoding RNA",
    7: "Nab-paclitaxel treatment",
    8: "Neuroendocrine tumors",
    9: "Nonfunctional pNETs",
}

animated_df["Topic_Label"] = animated_df["Topic"].map(topic_names)

# Sort the rows for cleaner animation
animated_df = animated_df.sort_values(
    ["Timestamp", "Frequency"],
    ascending=[True, True]
)

# Use the same maximum x-axis value for every frame
max_frequency = animated_df["Frequency"].max()

fig = px.bar(
    animated_df,
    x="Frequency",
    y="Topic_Label",
    animation_frame="Timestamp",
    animation_group="Topic",
    orientation="h",
    hover_data={
        "Topic": True,
        "Words": True,
        "Frequency": True,
        "Timestamp": True,
        "Topic_Label": False,
    },
    range_x=[0, max_frequency * 1.1],
    title="Pancreatic Cancer Research Topics Over Time",
    labels={
        "Frequency": "Number of papers",
        "Topic_Label": "Topic",
        "Timestamp": "Publication year",
    },
)

fig.update_layout(
    xaxis_title="Number of papers",
    yaxis_title="",
)

Path("plots").mkdir(exist_ok=True)

fig.show()
fig.write_html("plots/animated_topic_frequency.html")
