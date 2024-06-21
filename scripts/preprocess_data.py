import os
import polars as pl
import pandas as pd
import numpy as np
from tqdm import tqdm


DATA_ROOT = "data"
DATASET_PATH = "ebnerd_demo"
TARGET_PATH = os.path.join(DATA_ROOT, f"{DATASET_PATH}_modified")
TEXT_EMBEDDING_PATH = "data/artifacts/google_bert_base_multilingual_cased/google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet"

if not os.path.exists(TARGET_PATH):
    os.makedirs(TARGET_PATH)

# processing articles
articles = pl.read_parquet(os.path.join(DATA_ROOT, DATASET_PATH, "articles.parquet"))
articles = articles.with_columns(
    pl.col("image_ids")
    .map_elements(lambda x: x[0] if len(x) > 0 else None)
    .alias("image_id")
)
articles = articles.with_columns(
    pl.col("topics").map_elements(lambda x: len(x)).alias("topics_count")
)
articles = articles.with_columns(
    pl.col("topics")
    .map_elements(lambda x: x[0].lower() if len(x) > 0 else None)
    .alias("topic")
)
articles = articles.with_columns(
    pl.col("category_str").str.to_lowercase().alias("category")
)
articles = articles[
    [
        "article_id",
        "premium",
        "image_id",
        "article_type",
        "topic",
        "topics_count",
        "category",
        "sentiment_score",
    ]
]
articles.columns = [
    "article_id",
    "is_premium",
    "image_id",
    "article_type",
    "topic",
    "topics_count",
    "category",
    "sentiment_score",
]

print("Preparing embeddings")
# PREPARE AND SERIALIZE EMBEDDINGS

emb_df = pd.read_parquet(TEXT_EMBEDDING_PATH)
uniq_art_ids = (
    articles.select(pl.col("article_id")).unique().sort("article_id")["article_id"]
)

# Create mapping for article_id to position in the embedding matrix
# Offset by 1, as the first row is reserved for the zero vector (padding)
art_to_pos = {art_id: pos + 1 for pos, art_id in enumerate(uniq_art_ids)}

# Map article_id to its position
articles = articles.with_columns(
    pl.col("article_id").replace(art_to_pos).alias("article_emb_id")
)

# Filter the embeddings DataFrame and create the embedding matrix
emb_column = emb_df.columns[-1]
emb_np = np.stack(emb_df[emb_df.article_id.isin(uniq_art_ids)][emb_column].to_numpy())
del emb_df


# Adding padding embedding at 0th position
emb_dim = emb_np.shape[1]
pad_embedding = np.zeros(emb_dim)
emb_np = np.vstack([pad_embedding, emb_np])

os.makedirs(TARGET_PATH, exist_ok=True)
np.save(os.path.join(TARGET_PATH, "article_embeddings.npy"), emb_np)
articles.write_parquet(os.path.join(TARGET_PATH, "articles.parquet"))


print("Processing history")
# processing history
split_dfs = []
for split in ["train", "validation"]:
    history = pl.read_parquet(
        os.path.join(DATA_ROOT, DATASET_PATH, split, "history.parquet")
    )
    history = history.drop("scroll_percentage_fixed")
    history = history.explode(
        ["impression_time_fixed", "article_id_fixed", "read_time_fixed"]
    )
    history.columns = ["user_id", "impression_time", "article_id", "read_time"]

    # processing behaviors
    behaviors = pl.read_parquet(
        os.path.join(DATA_ROOT, DATASET_PATH, split, "behaviors.parquet")
    )
    behaviors = behaviors[
        ["user_id", "impression_time", "article_ids_clicked", "read_time"]
    ]
    behaviors = behaviors.with_columns(
        pl.col("article_ids_clicked")
        .map_elements(lambda x: x[0] if len(x) > 0 else None)
        .alias("article_id")
    )
    behaviors = behaviors.drop("article_ids_clicked")

    history = history.select(["user_id", "impression_time", "article_id", "read_time"])
    behaviors = behaviors.select(
        ["user_id", "impression_time", pl.col("article_id").cast(pl.Int32), "read_time"]
    )
    behaviors = pl.concat([history, behaviors])
    del history

    behaviors = behaviors.join(articles, on="article_id", how="left")
    split_dfs.append(behaviors)


data = pl.concat(split_dfs)
data = data.sort(by=["user_id", "impression_time"])

min_impressions = 3
max_impressions = 20
session_id_counter = 1


def create_sessions(group_df):
    global session_id_counter

    num_rows = group_df.height
    if num_rows <= min_impressions:
        return group_df.with_columns(pl.lit(session_id_counter).alias("session_id"))

    session_ids = []
    i = 0
    while i < num_rows:
        end_i = min(
            i + np.random.randint(min_impressions, max_impressions + 1), num_rows
        )
        session_ids.extend([session_id_counter] * (end_i - i))
        session_id_counter += 1
        i = end_i

    return group_df.with_columns(pl.Series("session_id", session_ids))


data = data.groupby("user_id").apply(create_sessions)
groups_counts = data.group_by("session_id").count().sort("count")
groups_with_enough_records = groups_counts.filter(pl.col("count") >= 2).select(
    "session_id"
)
result = data.join(groups_with_enough_records, on="session_id", how="inner")
print(f"Writing results to {TARGET_PATH}")
result.write_parquet(os.path.join(TARGET_PATH, "all.parquet"))
