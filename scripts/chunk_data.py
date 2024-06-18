import os
import pandas as pd
import nvtabular as nvt
import datetime

from nvtabular.ops import *
from merlin.schema.tags import Tags
from transformers4rec.utils.data_utils import save_time_based_splits


INPUT_DATA_DIR = "data/ebnerd_small_modified"
OUTPUT_DIR = os.path.join(INPUT_DATA_DIR, "sessions_by_ts")
PADDING = False

df = pd.read_parquet(f"{INPUT_DATA_DIR}/all.parquet")

df.sort_values(by="impression_ts")
df["impression_ts"] = (datetime.datetime.now() - df["impression_ts"]).dt.days
df["impression_ts"] = df["impression_ts"] - df["impression_ts"].min()

SESSIONS_MAX_LENGTH = 10

# Categorify categorical features
categ_feats = [
    "article_id",
    "article_is_premium",
    "article_type",
    "article_category",
] >> nvt.ops.Categorify()

# Define Groupby Workflow
groupby_feats = categ_feats + [
    "user_id",
    "article_emb_id",
    "impression_ts",
    "article_read_time",
    "article_total_read_time",
    "article_sentiment",
    "article_ctr",
]

# Group interaction features by session
groupby_features = groupby_feats >> nvt.ops.Groupby(
    groupby_cols=["user_id"],
    aggs={
        "article_id": ["list", "count"],
        "article_emb_id": ["list"],
        "article_is_premium": ["list"],
        "article_type": ["list"],
        "article_category": ["list"],
        "article_read_time": ["list"],
        "article_total_read_time": ["list"],
        "article_sentiment": ["list"],
        "article_ctr": ["list"],
        "impression_ts": ["min"],
    },
    name_sep="-",
)

sequence_features_truncated_item = (
    groupby_features["article_id-list"]
    >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH, pad=PADDING)
    >> nvt.ops.TagAsItemID()
)

sequence_features_truncated_cat = (
    groupby_features[
        "article_is_premium-list", "article_type-list", "article_category-list"
    ]
    >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH, pad=PADDING)
    >> nvt.ops.AddMetadata(tags=[Tags.CATEGORICAL])
)
sequence_features_truncated_cont = (
    groupby_features[
        "article_read_time-list",
        "article_total_read_time-list",
        "article_sentiment-list",
        "article_ctr-list",
    ]
    >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH, pad=PADDING)
    >> nvt.ops.AddMetadata(tags=[Tags.CONTINUOUS])
)

sequence_features_truncated_emb = (
    groupby_features["article_emb_id-list"]
    >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH, pad=PADDING)
    # >> nvt.ops.AddMetadata(tags=[Tags.EMBEDDING])
)


# Filter out sessions with length 1 (not valid for next-item prediction training and evaluation)
MINIMUM_SESSION_LENGTH = 2
selected_features = (
    groupby_features["article_id-count", "impression_ts-min", "user_id"]
    + sequence_features_truncated_item
    + sequence_features_truncated_cat
    + sequence_features_truncated_cont
    + sequence_features_truncated_emb
)

filtered_sessions = selected_features >> nvt.ops.Filter(
    f=lambda df: df["article_id-count"] >= MINIMUM_SESSION_LENGTH
)

seq_feats_list = (
    filtered_sessions[
        "article_id-list",
        "article_emb_id-list",
        "article_is_premium-list",
        "article_type-list",
        "article_category-list",
        "article_read_time-list",
        "article_total_read_time-list",
        "article_sentiment-list",
        "article_ctr-list",
    ]
    >> nvt.ops.ValueCount()
)

workflow = nvt.Workflow(
    filtered_sessions["user_id", "impression_ts-min"] + seq_feats_list
)

dataset = nvt.Dataset(df)

# Generate statistics for the features and export parquet files
# this step will generate the schema file
workflow.fit_transform(dataset).to_parquet(
    os.path.join(INPUT_DATA_DIR, "processed_nvt")
)
workflow.save(os.path.join(INPUT_DATA_DIR, "workflow_etl"))
sessions_gdf = pd.read_parquet(
    os.path.join(INPUT_DATA_DIR, "processed_nvt/part_0.parquet")
)
save_time_based_splits(
    data=nvt.Dataset(sessions_gdf),
    output_dir=OUTPUT_DIR,
    partition_col="impression_ts-min",
    timestamp_col="user_id",
    cpu=False,
)
