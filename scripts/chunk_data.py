import os
import pandas as pd
import nvtabular as nvt
import datetime

from nvtabular.ops import *
from merlin.schema.tags import Tags
from transformers4rec.utils.data_utils import save_time_based_splits


INPUT_DATA_DIR = "data/ebnerd_demo_modified"
OUTPUT_DIR = os.path.join(INPUT_DATA_DIR, "sessions_by_ts")
PADDING = False
PART_SIZE = "256MB"  # Change it to "1GB" if you have enough GPU mem, it will be faster

if not os.path.exists(f"{INPUT_DATA_DIR}/all_sorted.parquet"):
    df = pd.read_parquet(f"{INPUT_DATA_DIR}/all.parquet")
    df.sort_values(by="impression_time")
    df["impression_time"] = (datetime.datetime.now() - df["impression_time"]).dt.days
    df["impression_time"] = df["impression_time"] - df["impression_time"].min()
    df.to_parquet(f"{INPUT_DATA_DIR}/all_sorted.parquet")
    del df

# df = pd.read_parquet(f"{INPUT_DATA_DIR}/all.parquet")
# df.sort_values(by="impression_time")
# df["impression_time"] = (datetime.datetime.now() - df["impression_time"]).dt.days
# df["impression_time"] = df["impression_time"] - df["impression_time"].min()
# df.to_parquet(f"{INPUT_DATA_DIR}/all_sorted.parquet")
# del df

SESSIONS_MAX_LENGTH = 20

# Categorify categorical features
categ_feats = [
    "article_id",
    "is_premium",
    "article_type",
    "category",
] >> nvt.ops.Categorify()

# Define Groupby Workflow
groupby_feats = categ_feats + [
    "session_id",
    "article_emb_id",
    "impression_time",
    "read_time",
    # "article_total_read_time",
    "sentiment_score",
    # "article_ctr",
]

# Group interaction features by session
groupby_features = groupby_feats >> nvt.ops.Groupby(
    groupby_cols=["session_id"],
    aggs={
        "article_id": ["list", "count"],
        # "session_id": ["list"],
        "article_emb_id": ["list"],
        "is_premium": ["list"],
        "article_type": ["list"],
        "category": ["list"],
        "read_time": ["list"],
        # "article_total_read_time": ["list"],
        "sentiment_score": ["list"],
        # "article_ctr": ["list"],
        "impression_time": ["min"],
    },
    name_sep="-",
)

sequence_features_truncated_item = (
    groupby_features["article_id-list"]
    >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH, pad=PADDING)
    >> nvt.ops.TagAsItemID()
)

sequence_features_truncated_cat = (
    groupby_features["is_premium-list", "article_type-list", "category-list"]
    >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH, pad=PADDING)
    >> nvt.ops.AddMetadata(tags=[Tags.CATEGORICAL])
)
sequence_features_truncated_cont = (
    groupby_features[
        "read_time-list",
        # "article_total_read_time-list",
        "sentiment_score-list",
        # "article_ctr-list",
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
    groupby_features["article_id-count", "impression_time-min", "session_id"]
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
        "is_premium-list",
        "article_type-list",
        "category-list",
        "read_time-list",
        # "article_total_read_time-list",
        "sentiment_score-list",
        # "article_ctr-list",
    ]
    >> nvt.ops.ValueCount()
)

print("Creating workflow...")
workflow = nvt.Workflow(
    filtered_sessions["session_id", "impression_time-min"] + seq_feats_list
)

print("Creating dataset...")
dataset = nvt.Dataset(
    os.path.join(INPUT_DATA_DIR, "all_sorted.parquet"), part_size=PART_SIZE
)

# Generate statistics for the features and export parquet files
# this step will generate the schema file
print("Starting fit transform...")
workflow.fit_transform(dataset).to_parquet(
    os.path.join(INPUT_DATA_DIR, "processed_nvt")
)
workflow.save(os.path.join(INPUT_DATA_DIR, "workflow_etl"))
sessions_gdf = pd.read_parquet(
    os.path.join(INPUT_DATA_DIR, "processed_nvt/part_0.parquet")
)
print("Creating time based splits...")
save_time_based_splits(
    data=nvt.Dataset(sessions_gdf),
    output_dir=OUTPUT_DIR,
    partition_col="impression_time-min",
    timestamp_col="session_id",
    cpu=False,
)
print("Done!")
