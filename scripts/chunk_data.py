import datetime
import os
from pathlib import Path

import nvtabular as nvt
import pandas as pd
from merlin.schema.tags import Tags
from nvtabular.ops import *
from sklearn.model_selection import train_test_split

INPUT_DATA_DIR = "data/ebnerd_demo_modified"
OUTPUT_DIR = os.path.join(INPUT_DATA_DIR, "sessions_by_ts")
PADDING = False
PART_SIZE = "128MB"  # Change it to "1GB" if you have enough GPU mem, it will be faster

# if not os.path.exists(f"{INPUT_DATA_DIR}/all_sorted.parquet"):
df = pd.read_parquet(f"{INPUT_DATA_DIR}/all.parquet")
df.sort_values(by="impression_time", inplace=True)
df["impression_time"] = (datetime.datetime.now() - df["impression_time"]).dt.days
df["impression_time"] = df["impression_time"] - df["impression_time"].min()
df.rename(columns={"impression_time": "day_index"}, inplace=True)
df.to_parquet(f"{INPUT_DATA_DIR}/all_sorted.parquet")
del df

SESSIONS_MAX_LENGTH = 20

# Categorify categorical features
categ_feats = [
    "article_id",
    "is_premium",
    "article_type",
    "category",
    "topic",
] >> nvt.ops.Categorify()

# Define Groupby Workflow
groupby_feats = categ_feats + [
    "session_id",
    "article_emb_id",
    "read_time",
    "topics_count",
    "sentiment_score",
    "day_index",
]

# Group interaction features by session
groupby_features = groupby_feats >> nvt.ops.Groupby(
    groupby_cols=["session_id"],
    aggs={
        "article_id": ["list", "count"],
        "is_premium": ["list"],
        "article_type": ["list"],
        "category": ["list"],
        "topic": ["list"],
        "read_time": ["list"],
        "topics_count": ["list"],
        "sentiment_score": ["list"],
        # "impression_time": ["min"],
        "article_emb_id": ["list"],
        "day_index": ["min"],
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
        "is_premium-list", "article_type-list", "category-list", "topic-list"
    ]
    >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH, pad=PADDING)
    >> nvt.ops.AddMetadata(tags=[Tags.CATEGORICAL])
)
sequence_features_truncated_cont = (
    groupby_features[
        "read_time-list",
        "topics_count-list",
        "sentiment_score-list",
    ]
    >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH, pad=PADDING)
    >> nvt.ops.AddMetadata(tags=[Tags.CONTINUOUS])
)

sequence_features_truncated_emb = groupby_features[
    "article_emb_id-list"
] >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH, pad=PADDING)

# Filter out sessions with length 1 (not valid for next-item prediction training and evaluation)
MINIMUM_SESSION_LENGTH = 2
selected_features = (
    groupby_features["article_id-count", "session_id", "day_index-min"]
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
        "sentiment_score-list",
        "topics_count-list",
        "topic-list",
    ]
    >> nvt.ops.ValueCount()
)

print("Creating workflow...")
workflow = nvt.Workflow(
    filtered_sessions["session_id", "day_index-min"] + seq_feats_list
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


print("Creating time based splits...")
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR", os.path.join(INPUT_DATA_DIR, "sessions_by_ts")
)

data_dir = Path(os.path.join(INPUT_DATA_DIR, "processed_nvt"))
sessions_gdf = pd.concat(
    pd.read_parquet(parquet_file) for parquet_file in data_dir.glob("*.parquet")
)

print("=============DAYS FOR PARTITION==============")
print(sessions_gdf["day_index-min"].value_counts())

groups = sessions_gdf.groupby("day_index-min")

output_dir = OUTPUT_DIR
os.makedirs(output_dir, exist_ok=True)

for day, group in groups:
    train, val = train_test_split(group, test_size=0.1, random_state=42)

    # Create subdirectory for the current day index
    day_dir = os.path.join(output_dir, f"{day}")
    os.makedirs(day_dir, exist_ok=True)

    # Save train and eval sets in the day-specific directory
    train_file_path = os.path.join(day_dir, "train.parquet")
    val_file_path = os.path.join(day_dir, "valid.parquet")

    train.to_parquet(train_file_path, index=False)
    val.to_parquet(val_file_path, index=False)
print("Done!")
