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
PART_SIZE = "128MB"  # Change it to "1GB" if you have enough memory, it will be faster

if not os.path.exists(f"{INPUT_DATA_DIR}/all_sorted.parquet"):
    df = pd.read_parquet(f"{INPUT_DATA_DIR}/all.parquet")
    df.sort_values(by="impression_ts")
    df["impression_ts"] = (datetime.datetime.now() - df["impression_ts"]).dt.days
    df["impression_ts"] = df["impression_ts"] - df["impression_ts"].min()
    df.to_parquet(f"{INPUT_DATA_DIR}/all_sorted.parquet")
    del df

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

print("Creating workflow...")
workflow = nvt.Workflow(
    filtered_sessions["user_id", "impression_ts-min"] + seq_feats_list
)

print("Creating dataset...")
dataset = nvt.Dataset(
    os.path.join(INPUT_DATA_DIR, "all_sorted.parquet"), part_size="128MB"
)

# Generate statistics for the features and export parquet files
# this step will generate the schema file
print("Starting fit transform...")
workflow.fit_transform(dataset).to_parquet(
    os.path.join(INPUT_DATA_DIR, "processed_nvt")
)
workflow.save(os.path.join(INPUT_DATA_DIR, "workflow_etl"))


print("Creating time based splits...")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR",os.path.join(INPUT_DATA_DIR, "sessions_by_ts"))

data_dir = Path(os.path.join(INPUT_DATA_DIR, "processed_nvt"))
sessions_gdf = pd.concat(
    pd.read_parquet(parquet_file)
    for parquet_file in data_dir.glob('*.parquet')
)

groups = sessions_gdf.groupby('day_index')

output_dir = OUTPUT_DIR
os.makedirs(output_dir, exist_ok=True)

for day, group in groups:
    train, val = train_test_split(group, test_size=0.1, random_state=42)
    
    # Create subdirectory for the current day index
    day_dir = os.path.join(output_dir, f'{day}')
    os.makedirs(day_dir, exist_ok=True)
    
    # Save train and eval sets in the day-specific directory
    train_file_path = os.path.join(day_dir, 'train.parquet')
    val_file_path = os.path.join(day_dir, 'valid.parquet')
    
    train.to_parquet(train_file_path, index=False)
    val.to_parquet(val_file_path, index=False)
print("Done!")
