import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    DATASET_DIR = sys.argv[1]
    ts_path = os.path.join(DATASET_DIR, "sessions_by_ts")

    SEEDS = [42, 69]
    FRACTIONS = [0.1, 0.5]

    timestamps = [str(i) for i in sorted([int(i) for i in os.listdir(ts_path)])]
    test_timestamp = timestamps[-1]

    for ts in timestamps:
        valid_df = pd.read_parquet(os.path.join(ts_path, ts, "valid.parquet"))
        train_df = pd.read_parquet(os.path.join(ts_path, ts, "train.parquet"))

        full_df = pd.concat([train_df, valid_df])
        for seed in SEEDS:
            for fraction in FRACTIONS:
                sample_df = full_df.sample(frac=fraction, random_state=seed)
                ds_name = (
                    DATASET_DIR.split("/")[-1]
                    + f"_{str(fraction).replace('.', '')}_{seed}"
                )
                root = "/".join(DATASET_DIR.split("/")[:-1])
                if ts == test_timestamp:
                    Path(
                        os.path.join(root, ds_name, "sessions_by_ts", test_timestamp)
                    ).mkdir(exist_ok=True, parents=True)
                    shutil.copyfile(
                        os.path.join(ts_path, ts, "valid.parquet"),
                        os.path.join(
                            root, ds_name, "sessions_by_ts", ts, "valid.parquet"
                        ),
                    )
                    continue
                _train_df, _valid_df = train_test_split(sample_df, random_state=42)
                Path(os.path.join(root, ds_name, "sessions_by_ts", ts)).mkdir(
                    exist_ok=True, parents=True
                )
                _train_df.to_parquet(
                    os.path.join(root, ds_name, "sessions_by_ts", ts, "train.parquet")
                )
                _valid_df.to_parquet(
                    os.path.join(root, ds_name, "sessions_by_ts", ts, "valid.parquet")
                )

                if not Path(os.path.join(root, ds_name, "processed_nvt")).exists():
                    shutil.copytree(
                        os.path.join(DATASET_DIR, "processed_nvt"),
                        os.path.join(root, ds_name, "processed_nvt"),
                    )
                if not Path(os.path.join(root, ds_name, "workflow_etl")).exists():
                    shutil.copytree(
                        os.path.join(DATASET_DIR, "workflow_etl"),
                        os.path.join(root, ds_name, "workflow_etl"),
                    )
