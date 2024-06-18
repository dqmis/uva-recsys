import sys

import pandas as pd
import os
import polars as pl
import pickle

if __name__ == "__main__":
    data_root = sys.argv[1] if len(sys.argv) > 1 else "../data"
    DATASET_PATH = "ebnerd_testset/ebnerd_testset"

    # processing articles
    articles = pl.read_parquet(os.path.join(data_root, DATASET_PATH, "articles.parquet"))
    articles = articles.with_columns(
        pl.col("image_ids").map_elements(lambda x: x[0] if len(x) > 0 else None).alias("image_id")
    )
    articles = articles.with_columns(
        pl.col("topics").map_elements(lambda x: len(x)).alias("topics_count")
    )
    articles = articles.with_columns(
        pl.col("topics").map_elements(lambda x: x[0].lower() if len(x) > 0 else None).alias("topic")
    )
    articles = articles.with_columns(
        pl.col("category_str").str.to_lowercase().alias("category")
    )
    articles = articles[["article_id", "premium", "image_id", "article_type", "topic", "topics_count", "category", "sentiment_score"]]
    articles.columns = ["article_id", "is_premium", "image_id", "article_type", "topic", "topics_count", "category", "sentiment_score"]

    # processing history
    history = pl.read_parquet(os.path.join(data_root, DATASET_PATH, "test", "history.parquet"))
    history = history.drop("scroll_percentage_fixed")
    history = history.explode(["impression_time_fixed", "article_id_fixed", "read_time_fixed"])
    history = (
        history
        .sort(['user_id', 'impression_time_fixed'], descending=[False, True])
        .groupby('user_id')
        .agg([
            pl.all().head(20)
        ])
        .explode(pl.all().exclude("user_id"))
    )

    # processing behaviors
    behaviors = pl.read_parquet(os.path.join(data_root, DATASET_PATH, "test", "behaviors.parquet"))

    inviews = behaviors.select("article_ids_inview")

    behaviors = behaviors.drop('session_id')
    behaviors = behaviors.with_row_index(name='session_id')
    behaviors = behaviors.rename({col: col.replace("index", "session_id") for col in behaviors.columns})

    behaviors = behaviors[["session_id", "impression_id", "user_id"]]
    behaviors = behaviors.join(history, on="user_id", how="left")
    behaviors.columns = ["session_id", "impression_id", "user_id", "impression_time", "article_id", "read_time"]

    behaviors = behaviors.join(articles, on="article_id", how="left")

    inviews.write_parquet(os.path.join(data_root, DATASET_PATH, "test", "inviews.parquet"))
    behaviors.write_parquet(os.path.join(data_root, DATASET_PATH, "all.parquet"))


