import numpy as np
import pandas as pd
import os


# MARCEL MAKE SURE YOU DON'T NEED TO ADD/CHANGE ROOT PATHS
TEXT_EMBEDDING_PATH = "data/artifacts/google_bert_base_multilingual_cased/google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet"
SOURCE = "ebnerd_small"
TARGET_PATH = f"data/{SOURCE}_modified"
READ_TIME_MIN = 7
READ_TIME_MAX = 600

# PREPARE AND SERIALIZE EMBEDDINGS
# Add mapping from article_id to position in articles dataframe
articles = pd.read_parquet(f"data/{SOURCE}/articles.parquet")
emb_df = pd.read_parquet(TEXT_EMBEDDING_PATH)
uniq_art_ids = np.sort(articles.article_id.unique())

# Create mapping for article_id to position in the embedding matrix
# Offset by 1, as the first row is reserved for the zero vector (padding)
art_to_pos = {art_id: pos + 1 for pos, art_id in enumerate(uniq_art_ids)}
articles["article_emb_id"] = articles.article_id.map(art_to_pos)

# Create embedding matrix for articles
emb_column = emb_df.columns[-1]
emb_np = np.stack(emb_df[emb_df.article_id.isin(uniq_art_ids)][emb_column].to_numpy())

# Adding padding embedding at 0th position
emb_dim = emb_np.shape[1]
pad_embedding = np.zeros(emb_dim)
emb_np = np.vstack([pad_embedding, emb_np])

# Sanity check
pos = 237
id, pos = articles[["article_id", "article_emb_id"]].iloc[pos]
x = emb_df[emb_df.article_id == id][emb_column].to_numpy()[0]
y = emb_np[pos]
assert (x == y).all()

os.makedirs(f"data/{SOURCE}_modified/", exist_ok=True)
np.save(os.path.join(TARGET_PATH, "article_embeddings.npy"), emb_np)
articles.to_parquet(os.path.join(TARGET_PATH, "articles.parquet"))

# PREPARE ARTICLES
# fill total_read_time with median, where it is missing
articles = articles.fillna(articles["total_read_time"].median())
articles["article_ctr"] = articles["total_pageviews"] / articles["total_inviews"]
# fill ctr with median, where it is missing
articles = articles.fillna(articles["article_ctr"].median())
articles = articles[
    [
        "article_id",
        "article_emb_id",
        "premium",
        "article_type",
        "topics",
        "category_str",
        "total_read_time",
        "sentiment_score",
        "article_ctr",
    ]
]
articles.columns = [
    "article_id",
    "article_emb_id",
    "article_is_premium",
    "article_type",
    "article_topics",
    "article_category",
    "article_total_read_time",
    "article_sentiment",
    "article_ctr",
]

# MERGE SESSIONS AND HISTORY DATA
for split in ["train", "validation"]:
    data_path = f"data/{SOURCE}/{split}/"

    df_behave = pd.read_parquet(f"{data_path}/behaviors.parquet")
    df_behave = df_behave[
        ["user_id", "impression_time", "article_ids_clicked", "next_read_time"]
    ]
    # as the most clicked article is 1, we can take only the first article
    df_behave["article_id"] = df_behave["article_ids_clicked"].apply(lambda x: x[0])
    df_behave = df_behave.drop(columns=["article_ids_clicked"])
    df_behave.columns = ["user_id", "impression_ts", "article_read_time", "article_id"]

    df_history = pd.read_parquet(f"{data_path}/history.parquet")
    df_history = df_history.explode(
        [
            "impression_time_fixed",
            "scroll_percentage_fixed",
            "article_id_fixed",
            "read_time_fixed",
        ],
        ignore_index=True,
    )
    df_history = df_history[
        ["user_id", "impression_time_fixed", "article_id_fixed", "read_time_fixed"]
    ]
    df_history.columns = ["user_id", "impression_ts", "article_id", "article_read_time"]

    df_behave = pd.concat([df_behave, df_history], ignore_index=True)
    df_behave = df_behave.sort_values(by=["user_id", "impression_ts"])
    count_before = df_behave.shape[0]

    # removing records with low read time
    df_behave = df_behave.loc[df_behave["article_read_time"] >= READ_TIME_MIN]
    df_behave = df_behave.loc[df_behave["article_read_time"] <= READ_TIME_MAX]

    print(
        f"Removed {count_before - df_behave.shape[0]} records, which is {100 * (count_before - df_behave.shape[0]) / count_before}% of the data"
    )

    # merging articles with the main dataframe
    df_behave = df_behave.merge(articles, on="article_id", how="left")

    # add mapping between article id and embedding id
    aid_to_embid = articles.set_index("article_id")["article_emb_id"]
    df_behave["article_emb_id"] = df_behave["article_id"].map(aid_to_embid)
    df_behave.to_parquet(os.path.join(TARGET_PATH, f"{split}.parquet"), index=False)


# MERGE TRAIN AND VALIDATION DATA, UPDATE ARTICLES
df_behave_train = pd.read_parquet(os.path.join(TARGET_PATH, "train.parquet"))
df_behave_val = pd.read_parquet(os.path.join(TARGET_PATH, "validation.parquet"))

df_behave = pd.concat([df_behave_train, df_behave_val], ignore_index=True)
df_behave.to_parquet(os.path.join(TARGET_PATH, "all.parquet"), index=False)
