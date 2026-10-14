# 06. Temporal Topic Modeling: LDA and NMF

## Goal

Discover recurring themes in report text and examine how the
distribution of those themes changes over time.

## Method choice

-   **LDA** is conventionally trained on document-term counts.
-   **NMF** is commonly paired with TF-IDF features.
-   Topics are word-weight patterns; a human must inspect the top words
    and representative reports before assigning topic labels.

## Prepare weekly documents

``` python
import pandas as pd

df["week"] = df["date"].dt.to_period("W").dt.start_time
weekly_counts = df.groupby("week").size()
print("Weeks:", len(weekly_counts))
print(weekly_counts.head())
```

## Fit LDA and NMF

``` python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

N_TOPICS = 5

count_vec = CountVectorizer(
    stop_words="english", max_df=0.95, min_df=2,
    max_features=30000
)
X_counts = count_vec.fit_transform(df["model_text"])

lda = LatentDirichletAllocation(
    n_components=N_TOPICS, random_state=42, learning_method="batch"
)
lda_doc_topics = lda.fit_transform(X_counts)

tfidf_vec = TfidfVectorizer(
    stop_words="english", max_df=0.95, min_df=2,
    max_features=30000
)
X_tfidf = tfidf_vec.fit_transform(df["model_text"])

nmf = NMF(n_components=N_TOPICS, random_state=42, max_iter=500)
nmf_doc_topics = nmf.fit_transform(X_tfidf)
nmf_doc_topics = nmf_doc_topics / (nmf_doc_topics.sum(axis=1, keepdims=True) + 1e-12)
```

## Inspect topic words

``` python
def show_topics(model, feature_names, top_n=10):
    for i, weights in enumerate(model.components_):
        top = weights.argsort()[-top_n:][::-1]
        print(f"Topic {i+1}: " + ", ".join(feature_names[j] for j in top))

show_topics(lda, count_vec.get_feature_names_out())
show_topics(nmf, tfidf_vec.get_feature_names_out())
```

## Weekly topic proportions

LDA outputs topic proportions per document. For NMF, the code normalizes
each document's topic weights so they can be compared as relative
shares.

``` python
import numpy as np

def weekly_topic_shares(doc_topic_matrix, dates):
    tmp = pd.DataFrame(doc_topic_matrix)
    tmp["week"] = pd.to_datetime(dates).dt.to_period("W").dt.start_time.to_numpy()
    return tmp.groupby("week").mean().sort_index()

lda_weekly = weekly_topic_shares(lda_doc_topics, df["date"])
nmf_weekly = weekly_topic_shares(nmf_doc_topics, df["date"])

ax = lda_weekly.plot(figsize=(12, 5), title="LDA: Weekly Mean Topic Proportions")
ax.set_xlabel("Week")
ax.set_ylabel("Mean topic proportion")
plt.tight_layout()
plt.savefig("artifacts/lda_weekly_topic_trends.png", dpi=200)
plt.show()

ax = nmf_weekly.plot(figsize=(12, 5), title="NMF: Weekly Normalized Topic Weights")
ax.set_xlabel("Week")
ax.set_ylabel("Mean normalized weight")
plt.tight_layout()
plt.savefig("artifacts/nmf_weekly_topic_trends.png", dpi=200)
plt.show()
```

## Interpretation cautions

-   Topic number 1 in LDA is not automatically equivalent to topic
    number 1 in NMF.
-   Topic names must be assigned after inspecting top words and sample
    reports.
-   Weekly changes can reflect synthetic data generation patterns or
    changing sample volume.
-   A topic trend is a candidate signal for review, not proof of an
    emerging real-world attack.
