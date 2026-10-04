"""Feature extraction utilities."""
from sklearn.feature_extraction.text import TfidfVectorizer
from .config import MAX_FEATURES, NGRAM_RANGE, MIN_DF, MAX_DF

def build_tfidf(texts, **kwargs):
    params = dict(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        sublinear_tf=True,
    )
    params.update(kwargs)
    vec = TfidfVectorizer(**params)
    X = vec.fit_transform(texts)
    return X, vec
