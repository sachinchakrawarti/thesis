"""LDA and NMF topic modeling."""
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from .config import N_TOPICS, TOP_N_WORDS, RANDOM_STATE

def _top_words(components, feature_names, n=TOP_N_WORDS):
    out = []
    for row in components:
        idx = np.argsort(row)[::-1][:n]
        out.append([feature_names[i] for i in idx])
    return out

def run_lda(docs, n_topics=N_TOPICS):
    vec = CountVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X   = vec.fit_transform(docs)
    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=RANDOM_STATE,
        learning_method="batch", max_iter=20
    )
    lda.fit(X)
    return lda, _top_words(lda.components_, vec.get_feature_names_out())

def run_nmf(docs, n_topics=N_TOPICS):
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X   = vec.fit_transform(docs)
    nmf = NMF(n_components=n_topics, random_state=RANDOM_STATE,
              init="nndsvd", max_iter=400)
    nmf.fit(X)
    return nmf, _top_words(nmf.components_, vec.get_feature_names_out())
