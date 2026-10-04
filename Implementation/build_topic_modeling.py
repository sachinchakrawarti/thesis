"""Create notebooks/04_topic_modeling.ipynb — LDA & NMF Topic Modeling."""
from pathlib import Path
import json

ROOT = Path(r"D:\Projects\done\thesis\Implementation")
NB_PATH = ROOT / "notebooks" / "04_topic_modeling.ipynb"

NB_PATH.parent.mkdir(parents=True, exist_ok=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


CSS_CELL = """
<style>
:root { --primary:#0d47a1; --accent:#1976d2; --bg-soft:#f4f8fc; --border:#cfd8dc; }
h1,h2,h3 { color: var(--primary); font-family:'Segoe UI',Roboto,sans-serif; }
h1 { border-bottom:3px solid var(--accent); padding-bottom:8px; }
h2 { border-left:5px solid var(--accent); padding-left:12px; margin-top:26px; }
table { border-collapse:collapse; width:100%; margin:10px 0; font-size:0.95em; }
table th { background:var(--primary); color:#fff; padding:8px 12px; text-align:left; }
table td { border:1px solid var(--border); padding:8px 12px; }
table tr:nth-child(even) { background:var(--bg-soft); }
code { background:#eceff1; color:#c62828; padding:2px 6px; border-radius:3px;
       font-family:'Consolas',monospace; }
pre { background:#1e1e1e !important; color:#d4d4d4 !important;
      padding:14px; border-radius:6px; overflow-x:auto; }
.callout { background:linear-gradient(135deg,#e3f2fd 0%,#bbdefb 100%);
           border-left:5px solid #1976d2; padding:14px 18px;
           border-radius:6px; margin:14px 0; }
.badge { display:inline-block; padding:3px 10px; border-radius:12px;
         font-size:0.8em; font-weight:600; color:#fff; margin-right:6px; }
.badge-blue { background:#1976d2; }
.badge-green { background:#2e7d32; }
.badge-orange { background:#ef6c00; }
.badge-red { background:#c62828; }
.stat-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
             gap:12px; margin:14px 0; }
.stat { background:#fff; border:1px solid var(--border); border-radius:8px;
        padding:12px; text-align:center; box-shadow:0 2px 6px rgba(0,0,0,0.05); }
.stat .value { font-size:1.6em; font-weight:700; color:var(--primary); }
.stat .label { font-size:0.85em; color:#546e7a; margin-top:4px; }
</style>
"""


cells = [

    code([CSS_CELL]),

    md("# 04 — Topic Modeling (LDA & NMF)"),
    md(
        "> **Purpose** — Discover latent themes in the CTI corpus using two "
        "unsupervised techniques: **Latent Dirichlet Allocation (LDA)** and "
        "**Non-negative Matrix Factorization (NMF)**. Topics are used "
        "downstream for temporal analysis and emerging-threat scoring."
    ),

    md(
        '<span class="badge badge-blue">Unsupervised</span>'
        '<span class="badge badge-green">LDA</span>'
        '<span class="badge badge-orange">NMF</span>'
        '<span class="badge badge-red">Coherence</span>'
    ),

    md(
        "## Table of Contents\n"
        "\n"
        "1. [Setup & Imports](#1-setup--imports)\n"
        "2. [Load Cleaned Dataset](#2-load-cleaned-dataset)\n"
        "3. [Choose Number of Topics](#3-choose-number-of-topics)\n"
        "4. [Train LDA](#4-train-lda)\n"
        "5. [Train NMF](#5-train-nmf)\n"
        "6. [Side-by-Side Topic Comparison](#6-side-by-side-topic-comparison)\n"
        "7. [Topic Coherence Analysis](#7-topic-coherence-analysis)\n"
        "8. [Document-Topic Distribution](#8-document-topic-distribution)\n"
        "9. [Topic Prevalence](#9-topic-prevalence)\n"
        "10. [Topic ↔ Category Alignment](#10-topic--category-alignment)\n"
        "11. [Visualizations](#11-visualizations)\n"
        "12. [Save Models & Topics](#12-save-models--topics)\n"
        "13. [Summary](#13-summary)\n"
    ),

    # ==================================================
    md("## 1. Setup & Imports"),
    code([
        "import sys, pathlib, time",
        "sys.path.append(str(pathlib.Path.cwd().parent))",
        "",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "import joblib",
        "",
        "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer",
        "from sklearn.decomposition import LatentDirichletAllocation, NMF",
        "",
        "from src.config import (OUTPUT_DIR, MODELS_DIR, TOPICS_DIR,",
        "                        FIGURES_DIR, N_TOPICS, TOP_N_WORDS,",
        "                        RANDOM_STATE)",
        "",
        "sns.set_style('whitegrid')",
        "plt.rcParams['figure.dpi'] = 100",
        "plt.rcParams['savefig.dpi'] = 150",
        "",
        "print('N_TOPICS    :', N_TOPICS)",
        "print('TOP_N_WORDS :', TOP_N_WORDS)",
        "print('Random state:', RANDOM_STATE)",
    ]),

    # ==================================================
    md("## 2. Load Cleaned Dataset"),
    code([
        "df = pd.read_csv(OUTPUT_DIR / 'cleaned_cti.csv')",
        "print('Loaded:', df.shape)",
        "",
        "text_col = 'clean_full' if 'clean_full' in df.columns else 'clean_text'",
        "print('Using text column:', text_col)",
        "",
        "docs = df[text_col].astype(str).tolist()",
        "print('Documents:', len(docs))",
        "print()",
        "print('Sample document:')",
        "print(docs[0][:300])",
    ]),

    # ==================================================
    md("## 3. Choose Number of Topics"),
    md(
        "Train LDA with a range of K values and pick the K that maximizes "
        "topic coherence (or minimizes perplexity). "
        "Run this once, then fix K for the rest of the notebook."
    ),
    code([
        "K_RANGE = [4, 6, 8, 10, 12, 15]",
        "",
        "vec_counts = CountVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)",
        "X_counts = vec_counts.fit_transform(docs)",
        "print('Count matrix:', X_counts.shape)",
        "",
        "lda_scores = []",
        "for k in K_RANGE:",
        "    t0 = time.time()",
        "    lda = LatentDirichletAllocation(",
        "        n_components=k, random_state=RANDOM_STATE,",
        "        learning_method='batch', max_iter=15",
        "    )",
        "    lda.fit(X_counts)",
        "    perplexity = lda.perplexity(X_counts)",
        "    lda_scores.append({",
        "        'K': k,",
        "        'Perplexity': round(perplexity, 1),",
        "        'TimeSec': round(time.time() - t0, 1),",
        "    })",
        "    print(f'K={k:2d}  perplexity={perplexity:8.1f}  time={lda_scores[-1][\"TimeSec\"]}s')",
        "",
        "lda_scores_df = pd.DataFrame(lda_scores)",
        "lda_scores_df",
    ]),

    code([
        "fig, ax1 = plt.subplots(figsize=(9, 4))",
        "ax1.plot(lda_scores_df['K'], lda_scores_df['Perplexity'],",
        "         marker='o', color='steelblue')",
        "ax1.set_xlabel('Number of Topics (K)')",
        "ax1.set_ylabel('Perplexity', color='steelblue')",
        "ax1.set_title('LDA Perplexity vs K', fontweight='bold')",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / '04_lda_perplexity.png')",
        "plt.show()",
    ]),

    md(
        '<div class="callout">'
        '<b>Choosing K:</b> Lower perplexity is better, but interpretability '
        'matters too. If K=10 was set in <code>src/config.py</code>, '
        'compare it against this plot before overriding.'
        '</div>'
    ),

    # ==================================================
    md("## 4. Train LDA"),
    md(
        "LDA is a generative probabilistic model. Each document is modeled "
        "as a mixture of topics; each topic is a distribution over words."
    ),
    code([
        "print(f'Training LDA with K={N_TOPICS}...')",
        "t0 = time.time()",
        "lda_model = LatentDirichletAllocation(",
        "    n_components=N_TOPICS,",
        "    random_state=RANDOM_STATE,",
        "    learning_method='batch',",
        "    max_iter=20,",
        ")",
        "lda_model.fit(X_counts)",
        "print(f'  Time: {time.time()-t0:.2f}s')",
        "",
        "vocab_lda = vec_counts.get_feature_names_out()",
        "def top_words(components, vocab, n=TOP_N_WORDS):",
        "    result = []",
        "    for row in components:",
        "        idx = np.argsort(row)[::-1][:n]",
        "        result.append([vocab[i] for i in idx])",
        "    return result",
        "",
        "lda_topics = top_words(lda_model.components_, vocab_lda)",
        "for i, terms in enumerate(lda_topics):",
        "    print(f'LDA Topic {i:2d}: {', '.join(terms)}')",
    ]),

    # ==================================================
    md("## 5. Train NMF"),
    md(
        "NMF factorizes the TF-IDF matrix into two non-negative matrices "
        "(document-topic and topic-term). It tends to produce more "
        "interpretable, part-based topics than LDA."
    ),
    code([
        "print(f'Training NMF with K={N_TOPICS}...')",
        "vec_tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)",
        "X_tfidf = vec_tfidf.fit_transform(docs)",
        "print('TF-IDF matrix:', X_tfidf.shape)",
        "",
        "t0 = time.time()",
        "nmf_model = NMF(",
        "    n_components=N_TOPICS,",
        "    random_state=RANDOM_STATE,",
        "    init='nndsvd',",
        "    max_iter=400,",
        ")",
        "nmf_model.fit(X_tfidf)",
        "print(f'  Time: {time.time()-t0:.2f}s')",
        "",
        "vocab_nmf = vec_tfidf.get_feature_names_out()",
        "nmf_topics = top_words(nmf_model.components_, vocab_nmf)",
        "for i, terms in enumerate(nmf_topics):",
        "    print(f'NMF Topic {i:2d}: {', '.join(terms)}')",
    ]),

    # ==================================================
    md("## 6. Side-by-Side Topic Comparison"),
    code([
        "compare = pd.DataFrame({",
        "    'LDA_Topic': [f'LDA_T{i}' for i in range(N_TOPICS)],",
        "    'LDA_Terms': [', '.join(t[:8]) for t in lda_topics],",
        "    'NMF_Topic': [f'NMF_T{i}' for i in range(N_TOPICS)],",
        "    'NMF_Terms': [', '.join(t[:8]) for t in nmf_topics],",
        "})",
        "pd.set_option('display.max_colwidth', None)",
        "compare",
    ]),

    md(
        '<div class="callout">'
        '<b>Observation:</b> NMF topics tend to be more distinct and '
        'interpretable as single cybersecurity themes. LDA topics often '
        'overlap more and mix multiple threat concepts.'
        '</div>'
    ),

    # ==================================================
    md("## 7. Topic Coherence Analysis"),
    md(
        "**Coherence** (c_v) measures how semantically related the top words "
        "of a topic are. Higher = better. Uses Gensim's implementation."
    ),
    code([
        "try:",
        "    from gensim.corpora import Dictionary",
        "    from gensim.models.coherencemodel import CoherenceModel",
        "    HAVE_GENSIM = True",
        "except ImportError:",
        "    HAVE_GENSIM = False",
        "    print('gensim not installed. Run: pip install gensim')",
    ]),

    code([
        "if HAVE_GENSIM:",
        "    tokenized = [d.split() for d in docs]",
        "    dictionary = Dictionary(tokenized)",
        "",
        "    cm_lda = CoherenceModel(topics=lda_topics, texts=tokenized,",
        "                            dictionary=dictionary, coherence='c_v')",
        "    coh_lda = cm_lda.get_coherence()",
        "",
        "    cm_nmf = CoherenceModel(topics=nmf_topics, texts=tokenized,",
        "                            dictionary=dictionary, coherence='c_v')",
        "    coh_nmf = cm_nmf.get_coherence()",
        "",
        "    print(f'LDA coherence (c_v): {coh_lda:.4f}')",
        "    print(f'NMF coherence (c_v): {coh_nmf:.4f}')",
        "    print()",
        "    print(f'{\"NMF\" if coh_nmf > coh_lda else \"LDA\"} has higher coherence.')",
        "else:",
        "    coh_lda = coh_nmf = None",
    ]),

    code([
        "if HAVE_GENSIM:",
        "    fig, ax = plt.subplots(figsize=(6, 4))",
        "    ax.bar(['LDA', 'NMF'], [coh_lda, coh_nmf],",
        "           color=['steelblue', 'coral'], edgecolor='white')",
        "    ax.set_ylabel('Coherence (c_v)')",
        "    ax.set_title('Topic Coherence Comparison', fontweight='bold')",
        "    for i, v in enumerate([coh_lda, coh_nmf]):",
        "        ax.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=10)",
        "    plt.tight_layout()",
        "    plt.savefig(FIGURES_DIR / '04_coherence.png')",
        "    plt.show()",
    ]),

    # ==================================================
    md("## 8. Document-Topic Distribution"),
    md(
        "For each document, get its topic mixture. NMF's `transform()` "
        "returns the W matrix (documents × topics)."
    ),
    code([
        "doc_topic = nmf_model.transform(X_tfidf)",
        "print('Document-topic shape:', doc_topic.shape)",
        "",
        "topic_cols = [f'topic_{i+1}' for i in range(N_TOPICS)]",
        "doc_topic_df = pd.DataFrame(doc_topic, columns=topic_cols)",
        "doc_topic_df['dominant_topic'] = doc_topic_df.idxmax(axis=1)",
        "doc_topic_df['category'] = df['category'].values",
        "",
        "doc_topic_df.head()",
    ]),

    code([
        "# How many documents have each topic as dominant",
        "dominant_counts = doc_topic_df['dominant_topic'].value_counts().sort_index()",
        "print('Dominant topic counts:')",
        "print(dominant_counts.to_string())",
    ]),

    # ==================================================
    md("## 9. Topic Prevalence"),
    md(
        "Average topic weight across all documents — indicates how "
        "prominent each topic is in the corpus."
    ),
    code([
        "prevalence = doc_topic_df[topic_cols].mean().sort_values(ascending=False)",
        "print('Topic prevalence (mean weight):')",
        "print(prevalence.round(4).to_string())",
    ]),

    code([
        "fig, ax = plt.subplots(figsize=(10, 4))",
        "prevalence.plot(kind='bar', ax=ax, color='teal', edgecolor='white')",
        "ax.set_title('Topic Prevalence (NMF)', fontweight='bold')",
        "ax.set_ylabel('Mean Weight')",
        "ax.set_xlabel('Topic')",
        "plt.xticks(rotation=45)",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / '04_topic_prevalence.png')",
        "plt.show()",
    ]),

    # ==================================================
    md("## 10. Topic ↔ Category Alignment"),
    md(
        "Cross-tabulate dominant topic against the ground-truth threat "
        "category. Strong alignment suggests topics capture meaningful "
        "threat distinctions."
    ),
    code([
        "ct = pd.crosstab(doc_topic_df['dominant_topic'], df['category'])",
        "print('Dominant Topic vs Category:')",
        "print(ct.to_string())",
    ]),

    code([
        "plt.figure(figsize=(11, 6))",
        "sns.heatmap(ct, annot=True, fmt='d', cmap='Blues')",
        "plt.title('Dominant Topic vs Threat Category', fontweight='bold')",
        "plt.xlabel('Threat Category')",
        "plt.ylabel('Dominant Topic')",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / '04_topic_category_heatmap.png')",
        "plt.show()",
    ]),

    md(
        '<div class="callout">'
        '<b>Interpretation:</b> If a topic concentrates in one category, '
        'that topic captures a distinct threat type. Topics spread across '
        'many categories reflect generic cybersecurity vocabulary.'
        '</div>'
    ),

    # ==================================================
    md("## 11. Visualizations"),
    md("### 11.1 Top Terms per Topic (NMF)"),
    code([
        "n_rows = (N_TOPICS + 1) // 2",
        "fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows))",
        "axes = axes.flatten()",
        "",
        "for i, terms in enumerate(nmf_topics):",
        "    scores = nmf_model.components_[i]",
        "    top_idx = np.argsort(scores)[::-1][:10]",
        "    words = [vocab_nmf[j] for j in top_idx]",
        "    vals  = [scores[j] for j in top_idx]",
        "    axes[i].barh(range(len(words)), vals, color='coral')",
        "    axes[i].set_yticks(range(len(words)))",
        "    axes[i].set_yticklabels(words, fontsize=8)",
        "    axes[i].invert_yaxis()",
        "    axes[i].set_title(f'NMF Topic {i}', fontweight='bold')",
        "",
        "for j in range(N_TOPICS, len(axes)):",
        "    axes[j].axis('off')",
        "",
        "plt.suptitle('Top 10 Terms per NMF Topic', fontsize=13, fontweight='bold')",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / '04_nmf_top_terms.png')",
        "plt.show()",
    ]),

    md("### 11.2 Top Terms per Topic (LDA)"),
    code([
        "fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows))",
        "axes = axes.flatten()",
        "",
        "for i, terms in enumerate(lda_topics):",
        "    scores = lda_model.components_[i]",
        "    top_idx = np.argsort(scores)[::-1][:10]",
        "    words = [vocab_lda[j] for j in top_idx]",
        "    vals  = [scores[j] for j in top_idx]",
        "    axes[i].barh(range(len(words)), vals, color='steelblue')",
        "    axes[i].set_yticks(range(len(words)))",
        "    axes[i].set_yticklabels(words, fontsize=8)",
        "    axes[i].invert_yaxis()",
        "    axes[i].set_title(f'LDA Topic {i}', fontweight='bold')",
        "",
        "for j in range(N_TOPICS, len(axes)):",
        "    axes[j].axis('off')",
        "",
        "plt.suptitle('Top 10 Terms per LDA Topic', fontsize=13, fontweight='bold')",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / '04_lda_top_terms.png')",
        "plt.show()",
    ]),

    # ==================================================
    md("## 12. Save Models & Topics"),
    code([
        "# Save topic tables",
        "lda_rows = [{'topic': f'LDA_T{i}', 'top_terms': ', '.join(t)}",
        "            for i, t in enumerate(lda_topics)]",
        "pd.DataFrame(lda_rows).to_csv(TOPICS_DIR / 'lda_topics.csv', index=False)",
        "print(f'Saved: {TOPICS_DIR / \"lda_topics.csv\"}')",
        "",
        "nmf_rows = [{'topic': f'NMF_T{i}', 'top_terms': ', '.join(t)}",
        "            for i, t in enumerate(nmf_topics)]",
        "pd.DataFrame(nmf_rows).to_csv(TOPICS_DIR / 'nmf_topics.csv', index=False)",
        "print(f'Saved: {TOPICS_DIR / \"nmf_topics.csv\"}')",
        "",
        "# Save doc-topic distribution",
        "doc_topic_df.to_csv(TOPICS_DIR / 'doc_topic_distribution.csv', index=False)",
        "print(f'Saved: {TOPICS_DIR / \"doc_topic_distribution.csv\"}')",
    ]),

    code([
        "# Save trained models + vectorizers",
        "joblib.dump(lda_model, MODELS_DIR / 'lda_model.pkl')",
        "joblib.dump(nmf_model, MODELS_DIR / 'nmf_model.pkl')",
        "joblib.dump(vec_counts, MODELS_DIR / 'count_vectorizer.pkl')",
        "joblib.dump(vec_tfidf, MODELS_DIR / 'tfidf_topic_vectorizer.pkl')",
        "",
        "print(f'Saved LDA model        : {MODELS_DIR / \"lda_model.pkl\"}')",
        "print(f'Saved NMF model        : {MODELS_DIR / \"nmf_model.pkl\"}')",
        "print(f'Saved count vectorizer : {MODELS_DIR / \"count_vectorizer.pkl\"}')",
        "print(f'Saved tfidf vectorizer : {MODELS_DIR / \"tfidf_topic_vectorizer.pkl\"}')",
    ]),

    # ==================================================
    md("## 13. Summary"),
    md(
        '<div class="stat-grid">'
        '<div class="stat"><div class="value">K</div>'
        '<div class="label">Topics per model</div></div>'
        '<div class="stat"><div class="value">2</div>'
        '<div class="label">Topic Models</div></div>'
        '<div class="stat"><div class="value">V</div>'
        '<div class="label">Vocabulary size</div></div>'
        '<div class="stat"><div class="value">N</div>'
        '<div class="label">Documents</div></div>'
        '</div>'
    ),

    code([
        "print('=== TOPIC MODELING SUMMARY ===')",
        "print(f'Documents         : {len(docs):,}')",
        "print(f'Vocabulary (LDA)  : {len(vocab_lda):,}')",
        "print(f'Vocabulary (NMF)  : {len(vocab_nmf):,}')",
        "print(f'Number of topics  : {N_TOPICS}')",
        "print()",
        "print('Artifacts saved:')",
        "print(f'  - {TOPICS_DIR / \"lda_topics.csv\"}')",
        "print(f'  - {TOPICS_DIR / \"nmf_topics.csv\"}')",
        "print(f'  - {TOPICS_DIR / \"doc_topic_distribution.csv\"}')",
        "print(f'  - {MODELS_DIR / \"lda_model.pkl\"}')",
        "print(f'  - {MODELS_DIR / \"nmf_model.pkl\"}')",
        "print()",
        "if HAVE_GENSIM and coh_lda is not None:",
        "    print(f'Coherence LDA : {coh_lda:.4f}')",
        "    print(f'Coherence NMF : {coh_nmf:.4f}')",
    ]),

    md(
        '<div class="callout">'
        '<b>Next step:</b> Proceed to <code>05_emerging_threats.ipynb</code> '
        'to compute temporal topic prevalence and emerging-threat scores.'
        '</div>'
    ),

    md("---\n*End of topic modeling notebook.*"),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Created: {NB_PATH}")
print(f"Cells:   {len(cells)}")