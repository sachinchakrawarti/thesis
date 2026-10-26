"""
Generates notebooks/05_bertopic_topic_discovery.ipynb — CORRECTED.
Fix: temporal-analysis merge now uses document_id (not doc_id),
     accepts either column name in doc_dates.csv, and diagnoses
     empty merges instead of silently producing a blank plot.

Run from project root:  python scripts/make_bertopic.py
"""
import json, os

os.makedirs("notebooks", exist_ok=True)
os.makedirs("artifacts/figs", exist_ok=True)

def md(src):  return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}

cells = []

# ─────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────
cells.append(md(
"# 05 — Emerging-Threat Detection with BERTopic\n"
"\n"
"**Inputs:** `artifacts/train.csv`, `artifacts/train_emb.npy`  \n"
"**Outputs:**\n"
"- `artifacts/bertopic_model/`\n"
"- `artifacts/rel_with_topics.csv`\n"
"- `artifacts/topic_info.csv`\n"
"- `artifacts/metrics_bertopic.json`\n"
"- `artifacts/figs/fig_topic_sizes.png`\n"
"- `artifacts/figs/fig_topic_barchart.html`\n"
"- `artifacts/figs/fig_topic_heatmap.html`\n"
"- *(optional)* `artifacts/topic_prevalence_by_quarter.csv` and `fig_topic_prevalence.png`\n"
"\n"
"**Feeds thesis:** Chapter 4.3 (topic discovery) and Chapter 4.4 (temporal, optional)\n"
"\n"
"### Methodology note\n"
"\n"
"BERTopic is applied to the **relevant sentences only** (label=1) from the training\n"
"set — mixing irrelevant sentences would pollute the topic space. Evaluation uses\n"
"c_v coherence and topic diversity.\n"
"\n"
"The temporal-analysis cell is **optional** and only runs if `artifacts/doc_dates.csv`\n"
"exists. The cell is **robust** to either a `document_id` or `doc_id` column in the\n"
"dates file (the raw SMAH-CTI schema used `doc_id`; preprocessing renamed it to\n"
"`document_id`). If the dates file has zero overlap with the relevant subset, the\n"
"cell reports this explicitly instead of producing a blank plot."
))

# ─────────────────────────────────────────────────────────
# Cell 1 — setup
# ─────────────────────────────────────────────────────────
cells.append(md("## 1. Setup"))
cells.append(code(
"import pandas as pd, numpy as np, json, os\n"
"import matplotlib.pyplot as plt\n"
"import seaborn as sns\n"
"from bertopic import BERTopic\n"
"from umap import UMAP\n"
"from hdbscan import HDBSCAN\n"
"from sentence_transformers import SentenceTransformer\n"
"from gensim.corpora import Dictionary\n"
"from gensim.models.coherencemodel import CoherenceModel\n"
"\n"
"os.makedirs('../artifacts/figs', exist_ok=True)\n"
"sns.set_style('whitegrid')\n"
"plt.rcParams['figure.dpi'] = 120\n"
"\n"
"train = pd.read_csv('../artifacts/train.csv')\n"
"rel = train[train['relevant'] == 1].reset_index(drop=True)\n"
"print('relevant sentences:', len(rel))\n"
"print('documents in relevant subset:', rel['document_id'].nunique())"
))

# ─────────────────────────────────────────────────────────
# Cell 2 — reuse cached embeddings
# ─────────────────────────────────────────────────────────
cells.append(md(
"## 2. Reuse cached SBERT embeddings\n"
"\n"
"Notebook 4 already computed `train_emb.npy`. Reuse those rows for the relevant subset."
))
cells.append(code(
"EMB_TRAIN = '../artifacts/train_emb.npy'\n"
"\n"
"if os.path.exists(EMB_TRAIN):\n"
"    all_emb = np.load(EMB_TRAIN)\n"
"    emb = all_emb[train['relevant'].values == 1]\n"
"    print('reused embeddings:', emb.shape)\n"
"else:\n"
"    sbert = SentenceTransformer('all-MiniLM-L6-v2')\n"
"    emb = sbert.encode(rel['sentence_text'].tolist(), batch_size=64,\n"
"                       show_progress_bar=True, convert_to_numpy=True)\n"
"    print('recomputed embeddings:', emb.shape)\n"
"\n"
"assert len(emb) == len(rel), 'Embedding / row mismatch'"
))

# ─────────────────────────────────────────────────────────
# Cell 3 — fit BERTopic
# ─────────────────────────────────────────────────────────
cells.append(md(
"## 3. Fit BERTopic\n"
"\n"
"Pipeline: SBERT embeddings → UMAP (5-d) → HDBSCAN (min_cluster_size=20) → c-TF-IDF."
))
cells.append(code(
"umap_model    = UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)\n"
"hdbscan_model = HDBSCAN(min_cluster_size=20, metric='euclidean',\n"
"                        prediction_data=True)\n"
"\n"
"topic_model = BERTopic(\n"
"    umap_model=umap_model,\n"
"    hdbscan_model=hdbscan_model,\n"
"    calculate_probabilities=False,\n"
"    verbose=True,\n"
")\n"
"\n"
"topics, probs = topic_model.fit_transform(rel['sentence_text'].tolist(), emb)\n"
"info = topic_model.get_topic_info()\n"
"print(info.head(25).to_string(index=False))"
))

# ─────────────────────────────────────────────────────────
# Cell 4 — save model & topic assignments
# ─────────────────────────────────────────────────────────
cells.append(md("## 4. Save model & topic assignments"))
cells.append(code(
"topic_model.save('../artifacts/bertopic_model', serialization='safetensors')\n"
"rel_out = rel.copy()\n"
"rel_out['topic'] = topics\n"
"rel_out.to_csv('../artifacts/rel_with_topics.csv', index=False)\n"
"info.to_csv('../artifacts/topic_info.csv', index=False)\n"
"print('saved model + rel_with_topics.csv + topic_info.csv')"
))

# ─────────────────────────────────────────────────────────
# Cell 5 — top words per topic
# ─────────────────────────────────────────────────────────
cells.append(md("## 5. Top-10 words per topic"))
cells.append(code(
"def print_topics(topic_model, n_topics=15, top_n=10):\n"
"    for tid in topic_model.get_topics():\n"
"        if tid == -1: continue\n"
"        words = [w for w, _ in topic_model.get_topic(tid)][:top_n]\n"
"        size = int(topic_model.get_topic_info().set_index('Topic').loc[tid, 'Count'])\n"
"        print(f'Topic {tid:>3d} (size={size:>4d}): ' + ', '.join(words))\n"
"        if tid >= n_topics: break\n"
"\n"
"print_topics(topic_model, n_topics=15)"
))

# ─────────────────────────────────────────────────────────
# Cell 6 — coherence (c_v)
# ─────────────────────────────────────────────────────────
cells.append(md(
"## 6. Topic coherence (c_v)\n"
"\n"
"Higher is better. Above ~0.5 = meaningful topics; below ~0.3 = noise."
))
cells.append(code(
"def topic_coherence(topic_model, docs, top_n=10):\n"
"    topics_words = []\n"
"    for tid in topic_model.get_topics():\n"
"        if tid == -1: continue\n"
"        topics_words.append([w for w, _ in topic_model.get_topic(tid)][:top_n])\n"
"    tokenized = [d.split() for d in docs]\n"
"    dic = Dictionary(tokenized)\n"
"    cm = CoherenceModel(topics=topics_words, texts=tokenized,\n"
"                        dictionary=dic, coherence='c_v')\n"
"    return cm.get_coherence()\n"
"\n"
"coh = topic_coherence(topic_model, rel['sentence_text'].tolist())\n"
"print('c_v coherence:', round(coh, 4))"
))

# ─────────────────────────────────────────────────────────
# Cell 7 — diversity
# ─────────────────────────────────────────────────────────
cells.append(md("## 7. Topic diversity"))
cells.append(code(
"def topic_diversity(topic_model, top_n=10):\n"
"    words = []\n"
"    for tid in topic_model.get_topics():\n"
"        if tid == -1: continue\n"
"        words += [w for w, _ in topic_model.get_topic(tid)][:top_n]\n"
"    return len(set(words)) / max(len(words), 1)\n"
"\n"
"div = topic_diversity(topic_model)\n"
"print('topic diversity:', round(div, 4))"
))

# ─────────────────────────────────────────────────────────
# Cell 8 — save metrics
# ─────────────────────────────────────────────────────────
cells.append(md("## 8. Save BERTopic metrics"))
cells.append(code(
"metrics_bt = {\n"
"    'n_topics': int(len(info) - 1),\n"
"    'n_outliers': int((np.array(topics) == -1).sum()),\n"
"    'coherence_cv': round(float(coh), 4),\n"
"    'diversity':    round(float(div), 4),\n"
"    'min_cluster_size': 20,\n"
"}\n"
"json.dump(metrics_bt, open('../artifacts/metrics_bertopic.json','w'), indent=2)\n"
"print(json.dumps(metrics_bt, indent=2))"
))

# ─────────────────────────────────────────────────────────
# Cell 9 — barchart
# ─────────────────────────────────────────────────────────
cells.append(md("## 9. Topic barchart (interactive HTML)"))
cells.append(code(
"try:\n"
"    fig = topic_model.visualize_barchart(top_n_topics=10, n_words=8)\n"
"    fig.write_html('../artifacts/figs/fig_topic_barchart.html')\n"
"    print('saved -> artifacts/figs/fig_topic_barchart.html')\n"
"except Exception as e:\n"
"    print('barchart skip:', e)"
))

# ─────────────────────────────────────────────────────────
# Cell 10 — heatmap
# ─────────────────────────────────────────────────────────
cells.append(md("## 10. Topic similarity heatmap (interactive HTML)"))
cells.append(code(
"try:\n"
"    fig = topic_model.visualize_heatmap(top_n_topics=15)\n"
"    fig.write_html('../artifacts/figs/fig_topic_heatmap.html')\n"
"    print('saved -> artifacts/figs/fig_topic_heatmap.html')\n"
"except Exception as e:\n"
"    print('heatmap skip:', e)"
))

# ─────────────────────────────────────────────────────────
# Cell 11 — topic-size bar chart
# ─────────────────────────────────────────────────────────
cells.append(md("## 11. Topic-size distribution"))
cells.append(code(
"sizes = info[info['Topic'] != -1][['Topic','Count']].sort_values('Count', ascending=False).head(20)\n"
"plt.figure(figsize=(10, 4))\n"
"plt.bar(sizes['Topic'].astype(str), sizes['Count'], color='#55a868')\n"
"plt.xlabel('topic id'); plt.ylabel('sentence count')\n"
"plt.title('Top-20 topics by size')\n"
"plt.tight_layout()\n"
"plt.savefig('../artifacts/figs/fig_topic_sizes.png')\n"
"plt.show()\n"
"print('saved -> artifacts/figs/fig_topic_sizes.png')"
))

# ─────────────────────────────────────────────────────────
# Cell 12 — FIXED temporal analysis
# ─────────────────────────────────────────────────────────
cells.append(md(
"## 12. Optional — temporal analysis (FIXED)\n"
"\n"
"**This cell is robust to both `document_id` and `doc_id`** in `doc_dates.csv`:\n"
"\n"
"- If the file is missing → clean skip message.\n"
"- If the file has `doc_id` → column is renamed to `document_id` before merge.\n"
"- If the file has `document_id` → used directly.\n"
"- If both exist → `document_id` wins.\n"
"- If the merged set is empty → a diagnostic is printed with sample IDs from both\n"
"  tables, and the plot is skipped (no silent blank figure).\n"
"- If dates don't parse → rows are dropped with a warning count."
))
cells.append(code(
"dates_path = '../artifacts/doc_dates.csv'\n"
"\n"
"if not os.path.exists(dates_path):\n"
"    print('SKIP — no doc_dates.csv. Create one to enable temporal analysis.')\n"
"    print('Format (either column name accepted):')\n"
"    print('  document_id,date    OR    doc_id,date')\n"
"    print('  Example: 01_aptnotes_report,2015-03-15')\n"
"else:\n"
"    dates = pd.read_csv(dates_path)\n"
"    print(f'Loaded dates: {len(dates)} rows, columns={list(dates.columns)}')\n"
"\n"
"    # ── Normalize the join key to `document_id` ─────────────\n"
"    if 'document_id' in dates.columns and 'doc_id' in dates.columns:\n"
"        dates = dates.drop(columns=['doc_id'])\n"
"    elif 'doc_id' in dates.columns and 'document_id' not in dates.columns:\n"
"        dates = dates.rename(columns={'doc_id': 'document_id'})\n"
"    elif 'document_id' not in dates.columns:\n"
"        raise KeyError(\n"
"            f\"doc_dates.csv must contain either 'document_id' or 'doc_id'. \"\n"
"            f\"Found: {list(dates.columns)}\"\n"
"        )\n"
"\n"
"    # ── Validate date column ────────────────────────────────\n"
"    if 'date' not in dates.columns:\n"
"        raise KeyError(\n"
"            f\"doc_dates.csv must contain a 'date' column. \"\n"
"            f\"Found: {list(dates.columns)}\"\n"
"        )\n"
"    dates['date'] = pd.to_datetime(dates['date'], errors='coerce')\n"
"    n_before = len(dates)\n"
"    dates = dates.dropna(subset=['date'])\n"
"    if len(dates) < n_before:\n"
"        print(f'  dropped {n_before - len(dates)} rows with unparseable dates')\n"
"\n"
"    # ── Merge on document_id ────────────────────────────────\n"
"    rel_t = rel_out.merge(dates[['document_id','date']],\n"
"                          on='document_id', how='inner')\n"
"    n_matched = rel_t['document_id'].nunique()\n"
"    print(f'Matched: {n_matched} documents ({len(rel_t)} relevant sentences with dates).')\n"
"\n"
"    if n_matched == 0:\n"
"        print('⚠  NO OVERLAP between doc_dates.csv and the relevant subset.')\n"
"        print('   Sample document_id from rel_out:')\n"
"        print('  ', sorted(rel_out['document_id'].unique())[:5])\n"
"        print('   Sample document_id from dates:')\n"
"        print('  ', sorted(dates['document_id'].unique())[:5])\n"
"        print('   Temporal analysis skipped.')\n"
"    else:\n"
"        rel_t['quarter'] = rel_t['date'].dt.to_period('Q')\n"
"        prev = rel_t.groupby(['quarter','topic']).size().unstack(fill_value=0)\n"
"        prev_norm = prev.div(prev.sum(axis=1), axis=0)\n"
"        prev_norm.to_csv('../artifacts/topic_prevalence_by_quarter.csv')\n"
"\n"
"        plt.figure(figsize=(11, 5))\n"
"        for tid in prev_norm.columns:\n"
"            if tid == -1: continue\n"
"            plt.plot(prev_norm.index.astype(str), prev_norm[tid],\n"
"                     marker='o', label=f'T{tid}')\n"
"        plt.title(f'Topic prevalence by quarter ({n_matched} documents)')\n"
"        plt.ylabel('share of relevant sentences')\n"
"        plt.xlabel('quarter')\n"
"        plt.xticks(rotation=45)\n"
"        plt.legend(ncol=4, fontsize=8)\n"
"        plt.tight_layout()\n"
"        plt.savefig('../artifacts/figs/fig_topic_prevalence.png')\n"
"        plt.show()\n"
"        print('saved -> artifacts/topic_prevalence_by_quarter.csv')\n"
"        print('saved -> artifacts/figs/fig_topic_prevalence.png')"
))

# ─────────────────────────────────────────────────────────
# Cell 13 — summary
# ─────────────────────────────────────────────────────────
cells.append(md("## 13. BERTopic summary"))
cells.append(code(
"print('=' * 60)\n"
"print(f\"{'BERTOPIC — SUMMARY':^60s}\")\n"
"print('=' * 60)\n"
"print(f\"  Topics discovered   : {metrics_bt['n_topics']}\")\n"
"print(f\"  Outlier sentences   : {metrics_bt['n_outliers']}\")\n"
"print(f\"  Coherence (c_v)     : {metrics_bt['coherence_cv']}\")\n"
"print(f\"  Diversity           : {metrics_bt['diversity']}\")\n"
"print('=' * 60)"
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"}
    },
    "nbformat": 4, "nbformat_minor": 5
}

out = os.path.join("notebooks", "05_bertopic_topic_discovery.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Wrote:", os.path.abspath(out))
print("Cells:", len(cells))