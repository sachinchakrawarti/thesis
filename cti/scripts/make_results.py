"""
Generates notebooks/06_results_comparison.ipynb
Inputs : artifacts/metrics_tfidf.json, metrics_sbert.json, metrics_hybrid.json,
         metrics_bertopic.json, split_info.json, eda_summary.json
Outputs: artifacts/table_classification.csv
         artifacts/table_topic_discovery.csv
         artifacts/table_dataset_summary.csv
         artifacts/figs/fig_classification_comparison.png
         artifacts/figs/fig_dataset_summary.png
         artifacts/chapter5_draft.txt
Run from project root:  python scripts/make_results.py
"""
import json, os

os.makedirs("notebooks", exist_ok=True)
os.makedirs("artifacts/figs", exist_ok=True)

def md(src):  return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}

cells = []

cells.append(md(
"# 06 — Results Comparison & Discussion\n"
"\n"
"**Inputs:** all `artifacts/metrics_*.json` files  \n"
"**Outputs:** consolidated result tables, comparison figures, and a draft Chapter 5 paragraph  \n"
"**Feeds thesis:** Chapter 5 — Results and Discussion\n"
"\n"
"This notebook assembles every result from Notebooks 3–5 into the final tables and\n"
"figures that will appear in the thesis. It also auto-generates a draft results\n"
"paragraph based on the actual numbers — you edit it into fluent prose."
))

# ── Cell 1 — setup
cells.append(md("## 1. Setup"))
cells.append(code(
"import pandas as pd, numpy as np, json, os\n"
"import matplotlib.pyplot as plt\n"
"import seaborn as sns\n"
"\n"
"sns.set_style('whitegrid')\n"
"plt.rcParams['figure.dpi'] = 120\n"
"\n"
"required = {\n"
"    'eda':      '../artifacts/eda_summary.json',\n"
"    'split':    '../artifacts/split_info.json',\n"
"    'baseline': '../artifacts/metrics_tfidf.json',\n"
"    'sbert':    '../artifacts/metrics_sbert.json',\n"
"    'bertopic': '../artifacts/metrics_bertopic.json',\n"
"}\n"
"missing = [k for k, p in required.items() if not os.path.exists(p)]\n"
"if missing:\n"
"    print('Missing artifacts:', missing)\n"
"    print('Run Notebooks 1-5 first.')\n"
"else:\n"
"    print('All required artifacts present.')"
))

# ── Cell 2 — load all
cells.append(md("## 2. Load all metrics"))
cells.append(code(
"def load(path, default=None):\n"
"    return json.load(open(path)) if os.path.exists(path) else default\n"
"\n"
"eda      = load(required['eda'])\n"
"split    = load(required['split'])\n"
"m_base   = load(required['baseline'])\n"
"m_sbert  = load(required['sbert'])\n"
"m_hybrid = load('../artifacts/metrics_hybrid.json')     # optional\n"
"m_topic  = load(required['bertopic'])\n"
"\n"
"print('Loaded:')\n"
"for name, obj in [('EDA', eda), ('Split', split), ('Baseline', m_base),\n"
"                  ('SBERT', m_sbert), ('Hybrid', m_hybrid), ('BERTopic', m_topic)]:\n"
"    print(f'  {name:>10s}: {\"OK\" if obj else \"MISSING\"}')"
))

# ── Cell 3 — Table 1: dataset summary
cells.append(md("## 3. Table 1 — Dataset summary"))
cells.append(code(
"rows = [\n"
"    ('Total sentences (raw)',         eda['raw_rows']),\n"
"    ('Relevant sentences',            eda['relevant']),\n"
"    ('Non-relevant sentences',        eda['non_relevant']),\n"
"    ('Imbalance ratio',               f\"{eda['imbalance_ratio']}:1\"),\n"
"    ('Documents',                     eda['num_documents']),\n"
"    ('Sources',                       eda['num_sources']),\n"
"    ('Mean sentence length (words)',  eda['avg_words']),\n"
"    ('Median sentence length (words)',eda['median_words']),\n"
"    ('Vocabulary size',               eda['vocab_size']),\n"
"    ('Type-token ratio',              eda['type_token_ratio']),\n"
"    ('Hapax legomena (%)',            eda['hapax_pct']),\n"
"    ('Train sentences (80%)',         split['train_size']),\n"
"    ('Test sentences (20%)',          split['test_size']),\n"
"    ('Train documents',               split['train_docs']),\n"
"    ('Test documents',                split['test_docs']),\n"
"    ('Document overlap',              split['overlap_docs']),\n"
"]\n"
"table1 = pd.DataFrame(rows, columns=['Attribute', 'Value'])\n"
"print(table1.to_string(index=False))\n"
"table1.to_csv('../artifacts/table_dataset_summary.csv', index=False)"
))

# ── Cell 4 — Table 2: classification comparison
cells.append(md("## 4. Table 2 — Classification comparison"))
cells.append(code(
"rows = [m_base, m_sbert]\n"
"if m_hybrid: rows.append(m_hybrid)\n"
"\n"
"table2 = pd.DataFrame(rows)[[\n"
"    'model','accuracy','precision','recall','macro_f1','weighted_f1']].round(4)\n"
"table2.to_csv('../artifacts/table_classification.csv', index=False)\n"
"print(table2.to_string(index=False))\n"
"\n"
"best_idx = table2['macro_f1'].idxmax()\n"
"print(f\"\\nBest model: {table2.iloc[best_idx]['model']} \"\n"
"      f\"(macro-F1 = {table2.iloc[best_idx]['macro_f1']:.4f})\")"
))

# ── Cell 5 — Table 3: topic discovery
cells.append(md("## 5. Table 3 — Topic discovery"))
cells.append(code(
"table3 = pd.DataFrame([{\n"
"    'Metric': ['Topics discovered', 'Coherence (c_v)', 'Diversity',\n"
"               'Outlier sentences', 'Min cluster size'],\n"
"    'Value':  [m_topic['n_topics'], m_topic['coherence_cv'], m_topic['diversity'],\n"
"               m_topic['n_outliers'], m_topic['min_cluster_size']]\n"
"}])\n"
"print(table3.to_string(index=False))\n"
"table3.to_csv('../artifacts/table_topic_discovery.csv', index=False)\n"
"\n"
"print(f\"\\nTopic coherence = {m_topic['coherence_cv']:.4f} \"\n"
"      f\"({'good' if m_topic['coherence_cv']>=0.5 else 'acceptable' if m_topic['coherence_cv']>=0.3 else 'weak'})\")\n"
"print(f\"Topic diversity = {m_topic['diversity']:.4f} \"\n"
"      f\"({'good' if m_topic['diversity']>=0.7 else 'acceptable'})\")"
))

# ── Cell 6 — Figure 1: classification comparison
cells.append(md("## 6. Figure 1 — Classification comparison"))
cells.append(code(
"metrics_cols = ['accuracy','precision','recall','macro_f1']\n"
"x = np.arange(len(metrics_cols))\n"
"n_models = len(table2)\n"
"w = 0.8 / n_models\n"
"colors = ['#8888ff', '#ff5555', '#55aa55'][:n_models]\n"
"\n"
"fig, ax = plt.subplots(figsize=(10, 5))\n"
"for i, (_, row) in enumerate(table2.iterrows()):\n"
"    vals = [row[k] for k in metrics_cols]\n"
"    ax.bar(x + (i - (n_models-1)/2)*w, vals, w,\n"
"           label=row['model'][:30], color=colors[i])\n"
"    for j, v in enumerate(vals):\n"
"        ax.text(j + (i - (n_models-1)/2)*w, v + 0.01, f'{v:.3f}',\n"
"                ha='center', fontsize=7)\n"
"ax.set_xticks(x); ax.set_xticklabels(metrics_cols)\n"
"ax.set_ylim(0, 1); ax.set_ylabel('Score')\n"
"ax.set_title('Classification: baseline vs semantic models')\n"
"ax.legend(loc='lower right', fontsize=8)\n"
"plt.tight_layout()\n"
"plt.savefig('../artifacts/figs/fig_classification_comparison.png')\n"
"plt.show()\n"
"print('saved -> artifacts/figs/fig_classification_comparison.png')"
))

# ── Cell 7 — Figure 2: dataset imbalance pie
cells.append(md("## 7. Figure 2 — Dataset composition"))
cells.append(code(
"fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
"axes[0].pie([eda['relevant'], eda['non_relevant']],\n"
"            labels=['Relevant','Non-relevant'],\n"
"            autopct='%1.1f%%', colors=['#4c72b0','#dd8452'])\n"
"axes[0].set_title(f\"Class distribution (n={eda['raw_rows']})\")\n"
"\n"
"bars = [eda['relevant'], eda['non_relevant']]\n"
"axes[1].bar(['Relevant','Non-relevant'], bars, color=['#4c72b0','#dd8452'])\n"
"for i, v in enumerate(bars):\n"
"    axes[1].text(i, v, str(v), ha='center', va='bottom')\n"
"axes[1].set_ylabel('Count'); axes[1].set_title('Sentence count per class')\n"
"plt.tight_layout()\n"
"plt.savefig('../artifacts/figs/fig_dataset_summary.png')\n"
"plt.show()"
))

# ── Cell 8 — Figure 3: per-metric delta
cells.append(md("## 8. Figure 3 — Per-metric delta vs baseline"))
cells.append(code(
"fig, ax = plt.subplots(figsize=(8, 4))\n"
"base_row = table2.iloc[0]\n"
"for i, (_, row) in enumerate(table2.iloc[1:].iterrows(), start=1):\n"
"    deltas = [row[k] - base_row[k] for k in metrics_cols]\n"
"    ax.plot(metrics_cols, deltas, marker='o', label=row['model'][:30])\n"
"ax.axhline(0, color='black', linewidth=0.5)\n"
"ax.set_ylabel('Δ vs baseline'); ax.set_title('Metric delta vs baseline')\n"
"ax.legend(fontsize=8)\n"
"plt.tight_layout()\n"
"plt.savefig('../artifacts/figs/fig_metric_delta.png')\n"
"plt.show()"
))

# ── Cell 9 — summary box
cells.append(md("## 9. Final summary"))
cells.append(code(
"print('=' * 70)\n"
"print(f\"{'THESIS RESULTS SUMMARY':^70s}\")\n"
"print('=' * 70)\n"
"print(f\"Dataset            : SMAH-CTI ({eda['raw_rows']} sentences, {eda['num_documents']} docs)\")\n"
"print(f\"Split              : {split['train_size']} train / {split['test_size']} test (document-level)\")\n"
"print()\n"
"print('CLASSIFICATION (baseline → semantic):')\n"
"for _, row in table2.iterrows():\n"
"    print(f\"  {row['model'][:45]:<45s}  macro-F1 = {row['macro_f1']:.4f}\")\n"
"print()\n"
"print(f\"TOPIC DISCOVERY (BERTopic):\")\n"
"print(f\"  Topics            = {m_topic['n_topics']}\")\n"
"print(f\"  Coherence (c_v)   = {m_topic['coherence_cv']:.4f}\")\n"
"print(f\"  Diversity         = {m_topic['diversity']:.4f}\")\n"
"print('=' * 70)"
))

# ── Cell 10 — auto-draft Chapter 5 paragraph
cells.append(md("## 10. Auto-draft Chapter 5 paragraph"))
cells.append(code(
"best = table2.iloc[table2['macro_f1'].idxmax()]\n"
"baseline_f1 = table2.iloc[0]['macro_f1']\n"
"delta = best['macro_f1'] - baseline_f1\n"
"\n"
"para = f\"\"\"\n"
"CHAPTER 5 — DRAFT RESULTS PARAGRAPH\n"
"\n"
"On the SMAH-CTI document-level split ({split['train_size']} training and\n"
"{split['test_size']} test sentences), the tuned TF-IDF baseline achieved\n"
"macro-F1 = {baseline_f1:.4f}. \n"
"\n"
"\"\"\"\n"
"if abs(delta) < 0.01:\n"
"    para += f\"\"\"Semantic and hybrid approaches produced no measurable improvement\n"
"(best macro-F1 = {best['macro_f1']:.4f}, delta = {delta:+.4f}). This indicates that\n"
"short, terminology-dense CTI sentences are already well captured by lexical features,\n"
"and that pre-trained sentence embeddings trained on general-domain text do not add\n"
"discriminative signal for this corpus.\n"
"\"\"\"\n"
"elif delta > 0:\n"
"    para += f\"\"\"The best configuration ({best['model']}) improved macro-F1 to\n"
"{best['macro_f1']:.4f} (+{delta:.4f} vs baseline), showing that semantic\n"
"information complements lexical features on this task.\n"
"\"\"\"\n"
"else:\n"
"    para += f\"\"\"Semantic approaches underperformed the lexical baseline\n"
"(best = {best['macro_f1']:.4f}, delta = {delta:+.4f}). This is attributed to the\n"
"short length (mean 21.7 words) and technical vocabulary of the corpus, where exact-term\n"
"features remain more informative than dense semantic embeddings.\n"
"\"\"\"\n"
"\n"
"para += f\"\"\"\n"
"For emerging-threat detection, BERTopic discovered {m_topic['n_topics']} coherent\n"
"topics with c_v coherence = {m_topic['coherence_cv']:.4f} and diversity =\n"
"{m_topic['diversity']:.4f}. The discovered topics correspond to analyst-recognisable\n"
"threat campaigns including North Korean (DPRK) operations, ransomware families\n"
"(CyberVolk, HexaLocker), Fortinet device compromise (TableFlip), Scattered Spider's\n"
"RaaS activity, Sandworm's CaddyWiper OT attacks, and Israeli-Iranian cyber operations,\n"
"indicating that BERTopic produces semantically meaningful clusters suitable for\n"
"CTI analyst triage.\n"
"\"\"\"\n"
"\n"
"print(para)\n"
"open('../artifacts/chapter5_draft.txt', 'w').write(para)\n"
"print('\\nsaved -> artifacts/chapter5_draft.txt')"
))

# ── Cell 11 — Checklist for thesis writing
cells.append(md(
"## 11. Writing checklist\n"
"\n"
"Use this checklist when composing Chapter 5:\n"
"\n"
"- [ ] Table 1 (dataset summary) → Section 5.1\n"
"- [ ] Table 2 (classification) → Section 5.2\n"
"- [ ] Table 3 (topic discovery) → Section 5.3\n"
"- [ ] Figure 1 (classification bars) → Section 5.2\n"
"- [ ] Figure 2 (dataset pie) → Section 5.1\n"
"- [ ] Figure 3 (delta plot) → Section 5.2\n"
"- [ ] Chapter 5 draft paragraph → Section 5.4 (edit into fluent prose)\n"
"- [ ] Cross-reference the BERTopic topic table (saved by Notebook 5) for Chapter 4.3\n"
"- [ ] Add limitations paragraph (short sentences, 51% outliers, no temporal data)\n"
"\n"
"This completes the six-notebook thesis pipeline. Chapter 6 (Conclusion) can now be\n"
"written directly from the summary box in Cell 9 plus the chapter5_draft.txt."
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"}
    },
    "nbformat": 4, "nbformat_minor": 5
}

out = os.path.join("notebooks", "06_results_comparison.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Wrote:", os.path.abspath(out))
print("Cells:", len(cells))