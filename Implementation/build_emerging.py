"""Create notebooks/05_emerging_threats.ipynb."""
from pathlib import Path
import json

ROOT = Path(r"D:\Projects\done\thesis\Implementation")
NB_PATH = ROOT / "notebooks" / "05_emerging_threats.ipynb"

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

    md("# 05 — Emerging Threat Detection"),
    md(
        "> **Purpose** — Analyze how discovered topics evolve over time and "
        "compute an **Emerging Threat Score (ETS)** that combines five signals: "
        "novelty, growth, persistence, activity, and term change."
    ),

    md(
        '<span class="badge badge-blue">Temporal</span>'
        '<span class="badge badge-green">Topic Evolution</span>'
        '<span class="badge badge-orange">ETS Scoring</span>'
        '<span class="badge badge-red">Prioritization</span>'
    ),

    md(
        "## Table of Contents\n"
        "\n"
        "1. [Setup & Imports](#1-setup--imports)\n"
        "2. [Load Artifacts](#2-load-artifacts)\n"
        "3. [Topic Prevalence Over Time](#3-topic-prevalence-over-time)\n"
        "4. [Topic Growth Analysis](#4-topic-growth-analysis)\n"
        "5. [Topic Persistence](#5-topic-persistence)\n"
        "6. [Novelty Detection](#6-novelty-detection)\n"
        "7. [Activity & Term Change](#7-activity--term-change)\n"
        "8. [Emerging Threat Score (ETS)](#8-emerging-threat-score-ets)\n"
        "9. [Candidate Emerging Topics](#9-candidate-emerging-topics)\n"
        "10. [Temporal Visualizations](#10-temporal-visualizations)\n"
        "11. [Case Study — Top Emerging Topic](#11-case-study--top-emerging-topic)\n"
        "12. [Save Results](#12-save-results)\n"
        "13. [Summary](#13-summary)\n"
    ),

    # ==================================================
    md("## 1. Setup & Imports"),
    code([
        "import sys, pathlib",
        "sys.path.append(str(pathlib.Path.cwd().parent))",
        "",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "import joblib",
        "",
        "from src.config import (OUTPUT_DIR, MODELS_DIR, TOPICS_DIR,",
        "                        FIGURES_DIR, REPORTS_DIR, ETS_WEIGHTS)",
        "from src.temporal import topic_prevalence_by_period",
        "from src.emerging import compute_emerging_scores",
        "",
        "sns.set_style('whitegrid')",
        "plt.rcParams['figure.dpi'] = 100",
        "plt.rcParams['savefig.dpi'] = 150",
        "",
        "print('ETS weights:')",
        "for k, v in ETS_WEIGHTS.items():",
        "    print(f'  {k:12s} {v}')",
        "print('Sum =', sum(ETS_WEIGHTS.values()))",
    ]),

    # ==================================================
    md("## 2. Load Artifacts"),
    md(
        "Load the cleaned dataset, NMF model, and topic-term matrix "
        "produced in previous notebooks."
    ),
    code([
        "df = pd.read_csv(OUTPUT_DIR / 'cleaned_cti.csv')",
        "df['date'] = pd.to_datetime(df['date'])",
        "df['period'] = df['date'].dt.to_period('M').astype(str)",
        "",
        "text_col = 'clean_full' if 'clean_full' in df.columns else 'clean_text'",
        "docs = df[text_col].astype(str).tolist()",
        "",
        "nmf_model = joblib.load(MODELS_DIR / 'nmf_model.pkl')",
        "print('NMF model loaded. Components:', nmf_model.components_.shape)",
        "print('Documents:', len(docs))",
        "print('Date range:', df['date'].min().date(), '->', df['date'].max().date())",
        "print('Periods   :', df['period'].nunique())",
    ]),

    code([
        "from sklearn.feature_extraction.text import TfidfVectorizer",
        "vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=5)",
        "X = vec.fit_transform(docs)",
        "doc_topic = nmf_model.transform(X)",
        "print('Document-topic matrix:', doc_topic.shape)",
    ]),

    # ==================================================
    md("## 3. Topic Prevalence Over Time"),
    md(
        "Compute average topic weight per time period. This is the foundation "
        "for all subsequent analysis."
    ),
    code([
        "prev = topic_prevalence_by_period(doc_topic, df['date'], freq='M')",
        "prev.index = prev.index.astype(str)",
        "",
        "print('Temporal prevalence matrix shape:', prev.shape)",
        "prev.round(4)",
    ]),

    code([
        "fig, ax = plt.subplots(figsize=(13, 6))",
        "for col in prev.columns:",
        "    ax.plot(prev.index, prev[col], marker='o', linewidth=1.5, label=col)",
        "ax.set_title('Topic Prevalence Over Time', fontweight='bold', fontsize=13)",
        "ax.set_xlabel('Period')",
        "ax.set_ylabel('Average Topic Weight')",
        "ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=2)",
        "plt.xticks(rotation=45)",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / '05_topic_prevalence_over_time.png')",
        "plt.show()",
    ]),

    # ==================================================
    md("## 4. Topic Growth Analysis"),
    md(
        "Growth = percentage change of topic prevalence between consecutive "
        "periods. Positive growth indicates increasing attention."
    ),
    code([
        "growth = prev.pct_change().replace([np.inf, -np.inf], 0).fillna(0)",
        "",
        "print('Growth matrix (first 3 periods):')",
        "growth.head(3).round(4)",
    ]),

    code([
        "max_growth = growth.max().sort_values(ascending=False)",
        "print('Max growth per topic (across all periods):')",
        "print(max_growth.round(4).to_string())",
        "",
        "fig, ax = plt.subplots(figsize=(10, 4))",
        "max_growth.plot(kind='bar', ax=ax, color='coral', edgecolor='white')",
        "ax.set_title('Maximum Growth Rate per Topic', fontweight='bold')",
        "ax.set_ylabel('Max pct change')",
        "ax.set_xlabel('Topic')",
        "plt.xticks(rotation=45)",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / '05_max_growth.png')",
        "plt.show()",
    ]),

    # ==================================================
    md("## 5. Topic Persistence"),
    md(
        "Persistence = fraction of periods in which a topic's weight is "
        "above its mean. High persistence = sustained theme."
    ),
    code([
        "persistence = (prev > prev.mean()).sum() / len(prev)",
        "persistence = persistence.sort_values(ascending=False)",
        "",
        "print('Topic persistence scores:')",
        "print(persistence.round(4).to_string())",
        "",
        "fig, ax = plt.subplots(figsize=(10, 4))",
        "persistence.plot(kind='bar', ax=ax, color='seagreen', edgecolor='white')",
        "ax.set_title('Topic Persistence', fontweight='bold')",
        "ax.set_ylabel('Fraction of periods above mean')",
        "ax.set_xlabel('Topic')",
        "ax.set_ylim(0, 1.05)",
        "plt.xticks(rotation=45)",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / '05_persistence.png')",
        "plt.show()",
    ]),

    # ==================================================
    md("## 6. Novelty Detection"),
    md(
        "A topic is **novel** in a period if its prevalence was ~0 in the "
        "previous period and rises above a threshold in the current one."
    ),
    code([
        "def compute_novelty(prevalence_df, threshold=0.005):",
        "    \"\"\"Return binary novelty flag per (topic, period).\"\"\"",
        "    novelty = pd.DataFrame(0, index=prevalence_df.index,",
        "                                columns=prevalence_df.columns)",
        "    prev_row = None",
        "    for idx in prevalence_df.index:",
        "        row = prevalence_df.loc[idx]",
        "        if prev_row is None:",
        "            novelty.loc[idx] = (row > threshold).astype(int)",
        "        else:",
        "            novelty.loc[idx] = ((prev_row < threshold) &",
        "                                (row > threshold)).astype(int)",
        "        prev_row = row",
        "    return novelty",
        "",
        "novelty = compute_novelty(prev, threshold=0.005)",
        "print('Total novelty events per topic:')",
        "print(novelty.sum().sort_values(ascending=False).to_string())",
    ]),

    # ==================================================
    md("## 7. Activity & Term Change"),
    md(
        "**Activity** = mean topic weight across all periods.  \n"
        "**Term change** = standard deviation of topic weight (measures volatility)."
    ),
    code([
        "activity    = prev.mean()",
        "term_change = prev.std()",
        "",
        "print('Activity (mean weight):')",
        "print(activity.round(4).to_string())",
        "print()",
        "print('Term change (std of weight):')",
        "print(term_change.round(4).to_string())",
    ]),

    # ==================================================
    md("## 8. Emerging Threat Score (ETS)"),
    md(
        "Combine the five signals with weights from `src/config.py`. "
        "Every component is min-max normalized to [0, 1] before combining."
    ),
    code([
        "ets = compute_emerging_scores(prev)",
        "print('Emerging Threat Scores:')",
        "ets.round(4)",
    ]),

    code([
        "fig, ax = plt.subplots(figsize=(10, 5))",
        "ets_sorted = ets.sort_values('ETS', ascending=True)",
        "ax.barh(ets_sorted['Topic'], ets_sorted['ETS'], color='crimson')",
        "ax.set_title('Emerging Threat Score by Topic', fontweight='bold')",
        "ax.set_xlabel('ETS')",
        "for i, (t, v) in enumerate(zip(ets_sorted['Topic'], ets_sorted['ETS'])):",
        "    ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / '05_emerging_threat_scores.png')",
        "plt.show()",
    ]),

    code([
        "components = ['Novelty','Growth','Persistence','Activity','TermChange']",
        "ets_plot = ets.set_index('Topic')[components]",
        "",
        "fig, ax = plt.subplots(figsize=(11, 5))",
        "ets_plot.plot(kind='bar', stacked=True, ax=ax,",
        "              colormap='viridis', edgecolor='white')",
        "ax.set_title('ETS Component Breakdown', fontweight='bold')",
        "ax.set_ylabel('Normalized Score')",
        "ax.legend(loc='upper right', fontsize=9)",
        "plt.xticks(rotation=45)",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / '05_ets_components.png')",
        "plt.show()",
    ]),

    # ==================================================
    md("## 9. Candidate Emerging Topics"),
    md(
        "Topics above a chosen threshold are flagged as **candidate emerging "
        "threats**. Threshold is data-driven — use the 75th percentile or mean+1σ."
    ),
    code([
        "threshold = ets['ETS'].mean() + ets['ETS'].std()",
        "print(f'Threshold (mean + 1 sigma) = {threshold:.4f}')",
        "",
        "candidates = ets[ets['ETS'] > threshold].copy()",
        "candidates['Status'] = 'Candidate'",
        "",
        "print(f'Candidate topics: {len(candidates)}')",
        "candidates.round(4)",
    ]),

    code([
        "nmf_rows = pd.read_csv(TOPICS_DIR / 'nmf_topics.csv')",
        "nmf_rows['topic'] = nmf_rows['topic'].str.replace('NMF_', 'topic_')",
        "term_map = dict(zip(nmf_rows['topic'], nmf_rows['top_terms']))",
        "",
        "candidates['TopTerms'] = candidates['Topic'].map(term_map)",
        "print('Candidate emerging topics with representative terms:')",
        "for _, row in candidates.iterrows():",
        "    print(f'\\n[{row[\"Topic\"]}]  ETS={row[\"ETS\"]:.4f}')",
        "    print(f'  Terms: {row[\"TopTerms\"]}')",
    ]),

    # ==================================================
    md("## 10. Temporal Visualizations"),
    md("### 10.1 Topic Prevalence Heatmap"),
    code([
        "plt.figure(figsize=(13, 6))",
        "sns.heatmap(prev.T, cmap='YlOrRd', annot=False, cbar_kws={'label':'Prevalence'})",
        "plt.title('Topic Prevalence Heatmap (Topic x Period)', fontweight='bold')",
        "plt.xlabel('Period')",
        "plt.ylabel('Topic')",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / '05_prevalence_heatmap.png')",
        "plt.show()",
    ]),

    md("### 10.2 Growth Heatmap"),
    code([
        "plt.figure(figsize=(13, 6))",
        "sns.heatmap(growth.T, cmap='RdBu_r', center=0, annot=False,",
        "            cbar_kws={'label':'Pct change'})",
        "plt.title('Topic Growth Heatmap', fontweight='bold')",
        "plt.xlabel('Period')",
        "plt.ylabel('Topic')",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / '05_growth_heatmap.png')",
        "plt.show()",
    ]),

    # ==================================================
    md("## 11. Case Study — Top Emerging Topic"),
    code([
        "if len(candidates) > 0:",
        "    top_topic = candidates.iloc[0]['Topic']",
        "    print(f'Case study: {top_topic}')",
        "    print(f'  ETS       : {candidates.iloc[0][\"ETS\"]:.4f}')",
        "    print(f'  Top terms : {candidates.iloc[0][\"TopTerms\"]}')",
        "    print()",
        "    print('Prevalence trajectory:')",
        "    print(prev[top_topic].round(4).to_string())",
        "else:",
        "    print('No candidate emerging topics at this threshold.')",
    ]),

    code([
        "if len(candidates) > 0:",
        "    top_topic = candidates.iloc[0]['Topic']",
        "    fig, ax = plt.subplots(figsize=(11, 4))",
        "    ax.plot(prev.index, prev[top_topic], marker='o', color='crimson',",
        "            linewidth=2)",
        "    ax.fill_between(prev.index, prev[top_topic], alpha=0.2, color='crimson')",
        "    ax.set_title(f'Prevalence Trajectory — {top_topic}', fontweight='bold')",
        "    ax.set_xlabel('Period')",
        "    ax.set_ylabel('Average Topic Weight')",
        "    plt.xticks(rotation=45)",
        "    plt.tight_layout()",
        "    plt.savefig(FIGURES_DIR / '05_case_study.png')",
        "    plt.show()",
    ]),

    # ==================================================
    md("## 12. Save Results"),
    code([
        "prev.to_csv(TOPICS_DIR / 'temporal_prevalence.csv')",
        "print(f'Saved: {TOPICS_DIR / \"temporal_prevalence.csv\"}')",
        "",
        "growth.to_csv(TOPICS_DIR / 'temporal_growth.csv')",
        "print(f'Saved: {TOPICS_DIR / \"temporal_growth.csv\"}')",
        "",
        "ets.to_csv(REPORTS_DIR / 'emerging_threat_scores.csv', index=False)",
        "print(f'Saved: {REPORTS_DIR / \"emerging_threat_scores.csv\"}')",
        "",
        "candidates.to_csv(REPORTS_DIR / 'candidate_emerging_topics.csv', index=False)",
        "print(f'Saved: {REPORTS_DIR / \"candidate_emerging_topics.csv\"}')",
    ]),

    # ==================================================
    md("## 13. Summary"),
    code([
        "print('=== EMERGING THREAT ANALYSIS SUMMARY ===')",
        "print(f'Periods analyzed  : {len(prev)}')",
        "print(f'Topics analyzed   : {prev.shape[1]}')",
        "print(f'Threshold (mean+std): {threshold:.4f}')",
        "print(f'Candidate topics  : {len(candidates)}')",
        "print()",
        "print('Top 3 emerging topics:')",
        "for _, row in ets.head(3).iterrows():",
        "    print(f'  {row[\"Topic\"]:10s} ETS={row[\"ETS\"]:.4f}')",
        "print()",
        "print('Artifacts:')",
        "print(f'  - {TOPICS_DIR / \"temporal_prevalence.csv\"}')",
        "print(f'  - {REPORTS_DIR / \"emerging_threat_scores.csv\"}')",
        "print(f'  - {REPORTS_DIR / \"candidate_emerging_topics.csv\"}')",
    ]),

    md(
        '<div class="callout">'
        '<b>Interpretation note:</b> The Emerging Threat Score is a '
        '<i>prioritization signal</i>, not proof of a new or zero-day threat. '
        'High-ETS topics should be validated against external CTI sources '
        'before being treated as confirmed emerging threats.'
        '</div>'
    ),

    md(
        '<div class="callout">'
        '<b>All notebooks complete.</b> You now have classification results '
        'and emerging-threat analysis ready to populate Chapter 5.'
        '</div>'
    ),

    md("---\n*End of emerging threats notebook.*"),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (CTI Thesis)",
            "language": "python",
            "name": "cti-thesis",
        },
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Created: {NB_PATH}")
print(f"Cells:   {len(cells)}")