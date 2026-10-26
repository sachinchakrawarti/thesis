# scripts/make_03_technique.py
"""Generate notebooks/03_technique_classifier.ipynb"""
from pathlib import Path
import nbformat as nbf


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


CELLS = [

    md(
        "# Notebook 3: MITRE ATT&CK Technique Classifier\n"
        "\n"
        "**Purpose:** Multi-class classifier that maps text to MITRE ATT&CK parent techniques.\n"
        "\n"
        "**Input:** `dataset/cti_dataset_clean.csv` (from Notebook 01)\n"
        "**Strategy:** SBERT embeddings + LinearSVC, filtered to classes with ≥ 10 examples\n"
        "\n"
        "**Outputs**\n"
        "- `models/classification/sbert_technique.joblib`\n"
        "- `results/03_technique_classifier/technique_results.csv`\n"
        "- `results/03_technique_classifier/technique_confusion_matrix.png`"
    ),

    md("## Step 0 — Setup"),
    code(
        "import warnings, time, joblib\n"
        "from pathlib import Path\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "DATA_PATH    = PROJECT_ROOT / 'dataset' / 'cti_dataset_clean.csv'\n"
        "CLS_DIR      = PROJECT_ROOT / 'models' / 'classification'\n"
        "RESULTS_DIR  = PROJECT_ROOT / 'results'\n"
        "TECH_DIR     = RESULTS_DIR / '03_technique_classifier'\n"
        "for d in (CLS_DIR, RESULTS_DIR, TECH_DIR):\n"
        "    d.mkdir(parents=True, exist_ok=True)\n"
        "RANDOM_SEED = 42\n"
        "sns.set_theme(style='whitegrid')\n"
        "print('Charts will be saved to:', TECH_DIR)"
    ),

    md("## Step 1 — Load Clean Data"),
    code(
        "if not DATA_PATH.exists():\n"
        "    raise FileNotFoundError(f'{DATA_PATH} not found. Run Notebook 01 first.')\n"
        "df = pd.read_csv(DATA_PATH)\n"
        "print(f'Loaded {len(df):,} rows from {df[\"doc_id\"].nunique():,} documents.')"
    ),

    md("## Step 2 — Filter to Technique-Labeled Rows"),
    code(
        "mask = (df['relevant'].str.lower() == 'yes') & \\\n"
        "       (df['techniques'].fillna('').astype(str).str.strip() != '')\n"
        "df_tech = df[mask].copy()\n"
        "\n"
        "# Extract parent technique (strip sub-suffixes): T1003.008 -> T1003\n"
        "df_tech['parent'] = (\n"
        "    df_tech['techniques']\n"
        "    .astype(str).str.split(',').str[0].str.strip().str.split('.').str[0]\n"
        ")\n"
        "\n"
        "print(f'Rows with technique labels: {len(df_tech):,}')\n"
        "print(f'Distinct parent techniques: {df_tech[\"parent\"].nunique()}')\n"
        "print()\n"
        "print('Top 15:')"
        "print(df_tech['parent'].value_counts().head(15).to_string())"
    ),

    md("## Step 3 — Keep Classes with ≥ 10 Examples"),
    code(
        "counts = df_tech['parent'].value_counts()\n"
        "keep = counts[counts >= 10].index\n"
        "df_tech = df_tech[df_tech['parent'].isin(keep)].copy()\n"
        "\n"
        "print(f'Classes kept  : {df_tech[\"parent\"].nunique()} of {len(counts)}')\n"
        "print(f'Rows remaining: {len(df_tech):,}')\n"
        "print()\n"
        "print('Final class distribution:')\n"
        "print(df_tech['parent'].value_counts().to_string())\n"
        "print()\n"
        "print(f'Min class: {counts[keep].min()} | Max: {counts[keep].max()} | Median: {int(counts[keep].median())}')"
    ),

    md("## Step 4 — Class Distribution Chart"),
    code(
        "plt.figure(figsize=(10, 5))\n"
        "df_tech['parent'].value_counts().plot(kind='barh', color='#9467bd')\n"
        "plt.title(f'Technique Class Distribution ({df_tech[\"parent\"].nunique()} classes)')\n"
        "plt.xlabel('sentences')\n"
        "plt.gca().invert_yaxis()\n"
        "plt.tight_layout()\n"
        "plt.savefig(TECH_DIR / 'class_distribution.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 5 — Encode with SBERT"),
    code(
        "from sentence_transformers import SentenceTransformer\n"
        "sbert = SentenceTransformer('all-MiniLM-L6-v2')\n"
        "\n"
        "t0 = time.perf_counter()\n"
        "X = sbert.encode(df_tech['sentence_text'].tolist(),\n"
        "                 batch_size=64, show_progress_bar=True, normalize_embeddings=True)\n"
        "encode_t = time.perf_counter() - t0\n"
        "y = df_tech['parent'].values\n"
        "\n"
        "print(f'Encoded {len(X)} sentences in {encode_t:.1f}s')\n"
        "print(f'X shape: {X.shape}')\n"
        "print(f'Classes: {len(set(y))}')"
    ),

    md("## Step 6 — Train/Test Split"),
    code(
        "from sklearn.model_selection import StratifiedShuffleSplit\n"
        "sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)\n"
        "tr, te = next(sss.split(X, y))\n"
        "print(f'Train: {len(tr)} samples')\n"
        "print(f'Test : {len(te)} samples')"
    ),

    md("## Step 7 — Train LinearSVC"),
    code(
        "from sklearn.svm import LinearSVC\n"
        "from sklearn.metrics import accuracy_score, f1_score\n"
        "\n"
        "t0 = time.perf_counter()\n"
        "clf = LinearSVC(class_weight='balanced', random_state=RANDOM_SEED, max_iter=5000)\n"
        "clf.fit(X[tr], y[tr])\n"
        "train_t = time.perf_counter() - t0\n"
        "\n"
        "t0 = time.perf_counter()\n"
        "pred = clf.predict(X[te])\n"
        "infer_t = time.perf_counter() - t0\n"
        "\n"
        "acc = accuracy_score(y[te], pred)\n"
        "macro_f1 = f1_score(y[te], pred, average='macro', zero_division=0)\n"
        "weighted_f1 = f1_score(y[te], pred, average='weighted', zero_division=0)\n"
        "\n"
        "print(f'Training time : {train_t:.2f}s')\n"
        "print(f'Inference time: {infer_t:.3f}s')\n"
        "print()\n"
        "print(f'Accuracy    : {acc:.4f}')\n"
        "print(f'Macro-F1    : {macro_f1:.4f}')\n"
        "print(f'Weighted-F1 : {weighted_f1:.4f}')"
    ),

    md("## Step 8 — Per-Class Report"),
    code(
        "from sklearn.metrics import classification_report\n"
        "print(classification_report(y[te], pred, zero_division=0))"
    ),

    md("## Step 9 — Confusion Matrix"),
    code(
        "from sklearn.metrics import confusion_matrix\n"
        "labels = sorted(set(y))\n"
        "cm = confusion_matrix(y[te], pred, labels=labels)\n"
        "\n"
        "plt.figure(figsize=(10, 8))\n"
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
        "            xticklabels=labels, yticklabels=labels, cbar=False)\n"
        "plt.title(f'Technique Classifier Confusion Matrix (acc = {acc:.3f})')\n"
        "plt.ylabel('True')\n"
        "plt.xlabel('Predicted')\n"
        "plt.xticks(rotation=45, ha='right')\n"
        "plt.yticks(rotation=0)\n"
        "plt.tight_layout()\n"
        "plt.savefig(TECH_DIR / 'technique_confusion_matrix.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 10 — Save Results Table"),
    code(
        "report = classification_report(y[te], pred, output_dict=True, zero_division=0)\n"
        "rows = []\n"
        "for cls, metrics in report.items():\n"
        "    if isinstance(metrics, dict):\n"
        "        rows.append({\n"
        "            'class': cls,\n"
        "            'precision': metrics['precision'],\n"
        "            'recall': metrics['recall'],\n"
        "            'f1': metrics['f1-score'],\n"
        "            'support': metrics['support'],\n"
        "        })\n"
        "res_df = pd.DataFrame(rows)\n"
        "out = TECH_DIR / 'technique_results.csv'\n"
        "res_df.to_csv(out, index=False)\n"
        "print('Saved:', out)\n"
        "res_df.round(3)"
    ),

    md("## Step 11 — Save Trained Model"),
    code(
        "final = LinearSVC(class_weight='balanced', random_state=RANDOM_SEED, max_iter=5000)\n"
        "final.fit(X, y)\n"
        "model_path = CLS_DIR / 'sbert_technique.joblib'\n"
        "joblib.dump(final, model_path)\n"
        "print('Saved:', model_path)\n"
        "print(f'Trained on {len(X)} samples across {len(set(y))} classes')"
    ),

    md("## Step 12 — Save Config"),
    code(
        "import json\n"
        "config = {\n"
        "    'embedding_model': 'all-MiniLM-L6-v2',\n"
        "    'embedding_dim': 384,\n"
        "    'classifier': 'LinearSVC',\n"
        "    'class_weight': 'balanced',\n"
        "    'max_iter': 5000,\n"
        "    'trained_on_rows': int(len(X)),\n"
        "    'n_classes': int(len(set(y))),\n"
        "    'labels': sorted(set(y.tolist())),\n"
        "    'accuracy': round(float(acc), 4),\n"
        "    'macro_f1': round(float(macro_f1), 4),\n"
        "    'weighted_f1': round(float(weighted_f1), 4),\n"
        "}\n"
        "cfg_path = CLS_DIR / 'technique_config.json'\n"
        "with open(cfg_path, 'w') as f:\n"
        "    json.dump(config, f, indent=2)\n"
        "print('Saved:', cfg_path)"
    ),

    md("## Step 13 — Summary"),
    code(
        "print('=' * 55)\n"
        "print('TECHNIQUE CLASSIFIER SUMMARY')\n"
        "print('=' * 55)\n"
        "print(f'Total labeled rows     : {len(df_tech):,}')\n"
        "print(f'Number of classes      : {len(set(y))}')\n"
        "print(f'Train / Test           : {len(tr)} / {len(te)}')\n"
        "print(f'Smallest class size    : {int(counts[keep].min())}')\n"
        "print(f'Largest class size     : {int(counts[keep].max())}')\n"
        "print()\n"
        "print(f'Accuracy               : {acc:.4f}')\n"
        "print(f'Macro-F1               : {macro_f1:.4f}')\n"
        "print(f'Weighted-F1            : {weighted_f1:.4f}')\n"
        "print()\n"
        "print('Saved files:')\n"
        "print(f'  {CLS_DIR / \"sbert_technique.joblib\"}')\n"
        "print(f'  {CLS_DIR / \"technique_config.json\"}')\n"
        "print(f'  {TECH_DIR / \"technique_results.csv\"}')\n"
        "print(f'  {TECH_DIR / \"technique_confusion_matrix.png\"}')\n"
        "print(f'  {TECH_DIR / \"class_distribution.png\"}')\n"
        "print('=' * 55)"
    ),
]


def main():
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    }
    here = Path(__file__).resolve().parent
    root = here.parent if here.name == "scripts" else here
    out = root / "notebooks" / "03_technique_classifier.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()