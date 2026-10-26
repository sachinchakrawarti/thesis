"""
Generates notebooks/07_report_analyzer.ipynb — fresh version with CSS-styled output.
Inputs : artifacts/train.csv, artifacts/train_emb.npy, artifacts/rel_with_topics.csv,
         artifacts/bertopic_model/
Outputs: artifacts/models/*.pkl, artifacts/models/*.json
         artifacts/report_analysis.json (per analysed report)
Run from project root:  python scripts/make_inference.py
"""
import json, os

os.makedirs("notebooks", exist_ok=True)
os.makedirs("artifacts/models", exist_ok=True)

def md(src):  return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}

cells = []

# ── Title
cells.append(md(
"# 07 — End-to-End Threat Report Analyzer\n"
"\n"
"**Inputs:** trained components from Notebooks 3–5  \n"
"**Outputs:** `artifacts/models/*.pkl`, `artifacts/models/*.json`, per-report JSON  \n"
"**Feeds thesis:** Chapter 4.4 — End-to-End Threat Report Analysis\n"
"\n"
"This notebook turns the pipeline into a working analyst tool:\n"
"\n"
"1. Loads cached models (hybrid classifier + BERTopic).\n"
"2. Extracts text from a PDF or TXT report.\n"
"3. Classifies every sentence as relevant / irrelevant.\n"
"4. Assigns relevant sentences to topics.\n"
"5. Applies three emerging-threat rules.\n"
"6. Renders a **CSS-styled verdict card** directly in the notebook output.\n"
))

# ── Cell 1 — setup
cells.append(md("## 1. Setup"))
cells.append(code(
"import pandas as pd, numpy as np, json, os, re, joblib\n"
"from pathlib import Path\n"
"from sentence_transformers import SentenceTransformer\n"
"from bertopic import BERTopic\n"
"from sklearn.linear_model import LogisticRegression\n"
"from sklearn.feature_extraction.text import TfidfVectorizer\n"
"from sklearn.preprocessing import normalize\n"
"from scipy.sparse import hstack\n"
"from IPython.display import HTML, display\n"
"\n"
"os.makedirs('../artifacts/models', exist_ok=True)\n"
"print('ready')"
))

# ── Cell 2 — save artifacts
cells.append(md(
"## 2. Save trained models (one-time)\n"
"\n"
"Retrains the hybrid classifier on the training set (<1 s) and saves all four\n"
"artifacts needed for inference."
))
cells.append(code(
"train = pd.read_csv('../artifacts/train.csv')\n"
"X_train_emb = np.load('../artifacts/train_emb.npy')\n"
"\n"
"vec = TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)\n"
"X_tfidf = vec.fit_transform(train['sentence_text'])\n"
"X_hybrid = hstack([X_tfidf, normalize(X_train_emb)]).tocsr()\n"
"clf = LogisticRegression(C=2.0, class_weight='balanced',\n"
"                         max_iter=2000, solver='liblinear', random_state=42)\n"
"clf.fit(X_hybrid, train['relevant'])\n"
"\n"
"joblib.dump(vec, '../artifacts/models/tfidf_vectorizer.pkl')\n"
"joblib.dump(clf, '../artifacts/models/hybrid_classifier.pkl')\n"
"print('saved classifier + vectorizer')\n"
"\n"
"rel = pd.read_csv('../artifacts/rel_with_topics.csv')\n"
"baseline = rel['topic'].value_counts(normalize=True).to_dict()\n"
"baseline = {int(k): float(v) for k, v in baseline.items()}\n"
"json.dump(baseline, open('../artifacts/models/topic_baseline.json','w'), indent=2)\n"
"print('saved topic baseline:', len(baseline), 'entries')\n"
"\n"
"from collections import Counter\n"
"tokens = re.findall(r'\\b[a-z]{3,}\\b', ' '.join(train['sentence_text'].str.lower()))\n"
"vocab = set(w for w, _ in Counter(tokens).most_common(50000))\n"
"json.dump(sorted(vocab), open('../artifacts/models/vocab_reference.json','w'))\n"
"print('saved vocab reference:', len(vocab), 'tokens')"
))

# ── Cell 3 — load BERTopic
cells.append(md("## 3. Load the BERTopic model"))
cells.append(code(
"topic_model = BERTopic.load('../artifacts/bertopic_model')\n"
"print('loaded BERTopic model with', len(topic_model.get_topics())-1, 'topics')"
))

# ── Cell 4 — extraction helpers
cells.append(md(
"## 4. PDF / text extraction\n"
"\n"
"Requires `pdfplumber` for PDF files. Plain text works with no extra dependency."
))
cells.append(code(
"def extract_text(path):\n"
"    p = Path(path)\n"
"    if p.suffix.lower() == '.pdf':\n"
"        import pdfplumber\n"
"        with pdfplumber.open(p) as pdf:\n"
"            return '\\n'.join(page.extract_text() or '' for page in pdf.pages)\n"
"    return p.read_text(encoding='utf-8', errors='ignore')\n"
"\n"
"def split_sentences(text, min_words=5):\n"
"    text = re.sub(r'\\s+', ' ', text)\n"
"    parts = re.split(r'(?<=[.!?])\\s+', text)\n"
"    return [s.strip() for s in parts if len(s.split()) >= min_words]"
))

# ── Cell 5 — predict_report function
cells.append(md(
"## 5. `predict_report(path)`\n"
"\n"
"Full pipeline: extract → classify → assign topics → apply emerging-threat rules."
))
cells.append(code(
"SBERT = SentenceTransformer('all-MiniLM-L6-v2')\n"
"VOCAB = set(json.load(open('../artifacts/models/vocab_reference.json')))\n"
"TOPIC_BASE = {int(k): float(v) for k, v in\n"
"              json.load(open('../artifacts/models/topic_baseline.json')).items()}\n"
"\n"
"def predict_report(path):\n"
"    raw = extract_text(path)\n"
"    sentences = split_sentences(raw)\n"
"    if not sentences:\n"
"        return {'report': str(path), 'error': 'no sentences extracted'}\n"
"\n"
"    emb = SBERT.encode(sentences, batch_size=64, convert_to_numpy=True)\n"
"    X = hstack([vec.transform(sentences), normalize(emb)]).tocsr()\n"
"    preds = clf.predict(X)\n"
"    relevant_sents = [s for s, p in zip(sentences, preds) if p == 1]\n"
"    relevant_emb   = emb[preds == 1]\n"
"\n"
"    if len(relevant_sents) > 0:\n"
"        topics, _ = topic_model.transform(relevant_sents, relevant_emb)\n"
"    else:\n"
"        topics = []\n"
"\n"
"    topic_counts = {}\n"
"    for t in topics:\n"
"        t = int(t); topic_counts[t] = topic_counts.get(t, 0) + 1\n"
"\n"
"    reasons = []\n"
"    emerging = False\n"
"\n"
"    if len(topics) > 0 and -1 in topic_counts:\n"
"        frac = topic_counts[-1] / len(topics)\n"
"        if frac > 0.30:\n"
"            emerging = True\n"
"            reasons.append(f'{topic_counts[-1]} sentences ({frac*100:.0f}%) could not be assigned to any known topic')\n"
"\n"
"    for tid, count in topic_counts.items():\n"
"        if tid == -1: continue\n"
"        base = TOPIC_BASE.get(tid, 1e-9)\n"
"        surge = (count / len(topics)) / max(base, 1e-9)\n"
"        if surge > 3.0 and count >= 3:\n"
"            emerging = True\n"
"            reasons.append(f'topic {tid} appears {count}x, {surge:.1f}x historical baseline')\n"
"\n"
"    toks = []\n"
"    for s in relevant_sents:\n"
"        toks += re.findall(r'\\b[a-z]{3,}\\b', s.lower())\n"
"    if toks:\n"
"        novel = [t for t in toks if t not in VOCAB]\n"
"        if len(novel)/len(toks) > 0.40:\n"
"            emerging = True\n"
"            reasons.append(f'{len(novel)/len(toks)*100:.0f}% of tokens are novel')\n"
"\n"
"    dominant = None\n"
"    if topic_counts:\n"
"        dom_tid = max(topic_counts, key=topic_counts.get)\n"
"        if dom_tid != -1:\n"
"            dominant = {'topic_id': dom_tid,\n"
"                        'top_words': [w for w,_ in topic_model.get_topic(dom_tid)][:8],\n"
"                        'count': topic_counts[dom_tid]}\n"
"            reasons.append(f'dominant topic {dom_tid}: ' + ', '.join(dominant['top_words']))\n"
"\n"
"    if not relevant_sents:\n"
"        verdict = 'not_relevant'\n"
"    elif emerging:\n"
"        verdict = 'emerging'\n"
"    else:\n"
"        verdict = 'known'\n"
"\n"
"    return {\n"
"        'report': str(path),\n"
"        'total_sentences': len(sentences),\n"
"        'relevant_sentences': len(relevant_sents),\n"
"        'relevant_fraction': round(len(relevant_sents)/len(sentences), 3),\n"
"        'topic_counts': {str(k): v for k, v in topic_counts.items()},\n"
"        'dominant_topic': dominant,\n"
"        'verdict': verdict,\n"
"        'emerging_threat': emerging,\n"
"        'reasons': reasons,\n"
"    }"
))

# ── Cell 6 — run on inline CTI sample
cells.append(md(
"## 6. Run on an inline CTI sample\n"
"\n"
"Quick sanity test using text with recognisable CTI terms."
))
cells.append(code(
"sample_text = '''\n"
"The threat actor exploited CVE-2023-23397 to gain initial access via Outlook.\n"
"A malicious DLL named glib-2.0.dll was dropped in the %PUBLIC% folder.\n"
"The loader contacted a hardcoded C2 server at 185.220.101.15 over an SSL socket.\n"
"Credentials were harvested from compromised cloud accounts using a phishing kit.\n"
"Ransomware activity was observed against the subsidiary, with the CyberVolk family encrypting files.\n"
"North Korean actors were attributed to a separate campaign targeting financial institutions.\n"
"'''\n"
"\n"
"test_path = '../artifacts/test_sample.txt'\n"
"open(test_path, 'w').write(sample_text)\n"
"\n"
"result = predict_report(test_path)\n"
"print(json.dumps(result, indent=2))"
))

# ── Cell 7 — CSS-rendered verdict card
cells.append(md(
"## 7. Render the verdict as a CSS card\n"
"\n"
"The output below is styled HTML rendered directly inside Jupyter — suitable for\n"
"screenshotting into the thesis or presenting live in the viva."
))
cells.append(code(
"VERDICT_STYLES = {\n"
"    'emerging':     {'bg': '#fdecea', 'border': '#c0392b', 'label': 'EMERGING THREAT SIGNAL', 'icon': '⚠'},\n"
"    'known':        {'bg': '#eafaf1', 'border': '#27ae60', 'label': 'KNOWN THREAT PATTERN',    'icon': '✔'},\n"
"    'not_relevant': {'bg': '#f0f0f0', 'border': '#7f8c8d', 'label': 'NOT RELEVANT',           'icon': '—'},\n"
"}\n"
"\n"
"def render_verdict(result):\n"
"    v = result.get('verdict', 'not_relevant')\n"
"    s = VERDICT_STYLES.get(v, VERDICT_STYLES['not_relevant'])\n"
"    report_name = Path(result['report']).name\n"
"\n"
"    # -- summary rows --\n"
"    rows = [\n"
"        ('Total sentences',   result['total_sentences']),\n"
"        ('Relevant sentences', result['relevant_sentences']),\n"
"        ('Relevant fraction', f\"{result['relevant_fraction']*100:.1f}%\"),\n"
"        ('Emerging threat',   'Yes' if result['emerging_threat'] else 'No'),\n"
"    ]\n"
"    summary_html = ''.join(\n"
"        f'<tr><td style=\"padding:6px 12px;border-bottom:1px solid #eee;color:#555;\">{k}</td>'\n"
"        f'<td style=\"padding:6px 12px;border-bottom:1px solid #eee;font-weight:600;\">{v}</td></tr>'\n"
"        for k, v in rows\n"
"    )\n"
"\n"
"    # -- dominant topic card --\n"
"    dom = result.get('dominant_topic')\n"
"    if dom:\n"
"        words_html = ' '.join(\n"
"            f'<span style=\"display:inline-block;background:#1f3a93;color:#fff;'\n"
"            f'padding:3px 8px;border-radius:12px;margin:2px;font-size:11px;\">{w}</span>'\n"
"            for w in dom['top_words']\n"
"        )\n"
"        dom_html = f'''\n"
"        <div style=\"margin-top:16px;\">\n"
"          <div style=\"font-size:13px;color:#555;margin-bottom:8px;\">\n"
"            Dominant topic <b>#{dom['topic_id']}</b> — appears in <b>{dom['count']}</b> relevant sentences\n"
"          </div>\n"
"          <div>{words_html}</div>\n"
"        </div>\n"
"        '''\n"
"    else:\n"
"        dom_html = ''\n"
"\n"
"    # -- topic distribution --\n"
"    tc = result.get('topic_counts', {})\n"
"    if tc:\n"
"        tc_items = sorted(tc.items(), key=lambda x: -int(x[1]))\n"
"        tc_html = ' '.join(\n"
"            f'<span style=\"display:inline-block;background:#ecf0f1;color:#2c3e50;'\n"
"            f'padding:3px 10px;border-radius:12px;margin:2px;font-size:11px;\">'\n"
"            f'T{k} <b>{v}</b></span>'\n"
"            for k, v in tc_items\n"
"        )\n"
"        tc_section = f'''\n"
"        <div style=\"margin-top:16px;\">\n"
"          <div style=\"font-size:13px;color:#555;margin-bottom:8px;\">Topic distribution</div>\n"
"          <div>{tc_html}</div>\n"
"        </div>\n"
"        '''\n"
"    else:\n"
"        tc_section = ''\n"
"\n"
"    # -- reasons --\n"
"    if result.get('reasons'):\n"
"        reasons_html = ''.join(\n"
"            f'<li style=\"margin:4px 0;color:#333;\">{r}</li>'\n"
"            for r in result['reasons']\n"
"        )\n"
"        reasons_section = f'''\n"
"        <div style=\"margin-top:16px;\">\n"
"          <div style=\"font-size:13px;color:#555;margin-bottom:6px;\">Why this verdict</div>\n"
"          <ul style=\"margin:0;padding-left:20px;\">{reasons_html}</ul>\n"
"        </div>\n"
"        '''\n"
"    else:\n"
"        reasons_section = ''\n"
"\n"
"    html = f'''\n"
"    <div style=\"font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:720px;\n"
"                border:1px solid #ddd;border-radius:10px;padding:20px;\n"
"                box-shadow:0 2px 6px rgba(0,0,0,0.06);background:#fafbfc;\">\n"
"\n"
"      <div style=\"font-size:18px;font-weight:700;color:#1f3a93;margin-bottom:4px;\">\n"
"        CTI Threat Report Analysis\n"
"      </div>\n"
"      <div style=\"font-size:12px;color:#888;margin-bottom:16px;\">\n"
"        Report: <b>{report_name}</b>\n"
"      </div>\n"
"\n"
"      <div style=\"background:{s['bg']};border-left:6px solid {s['border']};\n"
"                  padding:14px 18px;border-radius:6px;font-size:16px;\n"
"                  font-weight:700;color:{s['border']};margin-bottom:16px;\">\n"
"        {s['icon']} &nbsp; {s['label']}\n"
"      </div>\n"
"\n"
"      <div style=\"background:#fff;border-radius:6px;padding:8px 4px;\">\n"
"        <table style=\"border-collapse:collapse;width:100%;font-size:13px;\">\n"
"          {summary_html}\n"
"        </table>\n"
"      </div>\n"
"\n"
"      {dom_html}\n"
"      {tc_section}\n"
"      {reasons_section}\n"
"\n"
"      <div style=\"margin-top:24px;padding-top:12px;border-top:1px solid #eee;\n"
"                  font-size:10px;color:#aaa;text-align:center;\">\n"
"        Generated by CTI Report Analyzer — Hybrid TF-IDF + SBERT classifier with BERTopic\n"
"      </div>\n"
"    </div>\n"
"    '''\n"
"    display(HTML(html))\n"
"\n"
"render_verdict(result)"
))

# ── Cell 8 — analyse a real PDF
cells.append(md(
"## 8. Analyse a real PDF report (optional)\n"
"\n"
"Point `PDF_PATH` at any PDF from the SMAH-CTI source collection (APTNotes, "
"CrowdStrike, Mandiant, MITRE, Microsoft). If the file is not present, this cell "
"prints a skip message — everything else still works."
))
cells.append(code(
"PDF_PATH = '../dataset/sample_report.pdf'\n"
"if os.path.exists(PDF_PATH):\n"
"    result_pdf = predict_report(PDF_PATH)\n"
"    print(json.dumps(result_pdf, indent=2))\n"
"    json.dump(result_pdf, open('../artifacts/report_analysis.json','w'), indent=2)\n"
"    print('saved -> artifacts/report_analysis.json')\n"
"    render_verdict(result_pdf)\n"
"else:\n"
"    print(f'No PDF at {PDF_PATH} — skipping. Put a CTI PDF there to test.')"
))

# ── Cell 9 — usage summary
cells.append(md(
"## 9. How to analyse a new report\n"
"\n"
"After this notebook runs once, the trained artifacts are saved. Analyse any\n"
"report from the command line:\n"
"\n"
"```bash\n"
"python scripts/predict_report.py path/to/new_report.pdf\n"
"```\n"
"\n"
"Or, inside Jupyter, use the `predict_report()` and `render_verdict()` functions\n"
"defined in Cells 5 and 7."
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"}
    },
    "nbformat": 4, "nbformat_minor": 5
}

out = os.path.join("notebooks", "07_report_analyzer.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Wrote:", os.path.abspath(out))
print("Cells:", len(cells))