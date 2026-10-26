"""
Adds two cells to notebooks/07_report_analyzer.ipynb:
  - Cell 6b: write_pdf_report() function definition
  - Cell 6c: call it on the sample result

Run from project root:  python scripts/add_pdf_cell.py
"""
import json, os
from pathlib import Path

NB_PATH = Path("notebooks/07_report_analyzer.ipynb")
if not NB_PATH.exists():
    raise SystemExit(f"Notebook not found: {NB_PATH}\nRun make_inference.py first.")

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]

# ── Cell 6b — the PDF generator (fully corrected) ─────────────────────────
cell_6b_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 6b — PDF report generator\n",
        "\n",
        "Renders a `predict_report()` result to a one-page A4 PDF with a coloured\n",
        "verdict banner, summary table, dominant topic, and reason list."
    ]
}

cell_6b_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from reportlab.lib.pagesizes import A4\n",
        "from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle\n",
        "from reportlab.lib.units import cm\n",
        "from reportlab.lib.colors import HexColor\n",
        "from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,\n",
        "                                 TableStyle)\n",
        "from reportlab.lib.enums import TA_LEFT, TA_CENTER\n",
        "from datetime import datetime\n",
        "from pathlib import Path\n",
        "\n",
        "def write_pdf_report(result, out_path='../artifacts/report_analysis.pdf'):\n",
        "    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)\n",
        "\n",
        "    doc = SimpleDocTemplate(\n",
        "        out_path, pagesize=A4,\n",
        "        leftMargin=2*cm, rightMargin=2*cm,\n",
        "        topMargin=2*cm, bottomMargin=2*cm,\n",
        "        title='CTI Threat Report Analysis',\n",
        "        author='CTI Analyzer',\n",
        "    )\n",
        "\n",
        "    styles = getSampleStyleSheet()\n",
        "    h1   = ParagraphStyle('H1', parent=styles['Heading1'],\n",
        "                          textColor=HexColor('#1f3a93'), fontSize=18, spaceAfter=12)\n",
        "    h2   = ParagraphStyle('H2', parent=styles['Heading2'],\n",
        "                          textColor=HexColor('#1f3a93'), fontSize=13, spaceAfter=8)\n",
        "    body = ParagraphStyle('Body', parent=styles['BodyText'],\n",
        "                          fontSize=10, leading=14, alignment=TA_LEFT)\n",
        "    small = ParagraphStyle('Small', parent=styles['BodyText'],\n",
        "                           fontSize=8, textColor=HexColor('#555555'),\n",
        "                           alignment=TA_CENTER)\n",
        "\n",
        "    flow = []\n",
        "    flow.append(Paragraph('CTI Threat Report Analysis', h1))\n",
        "    flow.append(Paragraph(f'Generated: {datetime.now():%Y-%m-%d %H:%M}', body))\n",
        "    flow.append(Paragraph(f\"Report file: <b>{Path(result['report']).name}</b>\", body))\n",
        "    flow.append(Spacer(1, 0.5*cm))\n",
        "\n",
        "    # --- verdict banner ---\n",
        "    verdict = result.get('verdict', 'unknown')\n",
        "    banner_color = {\n",
        "        'emerging':     '#c0392b',\n",
        "        'known':        '#27ae60',\n",
        "        'not_relevant': '#7f8c8d',\n",
        "    }.get(verdict, '#333333')\n",
        "    verdict_label = {\n",
        "        'emerging':     'EMERGING THREAT SIGNAL',\n",
        "        'known':        'KNOWN THREAT PATTERN',\n",
        "        'not_relevant': 'NOT RELEVANT',\n",
        "    }.get(verdict, verdict.upper())\n",
        "\n",
        "    verdict_style = ParagraphStyle(\n",
        "        'V', parent=body, fontSize=14, alignment=TA_CENTER,\n",
        "        textColor=HexColor('#ffffff')\n",
        "    )\n",
        "    verdict_tbl = Table(\n",
        "        [[Paragraph(f'<b>{verdict_label}</b>', verdict_style)]],\n",
        "        colWidths=[16*cm]\n",
        "    )\n",
        "    verdict_tbl.setStyle(TableStyle([\n",
        "        ('BACKGROUND', (0,0), (-1,-1), HexColor(banner_color)),\n",
        "        ('TOPPADDING', (0,0), (-1,-1), 10),\n",
        "        ('BOTTOMPADDING', (0,0), (-1,-1), 10),\n",
        "    ]))\n",
        "    flow.append(verdict_tbl)\n",
        "    flow.append(Spacer(1, 0.5*cm))\n",
        "\n",
        "    # --- summary table ---\n",
        "    flow.append(Paragraph('Summary', h2))\n",
        "    summary_rows = [\n",
        "        ['Total sentences',    str(result.get('total_sentences', '—'))],\n",
        "        ['Relevant sentences', str(result.get('relevant_sentences', '—'))],\n",
        "        ['Relevant fraction',  f\"{result.get('relevant_fraction', 0)*100:.1f}%\"],\n",
        "        ['Verdict',            verdict],\n",
        "        ['Emerging threat',    'Yes' if result.get('emerging_threat') else 'No'],\n",
        "    ]\n",
        "    summary_tbl = Table(summary_rows, colWidths=[6*cm, 10*cm])\n",
        "    summary_tbl.setStyle(TableStyle([\n",
        "        ('BACKGROUND', (0,0), (0,-1), HexColor('#ecf0f1')),\n",
        "        ('GRID', (0,0), (-1,-1), 0.25, HexColor('#bdc3c7')),\n",
        "        ('FONTSIZE', (0,0), (-1,-1), 10),\n",
        "        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),\n",
        "        ('TOPPADDING', (0,0), (-1,-1), 6),\n",
        "        ('BOTTOMPADDING', (0,0), (-1,-1), 6),\n",
        "    ]))\n",
        "    flow.append(summary_tbl)\n",
        "    flow.append(Spacer(1, 0.5*cm))\n",
        "\n",
        "    # --- dominant topic ---\n",
        "    dom = result.get('dominant_topic')\n",
        "    if dom:\n",
        "        flow.append(Paragraph('Dominant Topic', h2))\n",
        "        flow.append(Paragraph(\n",
        "            f\"Topic ID <b>{dom['topic_id']}</b> — appears in <b>{dom['count']}</b> relevant sentences.\",\n",
        "            body))\n",
        "        flow.append(Paragraph('Top words: ' + ', '.join(dom['top_words']), body))\n",
        "        flow.append(Spacer(1, 0.5*cm))\n",
        "\n",
        "    # --- topic distribution ---\n",
        "    if result.get('topic_counts'):\n",
        "        flow.append(Paragraph('Topic Distribution', h2))\n",
        "        tc_rows = [['Topic', 'Count']] + [\n",
        "            [str(k), str(v)] for k, v in sorted(\n",
        "                result['topic_counts'].items(), key=lambda x: -int(x[1]))\n",
        "        ]\n",
        "        tc_tbl = Table(tc_rows, colWidths=[6*cm, 10*cm])\n",
        "        tc_tbl.setStyle(TableStyle([\n",
        "            ('BACKGROUND', (0,0), (-1,0), HexColor('#1f3a93')),\n",
        "            ('TEXTCOLOR',  (0,0), (-1,0), HexColor('#ffffff')),\n",
        "            ('GRID', (0,0), (-1,-1), 0.25, HexColor('#bdc3c7')),\n",
        "            ('FONTSIZE', (0,0), (-1,-1), 9),\n",
        "            ('TOPPADDING', (0,0), (-1,-1), 4),\n",
        "            ('BOTTOMPADDING', (0,0), (-1,-1), 4),\n",
        "        ]))\n",
        "        flow.append(tc_tbl)\n",
        "        flow.append(Spacer(1, 0.5*cm))\n",
        "\n",
        "    # --- reasons ---\n",
        "    if result.get('reasons'):\n",
        "        flow.append(Paragraph('Why This Verdict', h2))\n",
        "        for r in result['reasons']:\n",
        "            flow.append(Paragraph(f'• {r}', body))\n",
        "        flow.append(Spacer(1, 0.3*cm))\n",
        "\n",
        "    flow.append(Spacer(1, 0.5*cm))\n",
        "    flow.append(Paragraph(\n",
        "        'Generated by the CTI Threat Report Analyzer — hybrid TF-IDF + SBERT '\n",
        "        'classifier with BERTopic emerging-threat detection.',\n",
        "        small))\n",
        "\n",
        "    doc.build(flow)\n",
        "    return out_path\n"
    ]
}

# ── Cell 6c — call it on the sample result ────────────────────────────────
cell_6c_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### 6c — Render the sample result as a PDF"]
}

cell_6c_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "pdf_path = write_pdf_report(result, out_path='../artifacts/report_analysis.pdf')\n",
        "print('saved ->', os.path.abspath(pdf_path))"
    ]
}

# ── Insert after the inline sample test (identify it by cell content) ─────
insert_at = None
for i, cell in enumerate(cells):
    src = "".join(cell.get("source", []))
    if "sample_text" in src and "predict_report" in src:
        insert_at = i + 1
        break

if insert_at is None:
    raise SystemExit("Could not find the inline-sample cell. Aborting.")

new_cells = cells[:insert_at] + [
    cell_6b_md, cell_6b_code, cell_6c_md, cell_6c_code
] + cells[insert_at:]

nb["cells"] = new_cells
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

print(f"Inserted 4 cells (6b md/code + 6c md/code) at index {insert_at}.")
print(f"Total cells now: {len(new_cells)}")