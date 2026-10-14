# CTI Threat Analysis

End-to-end prototype for Cyber-Threat-Intelligence report triage:

- **Text classification** -- TF-IDF + Logistic Regression / Naive Bayes / Linear SVM
- **Temporal topic modeling** -- NMF over monthly report text
- **Emerging-threat detection** -- heuristic threshold on topic-share drift
- **Document ingestion** -- PDF / DOCX / TXT / HTML / MD extraction
- **Optional web UI** -- FastAPI backend + static HTML/JS frontend (demo only)

## Project structure

```
ctc_ectd/
|-- artifacts/              # trained models, exported CSVs, final plots
|-- backend/                # FastAPI backend (demo only)
|-- dataset/
|   `-- synthetic_cti_reports.csv
|-- notebooks/
|   `-- cti_threat_analysis.ipynb
|-- scripts/
|   |-- document_extractor.py
|   |-- analyze_uploads.py
|   |-- generate_sample_pdfs.py
|   |-- phase_1.py
|   `-- archive/            # scaffolding, one-off fixers
|-- ui/                     # static frontend (demo only)
|-- uploads/                # drop CTI documents here
|-- README.md
`-- requirements.txt
```

## Install

```powershell
pip install -r requirements.txt
pip install pypdf python-docx beautifulsoup4 lxml reportlab
```

## Run the pipeline (CLI)

```powershell
python scripts/analyze_uploads.py
```

Reads every supported file in `uploads/`, extracts text, classifies each
document, runs temporal topic modeling on dated documents, and writes:

- `artifacts/uploaded_classifications.csv`
- `artifacts/emerging_threat_alerts.csv`

## Regenerate demo PDFs

```powershell
python scripts/generate_sample_pdfs.py
```

## Notebook

```powershell
jupyter notebook notebooks/cti_threat_analysis.ipynb
```

Run all cells top-to-bottom on a fresh kernel.

## Optional web UI (demo)

```powershell
# Terminal 1
python -m uvicorn backend.api:app --reload --port 8000

# Terminal 2
python -m http.server 5500 --directory ui

# Browser
# http://127.0.0.1:5500
```

The UI is a demonstration layer only and is not part of the thesis
evaluation.

## Reproducibility

- Fixed random seed: `42`
- Dataset: `dataset/synthetic_cti_reports.csv` (10,000 rows, 17 columns)
- All plots and CSVs referenced in the thesis live in `artifacts/final/`

## Limitations

- The dataset is **synthetic**. Category words may leak into report text.
- High classification scores reflect dataset construction, not real-world capability.
- Topic models require human interpretation.
- Emerging-threat alerts are heuristic and need analyst review.
- Real-world generalization requires evaluation on independently collected CTI data.

## License

Academic use only -- MSc thesis project.
