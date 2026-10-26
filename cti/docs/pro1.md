# Cyber Threat Classification and Emerging Threat Detection Using Sentence-BERT and BERTopic

## 1. Project File Structure

```text
cyber-threat-detection/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_baseline_tfidf_svm.ipynb
│   ├── 04_sbert_classification.ipynb
│   ├── 05_hybrid_classification.ipynb
│   ├── 06_bertopic_topic_discovery.ipynb
│   ├── 07_model_evaluation_comparison.ipynb
│   └── 08_pdf_analysis_demo.ipynb
│
├── server/
│   ├── app.py
│   ├── classifier.py
│   ├── topic_model.py
│   ├── pdf_reader.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── requirements.txt
│
├── models/
│   ├── classification/
│   │   ├── tfidf_svm.joblib
│   │   ├── sbert_svm.joblib
│   │   ├── hybrid_svm.joblib
│   │   ├── tfidf_vectorizer.joblib
│   │   ├── label_encoder.joblib
│   │   └── model_config.json
│   │
│   └── topic_modeling/
│       ├── bertopic_model/
│       └── topic_config.json
│
├── ui/
│   ├── index.html
│   ├── classification.html
│   ├── emerging-threats.html
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── app.js
│   │   ├── classification.js
│   │   └── emerging-threats.js
│   └── assets/
│       └── logo.png
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── historical_cti/
│       └── .gitkeep
│
├── sample_documents/
│   └── README.md
│
├── results/
│   ├── classification/
│   │   ├── base_paper_results.csv
│   │   ├── proposed_model_results.csv
│   │   └── confusion_matrix.png
│   ├── topic_modeling/
│   │   ├── topic_keywords.csv
│   │   └── topic_trends.csv
│   └── reports/
│
├── tests/
│   ├── test_classifier.py
│   ├── test_pdf_reader.py
│   └── test_server.py
│
├── README.md
├── .gitignore
└── requirements-dev.txt
```

**Folder responsibilities**

| Folder/file | Responsibility |
|---|---|
| `notebooks/` | Data exploration, experiments, training, evaluation, and visualizations |
| `server/` | Small Python server that handles uploads and connects the UI to ML models |
| `models/` | Saved classification models, vectorizers, encoders, and BERTopic artifacts |
| `ui/` | HTML pages, CSS styling, JavaScript interactions, and dashboard assets |
| `data/` | Raw, processed, and historical CTI datasets |
| `sample_documents/` | Instructions for sample PDFs used in demonstrations |
| `results/` | Base-paper comparisons, measured experimental results, charts, and generated reports |
| `tests/` | Tests for model predictions, PDF extraction, and API endpoints |

*Note:* Model files such as `.joblib` and the trained BERTopic model directory are generated after training. They do not need to exist in the initial project.

## 2. Small Server and UI Architecture

The project can use **Flask** for a lightweight Python server and plain HTML, CSS, and JavaScript for the user interface.

```text
User
  │
  ▼
HTML + CSS + JavaScript UI
  │
  │ Upload PDF / Submit text
  ▼
Flask Python Server
  │
  ├── PDF Reader
  │     └── Extract CTI text
  │
  ├── Preprocessing
  │     └── Split text into sentences
  │
  ├── Classification Module
  │     ├── Sentence-BERT + SVM
  │     ├── TF-IDF + SVM baseline
  │     └── Hybrid model
  │
  ├── BERTopic Module
  │     └── Discover topics
  │
  └── Results
        ├── Predicted labels
        ├── Topic keywords
        ├── Trend analysis
        └── Downloadable report
```

### Main UI pages

1. **Dashboard:** Summary of recent analyses and model evaluation.
2. **Classification:** Upload a PDF or enter text and view predicted labels.
3. **Emerging Threats:** Display discovered topics, representative passages, and historical topic trends.
4. **Model Comparison:** Compare base-paper results with measured results from the proposed models.

The emerging-threat page should distinguish topics found in a single report from trends calculated using a historical, dated corpus.

## 3. Classification Accuracy — Base Paper

The following table preserves the supplied base-paper results.

### Table 1. Binary Classification Accuracy (%)

| Learning type | Classifier | BOW (binary) | BOW (term frequency) | TF-IDF (unigram) | TF-IDF (bigram) | Word2Vec | Doc2Vec |
|---|---|---:|---:|---:|---:|---:|---:|
| Machine learning | RF | 89.88 | 89.89 | 88.30 | 69.01 | 64.64 | 73.85 |
| Machine learning | SVM | 93.67 | 92.75 | 90.69 | 68.87 | 64.53 | 73.59 |
| Machine learning | NB | 79.96 | 77.96 | 75.36 | 64.65 | — | — |
| Machine learning | XGBoost | 91.40 | 90.30 | 88.30 | 70.45 | 58.40 | 63.19 |
| Deep learning | CNN | 87.00 | 90.02 | 86.10 | 72.90 | 91.22 | 86.44 |
| Deep learning | LSTM | 88.90 | 90.50 | 88.30 | 67.03 | 93.13 | 88.90 |

### Table 2. Multiclass Classification Accuracy (%)

| Learning type | Classifier | BOW (binary) | BOW (term frequency) | TF-IDF (unigram) | TF-IDF (bigram) | Word2Vec | Doc2Vec |
|---|---|---:|---:|---:|---:|---:|---:|
| Machine learning | RF | 89.25 | 94.14 | 91.86 | 70.03 | 53.54 | 63.19 |
| Machine learning | SVM | 96.35 | 95.35 | 83.06 | 75.24 | 58.86 | 77.46 |
| Machine learning | NB | 85.67 | 83.06 | 70.68 | 68.50 | — | — |
| Machine learning | XGBoost | 92.30 | 91.13 | 86.00 | 72.50 | 54.53 | 66.10 |
| Deep learning | CNN | 88.98 | 92.34 | 83.81 | 78.83 | 90.11 | 84.10 |
| Deep learning | LSTM | 90.30 | 93.90 | 83.06 | 64.50 | 94.01 | 87.72 |

*Source note:* These values are the results supplied for the base paper. Verify them against the original paper before including them as final cited results in the thesis.

## 4. Proposed Classification Models — Your Experiments

The proposed methods use different feature representations from the traditional methods in the base-paper tables. Their results should therefore be reported separately, rather than filling the traditional feature columns with unmeasured values.

### Table 3. Proposed Model Evaluation

| Model | Feature representation | Classifier | Accuracy (%) | Precision | Recall | Macro-F1 |
|---|---|---|---:|---:|---:|---:|
| Baseline | TF-IDF | SVM | To be measured | To be measured | To be measured | To be measured |
| Proposed 1 | Sentence-BERT embeddings | SVM | To be measured | To be measured | To be measured | To be measured |
| Proposed 2 | TF-IDF + Sentence-BERT embeddings | SVM | To be measured | To be measured | To be measured | To be measured |

These values must be populated using actual experimental results. Do not use the base-paper accuracy values as predictions of how the proposed models will perform.

### Table 4. Reference Results and Proposed Models

| Classification setting | Base-paper reference | Proposed approach |
|---|---|---|
| Binary classification | Best supplied result: SVM + BOW (binary), 93.67% | Sentence-BERT + SVM; hybrid model — to be measured |
| Multiclass classification | Best supplied result: SVM + BOW (binary), 96.35% | Sentence-BERT + SVM; hybrid model — to be measured |

The base-paper results are reference points, not guaranteed targets. A valid comparison requires compatible datasets, labels, splits, and evaluation procedures.

## 5. Emerging-Threat Detection Experiments

| Experiment | Method | Output |
|---|---|---|
| Topic discovery | BERTopic | Topics, keywords, representative passages |
| Topic interpretation | Human review of representative passages | Interpreted threat themes |
| Temporal analysis | Topic prevalence by date or time period | Topic trend charts |
| Emerging-topic screening | Changes in prevalence and supporting evidence | Candidate emerging topics for review |

BERTopic can discover topics in an uploaded PDF, but reliable temporal emergence analysis requires historical CTI documents with dates. A topic's increasing prevalence should be treated as a signal for investigation, not proof of a new threat.

## 6. Recommended Development Order

1. Prepare the dataset and explore its labels in Jupyter notebooks.
2. Train and evaluate the TF-IDF + SVM baseline.
3. Train Sentence-BERT + SVM.
4. Train and evaluate the hybrid model.
5. Save the trained artifacts into `models/`.
6. Implement PDF extraction and prediction functions in `server/`.
7. Build the HTML/CSS/JavaScript interface in `ui/`.
8. Connect the UI to the Flask server.
9. Add BERTopic and historical trend analysis.
10. Save measured evaluation results and generate the comparison tables.

## 7. Final Deliverable

The final project will contain an experimental notebook workflow, a lightweight Python server, saved ML models, a browser-based interface, PDF analysis, classification comparisons against the base paper, and BERTopic-based topic discovery. The results will clearly separate published reference metrics from experimentally measured results.
