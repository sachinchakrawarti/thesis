# Cyber Threat Classification and Emerging Threat Detection Using Sentence-BERT and BERTopic

## 1. Proposed Approach

The proposed system consists of two primary modules:

1. **Cyber Threat Classification** using Sentence-BERT embeddings and a Support Vector Machine (SVM).
2. **Emerging Threat Detection** using BERTopic for topic discovery and temporal analysis of cyber threat intelligence (CTI) reports.

The system will also provide a web-based interface where users can upload a CTI PDF, analyze its content, view predicted classifications and discovered topics, and download an analysis report.

### 1.1 Cyber Threat Classification

**Objective:** Classify CTI sentences or extracted report passages into predefined categories.

**Approach:**

1. **Input:** Accept CTI sentences, text documents, or PDF reports.
2. **Text extraction:** Extract text from uploaded PDF reports.
3. **Preprocessing:** Remove irrelevant formatting, normalize whitespace, and split the text into meaningful sentences or passages.
4. **Semantic embedding:** Use Sentence-BERT (`all-MiniLM-L6-v2`) to convert each sentence into a dense semantic vector.
5. **Classification:** Train an SVM classifier on the generated embeddings using labeled training data.
6. **Prediction:** Predict the class of each sentence or passage.
7. **Result presentation:** Display predicted labels, confidence-related scores where appropriately calibrated, and supporting text.

**Proposed classification models:**

- **Baseline:** TF-IDF + SVM.
- **Proposed model 1:** Sentence-BERT + SVM.
- **Proposed model 2:** TF-IDF features combined with Sentence-BERT embeddings + SVM.

The hybrid model combines lexical features with semantic representations. The features should be scaled or normalized and combined using a defined feature-fusion strategy.

**Evaluation metrics:**

- Accuracy
- Precision
- Recall
- Macro-F1 score
- Confusion matrix
- Training and inference time

The proposed models will be evaluated using the same data splits and evaluation metrics as the baseline wherever possible. Results will be measured experimentally rather than assumed in advance.

**Important:** The classifier's output categories depend on the dataset's labels. If the training dataset uses `relevant` and `irrelevant`, the model will predict those two classes. It cannot automatically predict malware families, attack types, or MITRE ATT&CK techniques unless it has suitable labels or a separate technique-mapping component.

### 1.2 Emerging Threat Detection Using BERTopic

**Objective:** Discover recurring and potentially emerging themes in CTI reports.

**Approach:**

1. Collect CTI reports or passages with publication dates or observation timestamps.
2. Extract and preprocess the report text.
3. Generate semantic embeddings using Sentence-BERT.
4. Apply BERTopic to group semantically similar passages into topics.
5. Extract representative keywords and passages for each topic.
6. Calculate topic prevalence across time periods, such as days, weeks, or months.
7. Identify topics whose frequency or proportion increases over time.
8. Present topic distributions, trends, keywords, and representative evidence in the application.

**Emerging-threat indicators may include:**

- A newly observed topic in the available corpus.
- A sustained increase in a topic's prevalence.
- Increasing numbers of reports or independent sources mentioning the topic.
- Supporting evidence from analysts or trusted threat intelligence sources.

A newly discovered topic is not automatically a confirmed emerging cyber threat. The system should label its findings as *potentially emerging* until the evidence is validated.

**Important limitation:** A single uploaded PDF can be classified and analyzed for topics, but it cannot establish a reliable temporal trend on its own. Emerging-threat analysis requires a historical collection of dated CTI reports. A single-PDF analysis can be compared with an existing historical corpus if one is available.

### 1.3 Web Application Workflow

1. The user opens the web application.
2. The user uploads a CTI PDF or selects a text input.
3. The backend validates the file and extracts its text.
4. The text is split into sentences or passages.
5. Sentence-BERT generates embeddings.
6. The trained SVM predicts class labels.
7. BERTopic identifies topics in the text, when sufficient text is available.
8. If historical dated data is available, the system compares topic prevalence over time.
9. The frontend displays classification results, topic summaries, and trend charts.
10. The user downloads the analysis results as a report or CSV file.

## 2. Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Frontend | React.js | User interface and results display |
| UI styling | Tailwind CSS | Responsive interface design |
| Backend API | FastAPI | Expose classification and topic-analysis endpoints |
| Programming language | Python | ML pipeline and backend implementation |
| PDF processing | PyMuPDF | Extract text from PDF files |
| NLP embeddings | Sentence-Transformers | Generate semantic embeddings |
| Classification | scikit-learn SVM | Predict predefined threat-related labels |
| Baseline features | TF-IDF | Create lexical feature vectors |
| Topic modeling | BERTopic | Discover semantic topics |
| Dimensionality reduction | UMAP | Reduce embeddings for topic clustering |
| Clustering | HDBSCAN | Group similar text passages |
| Visualization | Plotly | Display classification summaries and topic trends |
| Model persistence | Joblib | Save and load trained classifiers |
| Data storage | CSV and JSON initially | Store results and metadata |
| Database (optional) | SQLite or PostgreSQL | Store reports, analysis history, and metadata |
| Testing | pytest | Test backend endpoints and ML components |
| API documentation | FastAPI Swagger UI | Inspect and test API endpoints |
| Deployment | Uvicorn; Docker optional | Run and package the application |

## 3. Project File Structure

```text
cyber-threat-analysis/
│
├── frontend/
│   ├── public/
│   │   └── favicon.ico
│   ├── src/
│   │   ├── components/
│   │   │   ├── FileUpload.jsx
│   │   │   ├── ClassificationResults.jsx
│   │   │   ├── TopicSummary.jsx
│   │   │   ├── TopicTrendChart.jsx
│   │   │   └── LoadingIndicator.jsx
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx
│   │   │   └── AnalysisPage.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── upload.py
│   │   │   ├── classification.py
│   │   │   └── topics.py
│   │   ├── services/
│   │   │   ├── pdf_extractor.py
│   │   │   ├── text_preprocessor.py
│   │   │   ├── classifier_service.py
│   │   │   ├── topic_service.py
│   │   │   └── report_service.py
│   │   ├── schemas/
│   │   │   └── responses.py
│   │   └── config.py
│   ├── requirements.txt
│   └── .env.example
│
├── ml/
│   ├── training/
│   │   ├── train_tfidf_svm.py
│   │   ├── train_sbert_svm.py
│   │   └── train_hybrid_svm.py
│   ├── topic_modeling/
│   │   ├── train_bertopic.py
│   │   └── analyze_topic_trends.py
│   ├── evaluation/
│   │   ├── evaluate_classifier.py
│   │   └── compare_models.py
│   └── utils/
│       └── metrics.py
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── historical_cti/
│       └── .gitkeep
│
├── models/
│   ├── classifier.joblib
│   ├── tfidf_vectorizer.joblib
│   └── bertopic_model/
│
├── reports/
│   ├── evaluation/
│   └── generated/
│
├── tests/
│   ├── test_classifier.py
│   ├── test_pdf_extractor.py
│   └── test_api.py
│
├── sample_documents/
│   └── README.md
│
├── .gitignore
├── README.md
└── requirements-dev.txt
```

The model files and generated reports shown above are expected runtime artifacts. They should be created during training or analysis rather than assumed to exist in a new project.

## 4. Main Backend API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Check whether the backend is running |
| `/api/analyze/pdf` | POST | Upload a PDF and run classification and topic analysis |
| `/api/classify/text` | POST | Classify a supplied text passage |
| `/api/topics/discover` | POST | Discover topics from a collection of passages |
| `/api/topics/trends` | POST | Analyze topic prevalence using dated CTI data |
| `/api/reports/{report_id}` | GET | Retrieve a generated analysis report |

The API should validate file type and size, handle malformed PDFs, and avoid executing or opening embedded PDF content. Uploaded reports may contain confidential intelligence, so temporary files should be managed securely.

## 5. Proposed Experimental Comparison

The classification experiment will compare the following models:

| Experiment | Feature representation | Classifier | Status |
|---|---|---|---|
| Baseline | TF-IDF | SVM | To be evaluated |
| Proposed 1 | Sentence-BERT embeddings | SVM | To be evaluated |
| Proposed 2 | TF-IDF + Sentence-BERT embeddings | SVM | To be evaluated |

The original paper's published accuracy results should be reproduced in a separate reference table. The proposed models should be reported in a separate experimental table unless the original study used directly comparable feature representations and evaluation conditions.

For a fair comparison, use the same target labels, a leakage-resistant train/test split, and consistent evaluation metrics. When multiple passages originate from the same document, consider grouping by document during splitting so that closely related passages do not appear in both training and test sets.

## 6. Expected Application Output

The application should display:

- Uploaded filename and extracted-text preview.
- Predicted label for each sentence or passage.
- Overall classification summary.
- Relevant supporting passages.
- Discovered topics, keywords, and representative passages.
- Topic-prevalence charts when dated historical data is available.
- Analysis duration and processing status.
- Downloadable CSV or JSON results and a summary report.

The purpose of the application is to demonstrate the complete pipeline from CTI document ingestion to semantic classification, topic discovery, and evidence-based analysis of potentially emerging threats. Model performance and threat-detection effectiveness must be established through experiments and validation.
