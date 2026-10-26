# Cyber Threat Classification and Emerging Threat Detection Using Sentence-BERT and BERTopic

## 1. Proposed Approach

The proposed system consists of two primary modules:

1. **Cyber Threat Classification** u Sentence-BERT 
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


Project File Structure

cyber-threat-detection/
│
├── notebooks/
│   └── cti.ipynb
│
├── server/
│   ├── app.py                  # Lightweight Flask server
│   ├── classifier.py           # Load model and predict labels
│   ├── topic_model.py          # BERTopic analysis
│   ├── pdf_reader.py           # Extract text from PDFs
│   ├── preprocessing.py        # Clean and split text
│   ├── evaluation.py           # Calculate evaluation metrics
│   └── requirements.txt
│
├── models/
│   ├── classification/
│   │   ├── tfidf_svm.joblib
│   │   ├── sbert_svm.joblib
│   │   ├── hybrid_svm.joblib
│   │   ├── tfidf_vectorizer.joblib
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
│   └── js/
│       ├── app.js
│       ├── classification.js
│       └── emerging-threats.js
│
├── dataset/
│   └── SMAH-CTI.csv
│
├── sample_documents/
│   └── sample_cti.pdf
│
├── results/
│   ├── base_paper_results.csv
│   ├── proposed_model_results.csv
│   ├── confusion_matrix.png
│   └── topic_trends.csv
│
├── tests/
│   ├── test_classifier.py
│   ├── test_pdf_reader.py
│   └── test_server.py
│
├── README.md
├── .gitignore
└── requirements-dev.txt


dataset exam  

uid,source,doc_id,page_number,sentence_id,context_before,sentence_text,relevant,tactic,techniques,procedure,entities
aptnotes-01_aptnotes_report-p2-s1-428914,aptnotes,01_aptnotes_report,2,1,,"EXECUTIVE SUMMARY
The Secureworks® Counter Threat Unit™ (CTU) research team analyzes security threats to help organizations protect their systems.",no,,,,
aptnotes-01_aptnotes_report-p2-s2-d05d6e,aptnotes,01_aptnotes_report,2,2,"EXECUTIVE SUMMARY
The Secureworks® Counter Threat Unit™ (CTU) research team analyzes security threats to help organizations protect their systems.","Based on observations in March and April, CTU™ researchers identified the following noteworthy issues and changes in the global threat landscape:
• Cascading supply chain attacks bring extra risk •",no,,,,
aptnotes-01_aptnotes_report-p2-s3-a8749b,aptnotes,01_aptnotes_report,2,3,"Based on observations in March and April, CTU™ researchers identified the following noteworthy issues and changes in the global threat landscape:
• Cascading supply chain attacks bring extra risk •",Infostealers are a gateway to ransomware attacks •,no,,,,
aptnotes-01_aptnotes_report-p2-s4-df3f5f,aptnotes,01_aptnotes_report,2,4,Infostealers are a gateway to ransomware attacks •,Avoid the cyber dangers that lurk outside your perimeter CASCADING SUPPLY CHAIN ATTACKS BRING EXTRA RISK,no,,,,
aptnotes-01_aptnotes_report-p2-s5-4e7c70,aptnotes,01_aptnotes_report,2,5,Avoid the cyber dangers that lurk outside your perimeter CASCADING SUPPLY CHAIN ATTACKS BRING EXTRA RISK,The origin of the 3CX compromise reveals added complexity for organizations assessing the risk of supply chain attacks.,no,,,,
aptnotes-01_aptnotes_report-p2-s6-35f6da,aptnotes,01_aptnotes_report,2,6,The origin of the 3CX compromise reveals added complexity for organizations assessing the risk of supply chain attacks.,The SolarWinds supply chain compromise by a Russian state-sponsored threat group in 2020 remains one of the most significant example of attacks that originate from compromised vendor products.,no,,,,
aptnotes-01_aptnotes_report-p2-s7-34615f,aptnotes,01_aptnotes_report,2,7,The SolarWinds supply chain compromise by a Russian state-sponsored threat group in 2020 remains one of the most significant example of attacks that originate from compromised vendor products.,"However, a supply chain attack discovered in March 2023 added a new dimension to the risk that organizations face.",no,,,,