thesis/
├── dataset/
│   └── SMAH-CTI.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb       ← NEW: pure EDA, no modeling
│   ├── 02_preprocessing_and_split.ipynb ← NEW: cleaning + train/test
│   ├── 03_baseline_tfidf_svm.ipynb
│   ├── 04_sbert_classification.ipynb
│   ├── 05_bertopic_topic_discovery.ipynb
│   └── 06_results_comparison.ipynb
├── artifacts/
│   ├── figs/
│   ├── train.csv
│   ├── test.csv
│   └── ...
└── requirements.txt