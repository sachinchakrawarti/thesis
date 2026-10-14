# 01. Project Setup and Execution

## Objective

Create a reproducible notebook pipeline for cyber-threat report
classification and temporal topic analysis.

## Minimum software

-   Python 3.10 or newer (use the Python version available in your
    environment and record it)
-   Jupyter Notebook or JupyterLab
-   pandas, numpy, matplotlib, seaborn, scikit-learn, joblib

Install dependencies from a notebook cell:

``` python
%pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Expected workspace

``` text
ctc_ectd/
├── dataset/
│   └── synthetic_cti_reports.csv
├── notebooks/
│   └── cti_threat_analysis.ipynb
├── artifacts/
└── results/
```

## First verification cell

``` python
import pandas as pd
from pathlib import Path

csv_path = Path(r"D:\Sachin Chakrawarti\Learn\done\thesis\ctc_ectd\dataset\synthetic_cti_reports.csv")
df = pd.read_csv(csv_path)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
display(df.head())
print(df.isna().sum())
```

If the path differs, update `csv_path` to the actual CSV location. Do
not continue until this cell runs.

## Execution protocol

1.  Restart the kernel.
2.  Run every cell in order.
3.  Stop at the first error and save the full traceback.
4.  Fix the error, restart the kernel, and rerun from the beginning.
5.  Confirm output files exist before referring to them in the thesis.

## Required evidence

-   Screenshot/output of dataset shape and columns
-   Python/package versions
-   Notebook execution date
-   Saved model and result files, if successfully generated
