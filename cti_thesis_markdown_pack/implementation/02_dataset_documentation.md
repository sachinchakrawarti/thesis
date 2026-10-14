# 02. Dataset Documentation

## Dataset name

`synthetic_cti_reports.csv`

## Role in the project

The dataset is used to prototype text classification and topic-trend
analysis on cyber-threat-intelligence-style reports.

## Important fields

Confirm all names in the CSV before running the pipeline. The uploaded
dataset has been inspected as containing 10,000 rows and 17 columns,
including the following relevant fields:

  -----------------------------------------------------------------------
  Field                               Intended use
  ----------------------------------- -----------------------------------
  `title`                             Short report title; text input

  `text`                              Report body; primary text input

  `category`                          Supervised target label

  `date`                              Temporal grouping

  `severity`                          Descriptive analysis, not
                                      necessarily a target

  `threat_actor`, `malware`           Contextual attributes

  `attack_vector`, `target_sector`    Contextual attributes

  `mitre_technique`,                  Technique-related metadata
  `mitre_technique_name`              

  `indicators`, `cve_id`              Indicator/vulnerability context
  -----------------------------------------------------------------------

## Required dataset checks

``` python
print(df.shape)
print(df.columns.tolist())
print(df["category"].value_counts(dropna=False))
print(df["date"].head())
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
```

Parse dates:

``` python
df["date"] = pd.to_datetime(df["date"], errors="coerce")
print("Invalid dates:", int(df["date"].isna().sum()))
print("Date range:", df["date"].min(), df["date"].max())
print("Unique dates:", df["date"].nunique())
print("Unique weeks:", df["date"].dt.to_period("W").nunique())
```

## Dataset limitations

-   The data are synthetic, not a representative sample of live CTI
    feeds.
-   Generated category words may leak the label into the input text.
-   High classification scores can therefore reflect dataset
    construction shortcuts.
-   A date field is useful only if its values were assigned with a
    defensible temporal process.
-   Do not claim external validity without testing on real,
    independently sourced reports.

## Thesis wording

Use wording such as: "A synthetic CTI-style dataset was used to validate
the proposed processing pipeline. The experiment demonstrates
implementation feasibility; real-world generalization remains to be
evaluated on independently collected CTI data."
