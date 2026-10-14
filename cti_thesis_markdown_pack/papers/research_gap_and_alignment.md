# Research Gap and Alignment with the Base Paper

## Research focus

This project prototypes two connected tasks: 1. Classify individual
CTI-style reports into threat categories. 2. Examine topic prevalence
over time and flag candidate changes for analyst review.

## Proposed practical gap

A modest, defensible project contribution is an integrated and
documented workflow that combines text classification with temporal
topic analysis on a prepared CTI-style dataset. The gap must be
confirmed through the reviewed literature; do not claim no prior work
exists without systematic evidence.

## Alignment table

  -----------------------------------------------------------------------
  Component               Base paper approach     Present prototype
  ----------------------- ----------------------- -----------------------
  Text classification     Multiple ML/DL methods; TF-IDF with baseline
                          verify details in paper classifiers

  Topic discovery         LDA and NMF             LDA and NMF, if
                                                  executed successfully

  Time aggregation        Weekly trends           Weekly topic
                                                  distributions

  Emerging detection      Interpretation of topic Configurable heuristic
                          shifts                  candidate alerts

  Validation              Paper-specific          Synthetic-data
                          evaluation              prototype; real-world
                                                  validation remains
                                                  future work
  -----------------------------------------------------------------------

## Contribution statements (use only if implemented)

-   A reproducible pipeline for CTI-style text preparation and baseline
    classification.
-   A comparison of LDA and NMF topic trends over weekly intervals.
-   A simple candidate-alert mechanism for prioritizing topics for
    review.
-   A documented discussion of synthetic-data limitations and validation
    needs.

## Do not claim

-   Novel algorithms unless a genuinely new method is developed and
    evaluated.
-   Real-time production deployment if the work runs only in a notebook.
-   Real-world attack discovery based solely on synthetic topic spikes.
-   Better performance than the base paper based on incomparable
    datasets.
