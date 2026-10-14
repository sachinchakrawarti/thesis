# 08. Reproducibility, Ethics, and Limitations

## Reproducibility checklist

-   [ ] Record Python and package versions.
-   [ ] Set and record random seeds.
-   [ ] Record dataset version and checksum if required.
-   [ ] Record the train/test split and class distribution.
-   [ ] Save the final notebook after a clean run.
-   [ ] Save the selected classifier pipeline.
-   [ ] Save plots and tables from the final run.
-   [ ] Ensure the thesis numbers match saved outputs.

## Security and privacy

The current dataset is synthetic. If future work uses real CTI reports,
review licensing, sensitive indicators, personal data, and
organizational sharing restrictions before publication.

## Limitations to discuss

-   Synthetic dataset may not represent operational CTI.
-   Text may contain explicit category cues that inflate metrics.
-   Temporal topic trends can be affected by synthetic generation rules
    and reporting volume.
-   Topic models require human interpretation.
-   The alert threshold is heuristic unless calibrated against validated
    incident data.
-   A notebook prototype is not equivalent to a production real-time
    monitoring service.
-   No live ingestion, streaming service, or operational security
    integration should be claimed unless actually built and tested.

## Future work

Test on independently sourced CTI data; use time-based evaluation;
conduct expert review of topic labels; calibrate alerts; assess
robustness against duplicate reports and changing vocabulary; and
implement live ingestion only as a later extension.
