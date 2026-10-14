# Chapter 6: Conclusion and Future Work

## 6.1 Conclusion

Summarize the implemented classification pipeline, topic modeling,
weekly trend analysis, and candidate-alert workflow. Mention only
modules that successfully ran.

## 6.2 Achievement of Objectives

Map each objective from Chapter 1 to evidence in Chapter 5.

  -----------------------------------------------------------------------
  Objective               Evidence                Status
  ----------------------- ----------------------- -----------------------
  Dataset preparation     Schema and cleaning     Fill after final run
                          output                  

  Threat classification   Metrics and confusion   Fill after final run
                          matrix                  

  LDA/NMF topic discovery Topic word tables       Fill after final run

  Weekly temporal         Trend plots             Fill after final run
  analysis                                        

  Candidate alerts        Alert CSV and reviewed  Fill after final run
                          examples                
  -----------------------------------------------------------------------

## 6.3 Limitations

Discuss synthetic data, label leakage risk, limited ground truth,
heuristic thresholds, and lack of live operational deployment if
applicable.

## 6.4 Future Work

-   Evaluate on independently sourced real CTI reports.
-   Use chronological holdout and external validation.
-   Evaluate alert precision/recall against incident ground truth.
-   Calibrate thresholds and monitor report-volume effects.
-   Explore robust semantic representations and analyst feedback.
-   Integrate a streaming or web interface only as future work unless
    implemented.

## 6.5 Final Remarks

Conclude with a balanced statement about prototype feasibility and the
need for external validation.
