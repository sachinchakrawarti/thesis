# 07. Emerging-Pattern Detection

## Goal

Flag unusual changes in topic prevalence for human review.

## Basic heuristic

A simple baseline compares each week's topic share with a preceding
baseline. The threshold is a configurable heuristic, not a validated
universal standard.

``` python
def flag_topic_changes(weekly_shares, min_reports=None, threshold=0.10):
    alerts = []
    for topic_col in weekly_shares.columns:
        series = weekly_shares[topic_col].dropna().sort_index()
        baseline = series.shift(1).rolling(4, min_periods=2).mean()
        delta = series - baseline
        for week in series.index:
            if pd.notna(baseline.loc[week]) and delta.loc[week] >= threshold:
                alerts.append({
                    "week": str(week.date()),
                    "topic": int(topic_col) + 1,
                    "previous_4_week_baseline": float(baseline.loc[week]),
                    "current_share": float(series.loc[week]),
                    "increase": float(delta.loc[week]),
                    "status": "Review candidate"
                })
    return pd.DataFrame(alerts)

candidate_alerts = flag_topic_changes(lda_weekly, threshold=0.10)
display(candidate_alerts.head(20))
candidate_alerts.to_csv("artifacts/candidate_topic_alerts.csv", index=False)
```

## Improvements before claiming robustness

1.  Check the number of reports in each week.
2.  Avoid alerts in weeks with very few reports.
3.  Compare against multiple previous weeks, not just one.
4.  Review representative reports for every candidate alert.
5.  If available, compare alerts against independent incident or
    threat-feed data.
6.  Evaluate false positives and false negatives only when a defensible
    ground-truth set exists.

## Thesis wording

"The module produces candidate alerts based on changes in weekly topic
prevalence. Alerts are intended to prioritize analyst review and are not
interpreted as confirmation of real-world incidents."
