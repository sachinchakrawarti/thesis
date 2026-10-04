"""Temporal aggregation of topic prevalence."""
import pandas as pd

def topic_prevalence_by_period(doc_topic, dates, freq="M"):
    df = pd.DataFrame(
        doc_topic,
        columns=[f"topic_{i+1}" for i in range(doc_topic.shape[1])]
    )
    df["date"] = pd.to_datetime(dates.values)
    df["period"] = df["date"].dt.to_period(freq)
    cols = [c for c in df.columns if c.startswith("topic_")]
    return df.groupby("period")[cols].mean()
