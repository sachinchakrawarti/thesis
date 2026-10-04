"""Emerging Threat Score (ETS) computation."""
import numpy as np
import pandas as pd
from .config import ETS_WEIGHTS

def _normalize(series):
    s = series.astype(float).fillna(0.0)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return s * 0.0
    return (s - lo) / (hi - lo)

def compute_emerging_scores(prevalence_df):
    rows = []
    topics = [c for c in prevalence_df.columns if c.startswith("topic_")]
    for topic in topics:
        series = prevalence_df[topic]
        novelty     = series.iloc[0] == 0
        growth_vals = series.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        growth      = growth_vals.max()
        persistence = (series > series.mean()).sum() / len(series)
        activity    = series.mean()
        term_change = series.std()
        rows.append({
            "Topic":       topic,
            "Novelty":     int(novelty),
            "Growth":      growth,
            "Persistence": persistence,
            "Activity":    activity,
            "TermChange":  term_change,
        })
    df = pd.DataFrame(rows)
    for c in ["Novelty", "Growth", "Persistence", "Activity", "TermChange"]:
        df[c] = _normalize(df[c])
    df["ETS"] = (
        ETS_WEIGHTS["novelty"]     * df["Novelty"]     +
        ETS_WEIGHTS["growth"]      * df["Growth"]      +
        ETS_WEIGHTS["persistence"] * df["Persistence"] +
        ETS_WEIGHTS["activity"]    * df["Activity"]    +
        ETS_WEIGHTS["term_change"] * df["TermChange"]
    )
    return df.sort_values("ETS", ascending=False).reset_index(drop=True)
