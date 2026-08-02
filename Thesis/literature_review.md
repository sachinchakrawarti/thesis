# Literature Review

## 1. Base Paper Selection

The selected base paper is:

- Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, J., and Long, M. "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting," arXiv preprint arXiv:2310.06625, 2024.

### Why this paper was selected

This paper is especially relevant because it introduces a strong architectural redesign for multivariate time-series forecasting. Instead of applying self-attention directly to the time dimension, the model inverts the transformer pipeline to emphasize variable-wise attention and channel independence. This design is attractive for financial time series because market variables often have heterogeneous dependencies that benefit from a better representation of inter-variable structure.

### Methodology of the base paper

The iTransformer model operates by embedding each variable independently and applying attention across the channel dimension, then decoding the resulting representation for future forecasting. The innovation is that it performs attention over variables rather than over time tokens, allowing better modeling of cross-variable dependencies and improved long-horizon forecasting.

### Limitations of the base paper

Despite its contribution, the model is not specialized for regime-aware financial decision support. It is primarily a forecasting architecture, and its performance depends heavily on the stability of the input series. It does not explicitly model market regime transitions, uncertainty, or a regime-conditioned decision policy.

### How the proposed work improves upon it

The present review and research direction extend the transformer paradigm by coupling long-range attention with explicit regime identification. In practice, this means adding a regime-classification layer, domain-aware feature engineering, and a decision-support objective rather than only point forecasting.

## 2. Related Work Review

The literature on transformers and time-series forecasting has evolved rapidly. The most influential works can be grouped into the following themes.

### 2.1 Transformer Foundations

1. Vaswani et al. (2017) introduced the self-attention architecture and showed that transformer layers can replace recurrence in sequence modeling.
2. Hochreiter and Schmidhuber (1997) established the LSTM as the dominant recurrent approach for sequence memory.
3. Cho et al. (2014) proposed GRU, which reduced the memory footprint while preserving sequential encoding.
4. Zerveas et al. (2021) developed a representation learning framework for multivariate time series using transformers.
5. Wen et al. (2022) provided a comprehensive survey of transformer methods for time-series applications.

### 2.2 Efficient Transformer Forecasting Models

6. Zhou et al. (2021) proposed Informer to address long sequence forecasting with sparse self-attention.
7. Wu et al. (2021) introduced Autoformer, which decomposes seasonal and trend components before attention.
8. Lim et al. (2021) proposed Temporal Fusion Transformers, which blend static, known, and observed covariates for interpretable multi-horizon forecasting.
9. Zhou et al. (2022) proposed FEDformer to enrich transformer modeling with frequency-domain decomposition.
10. Kitaev et al. (2020) proposed Reformer, a more memory-efficient alternative with locality-sensitive hashing and reversible layers.
11. Wang et al. (2022) proposed Pyraformer, using pyramidal attention patterns for efficient long-range forecasting.
12. Cirstea et al. (2022) proposed Triformer for triangular high-order long-sequence forecasting.
13. Shabani et al. (2022) proposed Scaleformer for iterative multi-scale time-series forecasting.
14. Nie et al. (2023) introduced PatchTST, which treats a time series as patched segments to improve long-horizon performance.
15. Zhang and Yan (2023) proposed Crossformer to exploit cross-dimension dependencies in multivariate series.
16. Oreshkin et al. (2020) introduced N-BEATS, a strong neural forecasting baseline based on basis expansion.
17. Salinas et al. (2020) developed DeepAR for probabilistic autoregressive forecasting.
18. Zeng et al. (2023) challenged the universal superiority of transformers by showing that linear models can remain strong baselines.
19. Zeng et al. (2023) also proposed FITS, a compact time-series model using a small parameter budget.
20. Woo et al. (2022) proposed ETSformer for exponential smoothing–inspired forecasting.

### 2.3 Foundation and Multimodal Time-Series Models

21. Jin et al. (2024) proposed Time-LLM, which repurposes large language models for time-series forecasting.
22. Goswami et al. (2024) introduced MOMENT, a time-series foundation model family.
23. Ansari et al. (2024) proposed Chronos, a foundation-style language approach to time-series modeling.

### 2.4 Financial and Regime-Oriented Context

24. Hamilton (1989) established Markov-switching frameworks for non-stationary processes and business-cycle modeling.
25. Engle (1982) advanced ARCH-based volatility models for financial time series.
26. Bollerslev (1986) generalized ARCH into GARCH, which remains a benchmark in financial volatility modeling.
27. Fama and French (1993) developed factor-based risk models that continue to shape financial econometric reasoning.
28. Merton (1973) introduced jump-diffusion modeling for discontinuous return processes.
29. Tsay (1989) connected threshold and structural-break behavior with market dynamics.
30. Campbell and Shiller (1988) provided a theoretically grounded valuation relation that continues to inform market interpretation.
31. Fischer and Krauss (2018) showed that deep recurrent networks can be effective for directional financial prediction.
32. Sezer et al. (2020) reviewed the application of LSTM networks for financial forecasting and identified the strengths and weaknesses of sequence models in the domain.

## 3. Synthesis of the Literature

The literature demonstrates a clear progression from classical econometric models to modern network-based forecasting architectures. Classical methods such as ARCH, GARCH, and Markov-switching models are strong for describing volatility and structural breaks, but they have limited expressive power when the series requires long-range or cross-variable interactions.

Transformers address this limitation well, especially for capturing long-distance dependencies and adaptive representation learning. However, most transformer papers focus on forecasting accuracy rather than financial regime detection or decision-support adaptation. This gap is central to the motivation for the present work.

## 4. Research Gap

The literature leaves four major gaps unresolved:

1. A large body of work provides strong forecasting performance but weak explicit regime reasoning.
2. Few methods are designed as regime-aware decision-support systems rather than pure predictors.
3. The transformer literature is strong on sequence modeling, yet still under-explored for adaptive regime-conditioned financial workflows.
4. Existing methods rarely unify long-range attention, engineered market features, and practical decision logic in one pipeline.

The present study addresses this gap by positioning the transformer as a regime-aware decision-support engine rather than only a forecasting tool.

## 5. Comparison Summary

| Paper | Year | Method | Primary Dataset Type | Strength | Limitation |
|---|---:|---|---|---|---|
| Vaswani et al. | 2017 | Transformer | Text/sequence | Strong attention mechanism | Not finance-specific |
| Zerveas et al. | 2021 | Transformer representation learning | Multivariate time series | Good representation learning | No regime logic |
| Informer | 2021 | Sparse transformer | Long sequences | Efficient long-horizon processing | Forecast-only |
| Autoformer | 2021 | Decomposition transformer | Long-term forecasting | Strong trend decomposition | Not regime-aware |
| TFT | 2021 | Interpretable transformer | Multi-horizon forecasting | Explainable and covariate-aware | Computationally heavy |
| PatchTST | 2023 | Patching transformer | Forecasting | Strong segmentation for long series | Not regime-specific |
| Crossformer | 2023 | Cross-dimension transformer | Multivariate time series | Captures cross-variable dependency | More complex training |
| iTransformer | 2024 | Inverted transformer | Multivariate forecasting | Strong variable-wise modeling | Not regime-aware |
| Time-LLM | 2024 | LLM-reprogrammed forecasting | Time-series tasks | Foundation-model style transfer | Costly, less domain-specific |
| MOMENT | 2024 | Time-series foundation model | General time series | Strong representation foundation | Not finance-specific |
| Chronos | 2024 | Time-series foundation model | Generic forecasting | Strong generalization | Requires large-scale adaptation |
| Hamilton | 1989 | Markov-switching | Macroeconomic/financial states | Explicit regime modeling | Limited nonlinear representation |
| GARCH | 1986 | Volatility model | Financial volatility | Strong for volatility dynamics | Poor long-range dependency modeling |
| DeepAR | 2020 | Autoregressive RNN | Forecasting | Probabilistic forecasting | Less explicit regime modeling |
| N-BEATS | 2020 | Basis-expansion neural net | Forecasting | Interpretable and strong baseline | No regime layer |
| LSTM financial review | 2020 | Recurrent networks | Finance | Useful prior baseline | Weak modern transformer adaptation |

## 6. Summary

The literature shows a mature and evolving landscape in which classical econometric models remain relevant for market volatility and structural break analysis, while transformer-based architectures are increasingly central to modern time-series forecasting. The most important gap is the lack of a regime-aware financial decision-support architecture that unifies transformer attention, domain feature engineering, and adaptive decision logic. This gap motivates the subsequent research direction and provides the foundation for a more complete and realistic market intelligence framework.
