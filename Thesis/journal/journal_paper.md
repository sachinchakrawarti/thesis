# Journal Paper Draft

## Abstract

This paper proposes a hybrid transformer-based regime-aware framework for adaptive financial decision support. The model addresses the need for sequence-aware market state identification and decision adaptation under regime switching. Experimental analysis shows strong gains over classical and sequential baselines.

## Keywords

Regime detection, transformer, financial decision support, time-series modeling, adaptive forecasting.

## Introduction

Financial markets constantly shift across complex regimes. A static model may perform well on average but fail during abrupt transitions. This motivates a regime-aware decision-support method that separates states and adapts decisions accordingly.

## Methodology

The proposed method extends a transformer encoder with engineered temporal features and a regime classifier. The system uses normalized windows and a composite loss function to jointly optimize regime classification and forecast prediction.

## Results

The proposed model achieves higher accuracy, F1 score, and ROC AUC than the chosen baselines. The confusion matrix further confirms improved separation of volatile and transitional states.

## Conclusion

The work demonstrates that regime-aware transformer modeling provides a stronger solution for financial decision support than classical static forecasting pipelines.

## References

As in the thesis references, structured in IEEE-style bibliography order.
