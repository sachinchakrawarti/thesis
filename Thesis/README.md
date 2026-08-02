# Regime-Aware Transformer-Based Financial Decision Support

This project provides the complete research implementation for a thesis on regime-aware transformer-based financial decision support.

## Project Structure

- `thesis.md`: merged academic thesis in Markdown.
- `references.bib`: bibliography file for citation support.
- `code/`: reproducible Python implementation.
- `results/`: saved outputs and plots.
- `notebooks/`: research notebooks.

## Research Summary

The study proposes a Regime-Aware Temporal Transformer (RATT) that uses transformer attention and engineered temporal features for adaptive financial market regime identification and decision support.

## Getting Started

1. Create a virtual environment.
2. Install dependencies from `code/requirements.txt`.
3. Configure the experiment in `code/config/config.yaml`.
4. Run the training entry point.

```bash
python train.py
```

## Reproducibility

The implementation is modular and organized around clear responsibility boundaries to enable reproducible experimentation.
