from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

from code.evaluation.evaluation import evaluate_model


def main():
    metrics = evaluate_model()
    print('Test metrics:', metrics)


if __name__ == '__main__':
    main()
