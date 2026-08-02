from pathlib import Path


class Logger:
    def __init__(self, path: str = 'results/run.log'):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str):
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
