"""Central configuration for the CTI thesis project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR  = PROJECT_ROOT / "dataset"
OUTPUT_DIR   = PROJECT_ROOT / "output"
MODELS_DIR   = OUTPUT_DIR / "models"
FIGURES_DIR  = OUTPUT_DIR / "figures"
REPORTS_DIR  = OUTPUT_DIR / "reports"
TOPICS_DIR   = OUTPUT_DIR / "topics"

for d in [MODELS_DIR, FIGURES_DIR, REPORTS_DIR, TOPICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DATASET_FILE = DATASET_DIR / "synthetic_cti_reports.csv"

MAX_FEATURES = 10000
NGRAM_RANGE  = (1, 2)
MIN_DF       = 2
MAX_DF       = 0.95
RANDOM_STATE = 42
TEST_SIZE    = 0.20

N_TOPICS     = 10
TOP_N_WORDS  = 10

THREAT_CATEGORIES = ["Malware", "Phishing", "Ransomware",
                     "DDoS", "Botnets", "Spam"]

ETS_WEIGHTS = {
    "novelty":     0.20,
    "growth":      0.30,
    "persistence": 0.20,
    "activity":    0.15,
    "term_change": 0.15,
}
