"""Text cleaning and normalization for CTI documents."""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

_STOP = set(stopwords.words("english"))
_LEM  = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s\.\-:/_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = []
    for tok in text.split():
        if tok in _STOP and len(tok) < 4:
            continue
        if re.fullmatch(r"\d+", tok):
            continue
        tokens.append(_LEM.lemmatize(tok))
    return " ".join(tokens)
