import re

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.utils import Bunch

from config import DATA_DIR, MIN_DOC_WORDS, RANDOM_SEED

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_data():
    data_train = fetch_20newsgroups(
        data_home=DATA_DIR,
        subset="train",
        random_state=RANDOM_SEED,
    )
    data_test = fetch_20newsgroups(
        data_home=DATA_DIR,
        subset="test",
        random_state=RANDOM_SEED,
        shuffle=False,
    )
    assert isinstance(data_train, Bunch)
    assert isinstance(data_test, Bunch)

    return data_train, data_test


def _strip_header(text: str):
    """Headers end at the first blank line; keep text unchanged if none is found."""
    _, sep, body = text.partition("\n\n")
    return body if sep else text


def _strip_footer(text: str):
    """`--` on its own line is the conventional signature marker."""
    lines = text.strip().split("\n")
    for i, line in enumerate(lines):
        if line.strip() == "--":
            return "\n".join(lines[:i])
    return text


def _strip_quotes(text: str):
    """Remove quoted lines from the text."""
    return "\n".join(
        line for line in text.split("\n") if not line.lstrip().startswith(">")
    )


def _clean_tokens(text: str):
    """Strip structural noise, lowercase, drop stop words and pure-number tokens."""
    text = _strip_quotes(_strip_footer(_strip_header(text))).strip().lower()
    return [
        w
        for w in _TOKEN_RE.findall(text)
        if w not in ENGLISH_STOP_WORDS and not w.isdigit()
    ]


def preprocess_data(data: Bunch):
    token_lists = [_clean_tokens(text) for text in data.data]
    cleaned = [" ".join(tokens) for tokens in token_lists]
    keep = [i for i, text in enumerate(cleaned) if len(text.split()) >= MIN_DOC_WORDS]

    return Bunch(
        data=[cleaned[i] for i in keep],
        target=data.target[keep],
        target_names=data.target_names,
    )
