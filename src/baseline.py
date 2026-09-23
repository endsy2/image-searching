"""Keyword baseline over ``search_text``.

A TF-IDF-weighted bag-of-words scorer implemented directly on the catalog, so
the comparison needs no extra dependency. It searches exactly the field the
catalog contract defines for it:

    productDisplayName | gender | masterCategory | subCategory | articleType
    | baseColour | usage

It is deliberately plain. The point of the baseline is to show what lexical
matching over the same catalog and the same filters achieves, not to be a
competitive search engine.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(str(text).lower())


class KeywordBaseline:
    def __init__(self, catalog: pd.DataFrame, field: str = 'search_text'):
        self.product_ids = catalog['id'].astype('int64').to_numpy()
        self.names = catalog['productDisplayName'].fillna('').astype(str).to_numpy()
        documents = [tokenize(t) for t in catalog[field].fillna('').astype(str)]

        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(document))
        count = len(documents)
        self.idf = {
            term: math.log((count + 1) / (frequency + 1)) + 1.0
            for term, frequency in document_frequency.items()
        }

        # Per-document L2-normalized tf-idf, stored sparsely as term -> weight.
        self.postings: dict[str, list[tuple[int, float]]] = {}
        for index, document in enumerate(documents):
            if not document:
                continue
            counts = Counter(document)
            weights = {t: (1.0 + math.log(c)) * self.idf[t] for t, c in counts.items()}
            norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
            for term, weight in weights.items():
                self.postings.setdefault(term, []).append((index, weight / norm))

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.product_ids), dtype=np.float32)
        terms = Counter(tokenize(query))
        for term, count in terms.items():
            if term not in self.postings:
                continue
            query_weight = (1.0 + math.log(count)) * self.idf[term]
            for index, weight in self.postings[term]:
                scores[index] += query_weight * weight
        return scores

    def search(self, query: str, top_k: int = 5,
               allowed: set[int] | None = None) -> list[dict]:
        scores = self.score(query)
        if allowed is not None:
            mask = np.isin(self.product_ids, list(allowed))
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:top_k]
        return [
            {'product_id': int(self.product_ids[i]), 'name': str(self.names[i]),
             'final_score': float(scores[i])}
            for i in order if np.isfinite(scores[i]) and scores[i] > 0
        ]
