"""
Evaluation utilities for the SHL Assessment Recommendation System.

Implements Recall@K and Mean Recall@K and applies them at:
- Dense retrieval stage (vector search only)
- Final recommendation stage (after intent + LLM re‑ranking)

This file is NOT used by the API – it is for offline experiments only.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlparse, urlunparse

import pandas as pd

from retriever import load_vectorstore, recommend


def _load_train_data(path: str) -> Dict[str, List[str]]:
    """
    Load training data with ground‑truth relevant assessments.

    Supports two formats:
    1) Multiple rows per query (Appendix 3 style):
         columns: Query, Assessment_url
    2) Single row per query with concatenated URLs:
         columns: query, relevant_urls (semicolon‑separated)
    """
    df = pd.read_csv(path)

    # Normalise column names
    cols = {c.lower(): c for c in df.columns}

    if "query" in cols and "assessment_url" in cols:
        q_col = cols["query"]
        a_col = cols["assessment_url"]
        grouped: Dict[str, List[str]] = defaultdict(list)
        for _, row in df.iterrows():
            q = str(row[q_col]).strip()
            url = str(row[a_col]).strip()
            if q and url:
                grouped[q].append(url)
        return dict(grouped)

    raise ValueError(
        "Train CSV must contain either (Query, Assessment_url) or "
        "(query, relevant_urls / relevant_url) columns."
    )


def _normalize_url(url: str) -> str:
    """
    Normalize URL for comparison by:
    - Converting to lowercase
    - Removing trailing slashes
    - Normalizing http/https (treat as same)
    - Removing query parameters and fragments
    - Removing /solutions/ from path (SHL URLs can have this or not)
    """
    if not url or not isinstance(url, str):
        return ""
    url = url.strip()
    if not url:
        return ""
    
    try:
        parsed = urlparse(url.lower())
        # Normalize path: remove /solutions/ if present, remove trailing slash
        path = parsed.path.rstrip('/')
        # Remove /solutions/ from path if it appears
        if '/solutions/' in path:
            path = path.replace('/solutions/', '/')
        
        # Remove query params and fragments, normalize scheme
        normalized = urlunparse((
            parsed.scheme.replace('http', 'https'),  # Normalize http->https
            parsed.netloc,
            path,
            parsed.params,
            '',  # Remove query
            ''   # Remove fragment
        ))
        return normalized
    except Exception:
        # Fallback: just lowercase and strip, try to remove /solutions/
        fallback = url.lower().strip().rstrip('/')
        if '/solutions/' in fallback:
            fallback = fallback.replace('/solutions/', '/')
        return fallback


def _recall_at_k(relevant: Iterable[str], predicted: Iterable[str], k: int) -> float:
    """
    Recall@K for a single query.
    
    Formula: (Number of relevant assessments in top K) / (Total relevant assessments)
    
    Note: K is independent of the number of relevant URLs. The denominator
    is always the total number of relevant URLs for the query.
    """
    rel = {_normalize_url(u) for u in relevant if isinstance(u, str) and u.strip()}
    rel.discard("")  # Remove empty strings
    if not rel:
        return 0.0
    
    top_k = {_normalize_url(u) for u in list(predicted)[:k] if isinstance(u, str)}
    top_k.discard("")  # Remove empty strings
    if not top_k:
        return 0.0
    
    hits = len(rel.intersection(top_k))
    # Denominator is total relevant URLs (not min(len(rel), k))
    return hits / len(rel)


def evaluate(
    train_csv_path: str,
    retrieval_k: int = 20,
    final_k: int = 10,
) -> Dict[str, float]:
    """
    Run evaluation over the provided train data.

    Args:
        train_csv_path: Path to train CSV with Query and Assessment_url columns
        retrieval_k: Number of docs retrieved in dense search stage
        final_k: Number of final recommendations returned by the system
        debug: If True, print detailed debugging information for first query

    Returns:
        {
          "retrieval_mean_recall@K": ...,
          "final_mean_recall@K": ...,
          "num_queries": N
        }
    """
    ground_truth = _load_train_data(train_csv_path)
    if not ground_truth:
        raise ValueError("No valid train examples found.")

    vectorstore = load_vectorstore()

    retrieval_recalls: List[float] = []
    final_recalls: List[float] = []

    for idx, (query, relevant_urls) in enumerate(ground_truth.items()):
        # 1) Retrieval stage
        retrieved_docs = vectorstore.similarity_search(query, k=retrieval_k)
        retrieved_urls = [
            (doc.metadata.get("assessment_url") or "").strip()
            for doc in retrieved_docs
        ]
        retrieval_recall = _recall_at_k(relevant_urls, retrieved_urls, final_k)
        retrieval_recalls.append(retrieval_recall)

        # 2) Full recommendation stage
        rec_output = recommend(query, final_k)
        final_urls = [
            (item.get("url") or "").strip()
            for item in rec_output.get("recommended_assessments", [])[:final_k]
        ]
        final_recall = _recall_at_k(relevant_urls, final_urls, final_k)
        final_recalls.append(final_recall)

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "retrieval_mean_recall@K": _mean(retrieval_recalls),
        "final_mean_recall@K": _mean(final_recalls),
        "num_queries": len(ground_truth),
    }


if __name__ == "__main__":
    metrics = evaluate("data/train_queries.csv")
    print("\n" + "="*60)
    print("Evaluation metrics (Mean Recall@K):")
    print("="*60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print("="*60)


