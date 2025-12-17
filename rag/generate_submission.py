"""
Utility script to generate the submission CSV in the format
required in Appendix 3:

Query,Assessment_url
Query 1,Recommendation 1 (URL)
Query 1,Recommendation 2 (URL)
...
"""

from typing import List

import pandas as pd

from retriever import recommend


def generate_submission(
    input_queries_csv: str,
    output_csv: str,
    top_k: int = 7,
) -> None:
    """
    Read an unlabeled test CSV containing queries and write predictions
    in the required two‑column format:

    - Input CSV must have a column named either `Query` or `query`.
    - Output CSV will have columns: `Query`, `Assessment_url`.
    """
    df = pd.read_csv(input_queries_csv)

    # Normalise column name
    query_col = None
    for c in df.columns:
        if c.lower() == "query":
            query_col = c
            break
    if query_col is None:
        raise ValueError("Input CSV must contain a `Query` or `query` column.")

    rows: List[dict] = []

    for _, row in df.iterrows():
        q = str(row[query_col]).strip()
        if not q:
            continue

        rec_output = recommend(q)
        recs = rec_output.get("recommended_assessments", [])[:top_k]

        for rec in recs:
            url = rec.get("url") or ""
            if not url:
                continue
            rows.append({"Query": q, "Assessment_url": url})

    out_df = pd.DataFrame(rows, columns=["Query", "Assessment_url"])
    out_df.to_csv(output_csv, index=False)
    print(f"✅ Submission file written to {output_csv} with {len(out_df)} rows.")


if __name__ == "__main__":
    generate_submission(
        input_queries_csv="data/unlabeled_test_queries.csv",
        output_csv="submission.csv",
        top_k=7,
    )


