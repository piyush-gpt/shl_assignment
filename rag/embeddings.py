import os

import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

# ================== CONFIG ==================
# Run this script from the project root (`python rag/embeddings.py`)
DATA_PATH = "data/shl_catelog.csv"
PERSIST_DIR = "vectorstore/chroma"
EMBEDDING_MODEL = "openai/text-embedding-3-large"


# ================== SHL TEST TYPE MAP ==================
TEST_TYPE_MAP = {
    "A": "Ability and aptitude assessments",
    "B": "Biodata and situational judgment assessments",
    "C": "Competency-based assessments",
    "D": "Development and 360-degree feedback assessments",
    "E": "Assessment exercises",
    "K": "Knowledge and skills assessments",
    "P": "Personality and behavioral assessments",
    "S": "Simulation-based assessments"
}


def load_catalog(csv_path: str) -> pd.DataFrame:
    """Load and validate SHL catalog CSV."""
    df = pd.read_csv(csv_path)

    required_columns = [
        "name",
        "url",
        "description",
        "test_type",
        "duration",
        "remote_testing",
        "adaptive_irt"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if len(df) < 377:
        raise ValueError(
            f"Expected ≥ 377 individual test solutions, found {len(df)}"
        )

    return df


def expand_test_types(test_type_str: str) -> str:
    """Convert test-type codes into semantic descriptions."""
    if pd.isna(test_type_str):
        return ""

    codes = [c.strip() for c in test_type_str.split(",")]
    expanded = [TEST_TYPE_MAP.get(code) for code in codes if code in TEST_TYPE_MAP]

    return ", ".join(expanded)


def row_to_document(row: pd.Series) -> Document:
    """
    Create a semantically optimized Document for embeddings
    and constraint-rich metadata for ranking/filtering.
    """

    semantic_test_types = expand_test_types(row["test_type"])

    page_content = f"""

Description:
{row['description']}

Assessment Categories:
{semantic_test_types}
    """.strip()

    # Raw SHL codes from the CSV, e.g. "K" or "K,P"
    raw_codes = str(row["test_type"])
    codes = [c.strip() for c in raw_codes.split(",") if c.strip()]

    # Boolean flags per test type for easy metadata filtering
    type_flags = {
        f"is_type_{code}": (code in codes)
        for code in TEST_TYPE_MAP.keys()
    }

    metadata = {
        "assessment_name": row["name"],
        "assessment_url": row["url"],
        # Keep original codes as a comma-separated string for backwards compatibility
        "test_type_codes": raw_codes,
        "duration": row["duration"],
        "remote_testing": row["remote_testing"],
        "adaptive_irt": row["adaptive_irt"],
        **type_flags,
    }

    return Document(
        page_content=page_content,
        metadata=metadata
    )


def build_chroma_vectorstore():
    """Build and persist ChromaDB vector store."""
    os.makedirs(PERSIST_DIR, exist_ok=True)

    df = load_catalog(DATA_PATH)
    documents = [row_to_document(row) for _, row in df.iterrows()]

    # Uses Gemini embeddings via langchain-google-genai.
    # Make sure GOOGLE_API_KEY is set in your environment.
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL,base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"))
    vectorstore = Chroma(
        collection_name="shl_catalog",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    # Add all assessment documents
    vectorstore.add_documents(documents)


    print("✅ Chroma vector store created successfully")
    print(f"📦 Total assessments indexed: {len(documents)}")


if __name__ == "__main__":
    build_chroma_vectorstore()
