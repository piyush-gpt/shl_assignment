import os
from typing import List, Dict, Literal

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from pydantic import BaseModel, ValidationError, conint
from dotenv import load_dotenv

load_dotenv()

# ================== CONFIG ==================
PERSIST_DIR = "vectorstore/chroma"
EMBEDDING_MODEL = "openai/text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"

TOP_K_RETRIEVE = 20
FINAL_K = 7  # must be between 5–10


# ================== TEST TYPE MAP ==================
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgment",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behaviour",
    "S": "Simulations"
}

# ================== DOMAIN → TEST TYPE ==================
DOMAIN_TO_TEST_TYPES = {
    "Ability & Aptitude": ["A"],
    "Biodata & Situational Judgment": ["B"],
    "Competencies": ["C"],
    "Development & 360": ["D"],
    "Assessment Exercises": ["E"],
    "Knowledge & Skills": ["K"],
    "Personality & Behaviour": ["P"],
    "Simulations": ["S"],
}


# ================== VECTORSTORE ==================
def load_vectorstore():
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    return Chroma(
        collection_name="shl_catalog",
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )


# ================== QUERY INTENT DETECTION (OPTION A) ==================
DomainType = Literal["Ability & Aptitude", "Biodata & Situational Judgment", "Competencies", "Development & 360", "Assessment Exercises", "Knowledge & Skills", "Personality & Behaviour", "Simulations"]

class QueryIntent(BaseModel):
    domains: List[DomainType]


def detect_query_intent(query: str) -> List[str]:
    llm = ChatOpenAI(
        model=LLM_MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
    )

    prompt = f"""
You are an HR assessment expert.

Identify which most relevant and important skill domains are required by the hiring query.

Possible domains (choose one or more):
- "Ability & Aptitude" – cognitive ability, problem solving, reasoning, numerical / verbal skills
- "Biodata & Situational Judgment" – biographical data, realistic work scenarios, judgment in work situations
- "Competencies" – behavioral competencies and soft skills required for effective performance
- "Development & 360" – development-focused assessments, feedback, and 360-degree reviews
- "Assessment Exercises" – assessment centres, work samples, and multi-exercise job simulations
- "Knowledge & Skills" – technical or professional knowledge and job-specific hard skills like Java, Python, SQL, etc.
- "Personality & Behaviour" – personality traits, work styles, and behavioral preferences
- "Simulations" – interactive or game-like simulations of real work environments or tasks

Rules:
- Return ONLY a JSON object.
- Do not explain your answer.

Query:
{query}
"""

    structured_llm = llm.with_structured_output(QueryIntent)

    try:
        result = structured_llm.invoke(prompt).domains
        return result
    except ValidationError:
        return []


def infer_required_test_types(domains: List[str]) -> List[str]:
    test_types = set()
    for domain in domains:
        test_types.update(DOMAIN_TO_TEST_TYPES.get(domain, []))
    return list(test_types)


# ================== HELPERS ==================
def extract_test_types(doc: Document) -> List[str]:
    """Return list of SHL test-type codes (e.g. ["K", "P"])."""
    raw = doc.metadata.get("test_type_codes", [])

    # Backward compatibility: handle both string and list storage
    if isinstance(raw, str):
        codes = [c.strip() for c in raw.split(",") if c.strip()]
    elif isinstance(raw, list):
        codes = [str(c).strip() for c in raw if str(c).strip()]
    else:
        codes = []

    return codes


def semantic_test_types(codes: List[str]) -> str:
    return ", ".join(TEST_TYPE_MAP[c] for c in codes if c in TEST_TYPE_MAP)


def extract_description(doc: Document) -> str:
    text = doc.page_content or ""
    if "Description:" in text:
        desc = text.split("Description:", 1)[1]
        return desc.strip()
    return text.strip()


# ================== DYNAMIC BALANCING ==================
def balanced_selection(
    docs: List[Document],
    required_test_types: List[str],
    k: int,
) -> List[Document]:
    """
    Select documents with test-type balancing while preserving original retrieval order.
    
    Strategy:
    1. First pass: Select documents matching required test types, preserving original order
    2. Second pass: Fill remaining slots with other documents in original order
    """
    selected = []
    seen_urls = set()  # Track URLs to avoid duplicates
    
    # Track how many documents we've selected per test type
    per_type_target = max(1, k // max(1, len(required_test_types)))
    type_counts = {t: 0 for t in required_test_types}
    
    # First pass: Select documents matching required test types (preserving original order)
    for doc in docs:
        if len(selected) >= k:
            break
        
        url = doc.metadata.get("assessment_url", "")
        if not url or url in seen_urls:
            continue
        
        doc_types = extract_test_types(doc)
        matching_types = [t for t in required_test_types if t in doc_types]
        
        # Select if it matches a required type and we haven't exceeded per-type limit
        if matching_types:
            # Check if we can add this document (haven't exceeded any matching type limit)
            can_add = any(type_counts[t] < per_type_target for t in matching_types)
            if can_add:
                selected.append(doc)
                seen_urls.add(url)
                # Increment counts for all matching types
                for t in matching_types:
                    type_counts[t] += 1
    
    # Second pass: Fill remaining slots with documents in original order
    for doc in docs:
        if len(selected) >= k:
            break
        
        url = doc.metadata.get("assessment_url", "")
        if url and url not in seen_urls:
            selected.append(doc)
            seen_urls.add(url)

    return selected[:k]


# ================== LLM SCORING ==================
class ScoreList(BaseModel):
    scores: List[conint(ge=1, le=5)]


def score_with_llm(query: str, docs: List[Document]) -> List[int]:
    llm = ChatOpenAI(
        model=LLM_MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
    )

    prompt = f"""
You are an SHL assessment expert helping recruiters choose the most relevant assessments.

Score each assessment from 1 (low relevance) to 5 (high relevance).

Consider:
- Skill alignment with the query
- Match between assessment categories and query intent
- Complementarity (avoid over-scoring duplicates)

Return ONLY a JSON list of integers in the SAME ORDER.

Query:
{query}

Assessments:
"""

    for i, doc in enumerate(docs):
        prompt += f"""
[{i+1}]
Name: {doc.metadata['assessment_name']}
Categories: {semantic_test_types(extract_test_types(doc))}
Description:
{extract_description(doc)}
"""

    structured_llm = llm.with_structured_output(ScoreList)
    return structured_llm.invoke(prompt).scores


# ================== MAIN RECOMMENDER ==================
def recommend(query: str, k: int = FINAL_K) -> Dict:
    """
    End-to-end recommendation pipeline:
    1. Dense retrieval over the SHL catalog vector store.
    2. LLM-based intent detection to infer required test-type families.
    3. Intent-aware balancing to keep a diverse assessment mix.
    4. LLM scoring and re-ranking to produce the final recommendations.
    """
    vectorstore = load_vectorstore()

    #
    # 1. Detect intent
    domains = detect_query_intent(query)
    required_test_types = infer_required_test_types(domains)

    # Safety fallback
    if not required_test_types:
        required_test_types = ["K", "P"]

    # 2. Intent-aware retrieval
    retrieved: List[Document] = []
    seen_urls = set()

    # Distribute retrieval budget across required types
    per_type_k = max(1, TOP_K_RETRIEVE // max(1, len(required_test_types)))

    for t in required_test_types:
        flag_key = f"is_type_{t}"
        
            # Use boolean metadata flags like is_type_K, is_type_P for filtering
        docs_for_type = vectorstore.similarity_search(
            query,
            k=per_type_k,
            filter={flag_key: True},
        )
        for doc in docs_for_type:
            url = doc.metadata.get("assessment_url")
            if url and url not in seen_urls:
                retrieved.append(doc)
                seen_urls.add(url)

    # Fallback: if we still have fewer than TOP_K_RETRIEVE docs, top up with standard retrieval
    if len(retrieved) < TOP_K_RETRIEVE:
        extra_docs = vectorstore.similarity_search(
            query,
            k=TOP_K_RETRIEVE - len(retrieved),
        )
        for doc in extra_docs:
            url = doc.metadata.get("assessment_url")
            if url and url not in seen_urls:
                retrieved.append(doc)
                seen_urls.add(url)

    # 3. Intent-aware balancing on retrieved set
    balanced = balanced_selection(
        retrieved,
        required_test_types=required_test_types,
        k=k,
    )

    # 4. LLM scoring
    scores = score_with_llm(query, balanced)

    ranked = sorted(
        zip(balanced, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    def duration_as_int(raw):
        if raw is None:
            return None
        if isinstance(raw, str):
            import re

            m = re.search(r"(\d+)", raw)
            if m:
                return int(m.group(1))
        try:
            return int(float(raw))
        except Exception:
            return None

    # Build final recommendations, ensuring no duplicates by URL
    recommended = []
    seen_urls = set()
    for doc, _ in ranked:
        url = doc.metadata.get("assessment_url")
        if url and url not in seen_urls:
            codes = extract_test_types(doc)
            recommended.append(
                {
                    "url": url,
                    "name": doc.metadata.get("assessment_name"),
                    "adaptive_support": doc.metadata.get("adaptive_irt"),
                    "description": extract_description(doc),
                    "duration": duration_as_int(doc.metadata.get("duration")),
                    "remote_support": doc.metadata.get("remote_testing"),
                    "test_type": [TEST_TYPE_MAP.get(c, c) for c in codes],
                }
            )
            seen_urls.add(url)
            if len(recommended) >= k:
                break
    return {"recommended_assessments": recommended}


# ================== LOCAL TEST ==================
if __name__ == "__main__":
    print(
        recommend(
            "Need a Java developer who is good in collaborating with external teams and stakeholders"
        )
    )
