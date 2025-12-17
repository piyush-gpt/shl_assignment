# SHL Assessment Recommendation System: Approach & Optimization

## Problem Statement

The task was to build a recommendation system that matches job descriptions or natural language queries with relevant SHL assessments from a catalog of 377+ assessments. The system must return 1-10 recommendations per query, with each recommendation including assessment metadata (URL, name, description, duration, test types, etc.). The solution is evaluated using Mean Recall@K, measuring how many relevant assessments are retrieved in the top K recommendations.

## Solution Architecture

### Data Pipeline

**1. Data Ingestion**
- Scraped SHL's product catalog (`https://www.shl.com/solutions/products/product-catalog/`) using BeautifulSoup
- Extracted assessment metadata: name, URL, description, test type codes, duration, remote testing support, adaptive IRT support
- Stored structured data in CSV format (`data/shl_catelog.csv`)

**2. Embedding & Indexing**
- Converted each assessment to a LangChain `Document` with description and semantic categories in `page_content`
- Stored rich metadata for each assessment:
  - Raw SHL test-type codes (A, B, C, D, E, K, P, S)
  - Boolean flags like `is_type_K`, `is_type_P`, etc. to enable intent-aware filtering
  - Duration, remote-testing, and adaptive-IRT support
- Generated embeddings using OpenAI's `text-embedding-3-large` model via OpenRouter API
- Stored embeddings and metadata in a ChromaDB vector store for efficient and intent-aware similarity search

### Recommendation Pipeline

The final system implements a 4-stage pipeline:

1. **Intent Detection**: LLM identifies which assessment domains are required (Ability & Aptitude, Competencies, Knowledge & Skills, Personality & Behaviour, Simulations, etc.)
2. **Intent-Aware Retrieval**: For each required domain, the system uses boolean metadata flags (e.g. `is_type_K`, `is_type_P`) to retrieve a diverse pool of candidates from the vector store while still respecting semantic similarity
3. **Balanced Selection**: From this pool, selects a balanced subset across the required domains while preserving similarity order
4. **LLM Scoring**: Re-ranks the balanced subset by relevance using LLM scoring to produce the final recommendations

## Optimization Journey

### Initial Approach: Simple Vector Retrieval

**Implementation**: Direct similarity search returning top K results from vector store.

**Performance**: Mean Recall@K = **0.0012** (extremely low)

**Limitations**:
- No domain awareness: Couldn't balance recommendations across different assessment types
- Homogeneous results: Top results often came from a single domain (e.g., all technical assessments)
- Poor diversity: Failed to match queries requiring multiple assessment categories
- No query understanding: Didn't infer what types of assessments the query actually needed

**Root Cause**: Vector similarity alone doesn't understand that a query like "Java developer with leadership skills" needs both technical assessments (Knowledge & Skills) and behavioral assessments (Personality & Behaviour). The system would return only the most semantically similar assessments, which were often all from one category.

### Iteration 1: LLM-Based Domain Detection

**Enhancement**: Added LLM-based query intent detection to identify required assessment domains.

**Implementation**:
- Used GPT-4o-mini via OpenRouter to analyze queries
- Prompted LLM to identify relevant domains from: Ability & Aptitude, Biodata & Situational Judgment, Competencies, Development & 360, Assessment Exercises, Knowledge & Skills, Personality & Behaviour, Simulations
- Mapped detected domains to SHL test type codes (A, B, C, D, E, K, P, S)

**Impact**: Enabled the system to understand query requirements and identify which assessment categories were needed.

### Iteration 2: Intent-Aware Retrieval & Balanced Selection

**Enhancement**: Made retrieval itself domain-aware and then applied balanced selection to ensure diversity across detected domains.

**Implementation**:
- After detecting the required domains with the LLM, map them to SHL test-type codes (A, B, C, D, E, K, P, S)
- For each required code (e.g. `K`, `P`), run a separate similarity search with metadata filters like `is_type_K=True`, `is_type_P=True` to build a diverse candidate pool
- Deduplicate candidates by URL and, if needed, top up with an unfiltered similarity search to reach the retrieval budget
- Apply balanced selection over this pool to choose up to `k // num_domains` assessments per domain while preserving similarity order
- Fill remaining slots with other relevant assessments in similarity order

**Key Design Decision**: Shifted domain-awareness earlier into the retrieval stage and then preserved similarity ranking within each domain, ensuring both high relevance and good domain coverage.

**Impact**: Significantly improved diversity and domain coverage in recommendations.

### Iteration 3: LLM Scoring & Re-ranking

**Enhancement**: Added LLM-based relevance scoring to refine final recommendations.

**Implementation**:
- For each candidate assessment, LLM scores relevance (1-5) considering:
  - Skill alignment with query
  - Match between assessment categories and query intent
  - Complementarity (avoiding duplicates)
- Re-ranks candidates by LLM scores before returning final recommendations

**Impact**: Further improved relevance by leveraging LLM's understanding of nuanced query-assessment relationships.

### Final Performance

Using the provided labeled train set, I evaluated the system with **Mean Recall@10**, as specified in the assignment:

- **Initial baseline** (simple top‑K vector retrieval, no intent detection or re‑ranking):  
  - MeanRecall@10 ≈ **0.0012**

- **Final system** (intent‑aware retrieval + balanced selection + LLM scoring):  
  - **Retrieval_mean_recall@10** ≈ **0.1244**  
  - **Final_mean_recall@10** ≈ **0.1789**

This represents more than an order‑of‑magnitude improvement over the baseline, while also producing more balanced recommendations across relevant assessment domains.

## Technical Decisions

### Why LLM-Based Domain Detection?

Traditional keyword matching or rule-based approaches fail to capture nuanced query requirements. For example, "leadership role" implicitly requires both behavioral (Personality & Behaviour) and competency assessments. LLMs excel at this implicit understanding.

### Why Balanced Selection Over Pure Ranking?

Pure similarity ranking often returns homogeneous results. Balanced selection ensures diversity while preserving relevance by maintaining similarity order within each domain.

### Why Two-Stage LLM Usage?

1. **Domain Detection**: Understands "what types of assessments are needed"
2. **Scoring**: Understands "which specific assessments are most relevant"

This separation allows each LLM call to focus on a specific task, improving both accuracy and efficiency.

### URL Normalization for Evaluation

Implemented robust URL normalization to handle format differences:
- Removes `/solutions/` path variations
- Normalizes http/https
- Removes query parameters and trailing slashes

This ensures fair evaluation when comparing ground truth URLs with predicted URLs.

## Evaluation Methodology

Implemented comprehensive evaluation using Mean Recall@K:
- **Retrieval Stage**: Measures how many relevant assessments appear in top 20 retrieved
- **Final Stage**: Measures how many relevant assessments appear in top 7 recommendations
- Evaluates at multiple K values (5, 7, 10) to understand performance across different cutoffs

The evaluation pipeline handles variable numbers of relevant URLs per query, correctly computing Recall@K as (hits in top K) / (total relevant URLs).

## Conclusion

The evolution from simple vector retrieval to a multi-stage LLM-enhanced pipeline resulted in a **10-15x improvement** in Mean Recall@K. Key success factors:

1. **Domain awareness**: Understanding what types of assessments queries require
2. **Balanced diversity**: Ensuring recommendations span multiple relevant categories
3. **LLM refinement**: Using LLMs for both intent detection and relevance scoring
4. **Preserving relevance**: Maintaining similarity-based ranking while achieving diversity

The final system successfully balances relevance, diversity, and domain coverage, making it practical for real-world HR assessment recommendation scenarios.
