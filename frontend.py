"""
SHL Assessment Recommendation System - Frontend
A Streamlit web application for testing the recommendation system.
"""

import streamlit as st
import requests
from typing import List, Dict, Optional
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .assessment-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .assessment-name {
        font-size: 1.3rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .badge-success {
        background-color: #4caf50;
        color: white;
    }
    .badge-danger {
        background-color: #f44336;
        color: white;
    }
    .badge-info {
        background-color: #2196f3;
        color: white;
    }
    .description {
        color: #555;
        line-height: 1.6;
        margin-top: 0.75rem;
    }
    .stats-container {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
    .stat-item {
        flex: 1;
        text-align: center;
        padding: 0.5rem;
        background-color: #f0f0f0;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = st.sidebar.text_input(
    "API Base URL",
    value=os.getenv("API_BASE_URL"),
    help="Enter the base URL of your FastAPI server"
)

USE_API = st.sidebar.checkbox(
    "Use API Endpoint",
    value=True,
    help="If checked, uses the FastAPI endpoint. Otherwise, calls retriever directly."
)


def call_api_recommend(query: str) -> Optional[Dict]:
    """Call the FastAPI recommendation endpoint."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/recommend",
            json={"query": query},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def call_direct_recommend(query: str) -> Optional[Dict]:
    """Call the retriever function directly."""
    try:
        from rag.retriever import recommend
        return recommend(query)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def format_duration(duration: Optional[int]) -> str:
    """Format duration in minutes."""
    if duration is None:
        return "N/A"
    return f"{duration} min"


def render_assessment_card(assessment: Dict, index: int):
    """Render a single assessment card using clean Streamlit components (no raw HTML dump)."""
    name = assessment.get("name", "Unknown Assessment")
    url = assessment.get("url", "#")
    description = assessment.get("description", "No description available.")
    duration = assessment.get("duration")
    adaptive = assessment.get("adaptive_support", "No")
    remote = assessment.get("remote_support", "No")
    test_types = assessment.get("test_type", [])

    # Outer card container
    with st.container():
        st.markdown(f"**{index + 1}. [{name}]({url})**")

        # Meta badges as a compact row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"- **Adaptive**: {adaptive}")
        with col2:
            st.markdown(f"- **Remote**: {remote}")
        with col3:
            st.markdown(f"- **Duration**: {format_duration(duration)}")

        # Test types
        if test_types:
            st.markdown(f"- **Test types**: {', '.join(test_types)}")

        # Description
        st.markdown("")
        st.markdown(description)


def main():
    # Header
    st.markdown('<div class="main-header">🎯 SHL Assessment Recommender</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Get personalized assessment recommendations for your hiring needs</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 About")
    st.sidebar.markdown("""
    This system uses advanced AI to recommend relevant SHL assessments 
    based on job descriptions or natural language queries.
    
    **Features:**
    - 🧠 LLM-based domain detection
    - ⚖️ Balanced selection across assessment types
    - 🎯 Relevance scoring and ranking
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Configuration")
    
    # Main input area
    st.markdown("### Enter Your Query")
    
    # Example queries
    example_queries = [
        "Need a Java developer who is good in collaborating with external teams and stakeholders",
        "Looking for a sales manager with leadership skills",
        "Hiring a data scientist with strong analytical capabilities",
        "Need a customer service representative with good communication skills"
    ]
    
    selected_example = st.selectbox(
        "Or select an example query:",
        ["Custom query"] + example_queries
    )
    
    if selected_example != "Custom query":
        default_query = selected_example
    else:
        default_query = ""
    
    query = st.text_area(
        "Job Description / Query:",
        value=default_query,
        height=100,
        placeholder="Enter a job description or natural language query. For example: 'Need a Java developer with leadership skills'"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        recommend_button = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    # Process recommendation
    if recommend_button:
        if not query.strip():
            st.warning("⚠️ Please enter a query before requesting recommendations.")
        else:
            with st.spinner("🔄 Analyzing query and generating recommendations..."):
                start_time = time.time()
                
                # Call recommendation function
                if USE_API:
                    result = call_api_recommend(query)
                else:
                    result = call_direct_recommend(query)
                
                elapsed_time = time.time() - start_time
                
                if result:
                    assessments = result.get("recommended_assessments", [])
                    
                    if assessments:
                        # Display stats
                        st.success(f"✅ Found {len(assessments)} recommendations (took {elapsed_time:.2f}s)")
                        
                        # Display recommendations
                        st.markdown("---")
                        st.markdown("### 📊 Recommended Assessments")
                        
                        for idx, assessment in enumerate(assessments):
                            render_assessment_card(assessment, idx)
                        
                        # Summary statistics
                        st.markdown("---")
                        st.markdown("### 📈 Summary Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            adaptive_count = sum(1 for a in assessments if a.get("adaptive_support") == "Yes")
                            st.metric("Adaptive Support", f"{adaptive_count}/{len(assessments)}")
                        
                        with col2:
                            remote_count = sum(1 for a in assessments if a.get("remote_support") == "Yes")
                            st.metric("Remote Support", f"{remote_count}/{len(assessments)}")
                        
                        with col3:
                            avg_duration = sum(
                                a.get("duration", 0) or 0 
                                for a in assessments 
                                if a.get("duration")
                            ) / max(1, sum(1 for a in assessments if a.get("duration")))
                            st.metric("Avg Duration", f"{avg_duration:.0f} min" if avg_duration > 0 else "N/A")
                        
                        with col4:
                            unique_types = set()
                            for a in assessments:
                                unique_types.update(a.get("test_type", []))
                            st.metric("Test Types", len(unique_types))
                    else:
                        st.warning("⚠️ No recommendations found. Try a different query.")
                else:
                    st.error("❌ Failed to get recommendations. Please check your API configuration or try again.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 2rem;'>"
        "Built with ❤️ using Streamlit | SHL Assessment Recommendation System"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

