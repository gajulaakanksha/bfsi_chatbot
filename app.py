"""BFSI Call Center AI Assistant – Streamlit Chat UI.

Provides a conversational chat interface that routes queries through
the 3-tier LangGraph pipeline (Dataset Match → SLM → RAG).
"""
import os
import sys
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Add project root to Python path
sys.path.insert(0, os.path.dirname(__file__))


# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="BFSI Call Center Assistant",
    page_icon="🏦",
    layout="centered",
)


# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .main-header h1 {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .main-header p {
        color: #a0aec0;
        font-size: 0.95rem;
    }
    .tier-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .tier-dataset {
        background: rgba(72, 187, 120, 0.2);
        color: #48bb78;
        border: 1px solid rgba(72, 187, 120, 0.3);
    }
    .tier-slm {
        background: rgba(66, 153, 225, 0.2);
        color: #4299e1;
        border: 1px solid rgba(66, 153, 225, 0.3);
    }
    .tier-rag {
        background: rgba(159, 122, 234, 0.2);
        color: #9f7aea;
        border: 1px solid rgba(159, 122, 234, 0.3);
    }
    .tier-guardrail {
        background: rgba(245, 101, 101, 0.2);
        color: #f56565;
        border: 1px solid rgba(245, 101, 101, 0.3);
    }
    .sidebar-info {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏦 BFSI Call Center Assistant</h1>
    <p>AI-powered banking, financial services &amp; insurance support</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.markdown("**Model:** TinyLlama-1.1B-Chat")
    st.markdown("**Quantisation:** 4-bit (QLoRA)")
    st.markdown("**Embedding:** all-MiniLM-L6-v2")
    st.markdown("**Vector Store:** ChromaDB")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 📊 Response Tiers")
    st.markdown("""
    <div class="sidebar-info">
        <span class="tier-badge tier-dataset">Tier 1</span> Dataset Match<br>
        <span class="tier-badge tier-slm">Tier 2</span> SLM Generation<br>
        <span class="tier-badge tier-rag">Tier 3</span> RAG Augmented<br>
        <span class="tier-badge tier-guardrail">Filtered</span> Guardrail Block
    </div>
    """, unsafe_allow_html=True)

    show_debug = st.checkbox("Show debug info", value=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Lazy-load pipeline components ────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models... (first time takes ~60s)")
def load_pipeline():
    """Load all pipeline components once and cache them."""
    from src.dataset_matcher import DatasetMatcher
    from src.guardrails import Guardrails
    from src.pipeline import BFSIPipeline

    dataset_matcher = DatasetMatcher()

    # Try loading SLM and RAG — they may not be available yet
    slm_engine = None
    rag_engine = None

    try:
        from src.slm_engine import SLMEngine
        slm_engine = SLMEngine(use_lora=True)
    except Exception as e:
        st.warning(f"SLM not loaded (model may not be downloaded yet): {e}")

    try:
        from src.rag_engine import RAGEngine
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
        if os.path.isdir(chroma_dir):
            rag_engine = RAGEngine()
        else:
            st.info("RAG vector store not built yet. Run `scripts/build_vectorstore.py`.")
    except Exception as e:
        st.warning(f"RAG not loaded: {e}")

    guardrails = Guardrails()
    pipeline = BFSIPipeline(dataset_matcher, slm_engine, rag_engine, guardrails)
    return pipeline


pipeline = load_pipeline()


# ── Chat History ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "tier" in msg and show_debug:
            tier = msg["tier"]
            badge_class = f"tier-{tier}"
            tier_labels = {
                "dataset": "Tier 1 · Dataset Match",
                "slm": "Tier 2 · SLM Generation",
                "rag": "Tier 3 · RAG Augmented",
                "guardrail": "Guardrail Filtered",
            }
            label = tier_labels.get(tier, tier)
            st.markdown(
                f'<span class="tier-badge {badge_class}">{label}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])


# ── Handle user input ────────────────────────────────────────────────
if prompt := st.chat_input("Ask a banking, finance, or insurance question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = time.time()
            result = pipeline.run(prompt)
            elapsed = time.time() - start

        tier = result.get("tier_used", "unknown")
        response = result.get("response", "I'm sorry, I couldn't process your query.")

        # Show tier badge
        if show_debug:
            badge_class = f"tier-{tier}"
            tier_labels = {
                "dataset": "Tier 1 · Dataset Match",
                "slm": "Tier 2 · SLM Generation",
                "rag": "Tier 3 · RAG Augmented",
                "guardrail": "Guardrail Filtered",
            }
            label = tier_labels.get(tier, tier)
            st.markdown(
                f'<span class="tier-badge {badge_class}">{label}</span>',
                unsafe_allow_html=True,
            )

        st.markdown(response)

        # Debug expander
        if show_debug:
            with st.expander("🔍 Debug Details"):
                st.json({
                    "tier_used": tier,
                    "dataset_score": round(result.get("dataset_score", 0), 4),
                    "rag_score": round(result.get("rag_score", 0), 4),
                    "response_time_sec": round(elapsed, 2),
                })

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "tier": tier,
    })
