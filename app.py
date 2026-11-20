"""
Streamlit Web Interface for the Transformer Research Paper RAG Assistant.
Provides a fast, clean, minimalist, page-aware RAG chat experience.
"""

import os
import streamlit as st
from dotenv import load_dotenv

import rag

# Page configuration
st.set_page_config(
    page_title="Transformer Research Paper Assistant",
    page_icon="📄",
    layout="wide"
)

# Load environment configuration
load_dotenv()

# Initialize session state for chat and settings
if "messages" not in st.session_state:
    st.session_state.messages = []

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")

# Initialize vector store once on startup
@st.cache_resource(show_spinner="Initializing vector store index...")
def get_index():
    return rag.initialize_or_update_index()

collection, index_stats = get_index()

# Sidebar: System Status & Configuration
with st.sidebar:
    st.header("⚙️ System Status")
    
    st.markdown(f"**PDF Corpus:** `transformer_rag_research_paper_50p.pdf`")
    st.markdown(f"**Indexed Chunks:** `{index_stats.get('chunks_count', 50)}`")
    st.markdown(f"**Vector Store:** `ChromaDB (Persistent)`")
    st.markdown(f"**Embeddings:** `all-MiniLM-L6-v2` (Local)")
    st.markdown(f"**LLM Model:** `{os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}`")

    st.divider()

    st.subheader("🔑 Groq API Configuration")
    user_key_input = st.text_input(
        "Groq API Key",
        value=st.session_state.groq_api_key,
        type="password",
        help="Enter your Groq API Key (gsk_...) or set GROQ_API_KEY in your .env file."
    )
    if user_key_input != st.session_state.groq_api_key:
        st.session_state.groq_api_key = user_key_input
        os.environ["GROQ_API_KEY"] = user_key_input
        st.success("API Key updated for current session!")

    if not st.session_state.groq_api_key or st.session_state.groq_api_key == "your_groq_api_key_here":
        st.warning("⚠️ No Groq API key set. Provide one above or in `.env` to enable answer generation.")
    else:
        st.success("✅ Groq API key configured.")

    st.divider()

    # Re-indexing button
    if st.button("🔄 Force Re-index Corpus"):
        with st.spinner("Re-indexing PDF corpus into ChromaDB..."):
            col, new_stats = rag.initialize_or_update_index(force_reindex=True)
            st.success(f"Re-indexed {new_stats['chunks_count']} chunks successfully!")

    # Clear chat button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("🧪 Assessment Questions")
    st.caption("Click to test any of the 5 benchmark evaluation questions:")

    test_queries = [
        ("Q1: Pre-training Hyperparameters", "What exact learning rate, number of warmup steps, and Adam optimizer beta values were used in the primary pre-training run described in Section 4?"),
        ("Q2: OpenAI CEO (Out of corpus)", "Who is the current Chief Executive Officer of OpenAI?"),
        ("Q3: EM Accuracy Improvement", "By what percentage did the proposed hybrid retrieval architecture improve Exact Match (EM) accuracy on the benchmark dataset compared with the dense-only baseline?"),
        ("Q4: 13B GPU Cluster Config", "What GPU cluster configuration (number and model of GPUs) and total training duration, in days, were required to train the 13-billion-parameter model?"),
        ("Q5: Oppenheimer Revenue (Out of corpus)", "What was the worldwide box-office revenue of the film Oppenheimer in 2023?")
    ]

    selected_query = None
    for label, query_text in test_queries:
        if st.button(label, use_container_width=True):
            selected_query = query_text

# Main Chat Interface
st.title("📄 Transformer Research Paper Assistant")
st.markdown(
    "*Fast, page-aware, document-grounded research assistant strictly bounded to "
    "`transformer_rag_research_paper_50p.pdf`.*"
)

# Display existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display page sources if available
        if msg.get("sources"):
            sources_str = ", ".join(f"Page {s}" for s in msg["sources"])
            st.markdown(f"📍 **Supporting Sources:** `{sources_str}`")
        elif msg["role"] == "assistant" and msg["content"] == "[not found]":
            st.caption("📍 **Supporting Sources:** `[]` (Insufficient evidence in document)")
        
        # Display expandable retrieved context viewer
        if msg.get("retrieved_chunks"):
            with st.expander("🔍 Retrieved Context Excerpts & Similarity Scores"):
                for idx, chunk in enumerate(msg["retrieved_chunks"]):
                    sim_pct = f"{chunk['similarity']:.4f}"
                    st.markdown(
                        f"**Rank {idx + 1}** — Page `{chunk['page']}` "
                        f"(Cosine Sim: `{sim_pct}`) | Chunk ID: `{chunk['chunk_id']}`"
                    )
                    st.text(chunk["text"])
                    st.markdown("---")

# Handle input from chat input or test question buttons
user_input = st.chat_input("Ask a question about the Transformer research paper...")
active_prompt = selected_query if selected_query else user_input

if active_prompt:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    # Generate answer with loading indicator
    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and verifying grounding..."):
            # Ensure API key in session is set in environment if provided
            if st.session_state.groq_api_key:
                os.environ["GROQ_API_KEY"] = st.session_state.groq_api_key

            response = rag.answer(active_prompt, return_debug=True)
            answer_text = response.get("answer", "[not found]")
            sources = response.get("sources", [])
            retrieved_chunks = response.get("retrieved_chunks", [])

            # Render assistant answer
            st.markdown(answer_text)

            # Render supporting sources
            if sources:
                sources_str = ", ".join(f"Page {s}" for s in sources)
                st.markdown(f"📍 **Supporting Sources:** `{sources_str}`")
            elif answer_text == "[not found]":
                st.caption("📍 **Supporting Sources:** `[]` (Insufficient evidence in document)")

            # Render expandable debug context viewer
            if retrieved_chunks:
                with st.expander("🔍 Retrieved Context Excerpts & Similarity Scores"):
                    for idx, chunk in enumerate(retrieved_chunks):
                        sim_pct = f"{chunk['similarity']:.4f}"
                        st.markdown(
                            f"**Rank {idx + 1}** — Page `{chunk['page']}` "
                            f"(Cosine Sim: `{sim_pct}`) | Chunk ID: `{chunk['chunk_id']}`"
                        )
                        st.text(chunk["text"])
                        st.markdown("---")

    # Save to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks
    })
