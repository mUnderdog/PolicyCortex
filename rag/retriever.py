import os
import json
import streamlit as st

_DIR = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(_DIR, "security_index.faiss")
meta_path = os.path.join(_DIR, "security_metadata.json")

@st.cache_resource(show_spinner="🧠 Loading embedding model... (Initial load may take 30-60s)")
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="📂 Loading security index...")
def get_security_resources():
    import faiss
    index = faiss.read_index(index_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    return index, metadata

def retrieve_docs(query, k=3):
    model = get_embedding_model()
    index, metadata = get_security_resources()

    query_vector = model.encode([query])

    distances, indices = index.search(query_vector, k)

    results = []
    for i, idx in enumerate(indices[0]):
        doc = metadata[idx]
        results.append({
            "control": doc["control"],
            "source": doc["source"],
            "content": doc["content"],
            "score": float(distances[0][i])
        })
    return results
