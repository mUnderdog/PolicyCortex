import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_DIR = os.path.dirname(os.path.abspath(__file__))          # .../rag/
_ROOT = os.path.dirname(_DIR)                              # .../PolicyCortex/

# Load embedding model (GPU if available)
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded")

# Load knowledge base
with open(os.path.join(_ROOT, "knowledge_base", "cis_controls.json"), "r") as f:
    docs = json.load(f)

texts = []
metadata = []

for doc in docs:
    text = f"{doc['control']} - {doc['content']}"
    texts.append(text)
    metadata.append(doc)

print(f"Loaded {len(texts)} documents")

# Generate embeddings
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index
faiss.write_index(index, os.path.join(_DIR, "security_index.faiss"))

# Save metadata
with open(os.path.join(_DIR, "security_metadata.json"), "w") as f:
    json.dump(metadata, f)

print("Vector database built successfully")