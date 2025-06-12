import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# 🔧 Load the embedding model
# -----------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# 📂 Load the FAISS index
# -----------------------------
db_path = "vectorstore"
if not os.path.exists(db_path):
    raise FileNotFoundError(f"❌ Vectorstore not found at: {db_path}")

db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
print("✅ FAISS vectorstore loaded.")

# -----------------------------
# 🔍 Sample Query
# -----------------------------
query = "What is the deadline for submitting the TDS project?"
top_k = 5

results = db.similarity_search(query, k=top_k)

print(f"\n📌 Top {top_k} results for query:\n\"{query}\"\n")
for i, doc in enumerate(results, 1):
    print(f"--- Result #{i} ---")
    print(f"Source: {doc.metadata.get('source', 'N/A')}")
    print(f"Title: {doc.metadata.get('title', 'N/A')}")
    print(f"Content:\n{doc.page_content[:500]}...\n")  # truncate long outputs
