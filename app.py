import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. load PDF
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# 2. split text
def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# 3. embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed(texts):
    return model.encode(texts)

# 4. create vector store
def create_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# 5. search
def search(query, chunks, index):
    q_emb = model.encode([query])
    D, I = index.search(q_emb, k=3)
    return [chunks[i] for i in I[0]]

# 6. ask ollama
def ask_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# ---- RUN ----
text = load_pdf("data/sample.pdf")
chunks = split_text(text)

embeddings = embed(chunks)
index = create_index(embeddings)

while True:
    query = input("\nAsk about the PDF: ")

    results = search(query, chunks, index)
    context = "\n".join(results)

    prompt = f"""
Answer ONLY using the context below:

{context}

Question:
{query}
"""

    answer = ask_ollama(prompt)
    print("\nAnswer:\n", answer)