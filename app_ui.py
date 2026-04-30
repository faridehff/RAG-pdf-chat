import streamlit as st
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.title("📄 Chat with your PDF (Local AI)")

# load model once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# load PDF
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# split
def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# embed
def embed(texts):
    return model.encode(texts)

# index
def create_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# search
def search(query, chunks, index):
    q_emb = model.encode([query])
    D, I = index.search(q_emb, k=3)
    return [chunks[i] for i in I[0]]

# ollama
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

uploaded_file = st.file_uploader("Upload a PDF")

if uploaded_file:
    text = load_pdf(uploaded_file)
    chunks = split_text(text)

    embeddings = embed(chunks)
    index = create_index(embeddings)

    query = st.text_input("Ask a question about the PDF")

    if query:
        results = search(query, chunks, index)
        context = "\n".join(results)

        prompt = f"""
Answer ONLY using the context below:

{context}

Question:
{query}
"""
        answer = ask_ollama(prompt)
        st.write(answer)