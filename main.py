# app.py
import streamlit as st
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss
import numpy as np
import pickle
import requests
import io
import os
from groq import Groq
from typing import List, Dict

st.set_page_config(page_title="RAG Chat (Groq + FAISS)", layout="wide")

# --------- Helpers & cached resources ----------
@st.cache_resource(show_spinner=False)
def load_embedder(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    if len(text) <= chunk_size:
        return [text.strip()]
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def build_index_from_uploaded(uploaded_files, chunk_size=800, overlap=200):
    embedder = load_embedder()
    chunks_meta = []  # list of dicts: {"text", "source"}
    for f in uploaded_files:
        name = f.name
        if name.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(f.read()))
            text = ""
            for p in reader.pages:
                txt = p.extract_text() or ""
                text += txt + "\n"
            pieces = chunk_text(text, chunk_size, overlap)
            chunks_meta += [{"text": p, "source": name} for p in pieces]
        elif name.lower().endswith(".txt"):
            text = f.read().decode("utf-8")
            pieces = chunk_text(text, chunk_size, overlap)
            chunks_meta += [{"text": p, "source": name} for p in pieces]
        else:
            # try reading as text fallback
            try:
                text = f.read().decode("utf-8")
                pieces = chunk_text(text, chunk_size, overlap)
                chunks_meta += [{"text": p, "source": name} for p in pieces]
            except:
                continue

    if not chunks_meta:
        return None

    texts = [c["text"] for c in chunks_meta]
    embs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs_norm = embs / norms
    dim = embs_norm.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embs_norm.astype("float32"))

    return {"index": index, "chunks": chunks_meta}

def retrieve_topk(session_index, query: str, k: int = 3):
    embedder = load_embedder()
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    D, I = session_index.search(q_emb.astype("float32"), k)
    results = []
    for score, idx in zip(D[0], I[0]):
        idx = int(idx)
        results.append({"text": st.session_state["chunks"][idx]["text"],
                        "source": st.session_state["chunks"][idx]["source"],
                        "score": float(score)})
    return results

def generate_llm_response(prompt: str, api_key: str, model: str = "llama3-70b-8192", temperature: float = 0.0, max_tokens: int = 512):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    j = resp.json()
    return j["choices"][0]["message"]["content"]

def save_index_to_disk(prefix="faiss_store"):
    faiss.write_index(st.session_state["index"], f"{prefix}.index")
    with open(f"{prefix}_chunks.pkl", "wb") as f:
        pickle.dump(st.session_state["chunks"], f)
    return f"{prefix}.index", f"{prefix}_chunks.pkl"

def load_index_from_disk(prefix="faiss_store"):
    idx_file = f"{prefix}.index"
    chunks_file = f"{prefix}_chunks.pkl"
    if os.path.exists(idx_file) and os.path.exists(chunks_file):
        index = faiss.read_index(idx_file)
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
        st.session_state["index"] = index
        st.session_state["chunks"] = chunks
        return True
    return False

# --------- Sidebar: settings ----------
with st.sidebar:
    st.header("Settings")
    groq_key = st.text_input("GROQ API Key", type="password")
    model = st.selectbox("Model", ["llama3-70b-8192", "mixtral-8x7b"], index=0)
    top_k = st.slider("Top-k retrieval", 1, 10, 3)
    chunk_size = st.number_input("Chunk size (chars)", value=800, min_value=200, max_value=4000, step=100)
    overlap = st.number_input("Chunk overlap (chars)", value=200, min_value=0, max_value=chunk_size-50, step=50)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
    max_tokens = st.slider("Max tokens (response)", 64, 2048, 512, step=64)

    st.markdown("---")
    if st.button("Load existing index from disk"):
        ok = load_index_from_disk()
        if ok:
            st.success("Loaded index + chunks from disk.")
        else:
            st.error("No saved index found.")

    if st.button("Save index to disk"):
        if "index" in st.session_state and "chunks" in st.session_state:
            idxp, cp = save_index_to_disk()
            st.success(f"Saved: {idxp}, {cp}")
        else:
            st.error("No index in session to save.")

# --------- Main layout ----------
st.title("RAG Chat — Streamlit UI (FAISS + Groq)")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload documents")
    uploaded = st.file_uploader("Upload PDFs or TXT (multi)", type=["pdf", "txt"], accept_multiple_files=True)
    if uploaded and st.button("Build / Rebuild index"):
        build_result = build_index_from_uploaded(uploaded, chunk_size=chunk_size, overlap=overlap)
        if build_result is None:
            st.warning("No text extracted from uploaded files.")
        else:
            st.session_state["index"] = build_result["index"]
            st.session_state["chunks"] = build_result["chunks"]
            st.success(f"Built index with {len(st.session_state['chunks'])} chunks.")

    st.markdown("---")
    st.subheader("Index status")
    if "index" in st.session_state:
        st.write(f"Indexed chunks: {len(st.session_state['chunks'])}")
    else:
        st.write("No index in session. Build or load one.")

    st.markdown("---")
    st.subheader("Controls")
    show_sources = st.checkbox("Show retrieved source chunks after answer", value=True)
    show_prompt = st.checkbox("Show final prompt sent to LLM", value=False)

with col2:
    st.subheader("Chat / Query")
    if "history" not in st.session_state:
        st.session_state["history"] = []

    query = st.text_input("Ask a question", key="query_input")
    query_submitted = st.button("Submit")

    if query_submitted and query:
        if "index" not in st.session_state:
            st.error("No index available. Upload and build index first.")
        elif not groq_key:
            st.error("Provide GROQ API key in sidebar.")
        else:
            # retrieve
            results = retrieve_topk(st.session_state["index"], query, k=top_k)
            # build prompt
            ctx_parts = []
            for i, r in enumerate(results, start=1):
                ctx_parts.append(f"[{i}] Source: {r['source']}\n{r['text']}")
            context_str = "\n\n".join(ctx_parts)
            prompt = (
                "Use ONLY the context below to answer the question. Be concise. "
                "If the answer is not present in the context, say 'Not found in documents.'\n\n"
            )

            answer = generate_llm_response(prompt, groq_key, model=model, temperature=temperature, max_tokens=max_tokens)

            entry = {"query": query, "answer": answer, "results": results, "prompt": prompt}
            st.session_state["history"].append(entry)
            st.session_state["query_input"] = ""  # clear input

    # show chat history latest-first
    for i, e in enumerate(reversed(st.session_state["history"])):
        idx = len(st.session_state["history"]) - i
        with st.container():
            st.markdown(f"**Q:** {e['query']}")
            st.markdown(f"**A:** {e['answer']}")
            cols = st.columns([1, 3])
            with cols[0]:
                if st.button(f"Regenerate #{idx}", key=f"regen_{idx}"):
                    # regenerate
                    if not groq_key:
                        st.error("Provide GROQ API key in sidebar.")
                    else:
                        new_answer = generate_llm_response(e["prompt"], groq_key, model=model, temperature=temperature, max_tokens=max_tokens)
                        e["answer"] = new_answer
                        st.experimental_rerun()
            with cols[1]:
                if show_prompt:
                    with st.expander("Final prompt sent to LLM", expanded=False):
                        st.code(e["prompt"][:5000])  # cut large prompts
                if show_sources:
                    with st.expander("Retrieved source chunks", expanded=False):
                        for j, r in enumerate(e["results"], start=1):
                            st.markdown(f"**[{j}]** — *{r['source']}*  (score={r['score']:.4f})")
                            st.write(r['text'])
            st.markdown("---")

st.markdown("### Notes")
st.markdown("- Index is stored in session memory. Use sidebar controls to save/load index to disk.")
