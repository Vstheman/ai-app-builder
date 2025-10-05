#!/usr/bin/env python3
# api/rag_chatbot.py
"""
RAG Chatbot with Conversational Memory + Auto-Embedding Refresh + PDF support

- Reads .txt, .md, .pdf from data/docs/
- Per-file cache with content-hash to avoid re-embedding unchanged files
- Conversational memory, summarization, moderation, logging

Usage:
  python api/rag_chatbot.py
"""

import os, re, sys, json, glob, hashlib, datetime, textwrap
import numpy as np
from pathlib import Path
from openai import OpenAI
from retriever import retrieve
from pypdf import PdfReader  # NEW

# ---------- CONFIG ---------- #
DATA_DIR = Path("data")
DOCS_DIR = DATA_DIR / "docs"
CACHE_DIR = DATA_DIR / "cache"
INDEX_JSONL = DATA_DIR / "index.jsonl"
VECTORS_NPY = DATA_DIR / "vectors.npy"

LOGS_DIR = Path("logs")
CHAT_HISTORY = LOGS_DIR / "rag_chat_history.json"
LOG_JSONL = LOGS_DIR / "day11-ragchat.jsonl"

MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

MAX_HISTORY_CHARS = 5000
KEEP_RECENT_TURNS = 6
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
EMBED_DIM = 1536  # text-embedding-3-small

# ---------- UTILS ---------- #
def now_iso():
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

def usage_to_dict(usage_obj):
    if not usage_obj:
        return None
    try:
        return {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
            "completion_tokens": getattr(usage_obj, "completion_tokens", None),
            "total_tokens": getattr(usage_obj, "total_tokens", None),
        }
    except Exception:
        return None

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text: str, max_chars=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    t = normalize_ws(text)
    out, i, n = [], 0, len(t)
    while i < n:
        j = min(i + max_chars, n)
        out.append(t[i:j])
        i = j - overlap if j < n else j
        if i < 0:
            i = 0
    return out

# ---------- PDF TEXT EXTRACTION ---------- #
def read_pdf_text(path: Path) -> str:
    """
    Extract text from a PDF using pypdf.
    Note: Scanned PDFs without embedded text will yield little/none.
    """
    try:
        reader = PdfReader(str(path))
        pages = []
        for pg in reader.pages:
            pages.append(pg.extract_text() or "")
        return "\n\n".join(pages)
    except Exception as e:
        # Fall back to empty string if unreadable
        return ""

# ---------- EMBEDDING REFRESH ---------- #
def embed_chunks(chunks):
    client = OpenAI()
    vecs = []
    for c in chunks:
        resp = client.embeddings.create(model=EMBED_MODEL, input=c)
        vecs.append(resp.data[0].embedding)
    return np.array(vecs, dtype=np.float32)

def cache_key(src_path: str) -> str:
    # Make windows-safe filename for cache files
    return re.sub(r"[^\w\-\.]+", "_", src_path)[-200:]

def load_file_cache(src_path: str, file_hash: str):
    key = cache_key(src_path)
    meta_path = CACHE_DIR / f"{key}.json"
    vec_path = CACHE_DIR / f"{key}.npy"
    if not meta_path.exists() or not vec_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("sha1") != file_hash or meta.get("source") != src_path:
            return None
        vecs = np.load(vec_path)
        if len(meta.get("chunks", [])) != vecs.shape[0]:
            return None
        return meta, vecs
    except Exception:
        return None

def save_file_cache(src_path: str, file_hash: str, chunks, vecs):
    key = cache_key(src_path)
    meta_path = CACHE_DIR / f"{key}.json"
    vec_path = CACHE_DIR / f"{key}.npy"
    meta = {"source": src_path, "sha1": file_hash, "chunks": chunks}
    np.save(vec_path, vecs)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

def ensure_embeddings_up_to_date():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_JSONL.parent.mkdir(parents=True, exist_ok=True)
    VECTORS_NPY.parent.mkdir(parents=True, exist_ok=True)

    # Collect sources: .txt, .md, .pdf
    items = []
    for p in glob.glob(str(DOCS_DIR / "**" / "*"), recursive=True):
        pl = p.lower()
        if pl.endswith((".txt", ".md", ".pdf")):
            path = Path(p)
            raw = path.read_bytes()
            h = sha1_bytes(raw)  # hash raw bytes (works for text & pdf)
            if pl.endswith(".pdf"):
                txt = read_pdf_text(path)
            else:
                txt = raw.decode("utf-8", errors="ignore")
            items.append((str(path), txt, h))

    all_meta, all_vecs, vec_offset = [], [], 0
    for src, txt, h in items:
        cached = load_file_cache(src, h)
        if cached:
            meta, vecs = cached
            chunks = meta["chunks"]
            print(f"• Reused cache ({len(chunks)} chunks) → {src}")
        else:
            chunks = chunk_text(txt)
            if not chunks:
                print(f"• Skipped (no extractable text) → {src}")
                vecs = np.zeros((0, EMBED_DIM), dtype=np.float32)
            else:
                vecs = embed_chunks(chunks)
                print(f"• Embedded {len(chunks)} chunks → {src}")
            save_file_cache(src, h, chunks, vecs)

        all_vecs.append(vecs)
        for k, c in enumerate(chunks):
            all_meta.append({
                "id": f"{src}::chunk{k}",
                "source": src,
                "chunk": c,
                "vector_idx": vec_offset + k
            })
        vec_offset += len(chunks)

    mat = np.vstack(all_vecs) if any(v.shape[0] for v in all_vecs) else np.zeros((0, EMBED_DIM), dtype=np.float32)
    np.save(VECTORS_NPY, mat)
    with INDEX_JSONL.open("w", encoding="utf-8") as f:
        for m in all_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"✅ Embedding index up-to-date ({mat.shape[0]} vectors)\n")

# ---------- RAG CHAT SECTION ---------- #
def check_moderation(client, text):
    try:
        result = client.moderations.create(model="omni-moderation-latest", input=text)
        return not result.results[0].flagged
    except Exception:
        return True

def load_history():
    if CHAT_HISTORY.exists():
        try:
            return json.loads(CHAT_HISTORY.read_text(encoding="utf-8"))
        except Exception:
            pass
    return [{"role": "system", "content": "You are a helpful assistant that answers using the user’s documents."}]

def save_history(history):
    CHAT_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    CHAT_HISTORY.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

def append_log(record):
    LOG_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with LOG_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def total_chars(history):
    return sum(len(m.get("content", "")) for m in history if m["role"] != "system")

def summarize_history(client, history):
    if len(history) <= KEEP_RECENT_TURNS + 2:
        return history
    persona = history[0]
    recent = history[-KEEP_RECENT_TURNS:]
    old = history[1:-KEEP_RECENT_TURNS]
    plain = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in old if m["role"] in ("user", "assistant")
    )
    prompt = "Summarize this conversation to preserve facts, goals, and context in under 150 words."
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.2,
        messages=[{"role":"system","content":prompt}, {"role":"user","content":plain}]
    )
    summary = (resp.choices[0].message.content or "").strip()
    return [persona, {"role": "system", "content": f"Summary memory: {summary}"}] + recent

def chat_loop():
    client = OpenAI()
    ensure_embeddings_up_to_date()
    history = load_history()
    print("\n🧠 RAG Chatbot (auto-updates embeddings; PDF-ready)\nType 'exit' to quit, 'clear' to reset memory.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Ending session.")
            break
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye.")
            break
        if user_input.lower() == "clear":
            history = [{"role": "system", "content": "You are a helpful assistant that answers using the user’s documents."}]
            save_history(history)
            print("🧹 Memory cleared.")
            continue

        if not check_moderation(client, user_input):
            print("⚠️ Message blocked by moderation API.")
            continue

        contexts = retrieve(user_input, k=3)
        ctx = "\n\n".join(
            f"[Source {i+1}: {c['source']}] {c['chunk']}"
            for i, c in enumerate(contexts)
        ) if contexts else "(No relevant documents found.)"

        augmented = f"Context:\n{ctx}\n\nQuestion: {user_input}"
        history.append({"role": "user", "content": augmented})

        if total_chars(history) > MAX_HISTORY_CHARS:
            history = summarize_history(client, history)
            print("📝 (Older chat summarized.)")

        try:
            resp = client.chat.completions.create(model=MODEL, temperature=0.4, messages=history)
            answer = resp.choices[0].message.content.strip()
            print(f"\nAI: {answer}\n")
            history.append({"role": "assistant", "content": answer})
            save_history(history)
            append_log({
                "ts": now_iso(),
                "user": user_input,
                "assistant": answer,
                "sources": [c["source"] for c in contexts] if contexts else [],
                "usage": usage_to_dict(getattr(resp, "usage", None))
            })
        except Exception as e:
            print(f"❌ Error: {e}")
            break

def main():
    chat_loop()

if __name__ == "__main__":
    main()
