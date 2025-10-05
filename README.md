🧠 AI-Powered RAG Chatbot

A self-updating, document-aware conversational assistant built end-to-end with OpenAI, Python, and smart caching.

🚀 Overview

This project is part of my AI App Builder program, where I’m mastering end-to-end development of LLM-powered apps.

It’s a Retrieval-Augmented Generation (RAG) Chatbot that can:

💬 Chat conversationally like ChatGPT

📚 Answer questions based on your own documents (TXT, Markdown, PDF)

🔁 Auto-detect & re-embed changed files (no manual refresh)

💾 Maintain conversation memory between turns

🧠 Auto-summarize long chats to save tokens

🛡️ Filter unsafe prompts with moderation

📊 Log all queries, answers, and usage metrics

🧩 Architecture
data/
 ├── docs/            # Your knowledge base (.txt, .md, .pdf)
 ├── cache/           # Per-file embedding caches
 ├── index.jsonl      # Global chunk metadata
 └── vectors.npy      # Global embedding matrix
api/
 ├── retriever.py     # Semantic search via cosine similarity
 ├── rag_cli.py       # One-shot RAG Q&A
 ├── chat_cli.py      # Basic multi-turn chat
 └── rag_chatbot.py   # Full RAG Chatbot (auto-refresh + memory)
logs/
 ├── rag_chat_history.json
 └── dayXX-*.jsonl

⚙️ Features
Feature	Description
🧩 RAG (Retrieval Augmented Generation)	Retrieves most relevant document chunks before every answer
🔁 Smart Cache	Reuses embeddings for unchanged files, re-embeds only modified ones
📄 PDF Support	Extracts text automatically using pypdf
💬 Conversational Memory	Maintains multi-turn chat context with summarization
🧱 Local Vector Index	Built with NumPy — no external vector DB required
🛡️ Moderation & Logging	Filters unsafe content and records all chat data
⚡ Fast Startup	Scans & validates docs in seconds, even for large folders
🛠️ Setup
1️⃣ Clone the repo
git clone https://github.com/<your-username>/ai-app-builder.git
cd ai-app-builder

2️⃣ Install dependencies
pip install -r requirements.txt


(Ensure you have openai>=1.40.0 and pypdf>=4.0.0.)

3️⃣ Add your documents

Place .txt, .md, or .pdf files in:

data/docs/

4️⃣ Add your OpenAI API key

Create an .env file or set the environment variable:

export OPENAI_API_KEY="sk-..."


(On Windows PowerShell:)

$env:OPENAI_API_KEY="sk-..."

5️⃣ Run the chatbot
python api/rag_chatbot.py


You’ll see:

• Reused cache (5 chunks) → data/docs/faq.txt
• Embedded 3 chunks → data/docs/policy.pdf
✅ Embedding index up-to-date (8 vectors)
🧠 RAG Chatbot (auto-updates embeddings; PDF-ready)

💬 Example Session
You: What is the new lunch policy?
AI: Employees now get one hour for lunch as per the updated policy. [Source: policy.pdf]

You: What about weekends?
AI: The lunch policy applies only on weekdays. [Source: policy.pdf]

🧠 How It Works (Under the Hood)
Step	Description
1️⃣ Ingestion	On startup, reads all files → hashes → detects changes
2️⃣ Embedding	Generates semantic vectors using text-embedding-3-small
3️⃣ Indexing	Stores vectors in vectors.npy and metadata in index.jsonl
4️⃣ Retrieval	At each query, finds top-k similar chunks via cosine similarity
5️⃣ Response	Builds augmented prompt (context + question) → GPT → answer
6️⃣ Memory	Saves full chat and auto-summarizes when too long
🧾 Logs

Every conversation turn is logged in:

logs/day11-ragchat.jsonl


Example log entry:

{
  "ts": "2025-10-05T10:21:43Z",
  "user": "What is the refund policy?",
  "assistant": "Refunds are processed within 7 business days.",
  "sources": ["data/docs/policy.pdf"],
  "usage": {"prompt_tokens": 520, "completion_tokens": 65}
}

✨ Author
Varun Shetty
Building AI-powered apps that combine data, design, and intelligence.