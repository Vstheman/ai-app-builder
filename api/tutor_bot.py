#!/usr/bin/env python3
# api/tutor_bot.py
"""
Day 7 – Tutor Bot with Persona + Auto-Summarizing Memory (+ Moderation & Logging)

Usage:
  python api/tutor_bot.py                          # start with default persona (tutor)
  python api/tutor_bot.py --role coach             # fitness coach persona
  python api/tutor_bot.py --role mentor            # career mentor persona
  python api/tutor_bot.py --role custom --system "You are a calm philosophy tutor."
  python api/tutor_bot.py --new                    # clear previous memory
  python api/tutor_bot.py --max-chars 6000 --keep 6  # adjust summarization thresholds
"""

import json
import os
import sys
import datetime
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI

# ---------- CONFIG ---------- #
LOGS_DIR = Path("logs")
CHAT_PATH = LOGS_DIR / "day07_tutor_history.json"
LOG_PATH  = LOGS_DIR / "day07-tutor.jsonl"

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.4

# Summarization trigger: if total characters of conversation (excluding system & summary) exceed this, summarize.
DEFAULT_MAX_CHARS = 4500
# How many most-recent messages (user+assistant turns combined) to keep verbatim when summarizing.
DEFAULT_KEEP_RECENT = 6

PERSONAS = {
    "tutor":  "You are a patient STEM tutor. Explain simply, use small steps, verify understanding.",
    "coach":  "You are a pragmatic fitness coach. Give concise, safe, evidence-based advice with simple plans.",
    "mentor": "You are a supportive career mentor. Provide concrete, actionable suggestions and examples."
}

# ---------- UTILS ---------- #
def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

def usage_to_dict(usage_obj):
    if not usage_obj:
        return None
    if isinstance(usage_obj, dict):
        return {
            "prompt_tokens": usage_obj.get("prompt_tokens"),
            "completion_tokens": usage_obj.get("completion_tokens"),
            "total_tokens": usage_obj.get("total_tokens"),
        }
    # object-like (SDK)
    return {
        "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
        "completion_tokens": getattr(usage_obj, "completion_tokens", None),
        "total_tokens": getattr(usage_obj, "total_tokens", None),
    }

def total_chars(messages: List[Dict[str, str]]) -> int:
    return sum(len(m.get("content", "")) for m in messages if m.get("role") in ("user", "assistant"))

def load_history(system_prompt: str) -> List[Dict[str, str]]:
    if CHAT_PATH.exists():
        try:
            with CHAT_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    return data
        except Exception:
            pass
    # Fresh conversation with persona system prompt and empty summary note
    return [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": "Conversation summary (start): none yet."}
    ]

def save_history(history: List[Dict[str, str]]):
    CHAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHAT_PATH.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def append_log(record: Dict[str, Any]):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ---------- SAFETY (MODERATION) ---------- #
def check_moderation(client: OpenAI, text: str) -> bool:
    """True if safe, False if flagged."""
    try:
        result = client.moderations.create(model="omni-moderation-latest", input=text)
        return not result.results[0].flagged
    except Exception:
        # If moderation fails, don’t block the flow: treat as safe but log
        append_log({"ts": now_iso(), "event": "moderation_failed"})
        return True

# ---------- SUMMARIZATION ---------- #
def summarize_history(client: OpenAI, history: List[Dict[str, str]], keep_recent: int) -> List[Dict[str, str]]:
    """
    Summarize older messages into a compact system note, keep last `keep_recent` messages verbatim.
    Returns a new history list: [system persona, system summary, ...recent messages]
    """
    if len(history) < 4:
        return history

    # Identify fixed system persona (first item) and current summary (second item)
    persona_msg = history[0]
    # history[1] is summary message; keep it but we’ll update its content.
    recent = history[-keep_recent:] if keep_recent > 0 else []

    # Select everything after the two system messages, excluding the recent window.
    chunk_to_summarize = history[2: max(2, len(history) - keep_recent)]
    if not chunk_to_summarize:
        return history

    # Build a compact text transcript for summarization
    plain = []
    for m in chunk_to_summarize:
        role = m.get("role", "")
        content = m.get("content", "")
        if role in ("user", "assistant"):
            plain.append(f"{role.upper()}: {content}")
    transcript = "\n".join(plain)

    prompt_summary = (
        "You are compressing a chat history into a concise memory for a tutor bot.\n"
        "Summarize ONLY the durable facts, user preferences, goals, definitions created, and pending tasks.\n"
        "Keep it under 180 words. No filler. Use bullet points where helpful."
    )

    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": prompt_summary},
            {"role": "user", "content": transcript}
        ],
    )
    summary = (resp.choices[0].message.content or "").strip()
    # Rebuild history: persona system + new summary + recent messages
    new_history = [
        persona_msg,
        {"role": "system", "content": f"Conversation summary (auto):\n{summary}"}
    ] + recent
    return new_history

# ---------- CHAT LOOP ---------- #
def chat_loop(role: str, custom_system: str, model: str, temperature: float,
              max_chars: int, keep_recent: int):
    client = OpenAI()

    # Build persona/system prompt
    if role == "custom":
        system_prompt = custom_system or PERSONAS["tutor"]
    else:
        system_prompt = PERSONAS.get(role, PERSONAS["tutor"])

    history = load_history(system_prompt)

    print("\n🧑‍🏫 Tutor Bot (type 'exit' to quit, 'clear' to reset memory)\n"
          f"Persona: {role}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Ending session.")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("👋 Ending session.")
            break
        if user_input.lower() == "clear":
            history = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": "Conversation summary (start): none yet."}
            ]
            save_history(history)
            print("🧹 Memory cleared.")
            continue

        # Safety first
        if not check_moderation(client, user_input):
            print("⚠️ Message blocked by moderation.")
            append_log({"ts": now_iso(), "event": "blocked_by_moderation", "user": user_input})
            continue

        # Add user message
        history.append({"role": "user", "content": user_input})

        # Summarize if conversation is getting large
        if total_chars(history[2:]) > max_chars:  # exclude the two system messages
            try:
                history = summarize_history(client, history, keep_recent=keep_recent)
                save_history(history)
                print("📝 (Context summarized to stay within budget.)")
            except Exception as e:
                append_log({"ts": now_iso(), "event": "summary_failed", "error": str(e)})

        # Ask model with full (possibly summarized) context
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=history,
            )
            answer = (resp.choices[0].message.content or "").strip()
            print(f"AI: {answer}\n")

            history.append({"role": "assistant", "content": answer})
            save_history(history)

            append_log({
                "ts": now_iso(),
                "role": role,
                "user": user_input,
                "assistant": answer,
                "usage": usage_to_dict(getattr(resp, "usage", None))
            })

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"❌ Error: {err}\n")
            append_log({"ts": now_iso(), "event": "error", "error": err})

# ---------- ENTRY ---------- #
def main():
    import argparse
    p = argparse.ArgumentParser(description="Tutor Bot with persona + summarizing memory.")
    p.add_argument("--role", choices=["tutor", "coach", "mentor", "custom"], default="tutor",
                   help="Persona/voice to use.")
    p.add_argument("--system", default="", help="Custom system prompt if --role custom.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                   help="Summarize when convo char count exceeds this (excludes system messages).")
    p.add_argument("--keep", type=int, default=DEFAULT_KEEP_RECENT,
                   help="How many recent messages to keep verbatim during summarization.")
    p.add_argument("--new", action="store_true", help="Start a fresh chat (clear memory).")
    args = p.parse_args()

    if args.new and CHAT_PATH.exists():
        try:
            CHAT_PATH.unlink()
            print("🆕 Started a new session (memory cleared).")
        except Exception:
            print("ℹ️ Could not delete old history; will overwrite on next save.")

    chat_loop(
        role=args.role,
        custom_system=args.system,
        model=args.model,
        temperature=args.temperature,
        max_chars=args.max_chars,
        keep_recent=args.keep
    )

if __name__ == "__main__":
    main()
