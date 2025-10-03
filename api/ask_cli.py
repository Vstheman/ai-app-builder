#!/usr/bin/env python3
# api/ask_cli.py
"""
Day 5 – Ask Anything CLI with Prompt Patterns

Patterns supported:
  role      → Sets the model role/persona
  json      → Forces JSON output
  chain     → Encourages step-by-step reasoning
  fewshot   → Provides in-context examples
  guardrail → Adds safe fallback behavior

Usage examples:
  python api/ask_cli.py --prompt "Explain compound interest" --pattern role
  python api/ask_cli.py --prompt "Summarize benefits of yoga" --pattern json
  python api/ask_cli.py --prompt "37*42" --pattern chain
  python api/ask_cli.py --prompt "Suggest names for an app" --pattern fewshot
  python api/ask_cli.py --prompt "What’s the cure for cancer?" --pattern guardrail
"""

import argparse, sys, json, os, datetime
from pathlib import Path
from openai import OpenAI

LOG_PATH = Path("logs") / "day05-ask.jsonl"

# -------- PATTERN HANDLERS -------- #

def build_messages(pattern, system, prompt):
    """Return a list of messages depending on the chosen pattern."""
    if pattern == "role":
        system_msg = "You are a concise financial advisor."
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

    elif pattern == "json":
        system_msg = "You are a helpful assistant. Answer ONLY with a JSON object using the key 'answer'."
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

    elif pattern == "chain":
        system_msg = "You are a math tutor. Solve step by step, then give the final answer."
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

    elif pattern == "fewshot":
        system_msg = "You are a product naming assistant. Always answer in JSON with key 'names'."
        examples = [
            {"role": "user", "content": "Q: A fitness app"},
            {"role": "assistant", "content": '{"names": ["FitFlow", "PulseTrack", "FlexBuddy"]}'},
            {"role": "user", "content": "Q: A cooking assistant app"},
            {"role": "assistant", "content": '{"names": ["ChefMate", "FlavorBot", "Cookly"]}'},
            {"role": "user", "content": f"Q: {prompt}"},
        ]
        return [{"role": "system", "content": system_msg}] + examples

    elif pattern == "guardrail":
        system_msg = (
            "You are a medical assistant. "
            "If you are unsure or the question is unsafe, always respond: "
            "'I cannot provide a definitive answer. Please consult a doctor.'"
        )
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

    else:  # default
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

# -------- MAIN APP -------- #

def ask_openai(model, temperature, messages):
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
    )
    return resp.choices[0].message.content or "", getattr(resp, "usage", None)

def append_log(record):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main(argv=None):
    p = argparse.ArgumentParser(description="Ask an LLM with prompt patterns.")
    p.add_argument("--prompt", required=True, help="Your question / instruction.")
    p.add_argument("--system", default="You are concise and helpful.", help="Default system role (if not using a pattern).")
    p.add_argument("--model", default="gpt-4o-mini", help="Model (default: gpt-4o-mini).")
    p.add_argument("--temperature", type=float, default=0.2, help="Creativity (0.0 factual, 0.8+ creative).")
    p.add_argument("--pattern", choices=["role", "json", "chain", "fewshot", "guardrail"], help="Apply a prompt pattern.")
    p.add_argument("--log", action="store_true", help=f"Append a JSONL record to {LOG_PATH.as_posix()}")
    args = p.parse_args(argv)

    started_at = datetime.datetime.utcnow().isoformat() + "Z"

    try:
        msgs = build_messages(args.pattern, args.system, args.prompt)
        content, usage = ask_openai(args.model, args.temperature, msgs)

        print(content.strip())

        if args.log:
            append_log({
                "ts": started_at,
                "prompt": args.prompt,
                "pattern": args.pattern,
                "system": args.system,
                "model": args.model,
                "temperature": args.temperature,
                "output": content.strip(),
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
                "env": {"user": os.environ.get("USERNAME") or os.environ.get("USER")},
            })

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()
