#!/usr/bin/env python3
# api/ask_cli.py
"""
Day 5 – Ask Anything CLI with:
- Prompt patterns (--pattern)
- Temperature control (--temperature)
- JSON output (--json via patterns)
- Logging (--log)
- Moderation (--moderate)

Usage:
  python api/ask_cli.py --prompt "What is 2+2?" --pattern guardrail --moderate
"""

import argparse, sys, json, os, datetime
from pathlib import Path
from openai import OpenAI

LOG_PATH = Path("logs") / "day05-ask.jsonl"

# -------- PATTERN HANDLERS -------- #

def build_messages(pattern, system, prompt):
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
            "You are a helpful assistant. "
            "If the user asks a general question you know (like basic math, geography, history, science), answer normally. "
            "If the question is about medicine, health, or anything you are not 100% sure of, "
            "respond ONLY with this exact text: "
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

def check_moderation(client, text: str) -> bool:
    """Return True if safe, False if flagged unsafe."""
    result = client.moderations.create(model="omni-moderation-latest", input=text)
    flagged = result.results[0].flagged
    return not flagged

def append_log(record):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main(argv=None):
    p = argparse.ArgumentParser(description="Ask an LLM with patterns + moderation.")
    p.add_argument("--prompt", required=True, help="Your question / instruction.")
    p.add_argument("--system", default="You are concise and helpful.", help="Default system role.")
    p.add_argument("--model", default="gpt-4o-mini", help="Model (default: gpt-4o-mini).")
    p.add_argument("--temperature", type=float, default=0.2, help="Creativity 0.0–1.0.")
    p.add_argument("--pattern", choices=["role", "json", "chain", "fewshot", "guardrail"], help="Prompt pattern.")
    p.add_argument("--log", action="store_true", help="Append JSONL record to logs/day05-ask.jsonl")
    p.add_argument("--moderate", action="store_true", help="Use OpenAI Moderation API before sending prompt")
    args = p.parse_args(argv)

    started_at = datetime.datetime.utcnow().isoformat() + "Z"
    client = OpenAI()

    try:
        # Moderation check
        if args.moderate:
            safe = check_moderation(client, args.prompt)
            if not safe:
                msg = "⚠️ Prompt blocked by moderation API as unsafe."
                print(msg)
                if args.log:
                    append_log({
                        "ts": started_at,
                        "prompt": args.prompt,
                        "pattern": args.pattern,
                        "moderation": "blocked",
                        "output": msg,
                    })
                sys.exit(1)

        # Build prompt + send to LLM
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
                "moderation": "passed" if args.moderate else "not_checked",
                "env": {"user": os.environ.get("USERNAME") or os.environ.get("USER")},
            })

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()
