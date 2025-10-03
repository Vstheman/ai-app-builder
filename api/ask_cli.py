#!/usr/bin/env python3
# api/ask_cli.py
"""
Day 4 – Ask Anything CLI (OpenAI) with:
- --temperature (creativity)
- --json (force JSON output)
- --log (append a JSONL record per run to logs/day04-ask.jsonl)

Usage:
  python api/ask_cli.py --prompt "Explain RAG in one sentence" --log
  python api/ask_cli.py --prompt "Give 3 startup ideas in healthtech" --temperature 0.8 --log
  python api/ask_cli.py --prompt "Transformers vs RNNs in 2 bullets" --json --log
"""

import argparse, sys, json, os, datetime
from pathlib import Path
from openai import OpenAI
from openai import APIStatusError

LOG_PATH = Path("logs") / "day04-ask.jsonl"

def ask_openai(prompt: str, system: str, model: str, temperature: float, as_json: bool):
    """
    Calls OpenAI Chat Completions.
    Returns (output_text: str, usage: dict|None).
    """
    client = OpenAI()

    sys_msg = system
    if as_json:
        # Strong instruction for strict JSON
        sys_msg += (
            "\nYou must respond using ONLY a single valid JSON object with the key 'answer'. "
            "Do not include any extra text, explanations, or markdown."
        )

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    if usage:
        usage = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    return content, usage

def safe_json_answer(raw: str):
    """
    Ensure we always print valid JSON when --json is set.
    If the model didn't return strict JSON, wrap it.
    """
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict) or "answer" not in obj:
            obj = {"answer": raw.strip()}
    except Exception:
        obj = {"answer": raw.strip()}
    return obj

def append_log(record: dict):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main(argv=None):
    p = argparse.ArgumentParser(description="Send a prompt to an LLM and print the reply.")
    p.add_argument("--prompt", required=True, help="Your question / instruction.")
    p.add_argument("--system", default="You are concise and helpful.", help="System instruction (tone/role).")
    p.add_argument("--model", default="gpt-4o-mini", help="Model name (default: gpt-4o-mini).")
    p.add_argument("--temperature", type=float, default=0.2, help="Creativity 0.0–1.0 (lower = more factual).")
    p.add_argument("--json", action="store_true", help="Force the model to return JSON with key 'answer'.")
    p.add_argument("--log", action="store_true", help=f"Append a JSONL record to {LOG_PATH.as_posix()}")
    args = p.parse_args(argv)

    started_at = datetime.datetime.utcnow().isoformat() + "Z"
    try:
        content, usage = ask_openai(
            prompt=args.prompt,
            system=args.system,
            model=args.model,
            temperature=args.temperature,
            as_json=args.json,
        )

        if args.json:
            obj = safe_json_answer(content)
            output_text = json.dumps(obj, ensure_ascii=False)
            print(output_text)
        else:
            output_text = content
            print(output_text)

        if args.log:
            append_log({
                "ts": started_at,
                "prompt": args.prompt,
                "system": args.system,
                "model": args.model,
                "temperature": args.temperature,
                "as_json": args.json,
                "output": obj if args.json else output_text,
                "usage": usage,
                "env": {
                    "user": os.environ.get("USERNAME") or os.environ.get("USER"),
                    "cwd": os.getcwd(),
                }
            })

    except APIStatusError as e:
        msg = f"OpenAI API error: {e.status_code} {e.message}"
        print(f"❌ {msg}")
        if args.log:
            append_log({
                "ts": started_at,
                "prompt": args.prompt,
                "system": args.system,
                "model": args.model,
                "temperature": args.temperature,
                "as_json": args.json,
                "error": msg
            })
        sys.exit(2)
    except Exception as e:
        msg = f"Unexpected error: {e}"
        print(f"❌ {msg}")
        if args.log:
            append_log({
                "ts": started_at,
                "prompt": args.prompt,
                "system": args.system,
                "model": args.model,
                "temperature": args.temperature,
                "as_json": args.json,
                "error": msg
            })
        sys.exit(3)

if __name__ == "__main__":
    main()
