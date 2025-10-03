#!/usr/bin/env python3
# api/ask_cli.py
"""
Day 4 – Ask Anything CLI (with temperature + JSON output)

Usage examples:
  # default (concise text)
  python api/ask_cli.py --prompt "Explain RAG in one sentence"

  # more creative
  python api/ask_cli.py --prompt "Give 3 creative startup ideas in fitness" --temperature 0.8

  # strict JSON output (always prints valid JSON)
  python api/ask_cli.py --prompt "Explain transformers in 2 bullets" --json

  # pick another model (if you have access)
  python api/ask_cli.py --prompt "Summarize the value prop of RAG" --model gpt-4o
"""

import argparse, sys, json
from openai import OpenAI
from openai import APIStatusError

def ask_openai(prompt: str, system: str, model: str, temperature: float, as_json: bool) -> str:
    """
    Calls OpenAI chat.completions.
    - If as_json=True, instructs the model to return ONLY JSON with the key "answer".
    - Returns the raw string content (we'll post-process when as_json=True).
    """
    client = OpenAI()

    sys_msg = system
    if as_json:
        # Strongly steer the model to return strict JSON (no prose, no markdown).
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
    return resp.choices[0].message.content or ""

def main(argv=None):
    p = argparse.ArgumentParser(description="Send a prompt to an LLM and print the reply.")
    p.add_argument("--prompt", required=True, help="Your question / instruction.")
    p.add_argument("--system", default="You are concise and helpful.", help="System instruction (tone/role).")
    p.add_argument("--model", default="gpt-4o-mini", help="Model name (default: gpt-4o-mini).")
    p.add_argument("--temperature", type=float, default=0.2, help="Creativity 0.0–1.0 (lower = more factual).")
    p.add_argument("--json", action="store_true", help="Force the model to return JSON with key 'answer'.")
    args = p.parse_args(argv)

    try:
        content = ask_openai(
            prompt=args.prompt,
            system=args.system,
            model=args.model,
            temperature=args.temperature,
            as_json=args.json,
        )

        if args.json:
            # Try to parse the model's output; if parsing fails, wrap it safely.
            try:
                obj = json.loads(content)
                # Ensure it has the "answer" key; if not, wrap.
                if not isinstance(obj, dict) or "answer" not in obj:
                    obj = {"answer": content.strip()}
            except Exception:
                obj = {"answer": content.strip()}
            # Always print valid, minified JSON (easy for piping/logging)
            print(json.dumps(obj, ensure_ascii=False))
        else:
            print(content.strip())

    except APIStatusError as e:
        print(f"❌ OpenAI API error: {e.status_code} {e.message}")
        sys.exit(2)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()
