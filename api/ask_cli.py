#!/usr/bin/env python3
# api/ask_cli.py
"""
Day 5 – Ask Anything CLI
Supports:
  • --pattern   role | json | chain | fewshot | guardrail
  • --temperature
  • --log
  • --moderate  (OpenAI moderation pre-check)
"""

import argparse, sys, json, os, datetime
from pathlib import Path
from openai import OpenAI

LOG_PATH = Path("logs") / "day05-ask.jsonl"

# ---------- PATTERN HANDLERS ---------- #
def build_messages(pattern, system, prompt):
    if pattern == "role":
        system_msg = "You are a concise financial advisor."
        return [{"role":"system","content":system_msg},
                {"role":"user","content":prompt}]

    elif pattern == "json":
        system_msg = "You are a helpful assistant. Answer ONLY with a JSON object using the key 'answer'."
        return [{"role":"system","content":system_msg},
                {"role":"user","content":prompt}]

    elif pattern == "chain":
        system_msg = "You are a math tutor. Solve step-by-step, then give the final answer."
        return [{"role":"system","content":system_msg},
                {"role":"user","content":prompt}]

    elif pattern == "fewshot":
        system_msg = "You are a product-naming assistant. Always answer in JSON with key 'names'."
        examples = [
            {"role":"user","content":"Q: A fitness app"},
            {"role":"assistant","content":'{"names":["FitFlow","PulseTrack","FlexBuddy"]}'},
            {"role":"user","content":"Q: A cooking assistant app"},
            {"role":"assistant","content":'{"names":["ChefMate","FlavorBot","Cookly"]}'},
            {"role":"user","content":f"Q: {prompt}"}
        ]
        return [{"role":"system","content":system_msg}]+examples

    elif pattern == "guardrail":
        system_msg = (
            "You are a helpful assistant. "
            "If the user asks a general question you know (math, geography, history), answer normally. "
            "If the question is about medicine, health, or anything you are not 100% sure of, "
            "respond ONLY with: 'I cannot provide a definitive answer. Please consult a doctor.'"
        )
        return [{"role":"system","content":system_msg},
                {"role":"user","content":prompt}]

    else:
        return [{"role":"system","content":system},
                {"role":"user","content":prompt}]

# ---------- MAIN LOGIC ---------- #
def check_moderation(client, text):
    result = client.moderations.create(model="omni-moderation-latest", input=text)
    return not result.results[0].flagged

def append_log(record):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH,"a",encoding="utf-8") as f:
        f.write(json.dumps(record,ensure_ascii=False)+"\n")

def main(argv=None):
    p = argparse.ArgumentParser(description="Ask an LLM with patterns + moderation.")
    p.add_argument("--prompt",required=True)
    p.add_argument("--system",default="You are concise and helpful.")
    p.add_argument("--model",default="gpt-4o-mini")
    p.add_argument("--temperature",type=float,default=0.2)
    p.add_argument("--pattern",choices=["role","json","chain","fewshot","guardrail"])
    p.add_argument("--log",action="store_true")
    p.add_argument("--moderate",action="store_true")
    args = p.parse_args(argv)

    started_at = datetime.datetime.utcnow().isoformat()+"Z"
    client = OpenAI()

    try:
        if args.moderate and not check_moderation(client,args.prompt):
            print("⚠️  Prompt blocked by moderation API.")
            if args.log:
                append_log({"ts":started_at,"prompt":args.prompt,
                            "pattern":args.pattern,"moderation":"blocked"})
            sys.exit(1)

        msgs = build_messages(args.pattern,args.system,args.prompt)
        resp = client.chat.completions.create(model=args.model,
                                              temperature=args.temperature,
                                              messages=msgs)
        answer = resp.choices[0].message.content.strip()
        print(answer)

        if args.log:
            append_log({
                "ts":started_at,"prompt":args.prompt,"pattern":args.pattern,
                "model":args.model,"temperature":args.temperature,
                "output":answer,
                "usage":getattr(resp,"usage",None),
                "moderation":"passed" if args.moderate else "not_checked",
                "env":{"user":os.environ.get("USERNAME") or os.environ.get("USER")}
            })

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(3)

if __name__=="__main__":
    main()
