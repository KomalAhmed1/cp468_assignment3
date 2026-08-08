"""
LLM baseline for the question generation task.

Runs an instruction-tuned model over the same test examples generate.py decodes
(same split, --n_examples and --seed) and writes the same jsonl schema, so
evaluate.py can score both systems.

Works with any OpenAI-compatible /chat/completions endpoint.

    export LLM_API_KEY=sk-...
    python llm_baseline.py --prompt zeroshot --model gpt-4o-mini \
        --out ../results/llm_zeroshot_predictions.jsonl

    python llm_baseline.py --prompt fewshot --model llama3.1:8b \
        --base_url http://localhost:11434/v1 --api_key_env "" \
        --out ../results/llm_fewshot_predictions.jsonl
"""
import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request

from data_utils import load_jsonl

SYSTEM_PROMPT = (
    "You are a question generation system. Given a passage and an answer span "
    "taken from it, you write the single question that the passage answers with "
    "exactly that span. Output only the question text, nothing else."
)

ZEROSHOT_TEMPLATE = (
    "Passage:\n{context}\n\n"
    "Answer: {answer}\n\n"
    "Write one question about the passage whose answer is exactly the answer "
    "span above. Output only the question, on a single line, ending with '?'."
)

FEWSHOT_INSTRUCTION = (
    "Write one question about the passage whose answer is exactly the answer "
    "span above. Match the style, length and phrasing conventions of the "
    "examples. Output only the question, on a single line, ending with '?'."
)

DEFAULT_PRICE_IN = 0.15
DEFAULT_PRICE_OUT = 0.60


def build_messages(ex, shots):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if not shots:
        msgs.append({"role": "user", "content": ZEROSHOT_TEMPLATE.format(
            context=ex["context"], answer=ex["answer_text"])})
        return msgs
    for s in shots:
        msgs.append({"role": "user", "content":
                     f"Passage:\n{s['context']}\n\nAnswer: {s['answer_text']}\n\n{FEWSHOT_INSTRUCTION}"})
        msgs.append({"role": "assistant", "content": s["question"]})
    msgs.append({"role": "user", "content":
                 f"Passage:\n{ex['context']}\n\nAnswer: {ex['answer_text']}\n\n{FEWSHOT_INSTRUCTION}"})
    return msgs


def chat(base_url, api_key, model, messages, max_tokens, temperature, timeout,
         reasoning_effort=None, retries=6):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_err = None
    for attempt in range(retries):
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                     data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            err_body = e.read().decode(errors="replace")[:500]
            last_err = f"HTTP {e.code}: {err_body}"
            retryable = e.code >= 500 or (e.code == 429 and "insufficient_quota" not in err_body)
            if not retryable:
                raise RuntimeError(last_err) from None
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM request failed after {retries} attempts: {last_err}")


def clean(text):
    for line in text.strip().splitlines():
        line = line.strip().strip('"')
        if line:
            return line
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../data/processed")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n_examples", type=int, default=500,
                    help="must match generate.py so the test set is identical")
    ap.add_argument("--seed", type=int, default=13,
                    help="must match generate.py so the test set is identical")
    ap.add_argument("--prompt", default="zeroshot", choices=["zeroshot", "fewshot"])
    ap.add_argument("--k_shots", type=int, default=4)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--base_url", default="https://api.openai.com/v1")
    ap.add_argument("--api_key_env", default="LLM_API_KEY",
                    help='env var holding the key; pass "" for a local server')
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--reasoning_effort", default=None,
                    help="set to 'none' for Gemini thinking models")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds between requests, to stay under rate limits")
    ap.add_argument("--price_in", type=float, default=DEFAULT_PRICE_IN,
                    help="USD per 1M input tokens (0 for a local model)")
    ap.add_argument("--price_out", type=float, default=DEFAULT_PRICE_OUT,
                    help="USD per 1M output tokens (0 for a local model)")
    ap.add_argument("--out", default="../results/llm_zeroshot_predictions.jsonl")
    ap.add_argument("--resume", action="store_true",
                    help="append to an existing --out file, skipping qids already done")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
    if args.api_key_env and not api_key:
        raise SystemExit(f"Set ${args.api_key_env}, or pass --api_key_env '' for a local server.")

    examples = load_jsonl(os.path.join(args.data_dir, f"{args.split}.jsonl"),
                          args.n_examples, args.seed)

    shots = []
    if args.prompt == "fewshot":
        train = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))
        shots = random.Random(args.seed).sample(train, args.k_shots)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if args.resume and os.path.exists(args.out):
        done = {json.loads(l)["qid"] for l in open(args.out)}
        print(f"Resuming: {len(done)} already done, {len(examples) - len(done)} to go")
    todo = [e for e in examples if e["qid"] not in done]

    in_tok = out_tok = empty = 0
    latencies = []
    t0 = time.time()
    aborted = None

    with open(args.out, "a" if args.resume else "w") as f:
        for i, ex in enumerate(todo, 1):
            msgs = build_messages(ex, shots)
            t = time.time()
            try:
                resp = chat(args.base_url, api_key, args.model, msgs,
                            args.max_tokens, args.temperature, args.timeout,
                            args.reasoning_effort)
            except (RuntimeError, KeyboardInterrupt) as e:
                aborted = str(e)[:300]
                print(f"ABORTED at {i}/{len(todo)}: {aborted}")
                break
            latencies.append(time.time() - t)
            choice = (resp.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content") or ""
            if not content.strip():
                empty += 1
            pred = clean(content)
            usage = resp.get("usage") or {}
            in_tok += usage.get("prompt_tokens", 0)
            out_tok += usage.get("completion_tokens", 0)
            f.write(json.dumps({
                "qid": ex["qid"], "context": ex["context"], "answer_text": ex["answer_text"],
                "reference_question": ex["question"], "prediction": pred,
            }) + "\n")
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}")
            if args.sleep:
                time.sleep(args.sleep)

    wall = time.time() - t0
    latencies.sort()
    cost = in_tok / 1e6 * args.price_in + out_tok / 1e6 * args.price_out
    report = {
        "model": args.model,
        "base_url": args.base_url,
        "prompt_variant": args.prompt,
        "k_shots": args.k_shots if args.prompt == "fewshot" else 0,
        "temperature": args.temperature,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "empty_responses": empty,
        "split": args.split,
        "n_examples": sum(1 for _ in open(args.out)),
        "n_requested": len(examples),
        "aborted": aborted,
        "seed": args.seed,
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "usd_per_1m_input": args.price_in,
        "usd_per_1m_output": args.price_out,
        "estimated_cost_usd": round(cost, 4),
        "wall_clock_s": round(wall, 1),
        "compute_hours": round(wall / 3600, 4),
        "median_latency_s": round(latencies[len(latencies) // 2], 2) if latencies else None,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": (ZEROSHOT_TEMPLATE if args.prompt == "zeroshot"
                                 else "Passage:\n{context}\n\nAnswer: {answer}\n\n" + FEWSHOT_INSTRUCTION),
        "fewshot_qids": [s["qid"] for s in shots],
    }
    meta_path = os.path.join(os.path.dirname(args.out),
                             f"llm_{args.prompt}_runmeta.json")
    json.dump(report, open(meta_path, "w"), indent=2)

    print("Wrote predictions to", args.out)
    print(f"tokens in/out={in_tok}/{out_tok}  est. cost=${cost:.4f}  "
          f"wall={wall:.1f}s ({wall/3600:.4f} h)  median latency={report['median_latency_s']}s")
    print("Wrote run metadata to", meta_path)


if __name__ == "__main__":
    main()
