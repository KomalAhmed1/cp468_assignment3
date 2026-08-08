"""
evaluate.py
===========
Computes BLEU (sacrebleu, corpus-level, standard for QG papers) and ROUGE-L
(rouge-score) for any predictions jsonl that has "prediction" and
"reference_question" fields. Works for LSTM output AND the LLM baseline output,
so both are scored identically for a fair comparison.

    python evaluate.py --pred ../results/lstm_predictions.jsonl --name lstm
    python evaluate.py --pred ../results/llm_zeroshot_predictions.jsonl --name llm_zeroshot
"""
import argparse
import json
import os

import sacrebleu
from rouge_score import rouge_scorer


def load_preds(path):
    rows = [json.loads(l) for l in open(path)]
    preds = [r["prediction"] if r["prediction"].strip() else "empty" for r in rows]
    refs = [r["reference_question"] for r in rows]
    return rows, preds, refs


def compute_metrics(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1 = r2 = rl = 0.0
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1 += s["rouge1"].fmeasure
        r2 += s["rouge2"].fmeasure
        rl += s["rougeL"].fmeasure
    n = len(preds)
    return {
        "bleu": bleu.score,
        "bleu_signature": str(bleu),
        "rouge1_f": 100 * r1 / n,
        "rouge2_f": 100 * r2 / n,
        "rougeL_f": 100 * rl / n,
        "n_examples": n,
    }


def bucket_by_length(rows):
    """Quantitative-gap-by-difficulty helper: bucket by passage length (proxy for
    difficulty -- longer context = harder to locate + phrase the right question)."""
    buckets = {"short(<=40 tok)": [], "medium(41-70)": [], "long(>70)": []}
    for r in rows:
        n_tok = len(r["context"].split())
        if n_tok <= 40:
            buckets["short(<=40 tok)"].append(r)
        elif n_tok <= 70:
            buckets["medium(41-70)"].append(r)
        else:
            buckets["long(>70)"].append(r)
    out = {}
    for name, rs in buckets.items():
        if not rs:
            continue
        preds = [r["prediction"] if r["prediction"].strip() else "empty" for r in rs]
        refs = [r["reference_question"] for r in rs]
        m = compute_metrics(preds, refs)
        out[name] = {"n": len(rs), "bleu": m["bleu"], "rougeL_f": m["rougeL_f"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out_dir", default="../results")
    args = ap.parse_args()

    rows, preds, refs = load_preds(args.pred)
    metrics = compute_metrics(preds, refs)
    metrics["length_buckets"] = bucket_by_length(rows)

    print(f"=== {args.name} ===")
    print(f"N={metrics['n_examples']}  BLEU={metrics['bleu']:.2f}  "
          f"ROUGE-1={metrics['rouge1_f']:.2f}  ROUGE-2={metrics['rouge2_f']:.2f}  ROUGE-L={metrics['rougeL_f']:.2f}")
    for bucket, m in metrics["length_buckets"].items():
        print(f"  {bucket}: n={m['n']} BLEU={m['bleu']:.2f} ROUGE-L={m['rougeL_f']:.2f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"metrics_{args.name}.json")
    json.dump(metrics, open(out_path, "w"), indent=2)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()