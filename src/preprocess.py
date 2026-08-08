"""
preprocess.py
=============
Turns the raw SQuAD-format json (data/raw/{train,dev,test}.json, the article-level
split from Du et al. 2017 "Learning to Ask") into flattened, tokenized
(passage-with-answer-marked -> question) examples ready for the LSTM seq2seq model
and for prompting the LLM baseline.

Why this split:
- Held-out dev/test are reserved BEFORE any modeling, and the split is done at the
  ARTICLE level (not question level), so no paragraph/question from a test article
  ever appears, even implicitly, in training. This avoids the leakage that a random
  question-level split would risk (many questions per SQuAD paragraph).

Usage:
    python preprocess.py --raw_dir ../data/raw --out_dir ../data/processed

Outputs (in out_dir):
    train.jsonl / val.jsonl / test.jsonl   -- one example per line:
        {"src_tokens": [...], "tgt_tokens": [...], "answer_text": "...",
         "question": "...", "context": "...", "qid": "..."}
    vocab_src.json / vocab_tgt.json        -- token -> id, built from TRAIN ONLY
"""
import argparse
import json
import os
import random
import re
from collections import Counter

SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>"]
ANS_START, ANS_END = "<ans>", "</ans>"

MAX_SRC_LEN = 60   # tokens, passage window (incl. answer markers)
MAX_TGT_LEN = 18   # tokens, question
CONTEXT_WINDOW = 25  # tokens kept on each side of the answer span

_token_re = re.compile(r"\w+|[^\w\s]")


def tokenize(text):
    return _token_re.findall(text.lower())


def build_source_sequence(context, answer_text, answer_start):
    """Tokenize the context, insert <ans> ... </ans> around the answer span,
    then truncate to a window around the answer so sequences stay short enough
    to train an LSTM on CPU while still giving the model real passage context."""
    before = context[:answer_start]
    answer = context[answer_start:answer_start + len(answer_text)]
    after = context[answer_start + len(answer_text):]

    before_toks = tokenize(before)
    ans_toks = tokenize(answer)
    after_toks = tokenize(after)

    # window around the answer so we don't feed a huge paragraph to the encoder
    before_toks = before_toks[-CONTEXT_WINDOW:]
    after_toks = after_toks[:CONTEXT_WINDOW]

    src = before_toks + [ANS_START] + ans_toks + [ANS_END] + after_toks
    return src[:MAX_SRC_LEN]


def flatten_split(raw_json_path):
    data = json.load(open(raw_json_path))
    examples = []
    for article in data:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                if not qa["answers"]:
                    continue
                ans = qa["answers"][0]
                src_tokens = build_source_sequence(context, ans["text"], ans["answer_start"])
                tgt_tokens = tokenize(qa["question"])[:MAX_TGT_LEN]
                if len(src_tokens) < 3 or len(tgt_tokens) < 3:
                    continue
                examples.append({
                    "qid": qa["id"],
                    "src_tokens": src_tokens,
                    "tgt_tokens": tgt_tokens,
                    "answer_text": ans["text"],
                    "question": qa["question"],
                    "context": context,
                })
    return examples


def build_vocab(examples, key, max_size=10000, min_freq=2):
    counter = Counter()
    for ex in examples:
        counter.update(ex[key])
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    if key == "src_tokens":
        vocab[ANS_START] = len(vocab)
        vocab[ANS_END] = len(vocab)
    for tok, freq in counter.most_common():
        if tok in vocab:
            continue
        if freq < min_freq:
            break
        if len(vocab) >= max_size:
            break
        vocab[tok] = len(vocab)
    return vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="../data/raw")
    ap.add_argument("--out_dir", default="../data/processed")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    split_map = {"train": "train.json", "val": "dev.json", "test": "test.json"}
    all_examples = {}
    for split, fname in split_map.items():
        exs = flatten_split(os.path.join(args.raw_dir, fname))
        random.shuffle(exs)
        all_examples[split] = exs
        print(f"{split}: {len(exs)} examples after filtering")

    vocab_src = build_vocab(all_examples["train"], "src_tokens", max_size=12000)
    vocab_tgt = build_vocab(all_examples["train"], "tgt_tokens", max_size=6000)
    print(f"src vocab size: {len(vocab_src)}  tgt vocab size: {len(vocab_tgt)}")

    json.dump(vocab_src, open(os.path.join(args.out_dir, "vocab_src.json"), "w"))
    json.dump(vocab_tgt, open(os.path.join(args.out_dir, "vocab_tgt.json"), "w"))

    for split, exs in all_examples.items():
        with open(os.path.join(args.out_dir, f"{split}.jsonl"), "w") as f:
            for ex in exs:
                f.write(json.dumps(ex) + "\n")

    meta = {
        "max_src_len": MAX_SRC_LEN,
        "max_tgt_len": MAX_TGT_LEN,
        "context_window": CONTEXT_WINDOW,
        "counts": {k: len(v) for k, v in all_examples.items()},
    }
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    print("Done. Wrote processed jsonl + vocab + meta.json to", args.out_dir)


if __name__ == "__main__":
    main()