"""
data_utils.py
=============
PyTorch Dataset + collate function for the processed QG jsonl files.
Handles: token->id lookup, <sos>/<eos> wrapping, padding, and building the
boolean padding masks the encoder/attention need.
"""
import json
import random

import torch
from torch.utils.data import Dataset

PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"


def load_vocab(path):
    return json.load(open(path))


def load_jsonl(path, max_examples=None, seed=13):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    if max_examples is not None and max_examples > 0 and max_examples < len(examples):
        rng = random.Random(seed)
        examples = rng.sample(examples, max_examples)
    return examples


def encode(tokens, vocab, add_sos_eos=False, max_len=None):
    ids = [vocab.get(t, vocab[UNK]) for t in tokens]
    if add_sos_eos:
        ids = [vocab[SOS]] + ids + [vocab[EOS]]
    if max_len:
        ids = ids[:max_len]
    return ids


class QGDataset(Dataset):
    """src: passage tokens with <ans>/</ans> markers. tgt: question tokens,
    wrapped with <sos>/<eos> so the decoder learns when to stop."""

    def __init__(self, examples, vocab_src, vocab_tgt, max_tgt_len=22):
        self.examples = examples
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        src_ids = encode(ex["src_tokens"], self.vocab_src)
        tgt_ids = encode(ex["tgt_tokens"], self.vocab_tgt, add_sos_eos=True,
                          max_len=self.max_tgt_len)
        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "qid": ex["qid"],
            "answer_text": ex["answer_text"],
            "question": ex["question"],
            "context": ex["context"],
            "src_tokens": ex["src_tokens"],
        }


def make_collate_fn(pad_src, pad_tgt):
    def collate(batch):
        src_lens = [len(b["src_ids"]) for b in batch]
        tgt_lens = [len(b["tgt_ids"]) for b in batch]
        max_src = max(src_lens)
        max_tgt = max(tgt_lens)

        src = torch.full((len(batch), max_src), pad_src, dtype=torch.long)
        tgt = torch.full((len(batch), max_tgt), pad_tgt, dtype=torch.long)
        for i, b in enumerate(batch):
            src[i, :len(b["src_ids"])] = torch.tensor(b["src_ids"], dtype=torch.long)
            tgt[i, :len(b["tgt_ids"])] = torch.tensor(b["tgt_ids"], dtype=torch.long)

        src_mask = (src != pad_src)  # True where real token
        meta = [{"qid": b["qid"], "answer_text": b["answer_text"],
                 "question": b["question"], "context": b["context"],
                 "src_tokens": b["src_tokens"]} for b in batch]
        return {
            "src": src, "tgt": tgt,
            "src_lens": torch.tensor(src_lens, dtype=torch.long),
            "src_mask": src_mask,
            "meta": meta,
        }
    return collate