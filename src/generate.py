"""
generate.py
===========
Loads a trained checkpoint and decodes questions for a data split, writing a
jsonl of {qid, context, answer_text, reference_question, prediction}.

    python generate.py --split test --n_examples 500 --decode greedy
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from data_utils import QGDataset, load_jsonl, load_vocab, make_collate_fn, PAD
from model import Seq2Seq


def ids_to_text(ids, inv_vocab, eos_idx):
    toks = []
    for i in ids:
        i = int(i)
        if i == eos_idx:
            break
        tok = inv_vocab.get(i, "<unk>")
        if tok in ("<pad>", "<sos>"):
            continue
        toks.append(tok)
    return " ".join(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../data/processed")
    ap.add_argument("--ckpt", default="../checkpoints/best_model.pt")
    ap.add_argument("--run_meta", default="../checkpoints/run_meta.json")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n_examples", type=int, default=500)
    ap.add_argument("--decode", default="greedy", choices=["greedy", "beam"])
    ap.add_argument("--beam_size", type=int, default=4)
    ap.add_argument("--out", default="../results/lstm_predictions.jsonl")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_src = load_vocab(os.path.join(args.data_dir, "vocab_src.json"))
    vocab_tgt = load_vocab(os.path.join(args.data_dir, "vocab_tgt.json"))
    inv_vocab_tgt = {v: k for k, v in vocab_tgt.items()}

    run_meta = json.load(open(args.run_meta))
    margs = run_meta["args"]

    model = Seq2Seq(vocab_src, vocab_tgt, emb_dim=margs["emb_dim"], hidden_dim=margs["hidden_dim"],
                     enc_layers=margs["enc_layers"], dropout=0.0,
                     pad_idx=vocab_tgt[PAD], sos_idx=vocab_tgt["<sos>"], eos_idx=vocab_tgt["<eos>"]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    examples = load_jsonl(os.path.join(args.data_dir, f"{args.split}.jsonl"), args.n_examples, args.seed)
    ds = QGDataset(examples, vocab_src, vocab_tgt)
    collate = make_collate_fn(vocab_src[PAD], vocab_tgt[PAD])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        if args.decode == "greedy":
            loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)
            for batch in loader:
                src, src_lens, src_mask = batch["src"].to(device), batch["src_lens"], batch["src_mask"].to(device)
                out_ids = model.greedy_decode(src, src_lens, src_mask, max_len=18)
                for i, meta in enumerate(batch["meta"]):
                    pred = ids_to_text(out_ids[i].tolist(), inv_vocab_tgt, vocab_tgt["<eos>"])
                    f.write(json.dumps({
                        "qid": meta["qid"], "context": meta["context"], "answer_text": meta["answer_text"],
                        "reference_question": meta["question"], "prediction": pred,
                    }) + "\n")
        else:
            loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
            for batch in loader:
                src, src_lens, src_mask = batch["src"].to(device), batch["src_lens"], batch["src_mask"].to(device)
                out_ids = model.beam_search_decode(src, src_lens, src_mask, beam_size=args.beam_size, max_len=18)
                pred = ids_to_text(out_ids, inv_vocab_tgt, vocab_tgt["<eos>"])
                meta = batch["meta"][0]
                f.write(json.dumps({
                    "qid": meta["qid"], "context": meta["context"], "answer_text": meta["answer_text"],
                    "reference_question": meta["question"], "prediction": pred,
                }) + "\n")
    print("Wrote predictions to", args.out)


if __name__ == "__main__":
    main()