"""
train.py
========
Trains the LSTM seq2seq-with-attention model for question generation.

Example (small reference run, CPU-friendly -- what this repo ships results for):
    python train.py --max_train 4000 --max_val 500 --epochs 8 \
        --emb_dim 128 --hidden_dim 256 --batch_size 64

Example (full-scale run, needs a GPU to finish in reasonable time):
    python train.py --max_train -1 --max_val -1 --epochs 15 \
        --emb_dim 300 --hidden_dim 512 --batch_size 64
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import QGDataset, load_jsonl, load_vocab, make_collate_fn, PAD
from model import Seq2Seq, count_parameters


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, criterion, device, tf_ratio, train=True):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_mask = batch["src_mask"].to(device)
            src_lens = batch["src_lens"]

            if train:
                optimizer.zero_grad()
            logits = model(src, src_lens, src_mask, tgt, teacher_forcing_ratio=tf_ratio if train else 0.0)
            # logits: [B, T-1, V], targets are tgt[:,1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../data/processed")
    ap.add_argument("--ckpt_dir", default="../checkpoints")
    ap.add_argument("--max_train", type=int, default=4000, help="-1 for full train set")
    ap.add_argument("--max_val", type=int, default=500, help="-1 for full val set")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--enc_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tf_ratio", type=float, default=0.6, help="teacher forcing ratio")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--resume", action="store_true",
                     help="continue from checkpoints/last_model.pt + run_meta.json "
                          "(this session's sandbox kills long-running commands, so "
                          "training is chunked across several resumed calls)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    vocab_src = load_vocab(os.path.join(args.data_dir, "vocab_src.json"))
    vocab_tgt = load_vocab(os.path.join(args.data_dir, "vocab_tgt.json"))

    train_ex = load_jsonl(os.path.join(args.data_dir, "train.jsonl"), args.max_train, args.seed)
    val_ex = load_jsonl(os.path.join(args.data_dir, "val.jsonl"), args.max_val, args.seed)
    print(f"train examples: {len(train_ex)}  val examples: {len(val_ex)}")

    train_ds = QGDataset(train_ex, vocab_src, vocab_tgt)
    val_ds = QGDataset(val_ex, vocab_src, vocab_tgt)
    collate = make_collate_fn(vocab_src[PAD], vocab_tgt[PAD])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = Seq2Seq(vocab_src, vocab_tgt, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim,
                     enc_layers=args.enc_layers, dropout=args.dropout,
                     pad_idx=vocab_tgt[PAD], sos_idx=vocab_tgt["<sos>"], eos_idx=vocab_tgt["<eos>"]).to(device)
    n_params = count_parameters(model)
    print(f"model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_tgt[PAD])

    history, best_val, start_epoch, total_time_so_far = [], float("inf"), 0, 0.0
    meta_path = os.path.join(args.ckpt_dir, "run_meta.json")
    last_path = os.path.join(args.ckpt_dir, "last_model.pt")
    if args.resume and os.path.exists(meta_path) and os.path.exists(last_path):
        prev = json.load(open(meta_path))
        history = prev["history"]
        best_val = prev.get("best_val", float("inf"))
        start_epoch = len(history)
        total_time_so_far = prev.get("total_train_seconds", 0.0)
        model.load_state_dict(torch.load(last_path, map_location=device))
        print(f"Resumed from epoch {start_epoch}. best_val so far={best_val:.4f}")

    t_start = time.time()
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, args.tf_ratio, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, device, args.tf_ratio, train=False)
        dt = time.time() - t0
        print(f"epoch {epoch}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({dt:.1f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": dt})
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "best_model.pt"))
        torch.save(model.state_dict(), last_path)
        total_time = total_time_so_far + (time.time() - t_start)
        run_meta = {
            "n_params": n_params, "device": str(device), "total_train_seconds": total_time,
            "best_val": best_val, "args": vars(args), "history": history,
            "n_train_examples": len(train_ex), "n_val_examples": len(val_ex),
        }
        json.dump(run_meta, open(meta_path, "w"), indent=2)

    print(f"Done. total_train_seconds so far={run_meta['total_train_seconds']:.1f}. "
          f"Best val loss={best_val:.4f}. Saved to {args.ckpt_dir}")


if __name__ == "__main__":
    main()