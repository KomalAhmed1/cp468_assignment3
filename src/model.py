"""
model.py
========
Classical LSTM seq2seq with attention, implemented from scratch (as required:
only nn.LSTM / nn.Embedding / nn.Linear building blocks are used -- no
Fairseq / OpenNMT / HF Seq2SeqTrainer).

Architecture:
    Embedding -> Bidirectional LSTM encoder -> Bahdanau (additive) attention
    -> LSTM decoder (unidirectional, fed the attention context each step)
    -> output projection over the target vocabulary.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=1, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers,
                             batch_first=True, bidirectional=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        # project concatenated final [fwd;bwd] states down to decoder hidden size
        self.fc_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim * 2, hidden_dim)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, src, src_lens):
        emb = self.dropout(self.embedding(src))  # [B, T, E]
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (h, c) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # outputs: [B, T, 2*H]  (encoder states the decoder will attend over)

        # combine last layer's forward/backward final states -> decoder init state
        h = h.view(self.num_layers, 2, -1, self.hidden_dim)  # [layers, dirs, B, H]
        c = c.view(self.num_layers, 2, -1, self.hidden_dim)
        h_cat = torch.cat([h[-1, 0], h[-1, 1]], dim=-1)  # [B, 2H]
        c_cat = torch.cat([c[-1, 0], c[-1, 1]], dim=-1)
        h0 = torch.tanh(self.fc_h(h_cat)).unsqueeze(0)  # [1, B, H]
        c0 = torch.tanh(self.fc_c(c_cat)).unsqueeze(0)
        return outputs, (h0, c0)


class BahdanauAttention(nn.Module):
    """Additive attention: score(s_t, h_i) = v^T tanh(W_s s_t + W_h h_i)."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)  # encoder states (bi)
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)      # decoder state
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, src_mask):
        # dec_hidden: [B, H]  enc_outputs: [B, T, 2H]  src_mask: [B, T] (True=real)
        proj_enc = self.W_h(enc_outputs)                       # [B, T, H]
        proj_dec = self.W_s(dec_hidden).unsqueeze(1)            # [B, 1, H]
        scores = self.v(torch.tanh(proj_enc + proj_dec)).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(~src_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)                # [B, T]
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)  # [B, 2H]
        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(hidden_dim)
        self.lstm = nn.LSTM(emb_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim * 3 + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_tok, h, c, enc_outputs, src_mask):
        # input_tok: [B] (single step)
        emb = self.dropout(self.embedding(input_tok)).unsqueeze(1)  # [B,1,E]
        context, attn_w = self.attention(h.squeeze(0), enc_outputs, src_mask)  # [B,2H]
        lstm_in = torch.cat([emb, context.unsqueeze(1)], dim=-1)  # [B,1,E+2H]
        out, (h, c) = self.lstm(lstm_in, (h, c))
        out = out.squeeze(1)  # [B,H]
        logits = self.out(torch.cat([out, context, emb.squeeze(1)], dim=-1))  # [B,V]
        return logits, h, c, attn_w


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb_dim=128, hidden_dim=256,
                 enc_layers=1, dropout=0.1, pad_idx=0, sos_idx=2, eos_idx=3):
        super().__init__()
        self.encoder = Encoder(len(src_vocab), emb_dim, hidden_dim, enc_layers, dropout, pad_idx)
        self.decoder = Decoder(len(tgt_vocab), emb_dim, hidden_dim, dropout, pad_idx)
        self.pad_idx, self.sos_idx, self.eos_idx = pad_idx, sos_idx, eos_idx
        self.tgt_vocab_size = len(tgt_vocab)

    def forward(self, src, src_lens, src_mask, tgt, teacher_forcing_ratio=0.5):
        """Training forward pass with scheduled teacher forcing."""
        enc_outputs, (h, c) = self.encoder(src, src_lens)
        B, T = tgt.shape
        logits_all = torch.zeros(B, T - 1, self.tgt_vocab_size, device=src.device)

        input_tok = tgt[:, 0]  # <sos>
        for t in range(1, T):
            logits, h, c, _ = self.decoder.forward_step(input_tok, h, c, enc_outputs, src_mask)
            logits_all[:, t - 1] = logits
            use_tf = torch.rand(1).item() < teacher_forcing_ratio
            input_tok = tgt[:, t] if use_tf else logits.argmax(-1)
        return logits_all

    @torch.no_grad()
    def greedy_decode(self, src, src_lens, src_mask, max_len=20):
        enc_outputs, (h, c) = self.encoder(src, src_lens)
        B = src.size(0)
        input_tok = torch.full((B,), self.sos_idx, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        outputs = torch.full((B, max_len), self.pad_idx, dtype=torch.long, device=src.device)
        for t in range(max_len):
            logits, h, c, _ = self.decoder.forward_step(input_tok, h, c, enc_outputs, src_mask)
            next_tok = logits.argmax(-1)
            next_tok = torch.where(done, torch.full_like(next_tok, self.pad_idx), next_tok)
            outputs[:, t] = next_tok
            done = done | (next_tok == self.eos_idx)
            input_tok = next_tok
            if done.all():
                break
        return outputs

    @torch.no_grad()
    def beam_search_decode(self, src, src_lens, src_mask, beam_size=4, max_len=20,
                            length_penalty=0.7):
        """Simple beam search, one example at a time (batch size 1 in, used at eval time)."""
        assert src.size(0) == 1, "beam_search_decode expects batch size 1"
        enc_outputs, (h, c) = self.encoder(src, src_lens)
        beams = [(0.0, [self.sos_idx], h, c)]
        completed = []
        for _ in range(max_len):
            new_beams = []
            for score, seq, h_b, c_b in beams:
                if seq[-1] == self.eos_idx:
                    completed.append((score, seq))
                    continue
                input_tok = torch.tensor([seq[-1]], device=src.device)
                logits, h2, c2, _ = self.decoder.forward_step(input_tok, h_b, c_b, enc_outputs, src_mask)
                logprobs = F.log_softmax(logits, dim=-1).squeeze(0)
                topk = torch.topk(logprobs, beam_size)
                for lp, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                    new_beams.append((score + lp, seq + [idx], h2, c2))
            if not new_beams:
                break
            new_beams.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
            beams = new_beams[:beam_size]
            if all(b[1][-1] == self.eos_idx for b in beams):
                completed.extend([(s, sq) for s, sq, _, _ in beams])
                break
        completed.extend([(s, sq) for s, sq, _, _ in beams])
        completed.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
        return completed[0][1]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)