"""
Full Transformer training script with an option to replace the position-wise Feed-Forward Network
with a simplified Kolmogorov-Arnold Network (KAN) implementation.

Single-file script that:
- Builds a tokenizer and Tiny Shakespeare dataset loader
- Implements Transformer Encoder / Decoder with a selectable FFN or KAN module
- Training and evaluation loop
- Checkpoint saving

Notes:
- This KAN implementation is a practical approximation intended for experiments: it implements
  the Kolmogorov–Arnold superposition idea by projecting the input to multiple scalar channels
  and applying small scalar MLPs (one per projection) followed by a learned linear readout.
- For serious research, replace the KAN class with a mathematically rigorous implementation.

Run: python transformer_kan.py --help
"""

import math
import argparse
import time
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ------------------------------
# Tiny Shakespeare dataset helper
# ------------------------------
class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s: str):
        return [self.stoi[ch] for ch in s]

    def decode(self, idxs):
        return ''.join(self.itos[i] for i in idxs)


class TinyShakespeareDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx: idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1: idx + 1 + self.seq_len], dtype=torch.long)
        return x, y


# ------------------------------
# Positional encoding
# ------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


# ------------------------------
# Multi-head attention (standard)
# ------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, n_heads, T, head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        return out


# ------------------------------
# Standard FeedForward
# ------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# ------------------------------
# KAN: simplified Kolmogorov-Arnold Network
# ------------------------------
class KANFeedForward(nn.Module):
    """
    Practical approximation: implement the KA superposition idea.
    We'll project the d_model vector into m scalar projections (via learned linear maps),
    pass each scalar through a small scalar MLP (same architecture but different parameters),
    then linearly combine results to produce d_model output.

    This module has an input projection W_in: (d_model -> m), scalar MLPs g_i: R->R, and
    readout W_out: (m -> d_model). Optionally a final non-linearity.
    """

    def __init__(self, d_model, d_ff, m=64, inner_hidden=16, dropout=0.1):
        super().__init__()
        # m is number of inner scalar channels (Kolmogorov "inner functions")
        self.m = m
        self.d_model = d_model
        self.in_proj = nn.Linear(d_model, m)
        # create m small scalar MLPs; for efficiency we'll batch them as a single MLP operating on (B, T, m, 1)
        # implement as a Conv1d with kernel=1 across the scalar dimension
        # but easier: use a ModuleList of tiny MLPs
        self.scalar_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, inner_hidden),
                nn.ReLU(),
                nn.Linear(inner_hidden, 1)
            ) for _ in range(m)
        ])
        self.out_proj = nn.Linear(m, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d_model)
        B, T, D = x.size()
        # project to m scalar channels
        s = self.in_proj(x)  # (B, T, m)
        # apply scalar mlps elementwise across B,T for each of the m channels
        s = s.view(-1, self.m)  # (B*T, m)
        out_channels = []
        # apply scalar MLPs per channel
        for i in range(self.m):
            si = s[:, i:i + 1]  # (B*T, 1)
            gi = self.scalar_mlps[i](si)  # (B*T, 1)
            out_channels.append(gi)
        out = torch.cat(out_channels, dim=1)  # (B*T, m)
        out = out.view(B, T, self.m)
        out = self.dropout(out)
        out = self.out_proj(out)  # (B, T, d_model)
        return out


# ------------------------------
# Encoder / Decoder layer
# ------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, ffn_type='ff', kan_m=64):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if ffn_type == 'ff':
            self.ff = FeedForward(d_model, d_ff, dropout)
        elif ffn_type == 'kan':
            self.ff = KANFeedForward(d_model, d_ff, m=kan_m, dropout=dropout)
        else:
            raise ValueError('ffn_type must be "ff" or "kan"')
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self attention
        attn_out = self.self_attn(x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, ffn_type='ff', kan_m=64):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        if ffn_type == 'ff':
            self.ff = FeedForward(d_model, d_ff, dropout)
        elif ffn_type == 'kan':
            self.ff = KANFeedForward(d_model, d_ff, m=kan_m, dropout=dropout)
        else:
            raise ValueError('ffn_type must be "ff" or "kan"')
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        attn_out = self.self_attn(x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        cross_out = self.cross_attn(x=x, mask=memory_mask) if False else self.cross_attn(x, mask=memory_mask)
        # cross_attn expects x being queries with memory as keys/values. We'll implement cross attention manually:
        # But our MultiHeadAttention is implemented only for q=k=v=x. To keep things simple, we reuse it by shaping.
        # For correct cross-attention, we will add a helper inline:
        # Implement cross-attn manually below
        # (Simpler approach: create q from x and k/v projections from memory)
        # Instead of rewriting MHA here, use a simple linear projections to emulate cross-attn.
        # For clarity and correctness, implement a CrossAttention class or reuse MultiHeadAttention but with queries/kv separate.

        # --- implement cross-attention properly ---
        # project queries
        B, Tq, C = x.size()
        Tb = memory.size(1)
        # project q
        q_proj = self.cross_attn.qkv_proj(x)  # returns qkv but we will split
        q_proj = q_proj.view(B, Tq, 3, self.cross_attn.n_heads, self.cross_attn.head_dim).permute(2, 0, 3, 1, 4)
        q = q_proj[0]
        # project k,v from memory using the same qkv_proj weights (but it's okay for a simple implementation)
        kv = self.cross_attn.qkv_proj(memory)
        kv = kv.view(B, Tb, 3, self.cross_attn.n_heads, self.cross_attn.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[1], kv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.cross_attn.head_dim)
        if memory_mask is not None:
            scores = scores.masked_fill(memory_mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        cross = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Tq, C)
        cross = self.cross_attn.out_proj(cross)

        x = self.norm2(x + self.dropout(cross))
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


# ------------------------------
# Encoder / Decoder stacks
# ------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, ffn_type='ff', kan_m=64):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, ffn_type, kan_m)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = self.tok_emb(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, ffn_type='ff', kan_m=64):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, ffn_type, kan_m)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = self.tok_emb(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.norm(x)


# ------------------------------
# Full Transformer model
# ------------------------------
class TransformerKAN(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=2, n_heads=4, d_ff=1024, dropout=0.1, ffn_type='ff', kan_m=64):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, n_layers, n_heads, d_ff, dropout, ffn_type, kan_m)
        self.decoder = Decoder(vocab_size, d_model, n_layers, n_heads, d_ff, dropout, ffn_type, kan_m)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, src_mask=src_mask)
        dec = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        logits = self.output_proj(dec)
        return logits


# ------------------------------
# Masks helpers
# ------------------------------
def subsequent_mask(size: int):
    # causal mask for target (1 = keep, 0 = mask)
    attn_shape = (1, size, size)
    subsequent = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.bool)
    return ~subsequent


# ------------------------------
# Training utilities
# ------------------------------

def train_epoch(model, dataloader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        # prepare target input and target output
        tgt_in = x  # for character-level next-token prediction typical simplification
        logits = model(x, tgt_in)
        B, T, V = logits.size()
        loss = criterion(logits.view(B * T, V), y.view(B * T))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            tgt_in = x
            logits = model(x, tgt_in)
            B, T, V = logits.size()
            loss = criterion(logits.view(B * T, V), y.view(B * T))
            total_loss += loss.item()
    return total_loss / len(dataloader)


# ------------------------------
# Main script: arguments & orchestration
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='tiny_shakespeare.txt', help='path to Tiny Shakespeare text')
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ffn_type', type=str, default='kan', choices=['ff', 'kan'], help='which feed-forward to use')
    parser.add_argument('--kan_m', type=int, default=64, help='number of scalar channels for KAN')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # load data
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"Please place tiny_shakespeare.txt at {args.data_path}. You can download the original file from public sources.")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    n = len(lines)
    train_end = int(0.8 * n)
    val_end = int(0.85 * n)  # 5% after train

    train_lines = lines[:train_end]
    val_lines = lines[train_end:val_end]
    test_lines = lines[val_end:]

    # keep copies for later evaluation script
    with open("train_lines.txt", "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    with open("val_lines.txt", "w", encoding="utf-8") as f:
        f.writelines(val_lines)
    with open("test_lines.txt", "w", encoding="utf-8") as f:
        f.writelines(test_lines)

    # join back into text for tokenizer
    train_text = "".join(train_lines)
    val_text = "".join(val_lines)
    test_text = "".join(test_lines)

    # tokenizer vocab built on full corpus
    tokenizer = CharTokenizer(train_text + val_text + test_text)
    train_data = tokenizer.encode(train_text)
    val_data = tokenizer.encode(val_text)
    test_data = tokenizer.encode(test_text)

    train_ds = TinyShakespeareDataset(train_data, args.seq_len)
    val_ds = TinyShakespeareDataset(val_data, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device(args.device)

    model = TransformerKAN(tokenizer.vocab_size,
                           d_model=args.d_model,
                           n_layers=args.n_layers,
                           n_heads=args.n_heads,
                           d_ff=args.d_ff,
                           ffn_type=args.ffn_type,
                           kan_m=args.kan_m).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        t1 = time.time()
        print(
            f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - time: {t1 - t0:.1f}s")
        # save
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'args': vars(args)
        }
        torch.save(ckpt, os.path.join(args.save_dir, f'ckpt_epoch_{epoch}.pt'))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.save_dir, 'best_ckpt.pt'))

    print('Training finished. Best val loss:', best_val)


if __name__ == '__main__':
    main()
