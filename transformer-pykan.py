"""
Full Transformer training script with an option to replace the position-wise Feed-Forward Network
with a Kolmogorov-Arnold Network (KAN) from pykan.

Single-file script that:
- Builds a tokenizer and Tiny Shakespeare dataset loader
- Implements Transformer Encoder / Decoder with a selectable FFN or KAN module
- Training and evaluation loop
- Checkpoint saving

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
from kan import KAN   # official pykan


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
# KAN FeedForward (PyKAN)
# ------------------------------
class KANFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, grid=3, k=3, seed=1, device=None):
        super().__init__()
        self.d_model = d_model
        self.kan = KAN(width=[d_model, d_ff, d_model],
                       grid=grid, k=k, seed=seed, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        x = x.reshape(B*T, D)   # flatten sequence
        x = self.kan(x)         # apply KAN
        x = x.reshape(B, T, D)  # restore shape
        return self.dropout(x)



# ------------------------------
# Encoder / Decoder layer
# ------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, ffn_type='ff',
                 grid=3, k=3, seed=1, device=None):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if ffn_type == 'ff':
            self.ff = FeedForward(d_model, d_ff, dropout)
        elif ffn_type == 'kan':
            self.ff = KANFeedForward(d_model, d_ff, dropout, grid, k, seed, device)
        else:
            raise ValueError('ffn_type must be "ff" or "kan"')

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_out = self.self_attn(x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, ffn_type='ff',
                 grid=3, k=3, seed=1, device=None):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        if ffn_type == 'ff':
            self.ff = FeedForward(d_model, d_ff, dropout)
        elif ffn_type == 'kan':
            self.ff = KANFeedForward(d_model, d_ff, dropout, grid, k, seed, device)
        else:
            raise ValueError('ffn_type must be "ff" or "kan"')

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        attn_out = self.self_attn(x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # simple cross-attention: query = x, key=value = memory
        B, Tq, C = x.size()
        Tb = memory.size(1)
        qkv_x = self.cross_attn.qkv_proj(x)
        qkv_x = qkv_x.view(B, Tq, 3, self.cross_attn.n_heads, self.cross_attn.head_dim).permute(2, 0, 3, 1, 4)
        q = qkv_x[0]

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
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff,
                 dropout=0.1, ffn_type='ff', grid=3, k=3, seed=1, device=None):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, ffn_type, grid, k, seed, device)
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
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff,
                 dropout=0.1, ffn_type='ff', grid=3, k=3, seed=1, device=None):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, ffn_type, grid, k, seed, device)
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
    def __init__(self, vocab_size, d_model=256, n_layers=2, n_heads=4, d_ff=1024,
                 dropout=0.1, ffn_type='ff', grid=3, k=3, seed=1, device=None):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, n_layers, n_heads, d_ff, dropout,
                               ffn_type, grid, k, seed, device)
        self.decoder = Decoder(vocab_size, d_model, n_layers, n_heads, d_ff, dropout,
                               ffn_type, grid, k, seed, device)
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
        tgt_in = x
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
# Main script
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='tiny_shakespeare.txt')
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ffn_type', type=str, default='kan', choices=['ff', 'kan'])
    parser.add_argument('--grid', type=int, default=3, help='KAN grid refinement level')
    parser.add_argument('--k', type=int, default=3, help='KAN basis size')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # load data
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Place tiny_shakespeare.txt at {args.data_path}")

    with open(args.data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    n = len(lines)
    train_end = int(0.8 * n)
    val_end = int(0.85 * n)

    train_lines = lines[:train_end]
    val_lines = lines[train_end:val_end]
    test_lines = lines[val_end:]

    with open("train_lines.txt", "w", encoding="utf-8") as f: f.writelines(train_lines)
    with open("val_lines.txt", "w", encoding="utf-8") as f: f.writelines(val_lines)
    with open("test_lines.txt", "w", encoding="utf-8") as f: f.writelines(test_lines)

    train_text = "".join(train_lines)
    val_text = "".join(val_lines)
    test_text = "".join(test_lines)

    tokenizer = CharTokenizer(train_text + val_text + test_text)
    train_data = tokenizer.encode(train_text)
    val_data = tokenizer.encode(val_text)
    test_data = tokenizer.encode(test_text)

    train_ds = TinyShakespeareDataset(train_data, args.seq_len)
    val_ds = TinyShakespeareDataset(val_data, args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device(args.device)

    model = TransformerKAN(
        tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=0.1,
        ffn_type=args.ffn_type,
        grid=args.grid,
        k=args.k,
        seed=args.seed,
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        t1 = time.time()
        print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - time: {t1 - t0:.1f}s")

        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'args': vars(args)
        }
        torch.save(ckpt, os.path.join(args.save_dir, f'ckpt_epoch_{epoch}.pt'))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pt'))


if __name__ == '__main__':
    main()
