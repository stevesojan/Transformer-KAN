import torch
import torch.nn as nn
import math
import argparse
import json
from transformer_kan import TransformerKAN, CharTokenizer

def load_tokenizer(vocab_file, full_text):
    if vocab_file:
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
        tokenizer = CharTokenizer(full_text)
        tokenizer.stoi = vocab_dict["stoi"]
        tokenizer.itos = vocab_dict["itos"]
        tokenizer.vocab_size = len(tokenizer.stoi)
    else:
        tokenizer = CharTokenizer(full_text)
    return tokenizer

def evaluate_model(checkpoint_path, data_path, device="cuda", vocab_file=None):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint["args"]

    # Load dataset
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    n = len(lines)
    test_lines = lines[int(0.85 * n):]  # last 15% of lines
    test_text = "".join(test_lines)

    # Tokenizer (rebuild or reload from saved file)
    with open(data_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    tokenizer = load_tokenizer(vocab_file, full_text)

    test_ids = torch.tensor(tokenizer.encode(test_text), dtype=torch.long)

    # Batchify test set
    batch_size = 16
    seq_len = model_args.get("seq_len", 128)
    num_batches = len(test_ids) // (batch_size * seq_len)
    test_ids = test_ids[: num_batches * batch_size * seq_len]
    test_data = test_ids.view(batch_size, -1).to(device)

    def get_batch(i):
        x = test_data[:, i:i+seq_len]
        y = test_data[:, i+1:i+1+seq_len]
        return x, y

    # Rebuild model with saved args
    model = TransformerKAN(
        vocab_size=tokenizer.vocab_size,
        d_model=model_args["d_model"],
        n_layers=model_args["n_layers"],
        n_heads=model_args["n_heads"],
        d_ff=model_args["d_ff"],
        dropout=0.1,
        ffn_type=model_args["ffn_type"],
        kan_m=model_args["kan_m"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    total_loss, total_tokens, correct = 0, 0, 0

    with torch.no_grad():
        for i in range(0, test_data.size(1) - seq_len, seq_len):
            x, y = get_batch(i)
            logits = model(x, x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    accuracy = correct / total_tokens

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Perplexity: {perplexity:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_ckpt.pt")
    parser.add_argument("--data_path", type=str, default="tiny_shakespeare.txt")
    parser.add_argument("--vocab_file", type=str, default=None, help="Optional path to saved tokenizer vocab")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    evaluate_model(args.checkpoint, args.data_path, args.device, args.vocab_file)
