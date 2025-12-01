import argparse
import time
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast

# import GPT model definition from starter.py
from starter import TransformerGPT

# ============================================
# Prefix-Tuning: modify only MultiHeadAttention
# ============================================

class PrefixMultiHeadAttention(nn.Module):
    """
    Multi-head attention with learnable prefix key/value.
    Only the prefix_k / prefix_v params are trainable.
    """
    def __init__(self, heads, d_model, seqlen, norm, dropout=0.1, prefix_len=10):
        super().__init__()
        self.h = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.prefix_len = prefix_len

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        # prefix key/value: shape [heads, prefix_len, d_k]
        self.prefix_k = nn.Parameter(torch.randn(heads, prefix_len, self.d_k) * 0.02)
        self.prefix_v = nn.Parameter(torch.randn(heads, prefix_len, self.d_k) * 0.02)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # project Q,K,V
        Q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        K = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        V = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)

        # prefix batch expansion
        pK = self.prefix_k.unsqueeze(0).expand(bs, -1, -1, -1)
        pV = self.prefix_v.unsqueeze(0).expand(bs, -1, -1, -1)

        # concat prefix + projected K,V
        K = torch.cat([pK, K], dim=2)
        V = torch.cat([pV, V], dim=2)

        # extend mask to match prefix length
        if mask is not None:
            prefix_mask = torch.ones(bs, 1, mask.size(-2), self.prefix_len, device=mask.device)
            mask = torch.cat([prefix_mask, mask], dim=-1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        output = torch.matmul(scores, V)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        return self.out(output)


# ============================================
# Replace only attention layer inside TransformerGPT
# ============================================

def get_modelGPT(opt, vocab_size):

    # build base model exactly like HW2
    model = TransformerGPT(
        vocab_size=vocab_size,
        d_model=opt.d_model,
        N=opt.n_layers,
        heads=opt.heads,
        dropout=opt.dropout,
        opt=opt   # HW2 uses opt to pass seqlen, device, etc.
    )

    print(f"Loading pretrained weights from {opt.loadname}/model_weights ...")
    model.load_state_dict(
        torch.load(f"{opt.loadname}/model_weights", map_location="cpu"),
        strict=False
    )

    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # inject prefix-tuning
    for layer in model.decoder.layers:
        h = layer.attn_1.h
        d = layer.attn_1.d_model

        layer.attn_1 = PrefixMultiHeadAttention(
            heads=h,
            d_model=d,
            seqlen=opt.seqlen,
            dropout=opt.dropout,
            prefix_len=10
        )

    # enable training for prefix parameters only
    for layer in model.decoder.layers:
        for p in layer.attn_1.parameters():
            p.requires_grad = True

    if opt.device.type == "cuda":
        model = model.cuda()

    return model


# ============================================
# Dataset + collate (same as HW2 / LoRA version)
# ============================================

class OpenBookQAGenDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len=512, train=True):
        self.samples = []
        label_map = {"A":0,"B":1,"C":2,"D":3}
        labels = ["A","B","C","D"]

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) != 7:
                    continue
                fact, q, A, B, C, D, ans = parts
                if ans not in label_map:
                    continue
                if train:
                    text = f"[START] {fact} {q} [A] {A} [B] {B} [C] {C} [D] {D} [ANSWER] {ans}"
                else:
                    text = f"[START] {fact} {q} [A] {A} [B] {B} [C] {C} [D] {D} [ANSWER]"
                ids = tokenizer.encode(text, max_length=seq_len, truncation=True)
                self.samples.append((torch.tensor(ids), label_map[ans]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    ids, labels = zip(*batch)
    max_len = max(len(x) for x in ids)
    pad = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(ids):
        pad[i, :len(x)] = x
    return pad, torch.tensor(labels)


# ============================================
# causal mask + train & eval
# ============================================

def create_causal_mask(seq, device):
    mask = np.triu(np.ones((1, seq, seq)), k=1)
    mask = torch.tensor(mask) == 0
    return mask.to(device)

def train_loop(model, loader, crit, optim, device, id_list):
    model.train()
    total = 0
    for ids, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        seq = ids.size(1)
        mask = create_causal_mask(seq, device)
        optim.zero_grad()
        logits = model(ids, mask)
        last = logits[:, -1, :]
        target = id_list[labels]
        loss = crit(last, target)
        loss.backward()
        optim.step()
        total += loss.item()
    return total / len(loader)

def evaluate(model, loader, id_list, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for ids, labels in loader:
            ids, labels = ids.to(device), labels.to(device)
            seq = ids.size(1)
            mask = create_causal_mask(seq, device)
            logits = model(ids, mask)
            last = logits[:, -1, :]
            four = last[:, id_list]
            pred = four.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total


# ============================================
# main()
# ============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/obqa.train.txt")
    parser.add_argument("--valid_path", default="data/obqa.valid.txt")
    parser.add_argument("--loadname", default="pretrain")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", args.device)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    vocab = len(tokenizer)

    model = get_modelGPT(args, vocab)

    # parameter summary (homework requires this)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nPrefix-Tuning parameter summary:")
    print("  Total params:     ", total)
    print("  Trainable params: ", trainable)
    print("  Trainable ratio:   {:.4f}%\n".format(100 * trainable / total))

    train_set = OpenBookQAGenDataset(args.train_path, tokenizer, args.seqlen, train=True)
    val_set = OpenBookQAGenDataset(args.valid_path, tokenizer, args.seqlen, train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, collate_fn=collate_fn)

    # A/B/C/D token ids
    ans_ids = {k: tokenizer.encode(" " + k)[-1] for k in "ABCD"}
    id_list = torch.tensor([ans_ids[c] for c in "ABCD"], device=args.device)

    crit = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    for e in range(args.epochs):
        loss = train_loop(model, train_loader, crit, optim, args.device, id_list)
        acc = evaluate(model, val_loader, id_list, args.device)
        print(f"Epoch {e+1:02d} | train_loss={loss:.4f} | val_acc={acc:.4f}")

    print("Prefix-Tuning done.")


if __name__ == "__main__":
    main()
