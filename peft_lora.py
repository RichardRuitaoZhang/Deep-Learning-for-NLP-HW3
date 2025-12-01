import argparse
import time
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast

# import base GPT implementation from starter.py
from starter import TransformerGPT


# ============================================================
# LoRA linear layer
# ============================================================

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) wrapper for a Linear layer.

    y = W x + (B A) x * (alpha / r)

    We freeze the original W outside this module and only train A and B.
    """
    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        self.r = rank
        self.scaling = alpha / rank

        # A: [r, in_features],  B: [out_features, r]
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # initialization: A ~ Kaiming, B = 0 -> initially no LoRA effect
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base path (W is frozen)
        result = self.original_layer(x)  # [*, out_features]

        # LoRA path
        # x: [*, in], A: [r, in], B: [out, r]
        # x @ A^T -> [*, r]; then @ B^T -> [*, out]
        lora_out = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return result + lora_out


# ============================================================
# Build GPT model with LoRA injected into attention projections
# ============================================================

def get_modelGPT(opt, vocab_size: int) -> TransformerGPT:
    """
    Construct TransformerGPT and inject LoRA modules into
    the attention Q/K/V/Out linear projections in each layer.

    All original parameters are frozen; only LoRA parameters are trainable.
    """
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1.0

    # instantiate base GPT
    model = TransformerGPT(vocab_size,
                           opt.d_model,
                           opt.n_layers,
                           opt.heads,
                           opt.dropout,
                           opt)

    # load pretrained weights if provided
    if opt.loadname is not None:
        ckpt_path = os.path.join(opt.loadname, "model_weights")
        print(f"Loading pretrained weights from {ckpt_path} ...")
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
    else:
        # random init (not recommended for HW3, but keep for completeness)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # 1) freeze all existing parameters
    print("Freezing base model parameters...")
    for param in model.parameters():
        param.requires_grad = False

    # 2) inject LoRA into attention projections
    print("Injecting LoRA modules into attention projections (Q, K, V, Out)...")
    lora_rank = opt.lora_rank

    # model.decoder.layers is a ModuleList of DecoderLayerGPT
    for layer in model.decoder.layers:
        attn = layer.attn_1  # MultiHeadAttention

        # replace q_linear, k_linear, v_linear, out with LoRALinear wrappers
        attn.q_linear = LoRALinear(attn.q_linear, rank=lora_rank, alpha=opt.lora_alpha)
        attn.k_linear = LoRALinear(attn.k_linear, rank=lora_rank, alpha=opt.lora_alpha)
        attn.v_linear = LoRALinear(attn.v_linear, rank=lora_rank, alpha=opt.lora_alpha)
        attn.out      = LoRALinear(attn.out,      rank=lora_rank, alpha=opt.lora_alpha)

    # move to device
    model.to(opt.device)

    return model


# ============================================================
# OpenBookQA dataset for generative GPT training
# ============================================================

class OpenBookQAGenDataset(Dataset):
    """
    Each training sample is a single sequence:

        [START] fact question [A] ... [B] ... [C] ... [D] ... [ANSWER] <correct_option>

    For validation, we remove the final option so that the model must
    predict it from the last position.
    """
    def __init__(self, path: str, tokenizer, seq_len: int = 512, train: bool = True):
        self.samples = []
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        labels = ["A", "B", "C", "D"]

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split("|")]
                if len(parts) != 7:
                    continue
                fact, question, A, B, C, D, ans = parts
                label_idx = label_map.get(ans, -1)
                if label_idx == -1:
                    continue

                if train:
                    text = (
                        f"[START] {fact} {question} "
                        f"[A] {A} [B] {B} [C] {C} [D] {D} "
                        f"[ANSWER] {labels[label_idx]}"
                    )
                else:
                    text = (
                        f"[START] {fact} {question} "
                        f"[A] {A} [B] {B} [C] {C} [D] {D} "
                        f"[ANSWER]"
                    )

                token_ids = tokenizer.encode(
                    text,
                    truncation=True,
                    max_length=seq_len,
                    add_special_tokens=False,
                )
                self.samples.append((torch.tensor(token_ids, dtype=torch.long),
                                     label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """
    Simple left-padded batch collation.
    """
    ids_list, labels = zip(*batch)
    max_len = max(len(x) for x in ids_list)

    batch_size = len(ids_list)
    padded = torch.zeros(batch_size, max_len, dtype=torch.long)

    for i, ids in enumerate(ids_list):
        padded[i, : len(ids)] = ids

    return padded, torch.tensor(labels, dtype=torch.long)


# ============================================================
# Training utilities
# ============================================================

def create_causal_mask(batch_size: int, seq_len: int, device) -> torch.Tensor:
    """
    Create a standard causal (no-peek) mask of shape [batch_size, seq_len, seq_len]
    compatible with the attention() implementation in starter.py.
    """
    # upper triangular (k=1) -> positions that should be masked
    nopeak = np.triu(np.ones((seq_len, seq_len), dtype=np.int32), k=1)
    nopeak = torch.from_numpy(nopeak)  # [seq, seq]
    mask = (nopeak == 0).unsqueeze(0).expand(batch_size, -1, -1)  # [B, seq, seq]
    return mask.to(device)


def train_one_epoch(model,
                    dataloader: DataLoader,
                    criterion,
                    optimizer,
                    device,
                    id_list: torch.Tensor) -> float:
    """
    Train for one epoch. Loss is computed only on the last token,
    which corresponds to the answer choice after [ANSWER].
    """
    model.train()
    total_loss = 0.0

    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)        # [B, L]
        labels = labels.to(device)              # [B]

        batch_size, seq_len = input_ids.size()
        mask = create_causal_mask(batch_size, seq_len, device)

        optimizer.zero_grad()
        logits = model(input_ids, mask)         # [B, L, V]

        # we want the distribution over the NEXT token after the last position
        # here assume last position in sequence corresponds to the answer token
        last_logits = logits[:, -1, :]          # [B, V]

        # target token ids for A/B/C/D
        target_ids = id_list[labels]            # [B]
        loss = criterion(last_logits, target_ids)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model,
             dataloader: DataLoader,
             device,
             id_list: torch.Tensor) -> float:
    """
    Evaluate accuracy on validation set.
    We only look at the probabilities assigned to A/B/C/D tokens.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)    # [1, L] typically
            labels = labels.to(device)

            batch_size, seq_len = input_ids.size()
            mask = create_causal_mask(batch_size, seq_len, device)

            logits = model(input_ids, mask)     # [B, L, V]
            last_logits = logits[:, -1, :]      # [B, V]

            # restrict to candidate tokens (A/B/C/D)
            # id_list: [4] containing token ids for " A", " B", " C", " D"
            cand_logits = last_logits[:, id_list]    # [B, 4]
            preds = cand_logits.argmax(dim=1)        # [B] in {0,1,2,3}

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0
    return correct / total


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/obqa.train.txt")
    parser.add_argument("--valid_path", type=str, default="data/obqa.valid.txt")
    parser.add_argument("--loadname", type=str, default="pretrain")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--norm", type=float, default=2.0)
    parser.add_argument("--tied", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    args = parser.parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(f"Using device: {device}")

    # tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    vocab_size = len(tokenizer)
    args.vocab_size = vocab_size

    # indices tensor used for tied embeddings in starter.py
    args.indices = torch.arange(vocab_size, dtype=torch.long, device=device)

    # build model with LoRA
    model = get_modelGPT(args, vocab_size)

    # parameter statistics for report
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("LoRA parameter summary:")
    print(f"  Total parameters:      {total_params}")
    print(f"  Trainable parameters:  {trainable_params}")
    print(f"  Trainable ratio:       {trainable_params / total_params:.4%}")

    # datasets and loaders
    train_dataset = OpenBookQAGenDataset(args.train_path, tokenizer,
                                         seq_len=args.seqlen, train=True)
    valid_dataset = OpenBookQAGenDataset(args.valid_path, tokenizer,
                                         seq_len=args.seqlen, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # map answer letters to GPT2 token ids (with leading space)
    ans_token_ids = {
        ch: tokenizer.encode(" " + ch, add_special_tokens=False)[-1]
        for ch in "ABCD"
    }
    id_list = torch.tensor(
        [ans_token_ids[ch] for ch in "ABCD"],
        dtype=torch.long,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()
    # only train parameters with requires_grad=True (i.e., LoRA)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device, id_list)
        val_acc = evaluate(model, valid_loader, device, id_list)
        print(f"Epoch {epoch:02d} | train_loss = {train_loss:.4f} "
              f"| val_acc = {val_acc:.4f}")

    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed / 60.0:.2f} minutes")


if __name__ == "__main__":
    main()
