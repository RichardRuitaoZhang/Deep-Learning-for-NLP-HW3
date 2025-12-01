import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast

# import original classes from starter.py (do NOT modify starter.py)
from starter import TransformerGPT, DecoderLayerGPT


# ====================================================
# Adapter module
# ====================================================

class Adapter(nn.Module):
    """
    Simple bottleneck adapter with residual:
      y = x + up(dropout(ReLU(down(x))))
    """
    def __init__(self, d_model, bottleneck=64, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        z = self.down(x)
        z = self.activation(z)
        z = self.dropout(z)
        z = self.up(z)
        return residual + z


# ====================================================
# Adapter-aware decoder layer (wraps original layer)
# ====================================================

class AdapterDecoderLayer(nn.Module):
    """
    Wraps an existing DecoderLayerGPT and inserts an Adapter
    after the FFN block.
    """
    def __init__(self, original_layer, bottleneck=64):
        super().__init__()
        # copy modules from the original layer
        self.norm_1 = original_layer.norm_1
        self.norm_2 = original_layer.norm_2
        self.dropout_1 = original_layer.dropout_1
        self.dropout_2 = original_layer.dropout_2
        self.attn_1 = original_layer.attn_1
        self.ff = original_layer.ff

        d_model = original_layer.norm_1.size
        self.adapter = Adapter(d_model=d_model, bottleneck=bottleneck)

    def forward(self, x, mask):
        # this is the same structure as in starter.py + adapter
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))

        # adapter after FFN
        x = self.adapter(x)
        return x


# ====================================================
# Inject adapters into the pretrained model
# ====================================================

def inject_adapters(model, bottleneck=64):
    """
    Replace each DecoderLayerGPT in model.decoder.layers
    with an AdapterDecoderLayer that wraps it.
    """
    print("Injecting adapters into decoder layers...")
    for i, layer in enumerate(model.decoder.layers):
        # layer is a DecoderLayerGPT from starter.py
        model.decoder.layers[i] = AdapterDecoderLayer(layer, bottleneck=bottleneck)
    return model


def freeze_all_except_adapters(model):
    """
    Freeze all base Transformer parameters and only train adapters.
    """
    print("Freezing base parameters...")
    for name, p in model.named_parameters():
        p.requires_grad = False

    print("Unfreezing adapter parameters...")
    for i, layer in enumerate(model.decoder.layers):
        for p in layer.adapter.parameters():
            p.requires_grad = True


# ====================================================
# OBQA dataset
# ====================================================

class OpenBookQAGenDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len=512, train=True):
        self.samples = []
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

        with open(path, "r") as f:
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


# ====================================================
# Causal mask (same shape pattern as starter.py)
# ====================================================

def causal_mask(seq_len, device):
    m = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    # starter.py builds mask with shape (batch, L, L); here we use (1, L, L) and rely on broadcasting
    return (m == 0).unsqueeze(0).to(device)


# ====================================================
# Training / evaluation loops
# ====================================================

def train_loop(model, loader, crit, optim, device, id_list):
    model.train()
    total = 0.0

    for ids, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        mask = causal_mask(ids.size(1), device)

        optim.zero_grad()
        logits = model(ids, mask)          # (B, L, V)
        last = logits[:, -1, :]            # (B, V)

        target = id_list[labels]           # (B,)
        loss = crit(last, target)
        loss.backward()
        optim.step()

        total += loss.item()

    return total / len(loader)


def evaluate(model, loader, id_list, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for ids, labels in loader:
            ids, labels = ids.to(device), labels.to(device)
            mask = causal_mask(ids.size(1), device)

            logits = model(ids, mask)
            last = logits[:, -1, :]
            four = last[:, id_list]        # logits for A/B/C/D
            pred = four.argmax(1)

            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ====================================================
# Main
# ====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/obqa.train.txt")
    parser.add_argument("--valid_path", default="data/obqa.valid.txt")
    parser.add_argument("--loadname", default="pretrain")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--norm", type=float, default=2.0)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    # starter.py expects opt.tied
    parser.add_argument("--tied", type=int, default=1)

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", args.device)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    vocab = len(tokenizer)

    # starter.py also expects opt.indices when tied embeddings are used
    args.vocab_size = vocab
    args.indices = torch.arange(vocab, device=args.device)

    # ===== load original pretrained GPT (from starter.py) =====
    model = TransformerGPT(
        vocab_size=vocab,
        d_model=args.d_model,
        N=args.n_layers,
        heads=args.heads,
        dropout=args.dropout,
        opt=args
    )
    print(f"Loading pretrained weights from {args.loadname}/model_weights ...")
    model.load_state_dict(torch.load(f"{args.loadname}/model_weights", map_location="cpu"))
    model.to(args.device)

    # ===== inject adapters and freeze base model =====
    inject_adapters(model, bottleneck=args.bottleneck)
    freeze_all_except_adapters(model)
    model.to(args.device)  # ensure patched modules are on GPU

    # ===== parameter summary =====
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nAdapter Parameter Summary:")
    print("  Total params:     ", total)
    print("  Trainable params: ", trainable)
    print("  Ratio:            {:.4f}%\n".format(100 * trainable / total))

    # ===== load data =====
    train_set = OpenBookQAGenDataset(args.train_path, tokenizer, args.seqlen, train=True)
    val_set = OpenBookQAGenDataset(args.valid_path, tokenizer, args.seqlen, train=False)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        collate_fn=collate_fn
    )

    # token ids for A/B/C/D (space before letter to match GPT2 tokenizer)
    ans_ids = {k: tokenizer.encode(" " + k)[-1] for k in "ABCD"}
    id_list = torch.tensor([ans_ids[c] for c in "ABCD"], device=args.device)

    # ===== training =====
    crit = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    for e in range(args.epochs):
        tl = train_loop(model, train_loader, crit, optim, args.device, id_list)
        acc = evaluate(model, val_loader, id_list, args.device)
        print(f"Epoch {e+1:02d} | train_loss = {tl:.4f} | val_acc = {acc:.4f}")

    print("Adapter Tuning complete.")


if __name__ == "__main__":
    main()
