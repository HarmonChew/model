with open(
    "data/input.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()

print("Number of characters:", len(text))

chars = sorted(list(set(text)))

print("Vocabulary size:", len(chars))
print(chars)
vocab_size = len(chars)

stoi = {
    ch: i
    for i, ch in enumerate(chars)
}

itos = {
    i: ch
    for i, ch in enumerate(chars)
}

def encode(s):
    return [stoi[c] for c in s]

def decode(numbers):
    return "".join(
        itos[i] for i in numbers
    )

test = "hello"

encoded = encode(test)

print(encoded)
print(decode(encoded))

import torch

data = torch.tensor(
    encode(text),
    dtype=torch.long
)

print(data.shape)
print(data[:100])

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration

torch.manual_seed(1337)

device = (
    "xpu"
    if hasattr(torch, "xpu") and torch.xpu.is_available()
    else "cpu"
)

batch_size = 32
block_size = 8

learning_rate = 1e-2
max_iters = 3000
eval_interval = 300
eval_iters = 100

print("Device:", device)

if device == "xpu":
    print("GPU:", torch.xpu.get_device_name(0))

print("Vocabulary size:", vocab_size)
print(
    "Uniform-loss baseline:",
    math.log(vocab_size)
)

# Batch creation

def get_batch(split):
    data = (
        train_data
        if split == "train"
        else val_data
    )

    ix = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x.to(device), y.to(device)

# Model

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.token_transition_table = nn.Embedding(
            vocab_size,
            vocab_size
        )

    def forward(self, idx, targets=None):

        # idx:
        # (B, T)

        logits = self.token_transition_table(idx)

        # logits:
        # (B, T, V)

        loss = None

        if targets is not None:

            B, T, V = logits.shape

            logits_flat = logits.reshape(
                B * T,
                V
            )

            targets_flat = targets.reshape(
                B * T
            )

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            logits, _ = self(idx)

            # Predictions at final time position
            logits = logits[:, -1, :]

            # Convert logits to probabilities
            probs = F.softmax(
                logits,
                dim=-1
            )

            # Sample next character
            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            # Append to sequence
            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

        return idx

# Instantiate

model = BigramLanguageModel(
    vocab_size
).to(device)

num_params = sum(
    p.numel()
    for p in model.parameters()
)

print("Parameters:", num_params)

# Shape sanity check

xb, yb = get_batch("train")

logits, loss = model(xb, yb)

print()
print("xb:", xb.shape)
print("yb:", yb.shape)
print("logits:", logits.shape)
print("initial loss:", loss.item())

# Evaluation

@torch.no_grad()
def estimate_loss():

    model.eval()

    out = {}

    for split in ["train", "val"]:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            xb, yb = get_batch(split)

            _, loss = model(xb, yb)

            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()

    return out


# Training

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

for step in range(max_iters):

    if step % eval_interval == 0:

        losses = estimate_loss()

        print(
            f"step {step:4d} | "
            f"train {losses['train']:.4f} | "
            f"val {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    optimizer.step()

# Final evaluation

losses = estimate_loss()

print()
print("Final losses:")
print("train:", losses["train"])
print("val:  ", losses["val"])

# Generate text

model.eval()

start_id = stoi.get("\n", 0)

context = torch.tensor(
    [[start_id]],
    dtype=torch.long,
    device=device
)

with torch.no_grad():

    generated = model.generate(
        context,
        max_new_tokens=500
    )

generated_text = decode(
    generated[0].cpu().tolist()
)

print()
print("----- generated text -----")
print(generated_text)


for current_char in ["q", "t", "e", " ", "\n"]:
    token_id = stoi[current_char]

    logits = model.token_transition_table.weight[token_id]
    probs = F.softmax(logits, dim=-1)

    values, indices = torch.topk(probs, k=10)

    print(f"\nAfter {repr(current_char)}:")

    for probability, next_id in zip(
        values.detach().cpu().tolist(),
        indices.detach().cpu().tolist()
    ):
        print(
            repr(itos[next_id]),
            f"{probability:.3f}"
        )