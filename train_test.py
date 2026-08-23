import torch
import torch.nn as nn

device = torch.device(
    "xpu" if torch.xpu.is_available() else "cpu"
)

print("Training on:", device)

model = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 1),
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

for step in range(1000):

    # Create fake training data
    x = torch.randn(512, 100, device=device)

    # Target = sum of all 100 inputs
    y = x.sum(dim=1, keepdim=True)

    prediction = model(x)

    loss = ((prediction - y) ** 2).mean()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 100 == 0:
        print(
            f"step {step:4d} | "
            f"loss {loss.item():.4f}"
        )

