import torch
import time

print("PyTorch:", torch.__version__)
print("XPU available:", torch.xpu.is_available())

if not torch.xpu.is_available():
    print("Intel GPU not detected by PyTorch.")
    raise SystemExit

device = torch.device("xpu")

print("GPU:", torch.xpu.get_device_name(0))

a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)

# warm-up
c = a @ b
torch.xpu.synchronize()

start = time.time()

for _ in range(10):
    c = a @ b

torch.xpu.synchronize()

elapsed = time.time() - start

print("Result device:", c.device)
print("10 matrix multiplies took:", elapsed, "seconds")
print("Success.")