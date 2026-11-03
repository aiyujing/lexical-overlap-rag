# 环境自检：确认 PyTorch 能用上 4060 显卡，并实际跑一次训练步骤
import time

import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("没检测到显卡，检查驱动或 PyTorch 是否装成了 CPU 版")

device = torch.device("cuda")
props = torch.cuda.get_device_properties(device)
print("显卡:", props.name)
print(f"显存: {props.total_memory / 1024**3:.1f} GB")

# 1. 矩阵乘法测速：半精度（fp16），训练时我们也会用半精度
a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
b = torch.randn(4096, 4096, device=device, dtype=torch.float16)
torch.matmul(a, b)  # 预热，第一次调用包含初始化开销
torch.cuda.synchronize()
start = time.time()
for _ in range(50):
    torch.matmul(a, b)
torch.cuda.synchronize()
elapsed = time.time() - start
tflops = 50 * 2 * 4096**3 / elapsed / 1e12
print(f"半精度矩阵乘法: {tflops:.1f} TFLOPS")

# 2. 跑一个最小的训练循环：前向 -> 算损失 -> 反向传播 -> 更新参数
model = torch.nn.Sequential(
    torch.nn.Linear(768, 768), torch.nn.ReLU(), torch.nn.Linear(768, 2)
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(256, 768, device=device)
y = torch.randint(0, 2, (256,), device=device)
for step in range(100):
    loss = torch.nn.functional.cross_entropy(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f"训练 100 步后损失: {loss.item():.4f}（应明显小于初始的约 0.69）")
print("环境正常，可以开始项目")
