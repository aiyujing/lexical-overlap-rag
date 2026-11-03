# 第二课：手写一层注意力 softmax(QKᵀ/√d)·V，看「挂」字在关注谁
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModel, AutoTokenizer

NAME = "BAAI/bge-small-zh-v1.5"
tokenizer = AutoTokenizer.from_pretrained(NAME)
# eager：用最朴素的写法算注意力，这样库才能把注意力权重返回给我们对比
bert = AutoModel.from_pretrained(NAME, attn_implementation="eager").to("cuda").eval()

text = "高数挂科怎么办"
inputs = tokenizer(text, return_tensors="pt").to("cuda")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
GUA = tokens.index("挂")
print("句子:", tokens)

with torch.no_grad():
    # ---------- 第 0 步：第 1 层的输入 ----------
    # 上节课的查表向量，再加上位置向量（告诉模型每个字在第几位），做一次归一化
    x = bert.embeddings(input_ids=inputs["input_ids"],
                        token_type_ids=inputs["token_type_ids"])[0]  # 字数 x 512

    # ---------- 第 1 步：每个字生成 Q、K、V ----------
    # 就是三个 512x512 的矩阵乘法，矩阵是训练出来的参数
    layer = bert.encoder.layer[0].attention.self
    Q, K, V = layer.query(x), layer.key(x), layer.value(x)

    # ---------- 第 2 步：拆成多个头 ----------
    # 512 维切成 8 份，每份 64 维，各算各的注意力，相当于 8 个人从不同角度看这句话
    n_heads = bert.config.num_attention_heads
    d = Q.shape[-1] // n_heads
    seq = len(tokens)
    Q, K, V = (t.view(seq, n_heads, d).transpose(0, 1) for t in (Q, K, V))  # 头数 x 字数 x 64

    # ---------- 第 3 步：打分 + softmax ----------
    # Q·Kᵀ：每个字的 Q 和每个字的 K 做点积，得到 字数 x 字数 的相关度表
    # 除以 √d：防止点积太大，softmax 后变成几乎全给一个字
    scores = Q @ K.transpose(-1, -2) / d ** 0.5
    weights = scores.softmax(dim=-1)  # 每一行加起来等于 1

    # ---------- 第 4 步：按权重混合 V ----------
    mixed = weights @ V                                  # 头数 x 字数 x 64
    mixed = mixed.transpose(0, 1).reshape(seq, -1)       # 拼回 字数 x 512

    # ---------- 验证：和库算出来的注意力权重对比 ----------
    lib = bert(**inputs, output_attentions=True).attentions  # 每层一个 1 x 头数 x 字数 x 字数
    print(f"\n验证：手写权重 vs 库，最大误差 = {(weights - lib[0][0]).abs().max().item():.2e}")


def show(w, title):
    print(f"\n{title}")
    for tok, p in sorted(zip(tokens, w.tolist()), key=lambda t: -t[1]):
        print(f"  {tok:>6} {p:.3f} " + "█" * round(p * 40))


# ---------- 结果：「挂」把注意力分给了谁 ----------
show(weights[:, GUA].mean(0), "第 1 层，8 个头平均，「挂」的注意力分布")

print("\n第 1 层，每个头最关注的字（「挂」那一行）:")
for h in range(n_heads):
    w = weights[h, GUA]
    top = w.argmax().item()
    print(f"  头{h}: {tokens[top]} ({w[top].item():.3f})")

show(lib[-1][0][:, GUA].mean(0), f"第 {len(lib)} 层（最后一层），8 个头平均")
