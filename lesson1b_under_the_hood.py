# 第一课（下）：拆开 model.encode()，手写一句话变成向量的全过程
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

NAME = "BAAI/bge-small-zh-v1.5"
tokenizer = AutoTokenizer.from_pretrained(NAME)
bert = AutoModel.from_pretrained(NAME).to("cuda").eval()

text = "高数挂科怎么办"

# ---------- 第 1 步：分词。文字 -> 编号 ----------
# 模型不认识汉字，只认识数字。词表里每个字/词都有一个编号
inputs = tokenizer(text, return_tensors="pt").to("cuda")
ids = inputs["input_ids"][0]
print("第 1 步 分词")
print("  切分结果:", tokenizer.convert_ids_to_tokens(ids))
print("  对应编号:", ids.tolist())
print("  词表大小:", tokenizer.vocab_size)
# [CLS] 和 [SEP] 是自动加的特殊标记：[CLS] 放开头，后面会用来代表整句话

# ---------- 第 2 步：查表。编号 -> 初始向量 ----------
# 模型里有一张 词表大小 x 512 的大表，每个编号查出一行，就是这个字的初始向量
table = bert.embeddings.word_embeddings.weight
print("\n第 2 步 查表")
print("  查找表形状:", tuple(table.shape))
print("  '高' 的初始向量前 5 个数:", table[ids[1]][:5].tolist())
# 这时每个字的向量是固定的，"高"在哪句话里都一样，还不懂上下文

# ---------- 第 3 步：过 Transformer。每个字看其他字，更新自己 ----------
# 注意力机制：每个字根据和其他字的相关程度，把它们的信息混进自己的向量
# 经过 4 层这样的更新，"挂科"的向量里就带上了"高数"的信息
with torch.no_grad():
    out = bert(**inputs).last_hidden_state[0]  # 形状：字数 x 512
print("\n第 3 步 Transformer")
print("  输出形状:", tuple(out.shape), "（每个字一个 512 维向量）")
print("  Transformer 层数:", bert.config.num_hidden_layers)

# ---------- 第 4 步：池化。很多个字的向量 -> 一个句子向量 ----------
# bge 的做法：直接取 [CLS] 位置的向量代表整句。它在每一层都看过所有字
sentence_vec = out[0]
print("\n第 4 步 池化：取 [CLS] 位置的向量，长度", sentence_vec.shape[0])

# ---------- 第 5 步：归一化。把向量长度缩放到 1 ----------
# 这样两个向量的点积就正好等于夹角的余弦，范围 -1 到 1
sentence_vec = sentence_vec / sentence_vec.norm()
print("\n第 5 步 归一化后长度:", round(sentence_vec.norm().item(), 4))

# ---------- 验证：手写结果和库的 encode() 一致吗 ----------
st = SentenceTransformer(NAME, device="cuda")
lib_vec = st.encode(text, normalize_embeddings=True, convert_to_tensor=True)
print("\n验证：手写向量 vs 库的结果，最大误差 =",
      f"{(sentence_vec - lib_vec).abs().max().item():.2e}")
