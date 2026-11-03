# 第一课：向量模型怎么判断两句话"意思相近"
import os

# 国内访问 Hugging Face 慢，走镜像站；必须在导入 sentence_transformers 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from sentence_transformers import SentenceTransformer

# 第一次运行会下载模型（约 100MB），之后从本地缓存读取
model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cuda")

# 1. 一句话 -> 一个向量
vec = model.encode("高数挂科怎么办")
print("向量长度:", vec.shape[0])
print("前 5 个数:", vec[:5].round(3))

# 2. 一个问题，和几段候选文本比相似度
question = "高数挂科怎么办"
candidates = [
    "微积分不及格的同学可以在下学期开学参加补考",  # 意思相同，但几乎没有共同的词
    "高数课本第三章讲的是导数的应用",  # 有共同的词"高数"，但没回答问题
    "食堂二楼周末不营业",  # 完全无关
]

# normalize_embeddings=True：把向量长度缩放到 1，这样点积就等于余弦相似度
q_vec = model.encode(question, normalize_embeddings=True)
c_vecs = model.encode(candidates, normalize_embeddings=True)
scores = c_vecs @ q_vec

print(f"\n问题：{question}")
for text, score in sorted(zip(candidates, scores), key=lambda p: -p[1]):
    print(f"  {score:.3f}  {text}")
