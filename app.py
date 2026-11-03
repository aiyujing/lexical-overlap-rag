"""网页问答 demo：输入问题 → 检索段落 → DeepSeek 根据段落回答。

可以切换检索方式，现场对比 BM25 和"微调模型 + reranker"。
用法：python app.py，然后浏览器打开 http://127.0.0.1:7860
"""
import json
import urllib.request
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from step2_gen_questions import API, MODEL, load_key
from step4_baseline import tokenize
from step5_exp_a_models import BGE_PREFIX

ROOT = Path(__file__).parent
TOP_K = 20  # reranker 重排的候选数，和实验C一致
SHOW = 5    # 页面上显示、并交给大模型的段落数

chunks = [json.loads(l) for l in (ROOT / "data/processed/chunks.jsonl").open(encoding="utf-8")]
docs = [c["title"] + " " + c["text"] for c in chunks]
bm25 = BM25Okapi([tokenize(d) for d in docs])
dense = SentenceTransformer(str(ROOT / "models/bge-base-ft-hardneg"), device="cuda",
                            model_kwargs={"torch_dtype": torch.float16})
dense.max_seq_length = 512
D = dense.encode(docs, normalize_embeddings=True, batch_size=32)
reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda", max_length=512,
                        model_kwargs={"torch_dtype": torch.float16})
KEY = load_key()

METHODS = ["微调 bge-base + reranker（最好）", "微调 bge-base", "BM25（关键词匹配）"]


def retrieve(q, method):
    if method.startswith("BM25"):
        s = bm25.get_scores(tokenize(q))
        return [(int(i), float(s[i])) for i in np.argsort(-s)[:SHOW]]
    s = dense.encode([BGE_PREFIX + q], normalize_embeddings=True)[0] @ D.T
    order = np.argsort(-s)[:TOP_K]
    if method == METHODS[1]:
        return [(int(i), float(s[i])) for i in order[:SHOW]]
    r = reranker.predict([(q, docs[i]) for i in order], show_progress_bar=False)
    return [(int(order[j]), float(r[j])) for j in np.argsort(-r)[:SHOW]]


def answer(q, hits):
    ctx = "\n\n".join(f"[{n + 1}] {chunks[i]['title']}\n{chunks[i]['text']}" for n, (i, _) in enumerate(hits))
    prompt = (f"根据下面的教材段落回答问题。只用段落里的信息，用 [编号] 标出处；段落里没有答案就直说。\n\n"
              f"{ctx}\n\n问题：{q}")
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.3}).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"]
    except Exception as e:  # demo 里网络出错不应让页面崩掉
        return f"（调用大模型失败：{e}）"


def run(q, method, use_llm):
    q = q.strip()
    if not q:
        return "请输入问题", ""
    hits = retrieve(q, method)
    md = "\n\n".join(f"**{n + 1}. {chunks[i]['title']}**　得分 {s:.3f}\n\n> {chunks[i]['text'][:300]}…"
                     for n, (i, s) in enumerate(hits))
    return (answer(q, hits) if use_llm else "（未勾选生成回答）"), md


with gr.Blocks(title="深度学习教材问答") as demo:
    gr.Markdown("## 深度学习教材问答（d2l-zh）\n换一种说法提问，对比不同检索方式的结果。")
    with gr.Row():
        q = gr.Textbox(label="问题", placeholder="例如：为什么去掉最后那几个全部连在一起的层，模型反而不容易死记训练数据？", scale=4)
        method = gr.Dropdown(METHODS, value=METHODS[0], label="检索方式", scale=2)
    use_llm = gr.Checkbox(value=True, label="让大模型根据检索结果生成回答")
    btn = gr.Button("提问", variant="primary")
    out_answer = gr.Markdown(label="回答")
    out_hits = gr.Markdown(label="检索到的段落")
    btn.click(run, [q, method, use_llm], [out_answer, out_hits])
    q.submit(run, [q, method, use_llm], [out_answer, out_hits])

if __name__ == "__main__":
    demo.launch()
