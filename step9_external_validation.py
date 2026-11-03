"""第9步：外部验证。在 DuRetrieval（百度真实搜索问题，C-MTEB）上检验两件事：
1. 字面重叠偏差在真人提问上是否同样存在
2. 在 d2l 教材上微调过的模型，换到完全不同的领域会不会变差

设置：2000 道题，全部 10 万段落作为候选；一道题可能有多个正确段落，任一个排进前 k 就算命中。
一道题的重叠率 = 它和各个正确段落重叠率中的最大值。
输出：results/external.json
"""
import json
import time
from pathlib import Path

import jieba
import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from step3_build_dataset import overlap
from step4_baseline import BINS, metrics
from step5_exp_a_models import BGE_PREFIX

ROOT = Path(__file__).parent
RAW = ROOT / "data/raw/cmteb"
OUT = ROOT / "results"
TOP_K = 20
MAX_CHARS = 1000  # 极少数段落有几万字，截断以免分词和编码太慢（模型本来也只看前 512 个 token）
DENSE = [("bge-base", "BAAI/bge-base-zh-v1.5"), ("bge-base-ft", str(ROOT / "models/bge-base-ft-hardneg"))]


def load():
    corpus = pd.read_parquet(next((RAW / "DuRetrieval").glob("corpus*.parquet")))
    queries = pd.read_parquet(next((RAW / "DuRetrieval").glob("queries*.parquet")))
    qrels = pd.read_parquet(next((RAW / "DuRetrieval-qrels").glob("*.parquet")))
    qrels = qrels[qrels["score"] > 0]
    pid2i = {p: i for i, p in enumerate(corpus["id"])}
    docs = [t[:MAX_CHARS] for t in corpus["text"]]
    gold = qrels.groupby("qid")["pid"].apply(lambda s: {pid2i[p] for p in s if p in pid2i}).to_dict()
    qs = [{"qid": r.id, "question": r.text, "gold": gold[r.id]} for r in queries.itertuples() if gold.get(r.id)]
    for q in qs:
        q["overlap"] = max(overlap(q["question"], docs[i]) for i in q["gold"])
    return docs, qs


def first_rank(order, gold):
    """order：按分数从高到低的段落下标。返回第一个正确段落的名次。"""
    for n, i in enumerate(order, 1):
        if i in gold:
            return n
    return len(order) + 1


def report(ranks, qs):
    res = {"全部": metrics(ranks)}
    for label, lo, hi in BINS:
        sub = [r for r, q in zip(ranks, qs) if lo <= q["overlap"] < hi]
        if sub:
            res[label] = metrics(sub)
    return res


def main():
    jieba.setLogLevel(60)
    docs, qs = load()
    print(f"DuRetrieval：{len(qs)} 道题，{len(docs)} 个段落；重叠率均值 {np.mean([q['overlap'] for q in qs]):.3f}")
    print("  各重叠组题数：", {l: sum(lo <= q["overlap"] < hi for q in qs) for l, lo, hi in BINS})

    tops = {}  # 检索器名 → 每道题的完整排序（只保留前 1000，后面的名次统一记为 1001）
    t0 = time.time()
    bm25 = BM25Okapi([jieba.lcut(d) for d in docs])
    tops["BM25"] = [np.argsort(-bm25.get_scores(jieba.lcut(q["question"])))[:1000] for q in qs]
    del bm25
    print(f"  BM25 完成 {time.time() - t0:.0f}s")

    for name, path in DENSE:
        t0 = time.time()
        model = SentenceTransformer(path, device="cuda", model_kwargs={"torch_dtype": torch.float16})
        model.max_seq_length = 512
        D = model.encode(docs, normalize_embeddings=True, batch_size=64, convert_to_tensor=True)
        Q = model.encode([BGE_PREFIX + q["question"] for q in qs], normalize_embeddings=True, convert_to_tensor=True)
        tops[name] = list(torch.topk(Q @ D.T, 1000, dim=1).indices.cpu().numpy())
        del model, D
        torch.cuda.empty_cache()
        print(f"  {name} 完成 {time.time() - t0:.0f}s")

    reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda", max_length=512,
                            model_kwargs={"torch_dtype": torch.float16})
    result = {"题数": len(qs), "段落数": len(docs),
              "重叠率均值": float(np.mean([q["overlap"] for q in qs]))}
    for name in list(tops):
        ranks = [first_rank(o, q["gold"]) for o, q in zip(tops[name], qs)]
        result[name] = report(ranks, qs)
        t0 = time.time()
        pairs = [(q["question"], docs[i]) for o, q in zip(tops[name], qs) for i in o[:TOP_K]]
        s = reranker.predict(pairs, batch_size=64, show_progress_bar=False).reshape(len(qs), TOP_K)
        rr = [first_rank(list(o[:TOP_K][np.argsort(-s[k])]) + list(o[TOP_K:]), q["gold"])
              for k, (o, q) in enumerate(zip(tops[name], qs))]
        result[name + "+rerank"] = report(rr, qs)
        print(f"  {name}+rerank 完成 {time.time() - t0:.0f}s")

    (OUT / "external.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n每格为 R@1 / R@10")
    groups = ["全部"] + [b[0] for b in BINS]
    for name, r in result.items():
        if isinstance(r, dict):
            print(f"  {name:<18} " + "  ".join(f"{g} {r[g]['R@1']:.3f}/{r[g]['R@10']:.3f}" for g in groups if g in r))


if __name__ == "__main__":
    main()
