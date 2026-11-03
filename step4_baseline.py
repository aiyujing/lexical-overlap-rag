"""第4步：Baseline 检索实验（BM25 vs 未微调的 bge-small）。

在全部 376 个段落里检索，用测试集的题目评估，并按"问题-段落字面重叠率"分组报结果。
输出：results/baseline.json，results/baseline_per_question.csv（错误分析用）
"""
import csv
import json
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
P = ROOT / "data/processed"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# 重叠率分组：低于 0.5 算"字面不像"，0.5~0.8 中等，0.8 以上"字面很像"
BINS = [("低重叠(<0.5)", 0, 0.5), ("中重叠(0.5-0.8)", 0.5, 0.8), ("高重叠(≥0.8)", 0.8, 1.01)]


def tokenize(s):
    return [w for w in jieba.lcut(s) if w.strip()]


def metrics(ranks):
    """ranks：每道题正确段落的名次（从1开始）。"""
    r = np.array(ranks)
    return {"n": len(r), "R@1": float((r <= 1).mean()), "R@5": float((r <= 5).mean()),
            "R@10": float((r <= 10).mean()), "MRR": float((1 / r).mean())}


def rank_of(scores, gold_idx):
    return int((scores > scores[gold_idx]).sum()) + 1


def main():
    jieba.setLogLevel(60)
    chunks = [json.loads(l) for l in (P / "chunks.jsonl").open(encoding="utf-8")]
    qs = [json.loads(l) for l in (P / "questions.jsonl").open(encoding="utf-8")]
    test = [q for q in qs if q["split"] == "test"]
    idx = {c["id"]: i for i, c in enumerate(chunks)}
    docs = [c["title"] + " " + c["text"] for c in chunks]

    # BM25：按词频打分的传统方法
    bm25 = BM25Okapi([tokenize(d) for d in docs])
    bm25_ranks = [rank_of(bm25.get_scores(tokenize(q["question"])), idx[q["chunk_id"]]) for q in test]

    # bge-small：向量模型，余弦相似度
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cuda")
    D = model.encode(docs, normalize_embeddings=True, batch_size=64)
    Q = model.encode(["为这个句子生成表示以用于检索相关文章：" + q["question"] for q in test],
                     normalize_embeddings=True, batch_size=64)
    S = Q @ D.T
    bge_ranks = [rank_of(S[i], idx[q["chunk_id"]]) for i, q in enumerate(test)]

    result = {}
    for name, ranks in (("BM25", bm25_ranks), ("bge-small", bge_ranks)):
        res = {"全部": metrics(ranks)}
        for kind in ("plain", "paraphrase"):
            res[kind] = metrics([r for r, q in zip(ranks, test) if q["kind"] == kind])
        for label, lo, hi in BINS:
            sub = [r for r, q in zip(ranks, test) if lo <= q["overlap"] < hi]
            if sub:
                res[label] = metrics(sub)
        result[name] = res

    (OUT / "baseline.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "baseline_per_question.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["qid", "kind", "overlap", "question", "bm25_rank", "bge_rank"])
        for q, a, b in zip(test, bm25_ranks, bge_ranks):
            w.writerow([q["qid"], q["kind"], q["overlap"], q["question"], a, b])

    print(f"测试题 {len(test)} 道，候选段落 {len(chunks)} 个")
    for name, res in result.items():
        print(f"\n== {name} ==")
        for g, m in res.items():
            print(f"  {g:<16} n={m['n']:<4} R@1={m['R@1']:.3f} R@5={m['R@5']:.3f} "
                  f"R@10={m['R@10']:.3f} MRR={m['MRR']:.3f}")


if __name__ == "__main__":
    main()
