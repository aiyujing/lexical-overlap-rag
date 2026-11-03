"""第7步：实验C，在第一阶段检索后面接 reranker（bge-reranker-base），看能否纠正被字面误导的题。

- 第一阶段：BM25 / 原版 bge-base / 微调后的 bge-base（ft-hardneg，seed 0）
- 第二阶段：reranker 把问题和前 K 个候选段落逐对打分、重新排序；K 名以外的段落保持原顺序
- 额外统计：按重叠分组，有多少题被"救回"（原来不是第1、重排后是第1），多少题被"带坏"（反过来）
输出：results/exp_c.json，results/exp_c_per_question.csv
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from step4_baseline import BINS, metrics, tokenize
from step5_exp_a_models import BGE_PREFIX
from step6_exp_b_finetune import group_metrics

ROOT = Path(__file__).parent
P = ROOT / "data/processed"
OUT = ROOT / "results"

TOP_K = 20
RERANKER = "BAAI/bge-reranker-base"
DENSE = [("bge-base", "BAAI/bge-base-zh-v1.5"), ("bge-base-ft", str(ROOT / "models/bge-base-ft-hardneg"))]


def first_stage_scores(chunks, docs, test):
    """返回 {检索器名: 分数矩阵(题数×段落数)}。"""
    out = {}
    bm25 = BM25Okapi([tokenize(d) for d in docs])
    out["BM25"] = np.stack([bm25.get_scores(tokenize(q["question"])) for q in test])
    for name, path in DENSE:
        model = SentenceTransformer(path, device="cuda", model_kwargs={"torch_dtype": torch.float16})
        model.max_seq_length = 512
        D = model.encode(docs, normalize_embeddings=True, batch_size=32)
        Q = model.encode([BGE_PREFIX + q["question"] for q in test], normalize_embeddings=True, batch_size=64)
        out[name] = Q @ D.T
        del model
        torch.cuda.empty_cache()
    return out


def rerank(reranker, docs, test, S, gold):
    """对每道题，前 TOP_K 个候选用 reranker 重排，返回正确段落的原名次、新名次，以及重排后排第一的段落。"""
    order = np.argsort(-S, axis=1)
    before = [int(np.where(order[i] == gold[i])[0][0]) + 1 for i in range(len(test))]
    pairs = [(q["question"], docs[j]) for i, q in enumerate(test) for j in order[i, :TOP_K]]
    scores = reranker.predict(pairs, batch_size=64, show_progress_bar=False).reshape(len(test), TOP_K)
    after, top1 = [], []
    for i in range(len(test)):
        new_order = order[i, :TOP_K][np.argsort(-scores[i])]
        top1.append(int(new_order[0]))
        if before[i] > TOP_K:  # 正确段落没进候选，reranker 救不回来
            after.append(before[i])
        else:
            after.append(int(np.where(new_order == gold[i])[0][0]) + 1)
    return before, after, top1


def rescue_stats(before, after, test):
    res = {}
    for label, lo, hi in [("全部", 0, 1.01)] + BINS:
        sub = [(b, a) for b, a, q in zip(before, after, test) if lo <= q["overlap"] < hi]
        res[label] = {"n": len(sub),
                      "救回": sum(b > 1 and a == 1 for b, a in sub),
                      "带坏": sum(b == 1 and a > 1 for b, a in sub),
                      "候选里没有答案": sum(b > TOP_K for b, a in sub)}
    return res


def main():
    chunks = [json.loads(l) for l in (P / "chunks.jsonl").open(encoding="utf-8")]
    qs = [json.loads(l) for l in (P / "questions.jsonl").open(encoding="utf-8")]
    test = [q for q in qs if q["split"] == "test"]
    idx = {c["id"]: i for i, c in enumerate(chunks)}
    docs = [c["title"] + " " + c["text"] for c in chunks]
    gold = [idx[q["chunk_id"]] for q in test]

    scores = first_stage_scores(chunks, docs, test)
    reranker = CrossEncoder(RERANKER, device="cuda", max_length=512, model_kwargs={"torch_dtype": torch.float16})

    result, per_q = {}, {}
    for name, S in scores.items():
        t0 = time.time()
        before, after, top1 = rerank(reranker, docs, test, S, gold)
        sec = time.time() - t0
        result[name] = {"重排前": group_metrics(before, test), "重排后": group_metrics(after, test),
                        "救回/带坏": rescue_stats(before, after, test), "重排耗时(s)": round(sec, 1)}
        per_q[f"{name}_rank"], per_q[f"{name}+rerank_rank"] = before, after
        per_q[f"{name}+rerank_top1"] = [chunks[j]["id"] for j in top1]

    (OUT / "exp_c.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "exp_c_per_question.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["qid", "kind", "overlap", "question", "chunk_id"] + list(per_q))
        for i, q in enumerate(test):
            w.writerow([q["qid"], q["kind"], q["overlap"], q["question"], q["chunk_id"]] + [per_q[k][i] for k in per_q])

    groups = ["全部", "paraphrase"] + [b[0] for b in BINS]
    print(f"候选数 K={TOP_K}；每格为 R@1 重排前→重排后")
    for name, r in result.items():
        cells = "  ".join(f"{g} {r['重排前'][g]['R@1']:.3f}→{r['重排后'][g]['R@1']:.3f}" for g in groups)
        print(f"  {name:<12} {cells}  ({r['重排耗时(s)']}s)")
        for g, s in r["救回/带坏"].items():
            print(f"      {g:<16} n={s['n']:<4} 救回 {s['救回']:<4} 带坏 {s['带坏']:<4} 候选里没有答案 {s['候选里没有答案']}")


if __name__ == "__main__":
    main()
