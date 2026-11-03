"""第5步：实验A，换不同的向量模型（都不微调），看模型变大能否减轻字面重叠偏差。

评估方式和 step4 一样：376 个段落里检索 568 道测试题，按 plain/paraphrase 和重叠率分组。
输出：results/exp_a.json，results/exp_a_per_question.csv
"""
import csv
import json
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from step4_baseline import BINS, metrics, rank_of

ROOT = Path(__file__).parent
P = ROOT / "data/processed"
OUT = ROOT / "results"

BGE_PREFIX = "为这个句子生成表示以用于检索相关文章："
# (显示名, 模型名, 问题前缀)；bge-zh-v1.5 官方建议给问题加前缀，m3e 和 bge-m3 不需要
MODELS = [
    ("bge-small", "BAAI/bge-small-zh-v1.5", BGE_PREFIX),
    ("bge-base", "BAAI/bge-base-zh-v1.5", BGE_PREFIX),
    ("bge-large", "BAAI/bge-large-zh-v1.5", BGE_PREFIX),
    ("m3e-base", "moka-ai/m3e-base", ""),
    ("bge-m3", "BAAI/bge-m3", ""),
]


def main():
    chunks = [json.loads(l) for l in (P / "chunks.jsonl").open(encoding="utf-8")]
    qs = [json.loads(l) for l in (P / "questions.jsonl").open(encoding="utf-8")]
    test = [q for q in qs if q["split"] == "test"]
    idx = {c["id"]: i for i, c in enumerate(chunks)}
    docs = [c["title"] + " " + c["text"] for c in chunks]

    result, all_ranks = {}, {}
    for name, path, prefix in MODELS:
        model = SentenceTransformer(path, device="cuda", model_kwargs={"torch_dtype": torch.float16})
        model.max_seq_length = 512
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        t0 = time.time()
        D = model.encode(docs, normalize_embeddings=True, batch_size=32)
        Q = model.encode([prefix + q["question"] for q in test], normalize_embeddings=True, batch_size=64)
        sec = time.time() - t0
        S = Q @ D.T
        ranks = [rank_of(S[i], idx[q["chunk_id"]]) for i, q in enumerate(test)]

        res = {"参数量(M)": round(n_params), "编码耗时(s)": round(sec, 1), "全部": metrics(ranks)}
        for kind in ("plain", "paraphrase"):
            res[kind] = metrics([r for r, q in zip(ranks, test) if q["kind"] == kind])
        for label, lo, hi in BINS:
            res[label] = metrics([r for r, q in zip(ranks, test) if lo <= q["overlap"] < hi])
        result[name], all_ranks[name] = res, ranks
        del model
        torch.cuda.empty_cache()

    (OUT / "exp_a.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "exp_a_per_question.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["qid", "kind", "overlap", "question"] + [f"{n}_rank" for n in all_ranks])
        for i, q in enumerate(test):
            w.writerow([q["qid"], q["kind"], q["overlap"], q["question"]] + [all_ranks[n][i] for n in all_ranks])

    groups = ["全部", "plain", "paraphrase"] + [b[0] for b in BINS]
    print(f"{'模型':<10} {'参数(M)':>7} {'耗时s':>6}  " + "  ".join(f"{g}" for g in groups) + "   (R@1 / R@10)")
    for name, res in result.items():
        cells = "  ".join(f"{res[g]['R@1']:.3f}/{res[g]['R@10']:.3f}" for g in groups)
        print(f"{name:<10} {res['参数量(M)']:>7} {res['编码耗时(s)']:>6}  {cells}")


if __name__ == "__main__":
    main()
