"""第10步：人工数据的评测（需要先人工填表）。

1. 抽查表 data/processed/spot_check.csv：统计"答案在段落里""像真人会问"的比例，按 plain/paraphrase 分开
2. 手写题 data/handwritten.csv：模拟学生提问，统计重叠率，并用 BM25 / bge-base / 微调 / 微调+reranker 评测
   段落编号可以在 data/processed/chunk_index.csv 里查（只列测试集两章）

用法：python step10_human_eval.py            # 两项都做（哪个表没填就跳过哪个）
      python step10_human_eval.py --index    # 只生成 chunk_index.csv 和空白的 handwritten.csv
输出：results/human_eval.json，results/handwritten_per_question.csv
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

from step3_build_dataset import TEST_CHAPTERS, overlap
from step4_baseline import BINS, metrics

ROOT = Path(__file__).parent
P = ROOT / "data/processed"
OUT = ROOT / "results"
HAND = ROOT / "data/handwritten.csv"
HAND_COLS = ["问题", "答案段落编号", "备注"]


def make_index(chunks):
    with (P / "chunk_index.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["段落编号", "标题", "开头"])
        for c in chunks:
            if c["chapter"] in TEST_CHAPTERS:
                w.writerow([c["id"], c["title"], c["text"][:80].replace("\n", " ")])
    if not HAND.exists():
        with HAND.open("w", encoding="utf-8-sig", newline="") as fp:
            csv.writer(fp).writerow(HAND_COLS)
    print(f"已生成 {P / 'chunk_index.csv'}" + ("" if HAND.stat().st_size > 30 else f"，空白模板 {HAND}"))


def spot_check():
    rows = list(csv.DictReader((P / "spot_check.csv").open(encoding="utf-8-sig")))
    kind = {q["qid"]: q["kind"] for q in map(json.loads, (P / "questions.jsonl").open(encoding="utf-8"))}
    cols = {"答案在段落里": "答案在段落里吗(1是/0否)", "像真人会问": "像真人会问的吗(1是/0否)"}
    filled = [r for r in rows if all(r[c].strip() in ("0", "1") for c in cols.values())]
    if not filled:
        print("spot_check.csv 还没填，跳过")
        return None
    res = {"已填": len(filled), "共": len(rows)}
    for k in ("全部", "plain", "paraphrase"):
        sub = [r for r in filled if k == "全部" or kind[r["qid"]] == k]
        res[k] = {"n": len(sub), **{name: round(np.mean([r[c] == "1" for r in sub]), 3) for name, c in cols.items()}}
    print(f"抽查（已填 {len(filled)}/{len(rows)}）：")
    for k in ("全部", "plain", "paraphrase"):
        print(f"  {k:<10} n={res[k]['n']:<3} 答案在段落里 {res[k]['答案在段落里']:.0%}  像真人会问 {res[k]['像真人会问']:.0%}")
    return res


def handwritten(chunks):
    if not HAND.exists():
        print("data/handwritten.csv 不存在，跳过（先运行 --index 生成模板）")
        return None
    idx = {c["id"]: i for i, c in enumerate(chunks)}
    rows = [r for r in csv.DictReader(HAND.open(encoding="utf-8-sig")) if r["问题"].strip()]
    bad = [r["答案段落编号"] for r in rows if r["答案段落编号"].strip() not in idx]
    if bad:
        sys.exit(f"这些段落编号不存在，请到 chunk_index.csv 核对：{bad}")
    if not rows:
        print("handwritten.csv 还没填，跳过")
        return None
    test = [{"question": r["问题"].strip(), "gold": idx[r["答案段落编号"].strip()]} for r in rows]
    for q in test:
        q["overlap"] = round(overlap(q["question"], chunks[q["gold"]]["text"]), 3)

    # 检索沿用 demo 的实现，保证和网页上看到的一致（导入时会加载模型）
    import app
    from step4_baseline import tokenize
    from step5_exp_a_models import BGE_PREFIX
    from sentence_transformers import SentenceTransformer
    import torch

    S = {"BM25": np.stack([app.bm25.get_scores(tokenize(q["question"])) for q in test])}
    base = SentenceTransformer("BAAI/bge-base-zh-v1.5", device="cuda", model_kwargs={"torch_dtype": torch.float16})
    base.max_seq_length = 512
    S["bge-base"] = (base.encode([BGE_PREFIX + q["question"] for q in test], normalize_embeddings=True)
                     @ base.encode(app.docs, normalize_embeddings=True, batch_size=32).T)
    S["bge-base-ft"] = app.dense.encode([BGE_PREFIX + q["question"] for q in test], normalize_embeddings=True) @ app.D.T
    ranks = {name: [int((s[i] > s[i, q["gold"]]).sum()) + 1 for i, q in enumerate(test)] for name, s in S.items()}
    rr = []
    for i, q in enumerate(test):
        order = np.argsort(-S["bge-base-ft"][i])[:app.TOP_K]
        if q["gold"] not in order:
            rr.append(ranks["bge-base-ft"][i])
            continue
        r = app.reranker.predict([(q["question"], app.docs[j]) for j in order], show_progress_bar=False)
        rr.append(int(np.where(order[np.argsort(-r)] == q["gold"])[0][0]) + 1)
    ranks["bge-base-ft+rerank"] = rr

    def grouped(rk):
        res = {"全部": metrics(rk)}
        for label, lo, hi in BINS:
            sub = [r for r, q in zip(rk, test) if lo <= q["overlap"] < hi]
            if sub:
                res[label] = metrics(sub)
        return res

    ov = [q["overlap"] for q in test]
    res = {"题数": len(test), "重叠率均值": round(float(np.mean(ov)), 3), "重叠率中位数": round(float(np.median(ov)), 3),
           "低重叠占比": round(float(np.mean([o < 0.5 for o in ov])), 3),
           "结果": {name: grouped(rk) for name, rk in ranks.items()}}
    with (OUT / "handwritten_per_question.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["问题", "答案段落", "overlap"] + [f"{n}_rank" for n in ranks])
        for i, q in enumerate(test):
            w.writerow([q["question"], chunks[q["gold"]]["id"], q["overlap"]] + [ranks[n][i] for n in ranks])
    print(f"手写题 {len(test)} 道，重叠率均值 {res['重叠率均值']}，低重叠占比 {res['低重叠占比']:.0%}")
    for name, g in res["结果"].items():
        print(f"  {name:<20} " + "  ".join(f"{k} R@1 {v['R@1']:.3f}(n={v['n']})" for k, v in g.items()))
    return res


def main():
    chunks = [json.loads(l) for l in (P / "chunks.jsonl").open(encoding="utf-8")]
    make_index(chunks)
    if "--index" in sys.argv:
        return
    res = {"抽查": spot_check(), "手写题": handwritten(chunks)}
    (OUT / "human_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
