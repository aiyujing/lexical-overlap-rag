"""第3步：过滤题目、按章节划分训练/测试集、导出人工抽查表。

输出：
- data/processed/questions.jsonl：过滤后的题目，带 kind（plain/paraphrase）、split、overlap
- data/processed/spot_check.csv：抽查表，用 Excel 打开填写
"""
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent
P = ROOT / "data/processed"

# 按章节划分，防止训练题和测试题出自同一章（数据泄漏）
TEST_CHAPTERS = {"chapter_convolutional-modern", "chapter_computer-vision"}
DEV_CHAPTERS = {"chapter_convolutional-neural-networks"}
BAD_WORDS = ("本段", "文中", "上述", "该段", "作者", "这段")


def overlap(q, p):
    """问题中的汉字有多大比例出现在段落中（去重后计算）。"""
    s = {c for c in q if "一" <= c <= "鿿"}
    return sum(c in p for c in s) / max(len(s), 1)


def main():
    chunks = {r["id"]: r for r in map(json.loads, (P / "chunks.jsonl").open(encoding="utf-8"))}
    raw = [r for r in map(json.loads, (P / "questions_raw.jsonl").open(encoding="utf-8")) if "error" not in r]

    rows, n_drop = [], 0
    for r in raw:
        c = chunks[r["chunk_id"]]
        split = "test" if c["chapter"] in TEST_CHAPTERS else "dev" if c["chapter"] in DEV_CHAPTERS else "train"
        for kind in ("plain", "paraphrase"):
            for q in r[kind]:
                q = q.strip()
                if not q or any(w in q for w in BAD_WORDS):
                    n_drop += 1
                    continue
                rows.append({"qid": f"q{len(rows):04d}", "question": q, "chunk_id": c["id"],
                             "kind": kind, "split": split, "overlap": round(overlap(q, c["text"]), 3)})

    with (P / "questions.jsonl").open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 抽查表：测试集里两类题各 50 道
    random.seed(0)
    test = [r for r in rows if r["split"] == "test"]
    sample = (random.sample([r for r in test if r["kind"] == "plain"], 50)
              + random.sample([r for r in test if r["kind"] == "paraphrase"], 50))
    random.shuffle(sample)
    with (P / "spot_check.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["qid", "问题", "段落标题", "段落内容", "答案在段落里吗(1是/0否)", "像真人会问的吗(1是/0否)", "备注"])
        for r in sample:
            c = chunks[r["chunk_id"]]
            w.writerow([r["qid"], r["question"], c["title"], c["text"], "", "", ""])

    print(f"保留 {len(rows)} 道，丢弃 {n_drop} 道")
    for s in ("train", "dev", "test"):
        sub = [r for r in rows if r["split"] == s]
        print(f"  {s}: {len(sub)} 道题，{len({r['chunk_id'] for r in sub})} 个段落")


if __name__ == "__main__":
    main()
