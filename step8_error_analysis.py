"""第8步：错误分析。看最好的组合（微调 bge-base + reranker）排错的题都错在哪。

两种分类方式：
1. 自动指标：比较问题和"错误第一名段落"、问题和"正确段落"的字面重叠率。错误段落更像 → 记为"字面更像"
2. 大模型判断（DeepSeek）：给它问题、正确段落、错误第一名段落，分成四类（见 PROMPT）

用法：python step8_error_analysis.py
输出：results/error_analysis.csv（每道错题一行，可直接挑例子写报告），results/error_analysis.json（统计）
"""
import csv
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from step2_gen_questions import API, MODEL, load_key
from step3_build_dataset import overlap
from step4_baseline import BINS

ROOT = Path(__file__).parent
P = ROOT / "data/processed"
OUT = ROOT / "results"
SYSTEM = "bge-base-ft"

CATS = {
    "A": "错误段落其实也能回答（标注只认一个答案，算不上真错）",
    "B": "被字面骗：错误段落和问题有很多相同的词，但并不能回答问题",
    "C": "题目本身有问题：太含糊、多个段落都沾边，或答案并不在正确段落里",
    "D": "其他：主题相近、需要更深的理解才能区分",
}
PROMPT = """一个检索系统要为问题找到教材里能回答它的段落。它选错了。请判断错误原因。

问题：{q}

【标注的正确段落】{gt}
{gx}

【系统选出的段落】{pt}
{px}

从下面四类中选一个最主要的原因：
A：{A}
B：{B}
C：{C}
D：{D}

只输出JSON：{{"category": "A/B/C/D 之一", "reason": "一句话说明"}}"""


def ask(key, item, retries=4):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(**item["prompt"], **CATS)}],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for i in range(retries):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120))
            out = json.loads(r["choices"][0]["message"]["content"])
            cat = str(out.get("category", "")).strip()[:1].upper()
            return {"category": cat if cat in CATS else "?", "reason": out.get("reason", ""),
                    "tokens": r.get("usage", {}).get("total_tokens", 0)}
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, TimeoutError) as e:
            if i == retries - 1:
                return {"category": "?", "reason": f"调用失败 {e!r}"[:200], "tokens": 0}
            time.sleep(2 ** i)


def main():
    chunks = {c["id"]: c for c in map(json.loads, (P / "chunks.jsonl").open(encoding="utf-8"))}
    rows = list(csv.DictReader((OUT / "exp_c_per_question.csv").open(encoding="utf-8-sig")))
    errors = [r for r in rows if int(r[f"{SYSTEM}+rerank_rank"]) > 1]
    print(f"{SYSTEM}+rerank 共 {len(rows)} 题，排错 {len(errors)} 题，开始分析")

    items = []
    for r in errors:
        g, w = chunks[r["chunk_id"]], chunks[r[f"{SYSTEM}+rerank_top1"]]
        items.append({"row": r, "gold": g, "wrong": w,
                      "ov_gold": overlap(r["question"], g["text"]), "ov_wrong": overlap(r["question"], w["text"]),
                      "prompt": {"q": r["question"], "gt": g["title"], "gx": g["text"],
                                 "pt": w["title"], "px": w["text"]}})

    key = load_key()
    with ThreadPoolExecutor(8) as pool:
        judged = list(pool.map(lambda it: ask(key, it), items))

    with (OUT / "error_analysis.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        wr = csv.writer(fp)
        wr.writerow(["qid", "kind", "overlap", "问题", "正确段落", "错误第一名", "正确名次(重排后)",
                     "答案进了前20吗", "与正确段落重叠", "与错误段落重叠", "字面更像错误段落", "大模型分类", "理由"])
        for it, j in zip(items, judged):
            r = it["row"]
            rank = int(r[f"{SYSTEM}+rerank_rank"])
            wr.writerow([r["qid"], r["kind"], r["overlap"], r["question"], it["gold"]["title"], it["wrong"]["title"],
                         rank, int(int(r[f"{SYSTEM}_rank"]) <= 20), round(it["ov_gold"], 3), round(it["ov_wrong"], 3),
                         int(it["ov_wrong"] > it["ov_gold"]), j["category"], j["reason"]])

    stats = {"系统": f"{SYSTEM}+rerank", "错题数": len(items), "大模型用量(tokens)": sum(j["tokens"] for j in judged)}
    groups = [("全部", 0, 1.01)] + BINS
    for label, lo, hi in groups:
        sub = [(it, j) for it, j in zip(items, judged) if lo <= float(it["row"]["overlap"]) < hi]
        stats[label] = {
            "错题数": len(sub),
            "字面更像错误段落": sum(it["ov_wrong"] > it["ov_gold"] for it, _ in sub),
            "大模型分类": dict(sorted(Counter(j["category"] for _, j in sub).items())),
        }
    stats["分类说明"] = CATS
    (OUT / "error_analysis.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"大模型用量 {stats['大模型用量(tokens)']} tokens")
    for label, *_ in groups:
        s = stats[label]
        print(f"  {label:<16} 错题 {s['错题数']:<4} 字面更像错误段落 {s['字面更像错误段落']:<4} 分类 {s['大模型分类']}")


if __name__ == "__main__":
    main()
