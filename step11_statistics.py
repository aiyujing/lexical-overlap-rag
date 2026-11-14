"""第11步：统计检验与补充分析（只读前面各步的逐题结果，不重新跑模型）。

1. 各方法 R@1 的 bootstrap 95% 置信区间（按题重采样，10000 次），整体和三个重叠组
2. 配对比较：McNemar 精确检验（两个方法在同一批题上的对错），以及"高低重叠差距"变化的配对 bootstrap 区间
3. 混杂控制：低重叠题几乎都是 paraphrase 题，用逻辑回归同时放入重叠率、题型、问题长度、段落长度，
   看重叠率的作用是否独立存在；并只在 paraphrase 题内部按重叠率分组
4. BM25 与向量模型的互补性：一方对、另一方错的题数
5. 错误分析：两种标准的一致性（Cohen's kappa）和自动标准的随机基线；换成词级重叠率后的稳健性检查
6. 人工抽查的 Wilson 置信区间、plain vs paraphrase 的 Fisher 检验，以及标注质量与检索对错的关系

用法：python step11_statistics.py
输出：results/statistics.json
"""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import binomtest, fisher_exact, spearmanr, wilcoxon
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).parent
P = ROOT / "data/processed"
R = ROOT / "results"
B = 10000
GROUPS = [("低重叠(<0.5)", 0, 0.5), ("中重叠(0.5-0.8)", 0.5, 0.8), ("高重叠(≥0.8)", 0.8, 1.01)]
SYSTEMS = {"BM25": "BM25_rank", "bge-base": "bge-base_rank", "bge-base-ft": "bge-base-ft_rank",
           "BM25+rerank": "BM25+rerank_rank", "bge-base+rerank": "bge-base+rerank_rank",
           "bge-base-ft+rerank": "bge-base-ft+rerank_rank"}


def load():
    df = pd.read_csv(R / "exp_c_per_question.csv")
    a = pd.read_csv(R / "exp_a_per_question.csv")
    for m in ("bge-small", "bge-large", "m3e-base", "bge-m3"):
        df[f"{m}_rank"] = a[f"{m}_rank"].values
        SYSTEMS[m] = f"{m}_rank"
    chunks = {c["id"]: c for c in map(json.loads, (P / "chunks.jsonl").open(encoding="utf-8"))}
    df["q_len"] = df["question"].str.len()
    df["p_len"] = df["chunk_id"].map(lambda i: len(chunks[i]["text"]))
    df["chapter"] = df["chunk_id"].map(lambda i: chunks[i]["chapter"])
    df["group"] = pd.cut(df["overlap"], [g[1] for g in GROUPS] + [1.01], right=False, labels=[g[0] for g in GROUPS])
    return df


def boot_ci(hit, rng):
    idx = rng.integers(0, len(hit), (B, len(hit)))
    m = hit[idx].mean(1)
    return [round(float(hit.mean()), 3), round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def mcnemar(a, b):
    """a、b 为两个方法每题是否命中（布尔数组）。返回只有一方对的题数和精确检验 p 值。"""
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return {"只有前者对": n10, "只有后者对": n01, "p": float(p)}


def gap_boot(df, col_a, col_b, rng):
    """方法 a→b 后，高低重叠组 R@1 差距的变化量（b 的差距 − a 的差距），按组内重采样。"""
    lo, hi = df[df.overlap < 0.5], df[df.overlap >= 0.8]
    la, lb = (lo[col_a] == 1).values, (lo[col_b] == 1).values
    ha, hb = (hi[col_a] == 1).values, (hi[col_b] == 1).values
    il = rng.integers(0, len(lo), (B, len(lo)))
    ih = rng.integers(0, len(hi), (B, len(hi)))
    d = (hb[ih].mean(1) - lb[il].mean(1)) - (ha[ih].mean(1) - la[il].mean(1))
    point = (hb.mean() - lb.mean()) - (ha.mean() - la.mean())
    return [round(float(point), 3), round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]


def main():
    rng = np.random.default_rng(0)
    df = load()
    out = {}

    # 1. bootstrap 置信区间
    ci = {}
    for name, col in SYSTEMS.items():
        hit = (df[col] == 1).values
        ci[name] = {"全部": boot_ci(hit, rng)}
        for g, _, _ in GROUPS:
            ci[name][g] = boot_ci(hit[(df.group == g).values], rng)
    out["R@1置信区间"] = ci

    # 2. 配对检验
    pairs = [("BM25", "bge-small"), ("BM25", "m3e-base"), ("BM25", "bge-large"), ("BM25", "bge-m3"), ("BM25", "bge-base"), ("bge-base", "bge-m3"), ("bge-base", "bge-large"), ("bge-base", "bge-base-ft"),
             ("bge-m3", "bge-base-ft"), ("bge-base-ft", "bge-base-ft+rerank"), ("bge-base", "bge-base+rerank"),
             ("BM25", "BM25+rerank")]
    test = {}
    for x, y in pairs:
        a, b = (df[SYSTEMS[x]] == 1).values, (df[SYSTEMS[y]] == 1).values
        low = (df.overlap < 0.5).values
        test[f"{x} vs {y}"] = {"全部": mcnemar(a, b), "低重叠": mcnemar(a[low], b[low]),
                                "高低差距变化": gap_boot(df, SYSTEMS[x], SYSTEMS[y], rng)}
    # 难负样本 vs 普通版：3 个种子逐对比较
    eb = pd.read_csv(R / "exp_b_per_question.csv")
    seeds = []
    for s in range(3):
        a, b = (eb[f"ft-inbatch_s{s}_rank"] == 1).values, (eb[f"ft-hardneg_s{s}_rank"] == 1).values
        seeds.append(mcnemar(a, b))
    test["微调普通版 vs 难负样本版（按种子）"] = seeds
    out["配对检验"] = test

    # 3. 混杂控制
    df["paraphrase"] = (df["kind"] == "paraphrase").astype(int)
    df["q_len10"] = df["q_len"] / 10
    df["p_len100"] = df["p_len"] / 100
    df["ov10"] = df["overlap"] * 10  # 系数解释为重叠率每升 0.1
    reg = {}
    for name in ("BM25", "bge-base", "bge-m3", "bge-base-ft", "bge-base-ft+rerank"):
        df["hit"] = (df[SYSTEMS[name]] == 1).astype(int)
        m = smf.logit("hit ~ ov10 + paraphrase + q_len10 + p_len100", data=df).fit(disp=0)
        m2 = smf.logit("hit ~ ov10 + q_len10 + p_len100", data=df[df.paraphrase == 1]).fit(disp=0)
        conf = np.exp(m.conf_int().loc["ov10"]).round(2).tolist()
        reg[name] = {"全部题·重叠率每升0.1的OR": [round(float(np.exp(m.params["ov10"])), 2)] + conf,
                     "全部题·重叠率p": float(m.pvalues["ov10"]),
                     "全部题·paraphrase的OR": round(float(np.exp(m.params["paraphrase"])), 2),
                     "全部题·paraphrase p": float(m.pvalues["paraphrase"]),
                     "全部题·问题长度每10字OR": round(float(np.exp(m.params["q_len10"])), 2),
                     "全部题·段落长度每100字OR": round(float(np.exp(m.params["p_len100"])), 2),
                     "全部题·段落长度p": float(m.pvalues["p_len100"]),
                     "仅paraphrase·重叠率每升0.1的OR": [round(float(np.exp(m2.params["ov10"])), 2)]
                     + np.exp(m2.conf_int().loc["ov10"]).round(2).tolist(),
                     "仅paraphrase·重叠率p": float(m2.pvalues["ov10"]),
                     "伪R2": round(float(m.prsquared), 3)}
    out["逻辑回归"] = reg
    para = df[df.paraphrase == 1]
    edges = [0, 0.4, 0.5, 0.6, 1.01]
    within = {}
    for name in ("BM25", "bge-base", "bge-base-ft", "bge-base-ft+rerank"):
        row = {}
        for i in range(len(edges) - 1):
            sub = para[(para.overlap >= edges[i]) & (para.overlap < edges[i + 1])]
            row[f"{edges[i]:.1f}-{min(edges[i + 1], 1):.1f}"] = [len(sub), round(float((sub[SYSTEMS[name]] == 1).mean()), 3)]
        within[name] = row
    out["paraphrase题内部按重叠率分组"] = within
    out["重叠率与名次的Spearman"] = {n: round(float(spearmanr(df.overlap, df[c]).statistic), 3) for n, c in SYSTEMS.items()}
    out["按章"] = {ch: {n: round(float((g[SYSTEMS[n]] == 1).mean()), 3) for n in ("BM25", "bge-base", "bge-base-ft", "bge-base-ft+rerank")}
                   | {"题数": len(g), "重叠率均值": round(float(g.overlap.mean()), 3)} for ch, g in df.groupby("chapter")}

    # 4. 互补性
    comp = {}
    for x, y in (("BM25", "bge-base-ft"), ("BM25", "bge-base")):
        a, b = (df[SYSTEMS[x]] == 1), (df[SYSTEMS[y]] == 1)
        res = {}
        for g, lo, hi in [("全部", 0, 1.01)] + GROUPS:
            m = (df.overlap >= lo) & (df.overlap < hi)
            res[g] = {"题数": int(m.sum()), "两者都对": int((a & b & m).sum()), f"只有{x}对": int((a & ~b & m).sum()),
                      f"只有{y}对": int((~a & b & m).sum()), "两者都错": int((~a & ~b & m).sum()),
                      "至少一个对(上限)": round(float((a | b)[m].mean()), 3)}
        a10, b10 = (df[SYSTEMS[x]] <= 20), (df[SYSTEMS[y]] <= 20)
        res["前20召回"] = {x: round(float(a10.mean()), 3), y: round(float(b10.mean()), 3), "并集": round(float((a10 | b10).mean()), 3)}
        comp[f"{x} vs {y}"] = res
    out["互补性"] = comp

    # 5. 错误分析两种标准的一致性
    e = pd.read_csv(R / "error_analysis.csv")
    auto = e["字面更像错误段落"] == 1
    llm = e["大模型分类"] == "B"
    po = (auto == llm).mean()
    pe = auto.mean() * llm.mean() + (1 - auto.mean()) * (1 - llm.mean())
    out["错误分析一致性"] = {"一致率": round(float(po), 3), "kappa": round(float((po - pe) / (1 - pe)), 3),
                       "自动判B比例": round(float(auto.mean()), 3), "大模型判B比例": round(float(llm.mean()), 3),
                       "B类·与正确段落重叠均值": round(float(e[llm]["与正确段落重叠"].mean()), 3),
                       "B类·与错误段落重叠均值": round(float(e[llm]["与错误段落重叠"].mean()), 3)}

    # 5b. 自动标准的基线：随便拿一个其他段落，它的重叠率高于正确段落的概率有多大
    from step3_build_dataset import overlap
    chunks = {c["id"]: c["text"] for c in map(json.loads, (P / "chunks.jsonl").open(encoding="utf-8"))}
    err = df[df["bge-base-ft+rerank_rank"] > 1]
    base_rows = []
    for _, r in err.iterrows():
        og = overlap(r.question, chunks[r.chunk_id])
        ow = overlap(r.question, chunks[r["bge-base-ft+rerank_top1"]])
        others = np.array([overlap(r.question, t) for i, t in chunks.items() if i != r.chunk_id])
        base_rows.append((r.overlap, ow > og, (others > og).mean(), og, ow, others.mean()))
    br = np.array(base_rows, dtype=float)
    base = {}
    for lab, lo, hi in [("全部", 0, 1.01)] + GROUPS:
        m = (br[:, 0] >= lo) & (br[:, 0] < hi)
        base[lab] = {"错题数": int(m.sum()), "错选段落重叠更高的比例": round(float(br[m, 1].mean()), 3),
                     "随机段落重叠更高的比例": round(float(br[m, 2].mean()), 3),
                     "重叠均值·正确": round(float(br[m, 3].mean()), 3), "重叠均值·错选": round(float(br[m, 4].mean()), 3),
                     "重叠均值·随机": round(float(br[m, 5].mean()), 3)}
    base["全部"]["Wilcoxon p(错选 vs 正确)"] = float(wilcoxon(br[:, 4], br[:, 3]).pvalue)
    out["错误分析·自动标准基线"] = base

    # 5c. 稳健性：换成按 jieba 词计算的重叠率（去掉单字词），分组结论是否不变
    import jieba
    jieba.setLogLevel(60)

    def word_overlap(q, p):
        ws = {w for w in jieba.lcut(q) if len(w) > 1 and any("一" <= c <= "鿿" for c in w)}
        return sum(w in p for w in ws) / max(len(ws), 1)

    df["word_ov"] = [word_overlap(q, chunks[c]) for q, c in zip(df.question, df.chunk_id)]
    t1, t2 = df.word_ov.quantile([1 / 3, 2 / 3]).tolist()
    rob = {"与字符重叠率的Spearman": round(float(spearmanr(df.overlap, df.word_ov).statistic), 3),
           "三分位点": [round(t1, 3), round(t2, 3)]}
    for name in ("BM25", "bge-base", "bge-m3", "bge-base-ft", "bge-base-ft+rerank"):
        hit = df[SYSTEMS[name]] == 1
        rob[name] = {"低": round(float(hit[df.word_ov <= t1].mean()), 3),
                     "中": round(float(hit[(df.word_ov > t1) & (df.word_ov <= t2)].mean()), 3),
                     "高": round(float(hit[df.word_ov > t2].mean()), 3)}
    rob["各组题数"] = [int((df.word_ov <= t1).sum()), int(((df.word_ov > t1) & (df.word_ov <= t2)).sum()),
                   int((df.word_ov > t2).sum())]
    out["稳健性·词级重叠率"] = rob

    # 6. 人工抽查
    sc =list(csv.reader((P / "spot_check.csv").open(encoding="utf-8-sig")))[1:]
    kind = df.set_index("qid")["kind"].to_dict()
    rank = df.set_index("qid")["bge-base-ft+rerank_rank"].to_dict()
    ov = df.set_index("qid")["overlap"].to_dict()
    hum = {}
    for k in ("全部", "plain", "paraphrase"):
        sub = [r for r in sc if k == "全部" or kind[r[0]] == k]
        for name, col in (("答案在段落里", 4), ("像真人会问", 5)):
            x = sum(r[col] == "1" for r in sub)
            lo, hi = proportion_confint(x, len(sub), method="wilson")
            hum[f"{k}·{name}"] = [x, len(sub), round(x / len(sub), 3), round(lo, 3), round(hi, 3)]
    tab = [[sum(r[4] == v and kind[r[0]] == k for r in sc) for v in ("1", "0")] for k in ("plain", "paraphrase")]
    hum["plain vs paraphrase 答案在段落里 Fisher p"] = float(fisher_exact(tab)[1])
    for v, lab in (("1", "答案在段落里"), ("0", "答案不在段落里")):
        sub = [r for r in sc if r[4] == v]
        hum[f"{lab}·最好系统R@1"] = [len(sub), round(float(np.mean([rank[r[0]] == 1 for r in sub])), 3),
                               round(float(np.mean([ov[r[0]] for r in sub])), 3)]
    sub_p = [r for r in sc if kind[r[0]] == "paraphrase"]
    t2 = [[sum(r[4] == v and (rank[r[0]] == 1) == h for r in sub_p) for h in (True, False)] for v in ("1", "0")]
    hum["paraphrase内·标注质量与R@1 Fisher p"] = float(fisher_exact(t2)[1])
    hum["paraphrase内·列联表[答案在/不在][对/错]"] = t2
    out["人工抽查"] = hum

    (R / "statistics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
