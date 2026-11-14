"""生成报告用的图（PNG，300dpi），输出到 figures/。

依赖前面各步的结果文件；外部验证的图在 results/external.json 存在时才画。
用法：python make_figures.py
"""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step3_build_dataset import overlap

ROOT = Path(__file__).parent
P = ROOT / "data/processed"
R = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# 配色：固定顺序的分类色（已用色盲模拟校验相邻色可区分）
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GROUPS = ["低重叠(<0.5)", "中重叠(0.5-0.8)", "高重叠(≥0.8)"]
GROUP_TICKS = ["低重叠\n(<0.5)", "中重叠\n(0.5–0.8)", "高重叠\n(≥0.8)"]

plt.rcParams.update({
    "font.family": ["Microsoft YaHei"], "axes.unicode_minus": False, "font.size": 10,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=300)
    plt.close(fig)
    print("  ", name)


def jl(path):
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def line_by_group(ax, series, ylabel="R@1（排第一即为正确段落的比例）"):
    """series：[(名字, [低, 中, 高])]。x 轴是重叠分组，每条线一个方法，线尾直接标名字。"""
    x = np.arange(len(GROUPS))
    ends = []
    for k, (name, ys) in enumerate(series):
        ax.plot(x, ys, color=C[k], lw=2, marker="o", ms=6, mec="#fcfcfb", mew=1.5, label=name)
        ends.append([ys[-1], f"{name} {ys[-1]:.2f}"])
    ends.sort(reverse=True)  # 线尾标签从上往下排，相邻至少隔 0.045，避免重叠
    for j in range(1, len(ends)):
        ends[j][0] = min(ends[j][0], ends[j - 1][0] - 0.045)
    for y, lab in ends:
        ax.text(x[-1] + 0.08, y, lab, va="center", fontsize=9, color=INK2)
    ax.set_xticks(x, GROUP_TICKS)
    ax.set_xlim(-0.2, len(GROUPS) - 1 + 0.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=9)


# ---------- 数据章 ----------
def fig_data():
    chunks = jl(P / "chunks.jsonl")
    qs = jl(P / "questions.jsonl")

    # 图1：段落长度分布
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    L = [len(c["text"]) for c in chunks]
    ax.hist(L, bins=30, color=C[0], edgecolor="#fcfcfb", linewidth=1)
    ax.axvline(np.median(L), color=INK2, lw=1, ls="--")
    ax.annotate(f"中位数 {np.median(L):.0f} 字", (np.median(L), ax.get_ylim()[1] * 0.9), xytext=(6, 0),
                textcoords="offset points", fontsize=9, color=INK2)
    ax.set_title(f"知识库段落长度分布（共 {len(chunks)} 段）")
    ax.set_xlabel("段落字数")
    ax.set_ylabel("段落数")
    save(fig, "fig01_chunk_length.png")

    # 图2：各章题目数，按划分着色
    order = list(dict.fromkeys(c["chapter"] for c in chunks))
    ch_of = {c["id"]: c["chapter"] for c in chunks}
    cnt = Counter(ch_of[q["chunk_id"]] for q in qs)
    split_of = {ch_of[q["chunk_id"]]: q["split"] for q in qs}
    names = [o.replace("chapter_", "") for o in order]
    colors = {"train": C[0], "dev": C[1], "test": C[2]}
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(names, [cnt[o] for o in order], color=[colors[split_of.get(o, "train")] for o in order],
            height=0.7, edgecolor="#fcfcfb", linewidth=2)
    for k, o in enumerate(order):
        ax.text(cnt[o] + 5, k, str(cnt[o]), va="center", fontsize=9, color=INK2)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    for s, lab in (("train", "训练集"), ("dev", "验证集"), ("test", "测试集")):
        ax.bar(0, 0, color=colors[s], label=f"{lab} {sum(q['split'] == s for q in qs)} 题")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("各章题目数与数据划分（按章划分，避免泄漏）")
    ax.set_xlabel("题目数")
    save(fig, "fig02_split_by_chapter.png")

    # 图3：字面重叠率分布，大模型两类题 vs 真人提问
    ov_real = real_query_overlap()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bins = np.linspace(0, 1, 21)
    sets = [("大模型 plain 题", [q["overlap"] for q in qs if q["kind"] == "plain"]),
            ("大模型 paraphrase 题", [q["overlap"] for q in qs if q["kind"] == "paraphrase"]),
            ("真人搜索提问（DuRetrieval）", ov_real)]
    for k, (name, v) in enumerate(sets):
        h, _ = np.histogram(v, bins=bins)
        ax.plot((bins[:-1] + bins[1:]) / 2, h / len(v), color=C[k], lw=2, marker="o", ms=4,
                label=f"{name}  均值 {np.mean(v):.2f}")
    ax.set_title("问题与答案段落的字面重叠率分布")
    ax.set_xlabel("重叠率（问题里的汉字有多少比例出现在答案段落中）")
    ax.set_ylabel("题目占比")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "fig03_overlap_distribution.png")


def real_query_overlap():
    raw = ROOT / "data/raw/cmteb"
    corpus = pd.read_parquet(next((raw / "DuRetrieval").glob("corpus*.parquet")))
    queries = pd.read_parquet(next((raw / "DuRetrieval").glob("queries*.parquet")))
    qrels = pd.read_parquet(next((raw / "DuRetrieval-qrels").glob("*.parquet")))
    text = dict(zip(corpus["id"], corpus["text"]))
    qtext = dict(zip(queries["id"], queries["text"]))
    out = []
    for qid, g in qrels[qrels["score"] > 0].groupby("qid"):
        ps = [text[p] for p in g["pid"] if p in text]
        if ps and qid in qtext:
            out.append(max(overlap(qtext[qid], p) for p in ps))
    return out


# ---------- 实验 ----------
def fig_experiments():
    base = json.loads((R / "baseline.json").read_text(encoding="utf-8"))
    a = json.loads((R / "exp_a.json").read_text(encoding="utf-8"))
    b = json.loads((R / "exp_b.json").read_text(encoding="utf-8"))
    c = json.loads((R / "exp_c.json").read_text(encoding="utf-8"))

    # 图4：Baseline + 实验A，各模型在三个重叠组的 R@1
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    series = [("BM25", [base["BM25"][g]["R@1"] for g in GROUPS])]
    series += [(n, [a[n][g]["R@1"] for g in GROUPS]) for n in ("bge-small", "bge-base", "bge-large", "m3e-base", "bge-m3")]
    line_by_group(ax, series)
    ax.set_title("实验A：所有模型都随字面重叠率下降而变差")
    save(fig, "fig04_exp_a_by_overlap.png")

    # 图5：实验A，模型大小 vs 整体与低重叠组 R@1
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    names = ["bge-small", "bge-base", "bge-large", "m3e-base", "bge-m3"]
    for k, (g, lab) in enumerate((("全部", "全部题目"), ("低重叠(<0.5)", "低重叠组"))):
        xs = [a[n]["参数量(M)"] for n in names]
        ys = [a[n][g]["R@1"] for n in names]
        ax.scatter(xs, ys, s=60, color=C[k], edgecolor="#fcfcfb", linewidth=1.5, label=lab, zorder=3)
        for n, x, y in zip(names, xs, ys):
            ax.annotate(n, (x, y), xytext=(6, -3 if k else 4), textcoords="offset points", fontsize=8, color=INK2)
    ax.axhline(base["BM25"]["全部"]["R@1"], color=C[0], lw=1, ls="--")
    ax.annotate("BM25（全部题目）", (20, base["BM25"]["全部"]["R@1"]), xytext=(0, 4), textcoords="offset points",
                fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("参数量（百万，对数刻度）")
    ax.set_ylabel("R@1")
    ax.set_ylim(0, 0.65)
    ax.set_title("模型变大，低重叠组的提升有限")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "fig05_exp_a_size.png")

    # 图6：实验B 训练曲线（验证集 MRR），细线为单个种子、粗线为均值
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, (name, lab) in enumerate((("ft-inbatch", "普通版"), ("ft-hardneg", "难负样本版"))):
        curves = np.array([[e["dev_MRR"] for e in run["log"]] for run in b[name]["runs"]])
        ep = np.arange(curves.shape[1])
        for cv in curves:
            ax.plot(ep, cv, color=C[k], lw=0.8, alpha=0.35)
        ax.plot(ep, curves.mean(0), color=C[k], lw=2, marker="o", ms=5, mec="#fcfcfb", label=f"{lab}（3 个种子均值）")
    ax.set_xticks(range(6), ["微调前"] + [f"第{i}轮" for i in range(1, 6)])
    ax.set_ylabel("验证集 MRR")
    ax.set_title("实验B：验证集成绩在第 1–2 轮后不再上升")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "fig06_exp_b_curve.png")

    # 图7：实验B，微调前后按重叠分组（带 3 个种子的标准差）
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(GROUPS))
    rows = [("bge-base 微调前", [a["bge-base"][g]["R@1"] for g in GROUPS], None),
            ("bge-m3（不微调）", [a["bge-m3"][g]["R@1"] for g in GROUPS], None),
            ("微调·普通版", [b["ft-inbatch"]["summary"][g]["R@1"]["mean"] for g in GROUPS],
             [b["ft-inbatch"]["summary"][g]["R@1"]["std"] for g in GROUPS]),
            ("微调·难负样本版", [b["ft-hardneg"]["summary"][g]["R@1"]["mean"] for g in GROUPS],
             [b["ft-hardneg"]["summary"][g]["R@1"]["std"] for g in GROUPS])]
    w = 0.2
    for k, (name, ys, err) in enumerate(rows):
        xs = x + (k - 1.5) * w
        ax.bar(xs, ys, w, color=C[k], edgecolor="#fcfcfb", linewidth=2, label=name,
               yerr=err, error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
        for xx, y in zip(xs, ys):
            ax.text(xx, y + 0.02, f"{y:.2f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, GROUP_TICKS)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("R@1")
    ax.set_title("实验B：微调后各组都提升，难负样本的额外收益在误差范围内")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    save(fig, "fig07_exp_b_by_overlap.png")

    # 图8：实验C，重排前后（哑铃图）
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.8), sharey=True)
    for ax, (name, lab) in zip(axes, (("BM25", "BM25"), ("bge-base", "bge-base 原版"), ("bge-base-ft", "bge-base 微调后"))):
        for k, g in enumerate(GROUPS):
            y0, y1 = c[name]["重排前"][g]["R@1"], c[name]["重排后"][g]["R@1"]
            ax.plot([k, k], [y0, y1], color=AXIS, lw=2, zorder=1)
            ax.scatter([k], [y0], s=50, color=C[0], zorder=2, edgecolor="#fcfcfb", linewidth=1.5)
            ax.scatter([k], [y1], s=50, color=C[1], zorder=2, edgecolor="#fcfcfb", linewidth=1.5)
            ax.annotate(f"+{y1 - y0:.2f}", (k, y1), xytext=(7, 0), textcoords="offset points", fontsize=8, color=INK2,
                        va="center")
        ax.set_xticks(range(3), GROUP_TICKS, fontsize=8)
        ax.set_xlim(-0.4, 2.6)
        ax.set_title(lab, fontsize=10)
    axes[0].set_ylabel("R@1")
    axes[0].set_ylim(0, 1)
    axes[0].scatter([], [], color=C[0], label="重排前")
    axes[0].scatter([], [], color=C[1], label="重排后")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("实验C：reranker 在高重叠组提升多、低重叠组提升少", x=0.01, ha="left", fontsize=12, fontweight="bold")
    save(fig, "fig08_exp_c_dumbbell.png")

    # 图9：主图，关键系统在细分重叠率区间上的 R@1
    rows = list(csv.DictReader((R / "exp_c_per_question.csv").open(encoding="utf-8-sig")))
    edges = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    mids = [f"{edges[i]:.1f}–{min(edges[i + 1], 1):.1f}" for i in range(len(edges) - 1)]
    systems = [("BM25", "BM25_rank"), ("bge-base 原版", "bge-base_rank"),
               ("bge-base 微调后", "bge-base-ft_rank"), ("微调后 + reranker", "bge-base-ft+rerank_rank")]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ov = np.array([float(r["overlap"]) for r in rows])
    ends = []
    for k, (lab, col) in enumerate(systems):
        rk = np.array([int(r[col]) for r in rows])
        ys = [(rk[(ov >= edges[i]) & (ov < edges[i + 1])] == 1).mean() for i in range(len(mids))]
        ax.plot(range(len(mids)), ys, color=C[k], lw=2, marker="o", ms=5, mec="#fcfcfb", label=lab)
        ends.append([ys[-1], lab])
    ends.sort(reverse=True)  # 线尾标签从上往下排，相邻至少隔 0.05，避免重叠
    for j in range(1, len(ends)):
        ends[j][0] = min(ends[j][0], ends[j - 1][0] - 0.05)
    for y, lab in ends:
        ax.text(len(mids) - 1 + 0.15, y, lab, fontsize=8.5, color=INK2, va="center")
    counts = [int(((ov >= edges[i]) & (ov < edges[i + 1])).sum()) for i in range(len(mids))]
    ax.set_xticks(range(len(mids)), [f"{m}\n(n={n})" for m, n in zip(mids, counts)], fontsize=8)
    ax.set_xlim(-0.3, len(mids) + 1.2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("问题与答案段落的字面重叠率")
    ax.set_ylabel("R@1")
    ax.set_title("字面重叠率越低，所有方法越容易找错")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "fig09_main_overlap_curve.png")

    # 图10：错误分析，各重叠组错题的原因构成
    err = list(csv.DictReader((R / "error_analysis.csv").open(encoding="utf-8-sig")))
    bins = [(GROUP_TICKS[0], 0, 0.5), (GROUP_TICKS[1], 0.5, 0.8), (GROUP_TICKS[2], 0.8, 1.01)]
    cats = [("被字面骗（两种标准一致）", lambda r: r["大模型分类"] == "B" and r["字面更像错误段落"] == "1"),
            ("被字面骗（仅大模型判定）", lambda r: r["大模型分类"] == "B" and r["字面更像错误段落"] == "0"),
            ("错选段落其实也能回答", lambda r: r["大模型分类"] == "A"),
            ("其他 / 题目有问题", lambda r: r["大模型分类"] in ("C", "D", "?"))]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    left = np.zeros(len(bins))
    for k, (lab, f) in enumerate(cats):
        v = np.array([sum(f(r) for r in err if lo <= float(r["overlap"]) < hi) for _, lo, hi in bins])
        ax.barh(range(len(bins)), v, left=left, color=C[k], height=0.6, edgecolor="#fcfcfb", linewidth=2, label=lab)
        for i, (vv, l0) in enumerate(zip(v, left)):
            if vv >= 6:
                ax.text(l0 + vv / 2, i, str(vv), ha="center", va="center", fontsize=8.5, color="#fcfcfb")
        left += v
    for i, t in enumerate(left):
        ax.text(t + 1.5, i, f"共 {int(t)}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(bins)), [b[0].replace("\n", " ") for b in bins])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("错题数（微调后 + reranker，测试集 568 题）")
    ax.set_title("错误分析：各重叠组错题的分类构成")
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, "fig10_error_analysis.png")


def fig_external():
    path = R / "external.json"
    if not path.exists():
        print("   （跳过外部验证图：results/external.json 还不存在）")
        return
    e = json.loads(path.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names = [("BM25", "BM25"), ("bge-base", "bge-base 原版"), ("bge-base-ft", "bge-base 微调后"),
             ("bge-base-ft+rerank", "微调后 + reranker")]
    line_by_group(ax, [(lab, [e[n][g]["R@1"] for g in GROUPS]) for n, lab in names if n in e])
    ax.set_title("外部验证（DuRetrieval 真人搜索提问，10 万段落）")
    save(fig, "fig11_external_by_overlap.png")


def fig_statistics():
    path = R / "statistics.json"
    if not path.exists():
        print("   （跳过统计图：先运行 step11_statistics.py）")
        return
    s = json.loads(path.read_text(encoding="utf-8"))

    # 图14：主要方法在三个重叠组的 R@1 及 bootstrap 95% 置信区间
    names = [("BM25", "BM25"), ("bge-base", "bge-base 原版"), ("bge-m3", "bge-m3"),
             ("bge-base-ft", "bge-base 微调后"), ("bge-base-ft+rerank", "微调后 + reranker")]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(GROUPS))
    w = 0.15
    for k, (n, lab) in enumerate(names):
        v = np.array([s["R@1置信区间"][n][g] for g in GROUPS])
        xs = x + (k - 2) * w
        ax.errorbar(xs, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", ms=6, color=C[k],
                    mec="#fcfcfb", mew=1.5, elinewidth=1.5, capsize=0, label=lab)
    ax.set_xticks(x, GROUP_TICKS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("R@1（竖线为 bootstrap 95% 置信区间）")
    ax.set_title("各方法 R@1 的 95% 置信区间")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "fig14_ci_by_overlap.png")

    # 图15：只看 paraphrase 题，按重叠率细分（排除题型的影响）
    within = s["paraphrase题内部按重叠率分组"]
    labs = {"BM25": "BM25", "bge-base": "bge-base 原版", "bge-base-ft": "bge-base 微调后",
            "bge-base-ft+rerank": "微调后 + reranker"}
    bins = list(within["BM25"].keys())
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ends = []
    for k, (n, lab) in enumerate(labs.items()):
        ys = [within[n][b][1] for b in bins]
        ax.plot(range(len(bins)), ys, color=C[k], lw=2, marker="o", ms=6, mec="#fcfcfb", mew=1.5, label=lab)
        ends.append([ys[-1], f"{lab} {ys[-1]:.2f}"])
    ends.sort(reverse=True)
    for j in range(1, len(ends)):
        ends[j][0] = min(ends[j][0], ends[j - 1][0] - 0.05)
    for y, lab in ends:
        ax.text(len(bins) - 1 + 0.1, y, lab, fontsize=8.5, color=INK2, va="center")
    ax.set_xticks(range(len(bins)), [f"{b.replace('-', '–')}\n(n={within['BM25'][b][0]})" for b in bins])
    ax.set_xlim(-0.3, len(bins) + 0.9)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("问题与答案段落的字面重叠率（仅 paraphrase 题，285 道）")
    ax.set_ylabel("R@1")
    ax.set_title("固定题型后，重叠率越低仍然越容易找错")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "fig15_within_paraphrase.png")

    # 图16：逻辑回归，重叠率每升 0.1 的优势比（控制题型、问题长度、段落长度）
    reg = s["逻辑回归"]
    rn = [("BM25", "BM25"), ("bge-base", "bge-base 原版"), ("bge-m3", "bge-m3"),
          ("bge-base-ft", "bge-base 微调后"), ("bge-base-ft+rerank", "微调后 + reranker")]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for k, (n, lab) in enumerate(rn):
        o, lo, hi = reg[n]["全部题·重叠率每升0.1的OR"]
        ax.plot([lo, hi], [k, k], color=C[0], lw=2, solid_capstyle="round")
        ax.scatter([o], [k], s=60, color=C[0], edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
        ax.text(hi + 0.06, k, f"{o:.2f}（{lo:.2f}–{hi:.2f}）", va="center", fontsize=8.5, color=INK2)
    ax.axvline(1, color=MUTED, lw=1, ls="--")
    ax.set_yticks(range(len(rn)), [lab for _, lab in rn])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0.8, 3.6)
    ax.set_xlabel("优势比（重叠率每升 0.1，答对的几率变为原来的倍数；虚线 = 1 表示无影响）")
    ax.set_title("控制题型与长度后，重叠率对所有方法仍有显著影响")
    save(fig, "fig16_logit_or.png")

    # 图17：BM25 与微调模型的互补性，各重叠组里"谁对谁错"的题数
    comp = s["互补性"]["BM25 vs bge-base-ft"]
    cats = [("两者都对", "两者都对"), ("只有BM25对", "只有 BM25 对"), ("只有bge-base-ft对", "只有微调模型对"),
            ("两者都错", "两者都错")]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    left = np.zeros(len(GROUPS))
    for k, (key, lab) in enumerate(cats):
        v = np.array([comp[g][key] for g in GROUPS])
        ax.barh(range(len(GROUPS)), v, left=left, color=C[k], height=0.6, edgecolor="#fcfcfb", linewidth=2, label=lab)
        for i, (vv, l0) in enumerate(zip(v, left)):
            if vv >= 10:
                ax.text(l0 + vv / 2, i, str(vv), ha="center", va="center", fontsize=8.5, color="#fcfcfb")
        left += v
    ax.set_yticks(range(len(GROUPS)), [g.replace("\n", " ") for g in GROUP_TICKS])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("题数（测试集，R@1 是否命中）")
    ax.set_title("BM25 只在高、中重叠组有独占的正确题", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8.5, ncol=4)
    save(fig, "fig17_complementarity.png")

    # 图18：四类问题的重叠率分布（含手写题），箱线图
    qs = jl(P / "questions.jsonl")
    hand = [float(r["overlap"]) for r in csv.DictReader((R / "handwritten_per_question.csv").open(encoding="utf-8-sig"))]
    groups = [("大模型 plain 题", [q["overlap"] for q in qs if q["kind"] == "plain"]),
              ("大模型 paraphrase 题", [q["overlap"] for q in qs if q["kind"] == "paraphrase"]),
              ("作者手写题", hand), ("真人搜索提问\n（DuRetrieval）", real_query_overlap())]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    bp = ax.boxplot([v for _, v in groups], vert=False, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "lw": 1.5}, whiskerprops={"color": MUTED}, capprops={"color": MUTED})
    for k, b in enumerate(bp["boxes"]):
        b.set(facecolor=C[k], alpha=0.85, edgecolor="#fcfcfb")
    for k, (_, v) in enumerate(groups):
        ax.text(1.02, k + 1, f"n={len(v)}  均值 {np.mean(v):.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(1, len(groups) + 1), [g for g, _ in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.32)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("字面重叠率")
    ax.set_title("四类问题的字面重叠率")
    save(fig, "fig18_overlap_four_sources.png")


if __name__ == "__main__":
    fig_data()
    fig_experiments()
    fig_external()
    fig_statistics()
