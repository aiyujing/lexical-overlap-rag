"""第6步：实验B，对比学习微调 bge-base（±难负样本）。

- 损失：InfoNCE。每道题的正确段落要比同一批里其他段落得分都高
- 普通版：同一批里别的题的答案段落当错误答案（in-batch negatives）
- 难负样本版：额外给每道题配一个 BM25 排名靠前但不是答案的段落，专治"字面像但答案错"
- 难负样本只从训练章节的段落里挑，训练时看不到测试章节的段落
- 每轮训练后在 dev（CNN 章）上评估，选 dev MRR 最好的一轮，再报 test 成绩

用法：python step6_exp_b_finetune.py            # 两种设置 × 3 个随机种子
输出：results/exp_b.json，results/exp_b_per_question.csv，models/（每种设置 seed0 的最佳模型）
"""
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from step4_baseline import BINS, metrics, rank_of, tokenize
from step5_exp_a_models import BGE_PREFIX

ROOT = Path(__file__).parent
P = ROOT / "data/processed"
OUT = ROOT / "results"
MODEL_DIR = ROOT / "models"

BASE = "BAAI/bge-base-zh-v1.5"
EPOCHS = 5
BATCH = 16
LR = 2e-5
TEMP = 0.05          # InfoNCE 温度：越小，模型越"较真"地拉开对错段落的分差
TRAIN_MAX_LEN = 384  # 训练时段落截断长度（省显存）；评估时用 512
HARD_TOP_K = 10      # 难负样本从 BM25 前 10 名（去掉正确段落）里随机挑
SEEDS = [0, 1, 2]


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def mine_hard_negatives(train_qs, chunks, train_chunk_idx):
    """对每道训练题，用 BM25 在训练段落里找排名靠前、但不是答案的段落。"""
    bm25 = BM25Okapi([tokenize(chunks[i]["title"] + " " + chunks[i]["text"]) for i in train_chunk_idx])
    idx = {c["id"]: i for i, c in enumerate(chunks)}
    hard = {}
    for q in train_qs:
        scores = bm25.get_scores(tokenize(q["question"]))
        order = [train_chunk_idx[j] for j in np.argsort(-scores)]
        hard[q["qid"]] = [i for i in order if i != idx[q["chunk_id"]]][:HARD_TOP_K]
    return hard


def make_batches(train_qs, idx):
    """打乱后分批，同一批里不出现两道题共用一个答案段落（否则会把正确答案当错误答案）。"""
    pool = train_qs[:]
    random.shuffle(pool)
    batches = []
    while pool:
        batch, used, rest = [], set(), []
        for q in pool:
            if len(batch) < BATCH and idx[q["chunk_id"]] not in used:
                batch.append(q)
                used.add(idx[q["chunk_id"]])
            else:
                rest.append(q)
        batches.append(batch)
        pool = rest
    return batches


def embed(model, texts, max_len):
    model.max_seq_length = max_len
    feats = model.tokenize(texts)
    feats = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in feats.items()}
    return F.normalize(model(feats)["sentence_embedding"], dim=-1)


@torch.no_grad()
def evaluate(model, docs, qs, idx):
    model.eval()
    model.max_seq_length = 512
    D = model.encode(docs, normalize_embeddings=True, batch_size=32, convert_to_tensor=True)
    Q = model.encode([BGE_PREFIX + q["question"] for q in qs], normalize_embeddings=True,
                     batch_size=64, convert_to_tensor=True)
    S = (Q @ D.T).float().cpu().numpy()
    return [rank_of(S[i], idx[q["chunk_id"]]) for i, q in enumerate(qs)]


def train_one(use_hard, seed, data, save_to=None):
    chunks, docs, idx, train_qs, dev_qs, test_qs, hard = data
    set_seed(seed)
    model = SentenceTransformer(BASE, device="cuda")
    model[0].auto_model.gradient_checkpointing_enable()  # 少存中间结果、反向时重算，省显存（8GB 显存满了会慢几十倍）
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    n_steps = EPOCHS * len(make_batches(train_qs, idx))
    sched = torch.optim.lr_scheduler.LambdaLR(  # 前 10% 步数慢慢升学习率，之后线性降到 0
        opt, lambda s: min((s + 1) / (0.1 * n_steps), max(0.0, (n_steps - s) / (0.9 * n_steps))))

    best = {"dev_MRR": evaluate(model, docs, dev_qs, idx), "epoch": 0}
    best["dev_MRR"] = metrics(best["dev_MRR"])["MRR"]
    best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    log = [{"epoch": 0, "dev_MRR": best["dev_MRR"]}]

    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch in make_batches(train_qs, idx):
            pos = [idx[q["chunk_id"]] for q in batch]
            passages = pos[:]
            if use_hard:
                taken = set(pos)
                for q in batch:  # 难负样本不能是本批任何一道题的答案
                    cand = [i for i in hard[q["qid"]] if i not in taken] or [i for i in hard[q["qid"]]]
                    passages.append(random.choice(cand))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qe = embed(model, [BGE_PREFIX + q["question"] for q in batch], 64)
                pe = embed(model, [docs[i] for i in passages], TRAIN_MAX_LEN)
                logits = (qe @ pe.T).float() / TEMP  # 每行：一道题对本批所有段落的打分
            loss = F.cross_entropy(logits, torch.arange(len(batch), device=logits.device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())

        dev = metrics(evaluate(model, docs, dev_qs, idx))
        log.append({"epoch": ep, "train_loss": float(np.mean(losses)), "dev_MRR": dev["MRR"]})
        print(f"    epoch {ep}  loss={np.mean(losses):.4f}  dev MRR={dev['MRR']:.3f}")
        if dev["MRR"] > best["dev_MRR"]:
            best = {"dev_MRR": dev["MRR"], "epoch": ep}
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ranks = evaluate(model, docs, test_qs, idx)
    if save_to:
        model.save(str(save_to))
    del model, opt
    torch.cuda.empty_cache()
    return ranks, best, log


def group_metrics(ranks, test_qs):
    res = {"全部": metrics(ranks)}
    for kind in ("plain", "paraphrase"):
        res[kind] = metrics([r for r, q in zip(ranks, test_qs) if q["kind"] == kind])
    for label, lo, hi in BINS:
        res[label] = metrics([r for r, q in zip(ranks, test_qs) if lo <= q["overlap"] < hi])
    return res


def main():
    chunks = [json.loads(l) for l in (P / "chunks.jsonl").open(encoding="utf-8")]
    qs = [json.loads(l) for l in (P / "questions.jsonl").open(encoding="utf-8")]
    idx = {c["id"]: i for i, c in enumerate(chunks)}
    docs = [c["title"] + " " + c["text"] for c in chunks]
    split = {s: [q for q in qs if q["split"] == s] for s in ("train", "dev", "test")}
    train_chunk_idx = sorted({idx[q["chunk_id"]] for q in split["train"]})
    hard = mine_hard_negatives(split["train"], chunks, train_chunk_idx)
    data = (chunks, docs, idx, split["train"], split["dev"], split["test"], hard)
    print(f"训练 {len(split['train'])} 题 / {len(train_chunk_idx)} 段，dev {len(split['dev'])}，test {len(split['test'])}")

    result, per_q = {}, {}
    for name, use_hard in (("ft-inbatch", False), ("ft-hardneg", True)):
        runs = []
        for seed in SEEDS:
            print(f"== {name} seed={seed} ==")
            t0 = time.time()
            save = MODEL_DIR / f"bge-base-{name}" if seed == 0 else None
            ranks, best, log = train_one(use_hard, seed, data, save)
            res = group_metrics(ranks, split["test"])
            runs.append({"seed": seed, "best_epoch": best["epoch"], "minutes": round((time.time() - t0) / 60, 1),
                         "log": log, "test": res})
            per_q[f"{name}_s{seed}"] = ranks
            print(f"    最佳 epoch={best['epoch']}  test R@1={res['全部']['R@1']:.3f}  "
                  f"低重叠 R@1={res[BINS[0][0]]['R@1']:.3f}  用时 {runs[-1]['minutes']} 分钟")
        groups = list(runs[0]["test"])
        summary = {g: {m: {"mean": float(np.mean([r["test"][g][m] for r in runs])),
                           "std": float(np.std([r["test"][g][m] for r in runs]))}
                       for m in ("R@1", "R@5", "R@10", "MRR")} for g in groups}
        result[name] = {"summary": summary, "runs": runs}

    (OUT / "exp_b.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "exp_b_per_question.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["qid", "kind", "overlap", "question"] + [f"{n}_rank" for n in per_q])
        for i, q in enumerate(split["test"]):
            w.writerow([q["qid"], q["kind"], q["overlap"], q["question"]] + [per_q[n][i] for n in per_q])

    print("\n汇总（3 个种子的均值±标准差，R@1）")
    for name, r in result.items():
        print(f"  {name:<11} " + "  ".join(f"{g} {s['R@1']['mean']:.3f}±{s['R@1']['std']:.3f}"
                                           for g, s in r["summary"].items()))


if __name__ == "__main__":
    main()
