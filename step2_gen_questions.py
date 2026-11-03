"""第2步：用大模型给每个段落出题。

每个段落出两类题：
- plain：常规出题提示词（业界常见做法）
- paraphrase：要求避开原文用词、模仿不熟悉内容的学生提问
两类题的字面重叠率对比是数据章的分析内容之一。

用法：
    python step2_gen_questions.py --limit 20   # 先试跑20个段落
    python step2_gen_questions.py              # 全量，已生成的段落会跳过
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
CHUNKS = ROOT / "data/processed/chunks.jsonl"
OUT = ROOT / "data/processed/questions_raw.jsonl"
API = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"

PROMPT = """下面是一本深度学习教材中的一个段落。请根据它出题，用于测试一个检索系统能否找到这个段落。

要求：
1. 出2道"plain"题：像正常出考题一样，问题的答案必须能在这个段落中找到。
2. 出2道"paraphrase"题：假设提问者是一个没读过这本书的学生，只能用自己的大白话描述想知道的东西，尽量不要使用段落里的原词和术语，但答案仍然必须在这个段落中。
3. 每道题是一句中文问句，不要出现"本段""文中""上述"这类指代。
4. 如果这个段落没有实质知识内容（例如只是过渡句、目录），所有列表返回空。

只输出JSON：{{"plain": ["...", "..."], "paraphrase": ["...", "..."]}}

段落标题：{title}
段落内容：
{text}"""


def load_key():
    for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("DEEPSEEK_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(".env 里没有 DEEPSEEK_API_KEY")


def ask(key, chunk, retries=4):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(title=chunk["title"], text=chunk["text"])}],
        "response_format": {"type": "json_object"},
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for i in range(retries):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120))
            q = json.loads(r["choices"][0]["message"]["content"])
            return {"chunk_id": chunk["id"], "model": r.get("model"), "usage": r.get("usage", {}),
                    "plain": q.get("plain", []), "paraphrase": q.get("paraphrase", [])}
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, TimeoutError) as e:
            if i == retries - 1:
                return {"chunk_id": chunk["id"], "error": repr(e)[:200]}
            time.sleep(2 ** i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
    done = set()
    if OUT.exists():
        rows = [json.loads(l) for l in OUT.open(encoding="utf-8")]
        done = {r["chunk_id"] for r in rows if "error" not in r}
    todo = [c for c in chunks if c["id"] not in done]
    if args.limit:
        todo = todo[:: max(1, len(todo) // args.limit)][: args.limit]  # 均匀抽样，覆盖各章
    print(f"待生成 {len(todo)} 个段落（已完成 {len(done)}）")

    key = load_key()
    n_ok = n_err = tokens = 0
    with OUT.open("a", encoding="utf-8") as fp, ThreadPoolExecutor(args.workers) as pool:
        for res in pool.map(lambda c: ask(key, c), todo):
            fp.write(json.dumps(res, ensure_ascii=False) + "\n")
            fp.flush()
            if "error" in res:
                n_err += 1
            else:
                n_ok += 1
                tokens += res["usage"].get("total_tokens", 0)
            if (n_ok + n_err) % 50 == 0:
                print(f"  {n_ok + n_err}/{len(todo)}")
    print(f"成功 {n_ok}，失败 {n_err}，共用 {tokens} tokens")


if __name__ == "__main__":
    main()
