"""第1步：清洗 d2l-zh 并切成段落。

输入：data/raw/d2l-zh/d2l-zh-master/ 下与课件 ML2–11 对应的 6 章
输出：data/processed/chunks.jsonl，每行一个段落
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "data/raw/d2l-zh/d2l-zh-master"
OUT = ROOT / "data/processed/chunks.jsonl"

# d2l-zh 章节 -> 对应的课件
CHAPTERS = {
    "chapter_introduction": "ML2 监督学习",
    "chapter_linear-networks": "ML3/ML4 回归与分类",
    "chapter_multilayer-perceptrons": "ML5 神经网络",
    "chapter_convolutional-neural-networks": "ML6 CNN",
    "chapter_convolutional-modern": "ML8/ML9 ResNet与DenseNet",
    "chapter_computer-vision": "ML10/ML11 检测与分割",
}

TARGET_ZH = 300  # 每段目标汉字数，满了就切
MIN_ZH = 60      # 少于这个数的段落并入前一段，仍不够就丢弃


def n_zh(s):
    return sum("一" <= c <= "鿿" for c in s)


def clean(md):
    md = re.sub(r"\n## 练习\n.*", "\n", md, flags=re.S)           # 练习题及之后的讨论链接
    md = re.sub(r"```.*?```", "", md, flags=re.S)                 # 代码块
    md = re.sub(r":begin_tab:`[^`]*`.*?:end_tab:", "", md, flags=re.S)
    md = re.sub(r"\$\$.*?\$\$", "（公式略）", md, flags=re.S)       # 独立公式行
    md = re.sub(r"^!\[.*$", "", md, flags=re.M)                   # 图片
    md = re.sub(r":(label|eqlabel|width):`[^`]*`", "", md)
    md = re.sub(r"[ \t]*:cite:`[^`]*`", "", md)  # 只吃空格不吃换行，否则会把下一行正文并进标题
    md = re.sub(r"[ \t]*:numref:`[^`]*`", "图", md)
    md = re.sub(r"[ \t]*:eqref:`[^`]*`", "公式", md)
    md = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md)              # 链接只留文字
    md = re.sub(r"[*_]{2}([^*_]+)[*_]{2}", r"\1", md)             # 加粗
    return md


def sections(md):
    """按标题切成 (标题路径, 正文) 。"""
    path, buf = [], []
    for line in md.splitlines():
        m = re.match(r"^(#{1,4}) (.+)", line)
        if m:
            if buf:
                yield " > ".join(path), "\n".join(buf)
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2).strip()]
            buf = []
        else:
            buf.append(line)
    if buf:
        yield " > ".join(path), "\n".join(buf)


def chunk(body):
    """把一个小节的自然段合并成约 TARGET_ZH 字的段落。"""
    paras = [re.sub(r"\s*\n\s*(?!\d+\.|[-*] )", "", p).strip() for p in re.split(r"\n\s*\n", body)]
    paras = [re.sub(r"\n\s*", "\n", p) for p in paras]  # 列表项保留换行
    paras = [p for p in paras if n_zh(p) >= 5]
    out, cur = [], ""
    for p in paras:
        if cur and n_zh(cur) + n_zh(p) > TARGET_ZH * 1.5:
            out.append(cur)
            cur = p
        else:
            cur = cur + "\n" + p if cur else p
        if n_zh(cur) >= TARGET_ZH:
            out.append(cur)
            cur = ""
    if cur:
        if out and n_zh(cur) < MIN_ZH:
            out[-1] += "\n" + cur
        else:
            out.append(cur)
    return [c for c in out if n_zh(c) >= MIN_ZH]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ch, lecture in CHAPTERS.items():
        for f in sorted((SRC / ch).glob("*.md")):
            if f.stem.endswith("_origin"):
                continue
            # 代码实现、比赛、数据集读取类小节：出的题多是"作者设的参数是多少"，不像学生提问
            if any(k in f.stem for k in ("scratch", "concise", "kaggle", "dataset")):
                continue
            md = clean(f.read_text(encoding="utf-8"))
            for title, body in sections(md):
                for text in chunk(body):
                    rows.append({
                        "id": f"d2l-{len(rows):04d}",
                        "chapter": ch,
                        "lecture": lecture,
                        "file": f.name,
                        "title": title,
                        "text": text,
                        "n_zh": n_zh(text),
                    })
    with OUT.open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"共 {len(rows)} 个段落 -> {OUT}")


if __name__ == "__main__":
    main()
