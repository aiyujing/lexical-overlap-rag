# 字面重叠偏差：中文教材问答中检索模型的分析与改进

机器学习课程期末项目。在《动手学深度学习》（d2l-zh）6 章内容上搭建 RAG 问答，
以"问题与答案段落的字面重叠率"为主线，分析 BM25、向量模型、微调和 reranker 在不同重叠程度下的表现。

## 主要结果（测试集 568 题，候选 376 段，R@1）

| 方法 | 全部 | 低重叠（<0.5） | 高重叠（≥0.8） |
|---|---|---|---|
| BM25 | 0.491 | 0.031 | 0.860 |
| bge-base-zh-v1.5（不微调） | 0.451 | 0.198 | 0.665 |
| bge-m3（不微调） | 0.505 | 0.229 | 0.726 |
| bge-base 对比学习微调（难负样本，3 种子均值） | 0.547 | 0.310 | 0.738 |
| 微调 + bge-reranker-base | 0.671 | 0.336 | 0.916 |

## 环境

- Windows 11，Python 3.11.3，NVIDIA RTX 4060 Laptop（8GB），驱动 581.57
- 安装：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

- 国内下载模型：`set HF_ENDPOINT=https://hf-mirror.com`
- Windows 终端输出中文：`set PYTHONIOENCODING=utf-8`
- 出题和错误分析调用 DeepSeek API：在项目根目录建 `.env`，写一行 `DEEPSEEK_API_KEY=你的key`（`.env` 已在 .gitignore 中）

## 数据

- d2l-zh（Apache-2.0）：从 https://github.com/d2l-ai/d2l-zh 下载 master 分支 zip，解压到 `data/raw/d2l-zh/d2l-zh-master/`
- DuRetrieval（C-MTEB）：从 Hugging Face 的 `C-MTEB/DuRetrieval` 和 `C-MTEB/DuRetrieval-qrels` 下载 parquet，
  放到 `data/raw/cmteb/DuRetrieval/` 和 `data/raw/cmteb/DuRetrieval-qrels/`
- 生成好的段落和题目在 `data/processed/`，可以跳过第 1–3 步直接做实验

## 运行顺序

| 步骤 | 命令 | 作用 | 耗时（RTX 4060） |
|---|---|---|---|
| 1 | `python step1_chunk_d2l.py` | 清洗 d2l-zh，切成 376 段 | 秒级 |
| 2 | `python step2_gen_questions.py` | DeepSeek 每段出 2 道 plain + 2 道 paraphrase 题（约 19 万 token） | 约 5 分钟 |
| 3 | `python step3_build_dataset.py` | 过滤、按章划分 train/dev/test，导出人工抽查表 | 秒级 |
| 4 | `python step4_baseline.py` | Baseline：BM25 vs bge-small | 约 1 分钟 |
| 5 | `python step5_exp_a_models.py` | 实验A：5 个向量模型对比 | 约 2 分钟 |
| 6 | `python step6_exp_b_finetune.py` | 实验B：InfoNCE 微调 bge-base（±难负样本，3 个种子） | 约 15 分钟 |
| 7 | `python step7_exp_c_rerank.py` | 实验C：bge-reranker-base 重排前 20 | 约 2 分钟 |
| 8 | `python step8_error_analysis.py` | 错误分析：DeepSeek 给错题分类（约 12 万 token） | 约 2 分钟 |
| 9 | `python step9_external_validation.py` | 外部验证：DuRetrieval 真人搜索提问，10 万段落 | 较长，主要花在 BM25 |
| 10 | `python step10_human_eval.py` | 人工抽查表和手写题的统计与评测（需先人工填 `data/processed/spot_check.csv`、`data/handwritten.csv`） | 约 1 分钟 |
| 11 | `python step11_statistics.py` | 统计检验：bootstrap 置信区间、McNemar 配对检验、逻辑回归控制题型、互补性、错误分析标准的一致性 | 约 1 分钟 |
| 图 | `python make_figures.py` | 生成 `figures/` 下的报告配图 | 约 1 分钟 |
| demo | `python app.py` | 网页问答，浏览器打开 http://127.0.0.1:7860 | — |

结果在 `results/`，微调后的模型保存在 `models/`（体积较大，未上传；运行第 6 步可重新得到）。

## 文件

```
step1–11_*.py         数据处理、实验与统计分析，按编号顺序运行
make_figures.py       报告配图
app.py                网页问答 demo
lesson*.py            学习过程中手写 embedding / attention 的练习代码
data/processed/       段落、题目、人工抽查表
results/              各实验的指标（json）和逐题结果（csv）
figures/              报告配图
```
