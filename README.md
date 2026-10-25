# rag-know-engine

文档入库管道 + RAG 检索服务（Java 版 KnowEngine 的 Python 改写）。设计见 [docs/design.md](docs/design.md)。

## 本地启动

需要 Docker（Docker Desktop 或 OrbStack 均可）。如果终端里找不到 `docker` 命令，OrbStack 的 CLI 在 `/Applications/OrbStack.app/Contents/MacOS/xbin`。

镜像都不走 Docker Hub（本机 Docker 虚拟机访问不了）：Postgres 用 AWS ECR 上的官方镜像，MinIO 用 Chainguard 镜像，ES 用 Elastic 官方仓库。

```bash
conda activate rag
pip install -e ".[dev]"

cp .env.example .env          # 填写 LLM_BASE_URL / LLM_API_KEY
docker compose up -d --build  # PostgreSQL / Elasticsearch(+IK) / MinIO
alembic upgrade head          # 建表
alembic check                 # 确认 ORM 模型与迁移一致
```

| 服务 | 地址 |
|---|---|
| PostgreSQL | `localhost:5432`（账号见 `.env`） |
| Elasticsearch | http://localhost:9200 |
| MinIO API / 控制台 | `localhost:9000` / http://localhost:9001 |

### MinerU（PDF / DOCX 解析）

MinerU 依赖较重，装在单独的 conda 环境 `mineru`（4.0.12）里，作为独立服务运行：

```bash
conda run -n mineru --no-capture-output mineru-api --host 127.0.0.1 --port 8000 --upload-dir ~/.cache/mineru-uploads
```

首次解析会自动下载模型（约 2 分钟），之后单个 10 页 PDF 约 7 秒。

### 启动服务

```bash
/opt/anaconda3/envs/rag/bin/uvicorn know_engine.main:app --port 8009  # 或 conda run -n rag uvicorn ...
```

打开 http://localhost:8009 即可使用原 Java 版的上传 / 文档管理页面。API 进程内会启动一个入库 worker；也可以单独运行更多 worker：

```bash
conda run -n rag --no-capture-output python -m know_engine.ingest.worker
```

打开 http://localhost:8009 后：上传 / 文档管理 / AI 对话 / 评测报告四个页面都可用（`chat.html`、`eval-report.html`）。

## 数据（不在仓库中）

测试文档和评测集不随仓库提交，需自行放到以下位置：

| 路径 | 内容 |
|---|---|
| `data/testFile/` | 待入库的示例文档（md / txt / docx / pdf / xlsx / csv） |
| `data/eval/original/*.jsonl` | 原始评测集，每行 `{"id", "question", "groundTruthAnswer", "groundTruthChunkIds", "groundTruthDocumentIds"}` |
| `data/corpus/` | 规模测试用的公开语料（如 Python 中文文档的纯文本包） |

## 评测

评测集在 `data/eval/`：`eval-all.jsonl`（100 条）按 case id 哈希固定拆成 `eval-tune.jsonl`（64 条，只用于调参）和 `eval-holdout.jsonl`（36 条，只用于汇报最终数字）。`data/eval/original/` 是 Java 项目原样的评测集。

```bash
# 文档重新切分后分块 ID 会变，需要重新映射评测集的真值
conda run -n rag python scripts/relabel_eval.py --sql ../know-engine/src/main/resources/sql/know_engine.sql

# 跑一轮评测（--set 覆盖 RagConfig 的任意字段，--no-judge 只算检索指标）
conda run -n rag python -m know_engine.eval.cli eval-tune.jsonl --label "试验说明" --set rerank_min_score=0.1

# 两次评测对比
conda run -n rag python -m know_engine.eval.cli --compare RUN_A RUN_B

# BadCase 链路排查：真值分块在向量 / 关键词 / RRF / 重排各阶段的名次，以及丢在哪一步
conda run -n rag python -m know_engine.eval.analyze eval-tune.jsonl aurora-md-010 aurora-txt-015

# 重排与向量分数分布（用数据定阈值）
conda run -n rag python -m know_engine.eval.analyze eval-tune.jsonl --distribution
```

## 数据工程工具

```bash
# 目录批量导入（幂等，已入库的文件按内容哈希跳过），输出吞吐
conda run -n rag python -m know_engine.ingest.bulk data/corpus/python-zh --pattern "*.txt" --workers 3

# PostgreSQL 与 ES 对账；--repair 把缺失的版本放回队列重建、删除孤儿数据
conda run -n rag python -m know_engine.ingest.reconcile [--repair]

# 生成 IK 领域词典（ES 通过 remote_ext_dict 每分钟拉取，热更新）
conda run -n rag python -m know_engine.ingest.dictionary

# 零停机重建索引：新建 vN+1 → _reindex 复制（不重新向量化）→ 抽样校验 → 原子切换别名
conda run -n rag python -m know_engine.ingest.reindex --delete-old

# Dagster 编排（文件发现 sensor、按文件分区的入库资产、每日对账任务）
conda run -n rag pip install -e ".[orchestration]"
conda run -n rag --no-capture-output dagster dev -m know_engine.orchestration.definitions
```

监控接口：`GET /api/ingest/stats`（各阶段次数、失败数、p50/p95 耗时、吞吐、向量复用率），`GET /api/document/versions/{versionId}/lineage`（某个版本经历的每个阶段及参数、结果）。

### 测试

```bash
conda run -n rag python -m pytest -q
```

验证 IK 分词插件：

```bash
curl -s -H 'Content-Type: application/json' localhost:9200/_analyze -d '{"analyzer":"ik_smart","text":"曙光智造园区运行手册"}'
```
