# rag-know-engine

一个完整的 RAG 知识库问答系统：支持 PDF / Word / Markdown / TXT / Excel / CSV 文档入库，经过解析、切分、向量化后写入 Elasticsearch，问答时做问题改写、向量与关键词双路检索、RRF 融合和 BGE 重排，再由大模型流式生成带引用的回答。配套一套带真值的评测体系，用 Hit@K / MRR / Recall@K 和 LLM 打分量化每一次调优的效果。

主要特性：

- **多格式文档解析**：MinerU 解析 PDF / DOCX，图片由视觉模型生成描述；表格按行切分
- **5 种切分策略**：按长度、按标题层级（父子块）、按全部标题、按正则、按分隔符
- **混合检索**：向量 KNN + BM25（IK 中文分词 + 领域词典），RRF 融合，BGE 重排，阈值按分数分布标定，空结果兜底
- **多轮对话**：意图识别（闲聊不检索）、结合历史的问题改写、SSE 流式输出进度与引用
- **评测与调优**：调参集 / 留出集分离，BadCase 逐阶段排查工具，每轮只改一个变量
- **可靠入库**：状态驱动的后台 worker、失败重试、文档多版本、增量向量化、数据质量检查、索引对账与零停机重建

完整设计与每一轮调优的依据见 [docs/design.md](docs/design.md)。

## 结果一览

| 指标 | 基线（初始参数） | 调优后 |
|---|---|---|
| 留出集 Hit@5（36 条，未参与调参） | 0.889 | **0.972** |
| 留出集 MRR | 0.803 | **0.830** |
| 留出集答案正确性（LLM 打分） | 0.903 | **0.947** |
| 忠实度 | 1.000 | 1.000 |

- 批量导入 552 篇文档 → 20,178 个分块，用时 601 秒（33.6 分块/秒，0 失败）。
- 加入 2 万条干扰分块后，留出集 Hit@5 不变，检索 p95 延迟约 55～60 ms。
- 文档新版本只改了一行时，复用 6 个已有向量、只重新计算 1 个。

## 架构

```
              ┌──────────────────────── 入库流水线 ────────────────────────┐
  上传 ─► UPLOADED ─► 转换 ─► CONVERTED ─► 切分 + 质量门禁 ─► CHUNKED ─► 向量化 + 索引 ─► VECTOR_STORED
           │           │ MinerU / 直读          │ 5 种策略            │ worker（SKIP LOCKED）
           ▼           ▼                        ▼                     ▼
        MinIO 原文件  MinIO 转换产物      PostgreSQL 分块       Elasticsearch（别名 → vN）
                                                                       ▲
  ┌──────────────────────── 问答链路 ────────────────────────┐          │
  问题 ─► 意图识别 ─► 问题改写 ─► 向量 KNN ┐                            │
                                 BM25(IK) ┴─► RRF ─► BGE 重排(+兜底) ─► 父块替换 ─► LLM 流式生成（SSE）

  ┌──────────────────────── 评测 ────────────────────────┐
  JSONL 评测集 ─► 跑真实链路 ─► Hit@K / MRR / Recall@K + LLM 打分 ─► 入库 / 对比 / BadCase 链路排查
```

## 技术栈

| 层 | 选型 |
|---|---|
| 语言 / Web | Python 3.12、FastAPI、Uvicorn |
| 存储 | PostgreSQL 16（SQLAlchemy 2 + Alembic）、Elasticsearch 8.15 + IK 中文分词、MinIO |
| 模型 | OpenAI 兼容接口：对话 / 改写 / 评测打分、`bge-m3` 向量（1024 维）、视觉模型（图片描述）；本地 `BAAI/bge-reranker-base` 重排 |
| 文档解析 | MinerU 4（PDF / DOCX → Markdown）、pandas（XLSX / CSV） |
| 编排 | 状态驱动 worker；Dagster（可选） |
| 工程化 | Docker Compose、pytest、ruff、structlog |

## 模块说明

```
src/know_engine/
├── main.py            FastAPI 入口：挂载路由与静态页面，启动时建索引、启动入库 worker、预加载重排模型
├── config.py          所有配置（pydantic-settings，读 .env）
├── db.py              数据库引擎与事务上下文
├── constants.py       文档 / 分块状态、切分方式等枚举
├── prompts.py         读取 prompts/ 下的提示词
├── models/            SQLAlchemy 模型：文档、版本、分块、会话、消息、评测批次 / 明细、入库阶段日志
├── clients/           外部服务客户端：Elasticsearch、MinIO、模型 API、MinerU
├── ingest/            入库流水线（见下）
├── rag/               检索与生成链路（见下）
├── chat/              流式对话
├── eval/              评测（见下）
├── api/               HTTP 接口
└── orchestration/     Dagster 定义
```

### `ingest/` 入库流水线

| 文件 | 作用 |
|---|---|
| `service.py` | 文档生命周期：上传（内容哈希去重）、上传新版本、切分、切换 / 失效版本、删除、分块编辑 |
| `converters.py` | 按文件类型转换：PDF / DOCX 走 MinerU，图片存 MinIO 并调用视觉模型生成描述；Markdown 补图片描述；TXT / 表格直接使用原文件 |
| `splitters/` | 5 种切分策略：LENGTH（长度）、TITLE（按标题层级的父子块）、SMART（全部标题层级）、REGEX、SEPARATOR；表格按行切分 |
| `quality.py` | 数据质量门禁：空块、短块、超长块、重复率、乱码率；严重问题直接拒绝本次切分 |
| `indexer.py` | 批量向量化并写入 ES（`_id` = chunk_id，重跑幂等）；按文本哈希复用已有向量；新版本完成后下线旧版本 |
| `worker.py` | 状态驱动 worker：`FOR UPDATE SKIP LOCKED` 领取待向量化版本，可多实例并行；失败指数退避，超过上限标记 FAILED |
| `state.py` | 状态流转与文档级锁（API 用 `NOWAIT`、worker 用 `SKIP LOCKED`，不会死锁） |
| `observe.py` | 每个阶段的耗时、条数、参数、错误写入 `ingest_stage_log`，汇总出吞吐、p50 / p95、向量复用率 |
| `reconcile.py` | PostgreSQL 与 ES 对账：缺失、过期版本、孤儿数据；`--repair` 自动修复 |
| `reindex.py` | 零停机重建索引：新建 `vN+1` → 复制（不重新向量化）→ 抽样自检 → 原子切换别名；重建期间用 advisory lock 挡住写入 |
| `dictionary.py` | 从数据中生成 IK 领域词典并发布到 MinIO，ES 定时拉取热更新 |
| `bulk.py` | 目录批量导入（中断后可重跑），输出吞吐 |

### `rag/` 检索与生成

| 文件 | 作用 |
|---|---|
| `query.py` | 意图识别（闲聊不检索）与问题改写（结合会话历史）；均有超时降级 |
| `retrieve.py` | 向量 KNN 与 BM25 双路召回、RRF 融合（按 chunk_id 合并）、BGE 重排（阈值 + 空结果兜底）、子块替换为父块 |
| `pipeline.py` | 串联整条链路，所有参数集中在 `RagConfig`，每一步计时 |
| `llm.py` | 对话调用封装：关闭思考模式、过滤 `<think>` 内容、流式输出 |

### `eval/` 评测

| 文件 | 作用 |
|---|---|
| `runner.py` | 批量评测：每条用例走真实链路，计算指标、LLM 打分，结果入库 |
| `metrics.py` | Hit@K、MRR、Recall@K，以及忠实度、相关性、正确性、上下文精确率 / 召回率五项 LLM 打分 |
| `report.py` | 报告、历史列表、两次评测对比、CSV 导出 |
| `analyze.py` | BadCase 链路排查：真值分块在向量 / 关键词 / RRF / 重排各阶段的名次；重排与向量分数分布 |
| `cli.py` | 命令行评测，`--set` 覆盖任意 `RagConfig` 字段做对照实验 |
| `dataset.py` | 评测集读取与上传 |

### 其他

| 路径 | 作用 |
|---|---|
| `chat/service.py` | 会话管理、意图分流、检索进度实时推送、SSE 流式回答、标题生成 |
| `api/` | `document.py`（文档 / 分块 / 入库统计 / 血缘）、`chat.py`、`eval.py`、`serializers.py`（接口 JSON 字段格式） |
| `orchestration/definitions.py` | Dagster：文件发现 sensor、按文件分区的入库资产、质量检查、每日对账任务 |
| `scripts/relabel_eval.py` | 文档重新切分后，把评测集真值映射到新的分块 ID |
| `alembic/` | 数据库迁移 |
| `docker/elasticsearch/` | ES 镜像（IK 插件 + 远程词典配置） |
| `prompts/` | 改写、意图、生成、评测打分等提示词 |
| `static/` | 上传、文档管理、对话、评测报告四个页面 |
| `tests/` | 切分器、质量门禁、评测指标、流式解析等单元测试 |

## 快速开始

### 1. 环境与依赖

需要 Docker（Docker Desktop 或 OrbStack）和 Python 3.12。

```bash
conda create -n rag python=3.12 -y && conda activate rag
pip install -e ".[dev]"            # 需要 Dagster 时加上 orchestration：".[dev,orchestration]"
```

### 2. 配置 `.env`

在项目根目录新建 `.env`。除模型接口外都有默认值：

| 变量 | 说明 | 默认值 |
|---|---|---|
| `LLM_BASE_URL` | OpenAI 兼容接口地址（以 `http(s)://` 开头，通常以 `/v1` 结尾） | 必填 |
| `LLM_API_KEY` | 接口密钥 | 必填 |
| `CHAT_MODEL` | 生成回答、评测打分 | `qwen3.8:27b` |
| `LIGHT_MODEL` | 意图识别、问题改写、会话标题 | `qwen3.5:9b` |
| `VISION_MODEL` | 图片描述；留空则不生成 | 空 |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMS` | 向量模型与维度（须与 ES 索引一致） | `bge-m3` / `1024` |
| `LLM_REASONING_EFFORT` | 传给模型的 `reasoning_effort`；`none` 关闭思考模式 | `none` |
| `RERANK_MODEL` | 本地重排模型 | `BAAI/bge-reranker-base` |
| `PG_HOST` / `PG_PORT` / `PG_USER` / `PG_PASSWORD` / `PG_DB` | PostgreSQL | `localhost` / `5432` / `know_engine` / `know_engine` / `know_engine` |
| `ES_URL` / `ES_INDEX_ALIAS` | Elasticsearch 地址与索引别名 | `http://localhost:9200` / `know-engine-chunks` |
| `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` / `MINIO_BUCKET` | MinIO | `localhost:9000` / `minioadmin` / `minioadmin` / `know-engine` |
| `MINERU_URL` | MinerU 服务地址 | `http://localhost:8000` |

最小示例：

```bash
LLM_BASE_URL=https://your-model-api/v1
LLM_API_KEY=your-key
VISION_MODEL=qwen3-vl:32b
```

### 3. 启动基础服务并建表

```bash
docker compose up -d --build   # PostgreSQL / Elasticsearch(+IK) / MinIO
alembic upgrade head
```

镜像不依赖 Docker Hub：Postgres 用 AWS ECR 上的官方镜像，MinIO 用 Chainguard 镜像，ES 用 Elastic 官方仓库。

| 服务 | 地址 |
|---|---|
| PostgreSQL | `localhost:5432` |
| Elasticsearch | http://localhost:9200 |
| MinIO API / 控制台 | `localhost:9000` / http://localhost:9001 |

### 4. 启动 MinerU（解析 PDF / DOCX 时需要）

MinerU 依赖较重，建议装在单独的环境里：

```bash
conda create -n mineru python=3.12 -y && conda run -n mineru pip install "mineru==4.0.12"
conda run -n mineru --no-capture-output mineru-api --host 127.0.0.1 --port 8000 --upload-dir ~/.cache/mineru-uploads
```

首次解析会下载模型（约 2 分钟），之后 10 页的 PDF 约 7 秒。

### 5. 启动服务

```bash
uvicorn know_engine.main:app --port 8009
```

打开 http://localhost:8009 ，上传、文档管理、AI 对话、评测报告四个页面都可用。API 进程内自带一个入库 worker，需要更快的向量化可再启动几个：

```bash
python -m know_engine.ingest.worker
```

## 数据（不在仓库中）

测试文档和评测集不随仓库提交，需自行放到以下位置：

| 路径 | 内容 |
|---|---|
| `data/testFile/` | 示例文档（md / txt / docx / pdf / xlsx / csv） |
| `data/eval/original/*.jsonl` | 评测集，每行 `{"id", "question", "groundTruthAnswer", "groundTruthChunkIds", "groundTruthDocumentIds"}` |
| `data/corpus/` | 规模测试用的公开语料，例如 Python 中文文档的纯文本包 |

## 评测

`scripts/relabel_eval.py` 把评测集真值映射到当前的分块 ID，并按用例 ID 哈希固定拆成调参集（64 条）和留出集（36 条）。调参只看调参集，留出集只用来汇报最终结果。

```bash
# 文档重新切分后重新映射真值
python scripts/relabel_eval.py --sql <包含旧分块文本的 SQL 导出>

# 跑一轮评测：--set 覆盖 RagConfig 字段，--no-judge 只算检索指标
python -m know_engine.eval.cli eval-tune.jsonl --label "试验说明" --set rerank_min_score=0.1

# 对比两次评测
python -m know_engine.eval.cli --compare RUN_A RUN_B

# BadCase 排查：真值分块在各阶段的名次以及丢在哪一步
python -m know_engine.eval.analyze eval-tune.jsonl <case_id> ...

# 重排 / 向量分数分布，用来定阈值
python -m know_engine.eval.analyze eval-tune.jsonl --distribution
```

## 运维工具

```bash
python -m know_engine.ingest.bulk data/corpus/python-zh --pattern "*.txt" --workers 3   # 批量导入
python -m know_engine.ingest.reconcile [--repair]                                       # PG / ES 对账
python -m know_engine.ingest.dictionary                                                 # 发布 IK 领域词典
python -m know_engine.ingest.reindex --delete-old                                       # 零停机重建索引
dagster dev -m know_engine.orchestration.definitions                                    # Dagster 界面
```

## 主要接口

| 接口 | 说明 |
|---|---|
| `POST /api/document/upload`、`/upload-version` | 上传文档 / 新版本 |
| `POST /api/document/split/{id}` | 切分（之后由 worker 自动向量化） |
| `POST /api/document/switch-version` | 切换生效版本 |
| `GET /api/document/page`、`/api/segment/page-by-document` | 文档、分块分页查询 |
| `POST /chat/send` | 流式问答（SSE：进度、引用、回答、结束） |
| `POST /eval/run`、`GET /eval/compare` | 运行评测、对比两次评测 |
| `GET /api/ingest/stats` | 各阶段次数、失败数、p50 / p95、吞吐、向量复用率 |
| `GET /api/document/versions/{id}/lineage` | 某个版本经历的全部处理阶段 |

## 测试

```bash
pytest -q
```

## License

[MIT](LICENSE)
