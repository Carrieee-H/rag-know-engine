# KnowEngine-Py 技术设计文档

> 版本：v1.0（阶段 0–4 已完成）　日期：2026-10-25
> 参考实现：`rag/know-engine`（Java / Spring Boot / LangChain4j）

## 1. 背景与目标

把现有的 Java 版 RAG 知识问答系统改写为 Python 版，并按**数据工程**方向重新定位：

- **核心卖点**：一条可靠的「非结构化文档 → 检索索引」数据管道（解析、切分、向量化、索引、版本、增量、质量、补偿）。
- **下游应用**：在索引之上提供混合检索 + 重排 + 流式问答服务。
- **质量闭环**：用带真值的评测集量化每次改动对检索效果的影响。

非目标：复杂前端、多租户权限、Agent 能力。

### 设计原则

1. **流水线显式化**：不依赖 LangChain4j `DefaultRetrievalAugmentor` 那种「框架隐式调用」，每一步都是可单测、可计时的普通函数。
2. **状态驱动 + 幂等**：每个入库阶段以数据库状态为准，任何一步都可以安全重跑。
3. **接口兼容**：保留 Java 版的 HTTP 路由和 SSE 协议，原 `static/` 前端直接复用。
4. **先平移、后增强**：阶段 1–3 复刻原功能并跑出基线指标，阶段 4 再做数据工程增强。

---

## 2. 技术栈

### 2.1 现有 Java 版

| 层 | 技术 |
|---|---|
| 语言 / 框架 | Java 21、Spring Boot 3.5、Spring AOP、Reactor（Flux/SSE） |
| LLM 编排 | LangChain4j（AiServices、DefaultRetrievalAugmentor、QueryTransformer、ContentAggregator 等） |
| 大模型 | 阿里云 DashScope（OpenAI 兼容接口）：`qwen3.6-flash` 对话 / 改写 / 意图 / 评测打分；`text-embedding-v4`（1536 维） |
| 重排 | BGE-Reranker 量化 ONNX，本地 onnxruntime 推理（输出原始 logit） |
| 文档解析 | MinerU（独立 HTTP 服务 `/file_parse`，PDF/DOCX → Markdown）、EasyExcel（XLSX/CSV）、视觉模型生成图片描述 |
| 存储 | MySQL 8（MyBatis-Plus + Druid）、Elasticsearch 8（向量 + 全文）、MinIO（原文件与解析产物）、Redis（会话记忆缓存 + Redisson 分布式锁） |
| 其他 | 雪花 ID、`@Scheduled` 定时补偿、Spring 事件异步向量化、fastjson2、Lombok |
| 前端 | 纯静态 HTML/CSS/JS（上传、文档管理、对话、评测报告 4 个页面） |

### 2.2 Python 版

| 层 | 选型 | 说明 |
|---|---|---|
| 语言 / 环境 | Python 3.12，conda 环境 `rag`（已创建） | 不改动已有的 `ai` 环境 |
| Web 框架 | FastAPI + Uvicorn | `StreamingResponse` 实现 SSE；挂载原静态前端 |
| 配置 | pydantic-settings + `.env` | 替代 `application.yml` |
| 数据库 | PostgreSQL 16 + SQLAlchemy 2.0 + Alembic | JSONB、`FOR UPDATE SKIP LOCKED`、advisory lock 都对数据工程场景有用 |
| 检索引擎 | Elasticsearch 8 + IK 中文分词插件，`elasticsearch-py` | dense_vector 做 KNN，`text` 字段用 IK 分词做 BM25 |
| 对象存储 | MinIO，`minio` SDK | 与原版一致 |
| 大模型 / Embedding | `openai` SDK 调自有模型 API（OpenAI 兼容），不再使用阿里云百炼 | 对话 / 评测打分：`qwen3.8:27b`；改写 / 意图：`qwen3.5:9b`；向量化：`bge-m3`（1024 维）；图片描述：`qwen3-vl:32b`。均通过 `.env` 配置，可随时替换 |
| 重排 | 本地 `BAAI/bge-reranker-base`，`sentence-transformers` CrossEncoder 加载 | 复用 Hugging Face 全局缓存中的已有模型；与 Java 版的模型不同，基线以 Python 版自己跑出的为准 |
| 文档解析 | MinerU（保留独立服务）、pandas（XLSX/CSV）、python-docx 兜底 | MinerU 依赖重，单独部署避免污染主环境 |
| 后台任务 | 基于数据库状态的 worker（`SKIP LOCKED` 轮询）；Dagster 编排批量入库、回填与每日对账 | 用一个 worker 同时取代 Spring 事件和定时补偿 |
| 评测 | 自研指标 + pandas 导出；可选对照 RAGAS | 指标口径与原版保持一致 |
| 工程化 | pytest、ruff、docker-compose、structlog | |

> **Redis 是否保留**：阶段 1–3 不引入。分布式锁用 PostgreSQL 行锁 / advisory lock + 唯一约束实现；会话记忆直接读数据库（最近 N 条）。如果后续需要缓存或限流再加。

### 2.3 模型配置（`.env`）

| 配置项 | 值 | 用途 |
|---|---|---|
| `LLM_BASE_URL` / `LLM_API_KEY` | 自有 API 地址与密钥 | 所有远程模型共用 |
| `CHAT_MODEL` | `qwen3.8:27b` | 对话生成、评测打分、会话标题 |
| `LIGHT_MODEL` | `qwen3.5:9b` | 问题改写、意图识别（每次提问都调用，用小模型降低延迟） |
| `VISION_MODEL` | `qwen3-vl:32b` | 图片描述；留空则关闭该功能 |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMS` | `bge-m3` / `1024` | 向量化；维度必须与 ES mapping 一致 |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | 本地重排 |

- 图片描述以 base64（data URL）传图：PDF 解析出的图片在本地 MinIO，外部 API 无法访问其 URL；发送前先把长边缩到约 1024 像素，控制请求体大小。
- 对话模型与轻量模型的分工是初始假设，阶段 2 用评测结果验证后再定。

---

## 3. 总体架构

```
                ┌──────────────── 入库管道（数据工程核心） ────────────────┐
 上传 ──► [UPLOADED] ──► 解析转换 ──► [CONVERTED] ──► 切分 ──► [CHUNKED] ──► 向量化+索引 ──► [VECTOR_STORED]
   │          │            │ MinerU / pandas            │ 5 种策略            │ 批量 embed、ES bulk upsert
   │          ▼            ▼                            ▼                     ▼
   │       MinIO 原文件  MinIO 解析产物(md+图片)     PG knowledge_segment   ES know-engine-chunks
   │
   └─ 所有状态推进由 ingest worker 执行：轮询 PG → 加行锁 → 执行一步 → 推进状态（失败可重试）

                ┌──────────────────── 问答链路 ────────────────────┐
 用户问题 ─► 意图识别 ─┬─ 闲聊 ─► LLM 直接回答
                       └─ 知识问答 ─► 问题改写 ─► 向量 KNN ┐
                                                  BM25     ┴► 父块替换 ─► RRF 融合 ─► BGE 重排(+兜底) ─► 拼提示词 ─► LLM 流式生成
                                     每一步通过 SSE 推送 [PROGRESS]，引用通过 [REFERENCE] 推送并落库

                ┌──────────────────── 评测 ────────────────────┐
 JSONL 测试集 ─► 强制走 RAG 链路 ─► 采集召回 chunk 与回答 ─► 检索指标 + LLM 打分 ─► 落库 / 对比 / 导出 CSV
```

---

## 4. 目录结构

```
rag-know-engine/
├── pyproject.toml
├── docker-compose.yml          # postgres / elasticsearch(+IK) / minio
├── .env.example
├── alembic/                    # 表结构迁移
├── docs/design.md
├── prompts/                    # 从 Java 版 resources/prompts 原样拷贝
├── static/                     # 从 Java 版 resources/static 原样拷贝
├── data/
│   ├── testFile/               # 测试文档
│   └── eval/                   # 评测集 JSONL
├── src/know_engine/
│   ├── main.py                 # FastAPI 应用入口，挂载路由与静态文件
│   ├── config.py               # Settings
│   ├── db.py                   # engine / session
│   ├── models/                 # SQLAlchemy ORM（7 张表）
│   ├── schemas/                # pydantic 请求/响应模型
│   ├── clients/                # llm.py / embedding.py / es.py / minio.py / mineru.py / reranker.py
│   ├── ingest/
│   │   ├── service.py          # 上传、新版本、切换版本
│   │   ├── converters/         # markdown.py / pdf_docx.py(MinerU) / txt.py / table.py
│   │   ├── splitters/          # length.py / title.py(父子块) / regex.py / separator.py / table.py
│   │   ├── indexer.py          # 批量 embed + ES bulk upsert / 删除
│   │   ├── state.py            # 状态机与状态推进
│   │   └── worker.py           # 状态驱动 worker（替代事件 + 定时补偿）
│   ├── rag/
│   │   ├── intent.py
│   │   ├── rewrite.py
│   │   ├── retrieve.py         # knn_search / bm25_search / 父块替换
│   │   ├── fusion.py           # RRF
│   │   ├── rerank.py           # 重排 + 阈值过滤 + 兜底
│   │   └── pipeline.py         # 串联整条链路，产出 SSE 事件
│   ├── chat/                   # 会话、消息、记忆、SSE 路由
│   ├── eval/                   # loader / metrics / judge / runner / compare / export
│   └── api/                    # document.py / segment.py / chat.py / eval.py 路由
└── tests/
```

---

## 5. 模块设计与 Java 对照

### 5.1 入库管道 `ingest/`

| Python | 对应 Java | 说明 |
|---|---|---|
| `ingest/service.py` | `DocumentProcessServiceImpl`、`DocumentServiceImpl`、`DocumentVersionServiceImpl` | 上传、新版本、切换 / 激活 / 失效版本 |
| `ingest/converters/*` | `MarkdownProcessServiceImpl`、`MinerUProcessBaseServiceImpl`、`FileProcessServiceFactory` | 按扩展名分派 |
| `ingest/splitters/*` | `DocumentSplitterFactory`、`MarkdownHeaderParentTextSplitter`、`ExcelSplitter` | 5 种策略 + 表格逐行 |
| `ingest/indexer.py` | `VectorStoreServiceImpl`、`DocumentCleanupService` | 批量 embed、写入 / 删除 ES |
| `ingest/worker.py` | `DocumentEventListener`、`DocumentCompensationJob` | 合二为一 |
| `infra` 锁 | `DistributeLock` + `DistributeLockAspect`（Redisson） | 改为 PG 行锁 + 唯一约束 |

**上传**

1. 计算文件 SHA-256；同一文档下 `content_hash` 已存在则直接返回（幂等）。
2. 写 `knowledge_document`（UPLOADED）+ `knowledge_document_version`，原文件存 MinIO：`raw/{doc_id}/{version_id}/{filename}`。
3. 转换：
   - Markdown：直接读取；含图片时调用视觉模型生成描述，处理后的 md 写回 MinIO。
   - TXT：不转换，原文件即转换结果。
   - PDF / DOCX：调用 MinerU `/file_parse` 拿到 zip，解压后图片上传 MinIO，替换 md 中的图片链接（同时补图片描述）。
   - XLSX / CSV：不转换，切分时由 pandas 解析。
4. 状态推进到 CONVERTED。

**切分**（由用户手动触发，与原版一致）

| 策略 | 原实现 | Python 实现 | 单位 |
|---|---|---|---|
| LENGTH | LangChain4j `DocumentByWordSplitter` | 按 token 的滑窗切分（tiktoken 或模型分词器） | 词 / token |
| TITLE | 自定义 `MarkdownHeaderParentTextSplitter` | 自写：按 `titleLevel` 切出父块，父块过长再切子块 | 字符 |
| SMART | 同上，`titleLevel=6`，`overlap=chunkSize×10%` | 复用 TITLE | 字符 |
| REGEX | `DocumentByRegexSplitter` | `re.split` + 长度兜底 | 字符 |
| SEPARATOR | 同 REGEX（分隔符按正则解释） | 复用 REGEX，分隔符默认 `re.escape` | 字符 |
| 表格 | `ExcelSplitter`（仅第一个 sheet，首行表头，逐行键值对） | pandas：每行转 `列名: 值` 文本 | 行 |

父子块规则：父块完整写入 `knowledge_segment`，标记 `skip_embedding=1`，不进 ES；子块带 `parent_chunk_id` 元数据，进 ES。检索命中子块后，用父块全文替换。

切分结果按 `(document_id, version_id, chunk_order)` 唯一；重复切分前先删除同版本旧分块（幂等）。

**向量化与索引**

- 取 `status=STORED and skip_embedding=0` 的分块，每批 100 条（可配置）：调用 embedding → ES `bulk` upsert，**文档 `_id` 直接用 `chunk_id`**，重跑不会产生重复向量。
- 每批成功后更新分块状态为 VECTOR_STORED；全部完成后版本 → VECTOR_STORED，并失效同文档的旧版本。

**状态机**

```
UPLOADED → CONVERTING → CONVERTED → CHUNKED → VECTOR_STORED
                                       ▲              │
                                       └── 版本失效 ───┘（删除 ES 向量，分块回到 STORED）
```

状态推进统一用条件更新：`UPDATE ... SET status=:to WHERE id=:id AND status=:from`，影响行数为 0 说明已被其他进程处理。

**worker（取代 Spring 事件 + 定时补偿）**

```sql
SELECT * FROM knowledge_document_version
WHERE status = 'CHUNKED'
ORDER BY updated_at
LIMIT 10
FOR UPDATE SKIP LOCKED;
```

- 轮询待向量化的版本，加行锁后执行向量化。多个 worker 并发也不会重复处理同一个版本。
- 同一循环里处理「需要清理旧版本」的文档（对应原版 `retryFailedCleanups`）。
- 记录每次尝试的次数、错误信息；超过重试上限标记为 FAILED 并告警（原版没有失败终态）。

### 5.2 检索链路 `rag/`

| Python | 对应 Java |
|---|---|
| `intent.py` | `IntentRecognitionService` + `intent-recognition-new-prompt.txt`（输出 `reasoning / related / intent`） |
| `rewrite.py` | `KnowEngineQueryTransformer` + `query-rewrite-prompt.txt` |
| `retrieve.py` | `KnowEngineElasticsearchContentRetriever` |
| `fusion.py` | `KnowEngineReciprocalRankFuser` + `KnowEngineDefaultContent` |
| `rerank.py` | `KnowEngineReRankingContentAggregator` + `BgeScoringModel` |
| `pipeline.py` | `ChatService.executeRagPipeline` + `ProgressAware*` 装饰器 |

**流程与参数**

| 步骤 | 实现要点 | 初版参数（沿用 Java 初版的数值，仅作起点） | 调优后参数（参考原版第 4 轮） |
|---|---|---|---|
| 问题改写 | 结合最近会话历史，LLM 输出 1 条改写问题；异步回写 `chat_message.transform_content` | — | 提示词要求保留关键实体与状态词 |
| 向量检索 | 问题 embed 后做 ES KNN | `top_k=5`，`min_score` 待标定 | `top_k=10` |
| 关键词检索 | ES `match` 查询 `text` 字段 | `top_k=5` | `top_k=10` |
| 两路并发 | `asyncio.gather` | — | — |
| 父块替换 | 子块 → 父块全文，同一父块只保留一次 | — | — |
| RRF 融合 | `score += 1/(k+rank)`，**以 `chunk_id` 去重**，合并检索来源（VECTOR / KEYWORD / BOTH） | `k=60` | — |
| 重排 | BGE 批量打分，过滤低于阈值的片段，取前 N | `min_score=0.6`，`top_n=5` | `min_score=-2.5`（原始 logit） |
| 兜底 | 重排后为空时回退到 RRF 前 N 条（原版初版代码缺失） | — | 开启 |
| 注入 | 用 `rag-content-injector-prompt.txt` 拼接片段 | — | — |

> 重排分数口径：Java 版的 ONNX 模型输出原始 logit（可能为负），调优后阈值为 -2.5。Python 版用 CrossEncoder 加载 `bge-reranker-base`，默认经过 sigmoid 归一化到 0–1，**上表中的阈值都不能照搬**。阶段 2 先不设阈值跑一轮，观察正确片段与无关片段的分数分布，再标定阈值，并把标定过程写进评测记录。
>
> 向量检索阈值同理：ES 的 cosine 打分为 `(1 + cos) / 2`，Java 版的 `min_score=0.5` 只过滤掉余弦值小于 0 的结果，几乎不起作用。阶段 2 先不设阈值，观察 `bge-m3` 下正确片段与无关片段的分数分布后再定。

### 5.3 对话 `chat/`

- `POST /chat/send` 返回 `text/event-stream`，内部是一个 async generator：
  1. 新会话时创建会话，并异步生成标题（`title-summary-*.txt`）。
  2. 预先插入 USER 和 ASSISTANT 两条消息。
  3. 意图识别：`related=false` 走通用对话（`common-chat-system-prompt.txt`），否则走 RAG。
  4. 依次输出进度、引用、token；生成结束后回写完整回答和 `model_name`。
- 客户端断开时取消 LLM 流（`request.is_disconnected()`）。
- 会话记忆：从 `chat_message` 取最近 N 轮，不做 Redis 缓存。

**SSE 协议**（必须与原版一致，前端靠它解析）

| 事件 | 含义 |
|---|---|
| `[PROGRESS]:<文案>` | 阶段进度，如「正在优化您的问题...」「正在检索知识库内容...」 |
| `[REFERENCE]:<json>` | 引用片段列表 |
| `[DONE]` | 结束 |
| 其他 | 回答 token |

### 5.4 评测 `eval/`

- 测试集格式与原版相同：每行一个 JSON，字段 `id / question / groundTruthAnswer / groundTruthChunkIds / groundTruthDocumentIds`。
- 推理：固定用户 `rag-eval`，强制走 RAG（跳过意图识别），收集回答、召回的 chunk / document ID 和耗时。
- 指标（共 8 项）：

| 层 | 指标 | 计算方式 |
|---|---|---|
| 检索 | Hit@K、MRR、Recall@K | 有 chunk 真值用 chunk 级，否则退化为文档级 |
| 生成 | Faithfulness、Answer Relevancy、Answer Correctness | LLM 打分（`eval-*-prompt.txt`） |
| 上下文 | Context Precision、Context Recall | LLM 打分 |

- 配置：`top_k=5`、`concurrency=1`、`enable_llm_judge=true`，以及 BadCase 判定阈值（`score<0.6`、`mrr<0.3`、`recall<0.3`）。
- 功能：批量运行、报告、历史列表、两次运行对比（指标差 + 每个 case 的对错变化）、导出 CSV。
- 改进：评测集拆成**调参集 / 测试集**，只在调参集上调参，测试集只用来汇报最终数字，避免过拟合。

---

## 6. 数据模型

沿用 Java 版的 7 张表，迁移到 PostgreSQL：

| 表 | 用途 | 关键字段 |
|---|---|---|
| `knowledge_document` | 文档 | `doc_id`、`doc_title`、`status`、`current_version_id` |
| `knowledge_document_version` | 文档版本 | `version_id`、`doc_id`、`version`、`doc_url`、`converted_doc_url`、`content_hash`、`status` |
| `knowledge_segment` | 分块 | `chunk_id`、`text`、`metadata`、`document_id`、`document_version`、`chunk_order`、`embedding_id`、`status`、`skip_embedding` |
| `chat_conversation` | 会话 | `conversation_id`、`user_id`、`title` |
| `chat_message` | 消息 | `message_id`、`conversation_id`、`type`、`content`、`transform_content`、`rag_references` |
| `eval_run` | 评测批次 | `run_id`、`config_json`、`summary_json`、`status` |
| `eval_run_case` | 评测明细 | `run_id`、`case_id`、召回 ID、指标、`latency_ms` |

调整：

- JSON 字段改为 `JSONB`；`knowledge_segment.metadata` 由 `varchar(2048)` 改为 `JSONB`（原版可能截断）。
- 去掉雪花 ID：主键用 `bigint identity`，`chunk_id` 用 UUID。
- `knowledge_segment` 加唯一约束 `(document_version, chunk_order)`。
- 版本表增加 `attempts`、`last_error` 字段，供 worker 重试和排查。
- 保留 `deleted` 软删除和 `lock_version` 乐观锁字段。

---

## 7. Elasticsearch 索引设计

原版没有显式定义 mapping，由 LangChain4j 自动建索引，`text` 字段用默认的 standard 分词器，对中文是**按单字切分**，BM25 效果打折扣。Python 版显式定义：

```json
PUT know-engine-chunks-v1
{
  "mappings": {
    "properties": {
      "chunk_id":        { "type": "keyword" },
      "doc_id":          { "type": "long" },
      "version_id":      { "type": "long" },
      "parent_chunk_id": { "type": "keyword" },
      "file_name":       { "type": "keyword" },
      "text":            { "type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart" },
      "vector":          { "type": "dense_vector", "dims": 1024, "index": true, "similarity": "cosine" },
      "metadata":        { "type": "object", "enabled": false }
    }
  }
}
```

- 通过别名 `know-engine-chunks` 读写，物理索引带版本号，为阶段 4 的零停机重建做准备。
- 文档 `_id = chunk_id`，写入用 upsert，保证幂等。
- 两路检索都带过滤条件：只查当前生效版本（`version_id` 在生效集合中）。原版的全文检索没有过滤，版本切换期间可能召回旧版本内容。

---

## 8. API 清单（与 Java 版保持一致）

| 模块 | 方法 | 路径 |
|---|---|---|
| 文档 | POST | `/api/document/upload` |
| | POST | `/api/document/upload-version` |
| | GET | `/api/document/versions/{docId}` |
| | POST | `/api/document/switch-version` |
| | POST | `/api/document/deactivate-version` |
| | POST | `/api/document/activate-version` |
| | POST | `/api/document/split/{documentId}` |
| | GET | `/api/document/page` |
| | GET / POST / PUT | `/api/document/{id}`、`/api/document` |
| | DELETE | `/api/document/{id}`、`/api/document/batch` |
| 分块 | GET | `/api/segment/{id}`、`/api/segment/page-by-document`、`/api/segment/count-by-document` |
| | PUT / DELETE | `/api/segment`、`/api/segment/{id}` |
| 对话 | POST | `/chat/send`（SSE） |
| | GET | `/chat/conversation/list`、`/chat/message/list` |
| | DELETE | `/chat/conversation/delete` |
| 评测 | POST | `/eval/run`、`/eval/dataset/upload` |
| | GET | `/eval/report/{runId}`、`/eval/runs`、`/eval/compare`、`/eval/report/{runId}/export.csv` |

响应体沿用原版 `Result` 结构：`success / message / data / total`（`total` 用于分页）。

---

## 9. 原版已知问题与本版处理

阅读 Java 代码时发现的问题，Python 版一并修正。这些也是面试时「你做了什么改进」的素材。

| # | 问题 | 位置 | 处理 |
|---|---|---|---|
| 1 | 重排阈值 0.6 套在原始 logit 上，大量正确片段被过滤 | `ChatService.createContentAggregator` | 阈值按模型输出口径标定，并写入评测记录 |
| 2 | 重排全部过滤后上下文为空，没有兜底 | `KnowEngineReRankingContentAggregator` | 回退到 RRF 结果 |
| 3 | ES 没有中文分词，BM25 按单字匹配 | 索引自动创建 | IK 分词 + 显式 mapping |
| 4 | 全文检索不带过滤条件，可能召回旧版本 | `doFullTextQuery` | 两路检索都按生效版本过滤 |
| 5 | 向量写入 ES 发生在数据库事务内，事务回滚时 ES 不会回滚 | `activateVersion` | ES 写入幂等（`_id=chunk_id`），先写 ES 再推进状态，失败由 worker 重试 |
| 6 | 补偿任务没有重试上限和失败终态 | `DocumentCompensationJob` | `attempts` + FAILED 状态 |
| 7 | 分块元数据字段 `varchar(2048)`，可能被截断 | `knowledge_segment.metadata` | 改为 JSONB |
| 8 | 评测只有一份测试集，在同一批题上反复调参 | `eval-test.jsonl` | 拆分调参集 / 测试集 |

---

## 10. 数据工程增强（阶段 4）

| 能力 | 做法 | 简历上能讲的点 |
|---|---|---|
| 增量入库 | 文档级 hash 判断是否变化；分块级 hash（文本 + 切分参数）只重算变化的分块 | 增量更新、embedding 成本节省比例 |
| 编排 | Dagster：`fetch → convert → split → embed → index` 作为资产（asset），支持分区回填 | DAG、重试、backfill |
| 数据质量 | 规则：空块、超长块、乱码比例、重复块比例、表格空值率；结果写入质量表，不合格的版本不进索引 | 质量门禁、拦截率 |
| 零停机重建索引 | 新建 `know-engine-chunks-v2` → 全量灌入 → 校验数量与抽样检索 → 切换别名 → 删除旧索引 | 蓝绿发布 |
| 规模化 | 引入公开语料（如 PyTorch / Python 官方文档、arXiv 论文），做到上万分块；记录吞吐 | 吞吐（文档/分钟）、耗时分布 |
| 可观测 | 每个阶段记录耗时、成功 / 失败数，汇总成报表 | 运行指标 |
| 数据血缘 | 文档 → 版本 → 分块 → 向量的映射可查询 | 血缘追踪 |

---

## 11. 分阶段计划与验收标准

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| 0 环境 | conda 环境、docker-compose（PG / ES+IK / MinIO）、MinerU 服务、Alembic 建表 | 服务都能启动，`alembic upgrade head` 成功 |
| 1 入库管道 | 上传、转换、5 种切分 + 表格、worker 向量化、版本切换 | 6 个测试文件全部到 VECTOR_STORED；重复上传、重复切分、worker 重跑都不产生重复数据 |
| 2 检索 + 评测 | 意图、改写、双路检索、父块替换、RRF、重排 + 兜底；评测运行、报告、对比、导出 | 用初版参数跑出 Python 版自己的基线并记录（embedding、重排、大模型都已更换，不与 Java 版的 Hit@K 0.75 直接比较）；之后每一轮调优都与该基线对比 |
| 3 对话 | `/chat/*` 接口、SSE、会话记忆、挂载原前端 | 原 `chat.html`、`upload.html`、`document.html`、`eval-report.html` 不改动即可使用 |
| 4 数据工程增强 | 第 10 节 | 每项都有可量化的数字 |

---

## 12. 阶段 1 实现记录（2026-10-25）

与上文设计的差异和补充：

| 项 | 实际做法 |
|---|---|
| MinerU | 4.0.12，独立 conda 环境 `mineru`；接口为 `POST /v1/file_parse`（文件以 octet-stream 上传，参数走 query：`response_format=zip`、`tier=standard`），不再是 2.x 的 `/file_parse` 表单接口 |
| LENGTH 单位 | 字符。LangChain4j 的 `DocumentByWordSplitter` 未配置 tokenizer 时同样按字符计，原教程「按词」的说法不准确 |
| 文档级锁 | 所有修改操作先锁 `knowledge_document` 行：API 用 `NOWAIT`（冲突即返回 409），worker 用 `SKIP LOCKED`；上传去重用 `pg_advisory_xact_lock(内容哈希)` 防并发重复 |
| 版本切换 | `activate-version` 等同于切换到该版本；不允许让当前版本失效（否则 worker 会立即重新向量化它）；切换到 FAILED 版本会重置重试次数 |
| 转换产物路径 | `converted/{doc_id}/{version_id}/`，原版按文档标题存放，新版本会覆盖旧版本文件 |
| 表格空行 | 全部值为空的数据行不再生成分块 |

验证：6 个测试文件全部到 VECTOR_STORED（PG 可索引分块 97 条与 ES 一致）；重复上传、重复切分、4 路并发切分、版本上传 / 切换 / 回切、分段编辑与乐观锁、删除清理、worker 重试 / 退避 / FAILED / 恢复均已测试通过。

---

## 13. 阶段 2 实现记录：检索链路与评测调优（2026-10-25）

### 13.1 评测集真值重映射

原评测集的 `groundTruthChunkIds` 是 Java 那次运行的分块 ID，换了切分实现后全部失效。Java 的 SQL 导出里带有这些分块的原文，因此用 `scripts/relabel_eval.py` 自动重映射：在同一文档中，与旧分块有重叠（字符 3-gram 包含度 ≥ 0.3）的新分块里，选「答案覆盖度 + 0.5 × 与旧分块相似度」最高的一个。

- 两轮迭代修正了规则：只按相似度选时，表格行（表头相同）和边界移动的 TXT 分块会选错；改成「答案优先、相似度兜底」后，抽查剩余的低覆盖度样本均为答案改写（如「5,000」对「五千」），标注正确。
- 100 条按 case id 哈希固定拆分：调参集 64 条（`eval-tune.jsonl`），留出集 36 条（`eval-holdout.jsonl`）。**所有调参只看调参集**，留出集只在基线和最终配置上各跑一次。
- 文档重新切分后重跑此脚本即可，过程可复现。

### 13.2 调优过程（调参集 64 条，每轮只改一处）

| 轮次 | 改动 | Hit@5 | MRR | Recall@5 | 正确性 | 上下文精确率 | 结论 |
|---|---|---|---|---|---|---|---|
| R0 | 基线：Java 初版参数（每路 5 条、重排阈值 0.6、无兜底、原改写提示词） | 0.906 | 0.844 | 0.891 | 0.900 | 0.765 | — |
| R1 | 重排阈值 0.6 → 0.1（按分数分布标定） | 0.969 | 0.872 | 0.961 | 0.947 | 0.606 | 采纳 |
| R2 | 重排对子块打分，再替换为父块全文 | 0.969 | 0.878 | 0.961 | 0.947 | 0.600 | 采纳 |
| R3 | 每路召回 5 → 10 | 0.984 | 0.878 | 0.977 | 0.961 | 0.590 | 采纳 |
| R4 | 改写提示词 v2（原样保留实体、编号、状态词，只删不扩） | 0.969 | 0.891 | 0.961 | 0.966 | 0.635 | 采纳（见下） |
| R5 | IK 领域词典 + 零停机重建索引 | 0.969 | 0.891 | 0.961 | 0.963 | 0.638 | 指标无变化，作为分词修正保留 |
| R6 | 打开重排兜底（最终配置） | 0.984 | 0.907 | 0.977 | 0.975 | 0.624 | 采纳 |

**留出集（36 条，从未参与调参）**：

| | Hit@5 | MRR | Recall@5 | 忠实度 | 正确性 | 上下文精确率 |
|---|---|---|---|---|---|---|
| R0 基线 | 0.889 | 0.803 | 0.889 | 1.000 | 0.903 | 0.803 |
| R6 最终 | **0.972** | **0.830** | **0.958** | 1.000 | **0.947** | 0.692 |

**每轮的依据**（工具：`know_engine.eval.analyze` 逐 case 给出真值分块在向量 / 关键词 / RRF / 重排各阶段的名次）：

- **R0 → R1**：6 个未命中里 4 个是「已召回、被重排阈值卡掉」（真值得分 0.12～0.34）。统计调参集上真值与非真值候选的重排分数：真值中位数 0.98，但 p10 为 0.70、p5 为 0.28；0.6 会误杀约 10% 的真值。阈值 0.1 保留约 95% 以上的真值，仍能过滤约 1/4 的噪声（非真值 p25 = 0.079）。代价是上下文精确率下降。
- **向量阈值**：同一统计中向量分数真值 p5 = 0.781、非真值中位数 = 0.791，两者几乎重叠，向量阈值无法区分，保持关闭。
- **R2**：交叉编码器最多读 512 token，长父块被截断、得分被稀释；改为对命中的子块打分，最后再换成父块全文。本数据集只有 1 个文档有父块，收益小但方向正确。
- **R3**：md-010（P2 首次响应目标）的真值被大量同样含「P2」的事件记录挤出前 5，扩大候选池后命中。
- **R4**：md-010 在新提示词下关键词召回名次 9 → 2，但重排排到第 10（0.933，低于若干 P2 事件行），属于重排的边界样本；docx-015 不再被改写成「……配置方法」。MRR 与上下文精确率提升，Hit 少 1 条，作为已知取舍记录。
- **R5**：词典修正了「智造园区 → 智 / 造园 / 区」「蓝鹭应急协调组 → 单字」「复盘中 → 复 / 盘 / 中」等切分，但调参集问题在召回 10 条 + 重排后并不依赖这些词，端到端指标不变。
- **R6**：xlsx-002（哪台资产处于维护状态）所有候选重排分都低于 0.1，兜底返回 RRF 前 5，命中。

### 13.3 稳健性：加入 2 万条干扰分块

在索引中导入 552 篇 Python 中文文档（20,178 个分块，约为原数据的 200 倍）后用最终配置重跑：调参集 Hit@5 0.984 / MRR 0.907 不变；留出集 Hit@5 0.972 不变，MRR 0.830 → 0.816。检索阶段 p95 延迟约 30 ms → 55～60 ms。

### 13.4 过程中发现并处理的问题

| 问题 | 处理 |
|---|---|
| 思考型模型（qwen3.x）一次一句话的改写耗时 17～52 秒 | 所有对话调用传 `reasoning_effort="none"`（`LLM_REASONING_EFFORT`），改写降到约 1.4 秒 |
| 模型 API 并发 4 时返回 429 | 客户端指数退避重试（尊重 Retry-After），评测默认并发 2 |
| 超时 300 秒 × 重试 8 次，一次卡住的调用能拖 40 分钟 | 单次超时 60 秒；意图识别与改写在请求路径上，改为 8 秒超时且不重试，失败分别降级为「默认检索」「使用原问题」 |
| 评测口径 bug：同一父块的多个子块合并后，只保留第一个子块 ID，真值子块被算作未召回 | 每条引用记录 `coveredChunkIds`，指标按覆盖的全部子块计算；修正后重跑基线 |
| 首次提问要等重排模型加载约 6 秒 | 服务启动时后台预加载 |

## 14. 阶段 3 实现记录：对话

- `/chat/send` 以 SSE 返回，事件顺序与 Java 版一致：`[PROGRESS]:正在识别您的意图...` → 改写 / 检索 / 排序筛选进度（检索在后台线程执行，进度实时推送）→ `[REFERENCE]:<json>` → 回答 token → `[DONE]:<conversationId>`。多行内容按 Spring 的格式拆成多条 `data:`。
- 意图为闲聊时跳过检索，直接用通用对话提示词。
- 会话记忆取数据库最近 10 条消息，同时用于改写（指代消解）和生成。验证：追问「它包含哪些生产区？」被改写为「曙光智造园区（代号 Aurora-7）的生产区域构成」。
- 新会话先用问题前 20 字作标题，后台用轻量模型生成正式标题。
- 客户端断开时，已生成的部分回答仍会写回数据库。
- 引用链接指向 `/api/document/{id}/file`，直接下载当前版本原文件。
- 原 `chat.html` 不改动即可使用。

## 15. 阶段 4 实现记录：数据工程增强

| 能力 | 实现 | 验证结果 |
|---|---|---|
| 增量向量化 | 分块保存 `text_hash = sha256(模型名 + 文本)`，同时写入 ES；新版本入库时，相同文本直接复用已有向量，不调用 embedding 接口 | 改动一行的新版本：复用 6 个、新算 1 个 |
| 数据质量门禁 | 切分后检查空块、短块、超长块、重复率、乱码率；乱码 > 2% / 重复 > 50% / 无可向量化分块直接拒绝（版本保持 CONVERTED，可换参数重切），其余为警告。报告写入阶段日志和文档 `extension`（文档页可查看） | 单元测试覆盖拒绝与警告规则 |
| 可观测 | `ingest_stage_log` 记录每次 convert / split / quality / embed 的耗时、条数、参数和错误；`GET /api/ingest/stats` 汇总 p50 / p95、吞吐、失败数、向量复用率；`GET /api/document/versions/{id}/lineage` 给出某个版本的完整处理链路 | 血缘：convert → split → quality → embed |
| PG / ES 对账 | `know_engine.ingest.reconcile`：按文档版本比较 PG 应有分块数与 ES 实际数，找出缺失、过期版本、孤儿数据；`--repair` 把缺失版本放回队列由 worker 幂等重建 | 手动删除 ES 中 2 个分块 → 对账发现 → 修复后一致；导入 2 万分块后 558 个版本全部一致 |
| IK 领域词典热更新 | 词典由 `know_engine.ingest.dictionary` 从表格的名称列（资产、团队、区域）加人工审核的种子词生成，发布到 MinIO（只有 `ik/` 前缀可匿名读）；IK 以 `remote_ext_dict` 每分钟拉取，按 ETag 热加载 | 发布后约 5 秒生效 |
| 零停机重建索引 | `know_engine.ingest.reindex`：新建 `vN+1` → `_reindex` 复制（不重新向量化）→ 文档数一致 + 抽样 20 条 KNN / BM25 自检 → 原子切换别名 → 删除旧索引。索引写入方持有共享 advisory lock，重建时持有排他锁，保证切换前不会有写入落在旧索引上丢失 | v1 → v2 → v3，各约 0.4～0.8 秒；重建期间写入方拿不到锁 |
| 批量导入 | `know_engine.ingest.bulk`：按内容哈希跳过已入库文件（中断后可直接重跑），线程池上传 / 转换 / 切分，多个 worker 并行向量化（`SKIP LOCKED` 各领各的版本） | 552 篇 → 20,178 分块，601 秒，55 篇/分钟、33.6 分块/秒，0 失败；上传 + 切分仅 6.2 秒，瓶颈在向量化 |
| Dagster 编排 | `know_engine/orchestration/definitions.py`：sensor 发现新文件并登记为动态分区；分区资产 `ingested_document`（上传 + 切分，幂等）→ `indexed_document`（等待 worker 完成，记录复用数）；`quality_gate` 资产检查；每日 03:00 对账任务 | sensor 发现 558 个文件；已入库文件物化为空操作；第二次 tick 无新请求；对账任务成功 |

**本阶段踩到的坑**：ES 容器里以单文件 bind mount 挂载 `IKAnalyzer.cfg.xml` 时，在 OrbStack 下 `open()` 会永久阻塞，IK 初始化卡住，分片无法恢复（集群 red），容器也无法被 kill，只能重启 OrbStack。改为在镜像构建时 `COPY` 进去后恢复正常。词典也从「由 API 提供」改为「由 MinIO 提供」，ES 启动不再依赖 API。

---

## 16. 决策记录

| 事项 | 结论 |
|---|---|
| 关系库 | PostgreSQL（JSONB、`SKIP LOCKED`、advisory lock 都用上了） |
| 分布式锁 | PG 行锁 + advisory lock，未引入 Redis |
| 编排工具 | Dagster（可选依赖 `.[orchestration]`），日常入库仍走状态驱动 worker |
| 重排模型 | 本地 `bge-reranker-base`（sigmoid 分数，阈值按分布标定为 0.1） |
| 大模型 / Embedding | 自有 API：改写 / 意图 `qwen3.5:9b`，生成 / 评测 `qwen3.8:27b`，向量 `bge-m3`（1024 维），关闭思考模式 |
| 语料 | 原 6 个测试文件用于评测；552 篇 Python 中文文档用于规模与稳健性测试 |
| 前端 | 复用原静态页面，四个页面均可用 |
