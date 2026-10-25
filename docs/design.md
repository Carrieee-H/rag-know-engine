# rag-know-engine 设计文档

> 版本：v1.0　日期：2026-10-25

## 1. 概述

rag-know-engine 是一个知识库问答系统：文档经解析、切分、向量化后写入 Elasticsearch，问答时做混合检索与重排，由大模型生成带引用的回答；配套一套带真值的评测体系，所有调优都以评测数据为依据。

非目标：复杂前端、多租户权限、Agent 能力。

### 设计原则

1. **链路显式化**：检索链路的每一步（改写、召回、融合、重排、注入）都是普通函数，可单独测试、单独计时，参数集中在一个配置对象里。
2. **状态驱动 + 幂等**：入库的每个阶段以数据库状态为准，任何一步都可以安全重跑；ES 写入以分块 ID 为文档 ID。
3. **用数据定参数**：阈值、召回数量、提示词的每次改动都要在评测集上验证，调参集与留出集分离。

---

## 2. 技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| 语言 / Web | Python 3.12、FastAPI + Uvicorn | `StreamingResponse` 实现 SSE |
| 配置 | pydantic-settings + `.env` | |
| 数据库 | PostgreSQL 16 + SQLAlchemy 2.0 + Alembic | JSONB、`FOR UPDATE SKIP LOCKED`、advisory lock |
| 检索引擎 | Elasticsearch 8.15 + IK 中文分词 | `dense_vector` 做 KNN，`text` 字段用 IK 分词做 BM25 |
| 对象存储 | MinIO | 原文件、解析产物、IK 词典 |
| 大模型 / 向量 | OpenAI 兼容接口（`openai` SDK） | 生成 / 评测打分、改写 / 意图、`bge-m3` 向量、视觉模型 |
| 重排 | 本地 `BAAI/bge-reranker-base`（`sentence-transformers` CrossEncoder） | 输出经 sigmoid，范围 0～1 |
| 文档解析 | MinerU 4（独立服务）、pandas | MinerU 依赖重，单独部署 |
| 后台任务 | 状态驱动 worker；Dagster（可选） | |
| 工程化 | pytest、ruff、Docker Compose、structlog | |

### 2.1 模型配置

| 配置项 | 默认值 | 用途 |
|---|---|---|
| `CHAT_MODEL` | `qwen3.8:27b` | 生成回答、评测打分 |
| `LIGHT_MODEL` | `qwen3.5:9b` | 问题改写、意图识别、会话标题 |
| `VISION_MODEL` | 空 | 图片描述；留空则关闭 |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMS` | `bge-m3` / `1024` | 维度必须与 ES mapping 一致 |
| `LLM_REASONING_EFFORT` | `none` | 关闭思考模式（见 11 节） |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | 本地重排 |

图片以 base64（data URL）发送给视觉模型：解析出的图片存在本地 MinIO，模型服务访问不到其地址；发送前把长边缩到约 1024 像素。

---

## 3. 总体架构

```
                ┌──────────────────────── 入库流水线 ────────────────────────┐
 上传 ──► [UPLOADED] ──► 解析转换 ──► [CONVERTED] ──► 切分 + 质量门禁 ──► [CHUNKED] ──► 向量化 + 索引 ──► [VECTOR_STORED]
             │            │ MinerU / pandas              │ 5 种策略              │ worker：批量 embed、ES bulk
             ▼            ▼                              ▼                       ▼
          MinIO 原文件  MinIO 解析产物(md + 图片)    PG knowledge_segment    ES know-engine-chunks（别名 → vN）

                ┌──────────────────────── 问答链路 ────────────────────────┐
 用户问题 ─► 意图识别 ─┬─ 闲聊 ─► LLM 直接回答
                       └─ 知识问答 ─► 问题改写 ─► 向量 KNN ┐
                                                  BM25(IK) ┴► RRF 融合 ─► BGE 重排(+兜底) ─► 父块替换 ─► 拼提示词 ─► LLM 流式生成
                                     每一步通过 SSE 推送 [PROGRESS]，引用通过 [REFERENCE] 推送并落库

                ┌──────────────────────── 评测 ────────────────────────┐
 JSONL 评测集 ─► 强制走 RAG 链路 ─► 采集召回分块与回答 ─► 检索指标 + LLM 打分 ─► 落库 / 对比 / 导出 / BadCase 排查
```

### 目录结构

```
rag-know-engine/
├── pyproject.toml
├── docker-compose.yml          # postgres / elasticsearch(+IK) / minio
├── docker/elasticsearch/       # ES 镜像：IK 插件 + 远程词典配置
├── alembic/                    # 表结构迁移
├── docs/design.md
├── prompts/                    # 改写、意图、生成、评测打分等提示词
├── static/                     # 上传、文档管理、对话、评测报告页面
├── scripts/relabel_eval.py     # 评测集真值重映射
├── data/                       # 示例文档、评测集、语料（不提交）
├── src/know_engine/
│   ├── main.py                 # FastAPI 入口
│   ├── config.py / db.py / constants.py / prompts.py
│   ├── models/                 # ORM：文档、版本、分块、会话、消息、评测、入库阶段日志
│   ├── clients/                # es.py / storage.py(MinIO) / llm.py / mineru.py
│   ├── ingest/                 # service / converters / splitters / quality / indexer / worker / state /
│   │                           # observe / reconcile / reindex / dictionary / bulk
│   ├── rag/                    # query(意图、改写) / retrieve(召回、融合、重排) / pipeline / llm
│   ├── chat/                   # 流式对话
│   ├── eval/                   # runner / metrics / report / analyze / cli / dataset
│   ├── api/                    # document / chat / eval / serializers
│   └── orchestration/          # Dagster 定义
└── tests/
```

---

## 4. 数据模型

| 表 | 用途 | 关键字段 |
|---|---|---|
| `knowledge_document` | 文档 | `doc_id`、`doc_title`、`status`、`current_version_id`、`extension`（切分参数、质量报告） |
| `knowledge_document_version` | 文档版本 | `version_id`、`doc_id`、`version`、`doc_url`、`converted_doc_url`、`content_hash`、`status`、`attempts`、`last_error` |
| `knowledge_segment` | 分块 | `chunk_id`、`text`、`text_hash`、`metadata`、`document_id`、`document_version`、`chunk_order`、`status`、`skip_embedding` |
| `chat_conversation` / `chat_message` | 会话与消息 | `conversation_id`、`title`；`type`、`content`、`transform_content`（改写结果）、`rag_references` |
| `eval_run` / `eval_run_case` | 评测批次与明细 | `config_json`、`summary_json`；召回 ID、各项分数、阶段耗时 |
| `ingest_stage_log` | 入库阶段日志 | `stage`、`status`、`duration_ms`、`items`、`detail` |

- JSON 类字段统一用 JSONB；主键用 `bigint identity`，`chunk_id` 用 UUID。
- 版本、分块对文档有外键并级联删除；`(document_version, chunk_order)` 唯一。
- 所有业务表带 `lock_version`（SQLAlchemy 乐观锁）和 `deleted` 软删除字段。

### Elasticsearch 索引

```json
{
  "settings": { "number_of_shards": 1, "number_of_replicas": 0 },
  "mappings": {
    "properties": {
      "chunk_id":        { "type": "keyword" },
      "text_hash":       { "type": "keyword" },
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

- 通过别名 `know-engine-chunks` 读写，物理索引带版本号（`-v1`、`-v2` …），支持零停机重建。
- 文档 `_id = chunk_id`，重复写入是覆盖而不是新增。
- 两路检索都只查各文档的当前生效版本（`version_id` 过滤）。

---

## 5. 入库流水线

### 5.1 上传与转换

1. 计算文件 SHA-256；内容已存在则拒绝（409）。同一内容的并发上传用 `pg_advisory_xact_lock(内容哈希)` 串行化，避免两个请求同时通过去重检查。
2. 写入文档与版本记录（UPLOADED），原文件存 MinIO：`raw/{doc_id}/{version_id}/{filename}`。
3. 转换（CONVERTING → CONVERTED）：
   - **PDF / DOCX**：调用 MinerU `POST /v1/file_parse`（文件以 octet-stream 上传，`response_format=zip`、`tier=standard`），得到 Markdown 和图片；图片存 MinIO 并由视觉模型生成描述，替换 Markdown 中的图片标签。
   - **Markdown**：补全图片描述后存 MinIO。
   - **TXT / XLSX / CSV**：不转换，切分时直接读原文件。
   - 转换产物路径为 `converted/{doc_id}/{version_id}/`，新版本不会覆盖旧版本的文件。
4. 转换失败时版本退回 UPLOADED，错误写入 `last_error`。

### 5.2 切分

| 策略 | 实现 | 单位 |
|---|---|---|
| LENGTH | 按空白拆词后贪心拼接到 `chunkSize`，超长片段按字符窗口切，块间保留 `overlap` | 字符 |
| TITLE | 按 `titleLevel` 及以上的 Markdown 标题切节（忽略代码块中的 `#`）；只有标题的节并入下一节；超长节保留为父块并切出子块 | 字符 |
| SMART | 所有标题层级、不合并，`overlap` 固定为 `chunkSize` 的 10% | 字符 |
| REGEX / SEPARATOR | 按正则切开后贪心拼接到 `chunkSize`（分隔符也按正则解释） | 字符 |
| 表格 | 只读第一个 sheet，首行为表头，每行写成 `列名：值; 列名：值`；全空行跳过 | 行 |

父子块：父块完整存 PG（`skip_embedding = 1`，不进 ES），子块带 `parentChunkId` 进 ES。检索命中子块后替换为父块全文，同一父块只保留一次。

切分时对当前版本加行锁：已切分过的版本直接返回分块数（幂等），并发的第二个请求得到 409。

### 5.3 数据质量门禁

切分结果在入库前检查：

| 规则 | 处理 |
|---|---|
| 没有可向量化的分块 | 拒绝 |
| 乱码字符（替换符、控制字符、私有区字符）占比 > 2% | 拒绝 |
| 重复分块占比 > 50% | 拒绝 |
| 空块、少于 10 字的短块占比 > 20%、超过 `chunkSize` 的块 | 警告 |

被拒绝的版本保持 CONVERTED，可以换参数重新切分。报告写入阶段日志和文档的 `extension`，文档页可以直接查看。

### 5.4 向量化与索引（worker）

```sql
SELECT * FROM knowledge_document_version v
JOIN knowledge_document d ON d.current_version_id = v.version_id
WHERE v.status = 'CHUNKED' AND v.attempts < 5
  AND v.updated_at <= now() - interval '5 seconds' * (2 ^ v.attempts - 1)   -- 指数退避
ORDER BY v.updated_at LIMIT 1
FOR UPDATE OF v SKIP LOCKED;
```

- worker 轮询数据库领取待向量化的版本；多个 worker 可以并行，`SKIP LOCKED` 保证不会重复处理同一版本。进程崩溃时版本仍是 CHUNKED，下一次轮询会自动接手，不需要单独的补偿任务。
- 每批 64 个分块：先按 `text_hash` 查 ES 中已有的向量（未变化的分块直接复用），其余调用 embedding 接口，再 bulk 写入 ES，最后更新 PG 状态。ES 写入幂等，失败重试只会覆盖。
- 写完后核对 ES 中该版本的文档数与 PG 一致，再把版本置为 VECTOR_STORED，并下线同文档的其他版本（删除其 ES 数据，分块退回 STORED）。
- 失败按 0、5、15、35、75 秒退避重试，5 次后标记 FAILED；切换到 FAILED 版本会重置重试次数。

### 5.5 状态机与并发控制

```
UPLOADED → CONVERTING → CONVERTED → CHUNKED → VECTOR_STORED
                                       ▲    │         │
                                       │    └→ FAILED │
                                       └── 版本下线 ──┘（删除 ES 数据，分块回到 STORED）
```

- 文档状态始终等于其当前版本的状态。
- 所有修改操作先锁 `knowledge_document` 行：API 用 `NOWAIT`（冲突立即返回 409），worker 用 `SKIP LOCKED`（跳过，下一轮再试）。没有人在持有锁时等待另一把锁，因此不会死锁。
- 新版本处理期间，旧版本仍可被检索；新版本向量化完成后旧版本才下线。不允许让当前版本失效（否则 worker 会立刻把它重新向量化）。

---

## 6. 检索与生成

### 6.1 链路与参数

| 步骤 | 实现 | 参数（最终配置） |
|---|---|---|
| 意图识别 | 轻量模型输出 `{reasoning, related, intent}`；闲聊不检索；8 秒超时，失败默认检索 | — |
| 问题改写 | 结合最近 10 条会话历史，输出一条检索用查询；8 秒超时，失败使用原问题 | 提示词 v2（保留实体、编号、状态词，只删不扩） |
| 向量检索 | 问题向量化后 ES KNN | `top_k = 10`，不设阈值 |
| 关键词检索 | ES `match`（IK 分词）| `top_k = 10` |
| 两路并发 | 线程池并行 | — |
| RRF 融合 | `score += 1 / (k + rank)`，按 `chunk_id` 合并，来源标记 VECTOR / KEYWORD / BOTH | `k = 60` |
| 重排 | BGE 交叉编码器对子块打分，过滤低于阈值的，取前 N | `min_score = 0.1`，`top_n = 5` |
| 兜底 | 重排后为空时回退到 RRF 前 N 条 | 开启 |
| 父块替换 | 子块替换为父块全文，同一父块合并（记录覆盖的全部子块 ID） | — |
| 生成 | 系统提示词 + 历史 + 注入检索片段的问题，流式输出 | — |

所有参数在 `RagConfig` 中，评测时可逐项覆盖。

### 6.2 分数口径

- **重排分数**：CrossEncoder 输出经 sigmoid，范围 0～1。阈值不能凭经验设，按评测集上真值与非真值的分数分布标定（见 9 节）。
- **向量分数**：ES 的 cosine 打分为 `(1 + cos) / 2`，0.5 只对应余弦值为 0。实测真值与非真值的分数几乎重叠，因此不设向量阈值。

---

## 7. 对话

- `POST /chat/send` 返回 `text/event-stream`：
  1. 新会话先用问题前 20 字作标题，后台再用轻量模型生成正式标题。
  2. 预先写入 USER 和 ASSISTANT 两条消息。
  3. 意图识别后分流：闲聊走通用对话，知识问答走 RAG。检索在后台线程执行，进度实时推送。
  4. 生成结束（或客户端断开）时，把已生成的回答、模型名、改写结果、引用写回数据库。
- 会话记忆取数据库最近 10 条消息，同时用于改写（指代消解）和生成。例如追问「它包含哪些生产区？」被改写为「曙光智造园区（代号 Aurora-7）的生产区域构成」。
- 引用链接指向 `/api/document/{id}/file`，直接下载当前版本原文件。

**SSE 协议**（多行内容拆成多条 `data:`）

| 事件 | 含义 |
|---|---|
| `[PROGRESS]:<文案>` | 阶段进度：识别意图、优化问题、检索、排序筛选、生成回答 |
| `[REFERENCE]:<json>` | 引用片段列表（文档、分块、相似度、重排分、来源） |
| `[DONE]:<conversationId>` | 结束 |
| 其他 | 回答 token |

---

## 8. 评测体系

### 8.1 数据集与真值

- 每行一个 JSON：`id / question / groundTruthAnswer / groundTruthChunkIds / groundTruthDocumentIds`。
- 真值分块 ID 依赖切分结果，文档重新切分后即失效。`scripts/relabel_eval.py` 根据旧分块的原文（SQL 导出）自动重映射：同一文档中，与旧分块有重叠（字符 3-gram 包含度 ≥ 0.3）的新分块里，选「答案覆盖度 + 0.5 × 与旧分块相似度」最高的一个。
  - 规则经过两轮修正：只按相似度选时，表格行（表头相同）和边界移动的文本块会选错；改为「答案优先、相似度兜底」后，抽查剩余的低覆盖度样本均为答案改写（如「5,000」对「五千」），标注正确。
- 100 条按 case id 哈希固定拆分：**调参集 64 条**（`eval-tune.jsonl`），**留出集 36 条**（`eval-holdout.jsonl`）。所有调参只看调参集，留出集只在基线和最终配置上各跑一次。

### 8.2 指标

| 层 | 指标 | 计算方式 |
|---|---|---|
| 检索 | Hit@K、MRR、Recall@K | 每条引用按其覆盖的全部子块计；没有分块真值时退化为文档级 |
| 生成 | 忠实度、相关性、答案正确性 | LLM 打分（0～1） |
| 上下文 | 上下文精确率、上下文召回率 | LLM 打分（0～1） |

推理时强制走 RAG（跳过意图识别），记录每个阶段的耗时（改写、向量化、检索、重排、生成）。

### 8.3 工具

- `eval.cli`：命令行评测，`--set` 覆盖任意 `RagConfig` 字段；`--compare` 对比两次评测（指标差、新命中 / 新丢失的用例、答案正确性变化 ≥ 0.1 的用例）。
- `eval.analyze`：BadCase 链路排查，给出真值分块在向量 / 关键词 / RRF / 重排各阶段的名次，并判断丢在哪一步；`--distribution` 输出真值与非真值的重排、向量分数分布。
- `/eval/*` 接口与评测报告页面：运行、报告、历史、对比、导出 CSV。

---

## 9. 调优过程与结果

### 9.1 调参集（64 条，每轮只改一处）

| 轮次 | 改动 | Hit@5 | MRR | Recall@5 | 正确性 | 上下文精确率 | 结论 |
|---|---|---|---|---|---|---|---|
| R0 | 基线：初始参数（每路 5 条、重排阈值 0.6、无兜底、初版改写提示词、对父块打分） | 0.906 | 0.844 | 0.891 | 0.900 | 0.765 | — |
| R1 | 重排阈值 0.6 → 0.1（按分数分布标定） | 0.969 | 0.872 | 0.961 | 0.947 | 0.606 | 采纳 |
| R2 | 重排对子块打分，再替换为父块全文 | 0.969 | 0.878 | 0.961 | 0.947 | 0.600 | 采纳 |
| R3 | 每路召回 5 → 10 | 0.984 | 0.878 | 0.977 | 0.961 | 0.590 | 采纳 |
| R4 | 改写提示词 v2（原样保留实体、编号、状态词，只删不扩） | 0.969 | 0.891 | 0.961 | 0.966 | 0.635 | 采纳（见下） |
| R5 | IK 领域词典 + 零停机重建索引 | 0.969 | 0.891 | 0.961 | 0.963 | 0.638 | 指标无变化，作为分词修正保留 |
| R6 | 打开重排兜底（最终配置） | 0.984 | 0.907 | 0.977 | 0.975 | 0.624 | 采纳 |

### 9.2 留出集（36 条，从未参与调参）

| | Hit@5 | MRR | Recall@5 | 忠实度 | 正确性 | 上下文精确率 |
|---|---|---|---|---|---|---|
| R0 基线 | 0.889 | 0.803 | 0.889 | 1.000 | 0.903 | 0.803 |
| R6 最终 | **0.972** | **0.830** | **0.958** | 1.000 | **0.947** | 0.692 |

### 9.3 每轮的依据

- **R0 → R1**：6 个未命中里 4 个是「已召回、被重排阈值卡掉」（真值得分 0.12～0.34）。统计调参集上真值与非真值候选的重排分数：真值中位数 0.98，但 p10 为 0.70、p5 为 0.28，阈值 0.6 会误杀约 10% 的真值。阈值 0.1 保留约 95% 以上的真值，仍能过滤约 1/4 的噪声（非真值 p25 = 0.079）。代价是上下文精确率下降。
- **向量阈值**：同一统计中向量分数真值 p5 = 0.781、非真值中位数 = 0.791，两者几乎重叠，向量阈值无法区分，保持关闭。
- **R2**：交叉编码器最多读 512 token，长父块被截断、得分被稀释；改为对命中的子块打分，最后再换成父块全文。本数据集只有 1 个文档有父块，收益小但方向正确。
- **R3**：md-010（P2 首次响应目标）的真值被大量同样含「P2」的事件记录挤出前 5，扩大候选池后命中。
- **R4**：md-010 在新提示词下关键词召回名次 9 → 2，但重排排到第 10（0.933，低于若干 P2 事件行），属于重排的边界样本；docx-015 不再被改写成「……配置方法」。MRR 与上下文精确率提升，Hit 少 1 条，作为已知取舍记录。
- **R5**：词典修正了「智造园区 → 智 / 造园 / 区」「蓝鹭应急协调组 → 单字」「复盘中 → 复 / 盘 / 中」等切分，但调参集问题在召回 10 条 + 重排后并不依赖这些词，端到端指标不变。
- **R6**：xlsx-002（哪台资产处于维护状态）所有候选重排分都低于 0.1，兜底返回 RRF 前 5，命中。

### 9.4 稳健性：加入 2 万条干扰分块

在索引中导入 552 篇 Python 中文文档（20,178 个分块，约为原数据的 200 倍）后用最终配置重跑：调参集 Hit@5 0.984 / MRR 0.907 不变；留出集 Hit@5 0.972 不变，MRR 0.830 → 0.816。检索阶段 p95 延迟约 30 ms → 55～60 ms。

---

## 10. 运维能力

| 能力 | 实现 | 验证结果 |
|---|---|---|
| 增量向量化 | 分块保存 `text_hash = sha256(模型名 + 文本)`，同时写入 ES；新版本入库时，相同文本直接复用已有向量 | 改动一行的新版本：复用 6 个、新算 1 个 |
| 可观测 | `ingest_stage_log` 记录每次 convert / split / quality / embed 的耗时、条数、参数和错误；`GET /api/ingest/stats` 汇总 p50 / p95、吞吐、失败数、向量复用率；`GET /api/document/versions/{id}/lineage` 给出某个版本的完整处理过程 | 一个版本的记录：convert → split → quality → embed |
| PG / ES 对账 | `ingest.reconcile`：按文档版本比较 PG 应有分块数与 ES 实际数，找出缺失、过期版本、孤儿数据；`--repair` 把缺失版本放回队列由 worker 幂等重建 | 手动删除 ES 中 2 个分块 → 对账发现 → 修复后一致；导入 2 万分块后 558 个版本全部一致 |
| IK 领域词典热更新 | `ingest.dictionary` 从表格的名称列（资产、团队、区域）加人工审核的种子词生成词典，发布到 MinIO（只有 `ik/` 前缀可匿名读）；IK 以 `remote_ext_dict` 每分钟拉取，按 ETag 热加载 | 发布后约 5 秒生效 |
| 零停机重建索引 | `ingest.reindex`：新建 `vN+1` → `_reindex` 复制（不重新向量化）→ 文档数一致 + 抽样 20 条 KNN / BM25 自检 → 原子切换别名 → 删除旧索引。写入方持有共享 advisory lock，重建时持有排他锁，保证切换前不会有写入落在旧索引上丢失 | v1 → v2 → v3，各约 0.4～0.8 秒；重建期间写入方拿不到锁 |
| 批量导入 | `ingest.bulk`：按内容哈希跳过已入库文件（中断后可直接重跑），线程池上传 / 转换 / 切分，多个 worker 并行向量化 | 552 篇 → 20,178 分块，601 秒，55 篇/分钟、33.6 分块/秒，0 失败；上传 + 切分仅 6.2 秒，瓶颈在向量化 |
| Dagster 编排 | sensor 发现新文件并登记为动态分区；分区资产 `ingested_document`（上传 + 切分，幂等）→ `indexed_document`（等待 worker 完成，记录复用数）；`quality_gate` 资产检查；每日 03:00 对账任务 | sensor 发现 558 个文件；已入库文件物化为空操作；第二次 tick 无新请求；对账任务成功 |

---

## 11. 工程问题与处理

| 问题 | 处理 |
|---|---|
| 思考型模型（qwen3.x）一次一句话的改写耗时 17～52 秒 | 所有对话调用传 `reasoning_effort="none"`，改写降到约 1.4 秒 |
| 模型 API 并发 4 时返回 429 | 客户端指数退避重试（尊重 Retry-After），评测默认并发 2 |
| 超时 300 秒 × 重试 8 次，一次卡住的调用能拖 40 分钟 | 单次超时 60 秒；意图识别与改写在请求路径上，改为 8 秒超时且不重试，失败分别降级为「默认检索」「使用原问题」 |
| 评测口径：同一父块的多个子块合并后只保留第一个子块 ID，真值子块被算作未召回 | 每条引用记录 `coveredChunkIds`，指标按覆盖的全部子块计算；修正后重跑基线 |
| 首次提问要等重排模型加载约 6 秒 | 服务启动时后台预加载 |
| 多个请求线程共用一个重排模型 | 推理加锁 |
| ES 容器以单文件 bind mount 挂载 IK 配置时，在 OrbStack 下 `open()` 永久阻塞，IK 初始化卡住、分片无法恢复 | 配置在镜像构建时 `COPY` 进去；词典由 MinIO 提供，ES 启动不依赖 API |
| 单节点 ES 的副本分片永远无法分配，集群一直是 yellow | 索引设置 `number_of_replicas: 0` |

---

## 12. API 清单

| 模块 | 方法 | 路径 |
|---|---|---|
| 文档 | POST | `/api/document/upload`、`/api/document/upload-version` |
| | GET | `/api/document/versions/{docId}`、`/api/document/page`、`/api/document/{id}`、`/api/document/{id}/file` |
| | POST | `/api/document/switch-version`、`/api/document/activate-version`、`/api/document/deactivate-version`、`/api/document/split/{documentId}` |
| | POST / PUT / DELETE | `/api/document`、`/api/document/{id}`、`/api/document/batch` |
| 分块 | GET | `/api/segment/{id}`、`/api/segment/page-by-document`、`/api/segment/count-by-document` |
| | PUT / DELETE | `/api/segment`、`/api/segment/{id}` |
| 入库 | GET | `/api/ingest/stats`、`/api/document/versions/{versionId}/lineage` |
| 对话 | POST | `/chat/send`（SSE） |
| | GET / DELETE | `/chat/conversation/list`、`/chat/message/list`、`/chat/conversation/delete` |
| 评测 | POST | `/eval/run`、`/eval/dataset/upload` |
| | GET | `/eval/report/{runId}`、`/eval/runs`、`/eval/compare`、`/eval/report/{runId}/export.csv` |

业务冲突（重复上传、状态不允许、并发修改、质量不达标等）返回 409，响应体为纯文本错误信息；评测与对话列表接口使用 `{success, message, data, total}` 结构。

---

## 13. 决策记录

| 事项 | 结论 |
|---|---|
| 关系库 | PostgreSQL（JSONB、`SKIP LOCKED`、advisory lock） |
| 锁 | PG 行锁 + advisory lock，不引入 Redis |
| 后台任务 | 状态驱动 worker；批量与定时任务由 Dagster 编排（可选依赖 `.[orchestration]`） |
| 重排模型 | 本地 `bge-reranker-base`，sigmoid 分数，阈值按分布标定为 0.1 |
| 大模型 / 向量 | 改写 / 意图 `qwen3.5:9b`，生成 / 评测 `qwen3.8:27b`，向量 `bge-m3`（1024 维），关闭思考模式 |
| 语料 | 6 份示例文档用于评测；552 篇 Python 中文文档用于规模与稳健性测试 |
