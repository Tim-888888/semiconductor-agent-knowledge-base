# API 与数据契约

全部 API 使用 `/api/v1` 前缀，所有 JSON 时间字段为带时区的 ISO 8601 格式。请求体中
提交的 `actor_scope` 不作为授权依据；网关只使用 JWT 中的 Scope。

## 鉴权

一期用 `Authorization: Bearer <JWT>`。`POST /api/v1/auth/demo-token` 仅供 Demo 生成预置
工程师/知识管理员令牌；生产环境应由 OIDC 适配器替换。评估数据集、运行列表、详情、启动和重试
均要求 `knowledge_admin` 或 `admin` 角色。

## 对话与连续会话

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `POST` | `/threads` | 创建 `thread_id`；服务端绑定 JWT Scope |
| `GET` | `/threads` | 仅列出当前用户的线程 |
| `GET` | `/threads/{thread_id}` | 恢复线程摘要、消息与引用 |
| `POST` | `/threads/{thread_id}/messages` | 先做约束检查，再追问或检索并写入 Checkpoint |

`thread_id` 是连续对话、LangGraph Checkpoint 和审计 Trace 的共同关联键。Demo 使用
LangGraph `InMemorySaver`；真实模式使用官方 `MongoDBSaver` 写入 `checkpoints`、
`checkpoint_writes`。消息接口返回 `clarification_required=true` 时，图已通过 `interrupt()`
持久化暂停；下一条同线程消息使用 `Command(resume=...)` 恢复。最多追问两轮，仍缺少关键字段时
返回 `insufficient_information`，不调用检索或制造工具。

除澄清中断/恢复外，T9-4.3.1 已实现通用历史装配，T9-4.3.2 已实现精确历史直答、受控意图和
“无需检索时跳过 RAG”。预定义组合执行、路由专用答案校验和前端路由标识仍属于 T9-4.3.3，不能把
当前能力描述为全部业务闭环已经完成。

T9-4.3.3 目标契约已于 2026-08-14 补充，但尚未实现：历史回顾和历史回答变换的新请求将迁移为
`chat_direct`，由服务端固定 `context_message_ids` 后交给 LLM 自然表达；`history_direct` 只保留为旧
请求台账和 Trace 的兼容枚举。该迁移不改变当前线上 T9-4.3.2 的已验收事实。

### T9-4.1 流式消息契约

`POST /threads/{thread_id}/messages/stream` 的线协议已在 T9-4.1 冻结，路由本身由 T9-4.2 实现。
请求体为 `StreamMessageRequest`，包含 `content` 和客户端生成、重试时复用的 `request_id`；原非流式
接口继续保留。

SSE 每条事件采用以下格式，其中 `data` 是单行 UTF-8 JSON，包含与 SSE 字段一致的完整事件信封：

```text
id: sse_<id>
event: stage
data: {"event":"stage","event_id":"sse_<id>","request_id":"req_<id>","thread_id":"thread_<id>","sequence":2,"emitted_at":"<UTC>","data":{"stage":"analyzing_request","message":"正在分析问题"}}

```

首事件必须是 `accepted`，序号从 1 连续递增，末事件必须是 `completed` 或 `error`。合法事件为
`accepted`、`stage`、`evidence`、`answer_delta`、`heartbeat`、`completed` 和 `error`。`completed`
携带与非流式接口相同的 `SendMessageResponse`；事件模型和稳定枚举位于
`semikb.contracts.streaming`。

鉴权、Scope、请求格式和线程存在性在发送 SSE 响应头前验证，失败沿用普通 `401/403/404/422` JSON。
响应头发出后的失败只能返回脱敏 `error` 事件。稳定错误码包括请求冲突/处理中、Provider 超时/限流/
不可用、答案校验失败、取消和内部错误；不得返回异常堆栈、内部 Prompt、思维过程或密钥。

取消与断连语义固定为：`AbortController` 或连接中断取消当前下游生成；已完整持久化的结果不回滚，
未完成的助手 Delta 不写入 `agent_threads`。客户端重新读取线程和请求台账进行对账，不能把本地半截文本
当作最终答案。

### T9-4.3.2 会话理解与按需路由契约

T9-4.3.1 上下文基础和 T9-4.3.2 意图/路由契约已经实现。现有消息 URL 和 SSE 事件信封保持兼容，
内部使用
`ConversationUnderstanding`：

```json
{
  "interaction_mode": "conversation",
  "primary_intent": "conversation",
  "task_items": [
    {
      "task_id": "task_1",
      "primary_intent": "conversation",
      "target_type": "previous_user_message",
      "action": "recall",
      "depends_on": [],
      "execution_policy": "execute"
    }
  ],
  "affect": {"sentiment": "neutral", "urgency": "normal", "complaint_signal": false},
  "slot_operations": [],
  "explicit_slots": {},
  "inherited_slots": {},
  "missing_slots": [],
  "context_message_ids": ["msg_xxx"],
  "standalone_query": "",
  "cancel_scope": null,
  "suggested_route": "chat_direct",
  "confidence": 0.99
}
```

上例是 T9-4.3.3c 已部署契约。同一语义的新请求把 `suggested_route` 和最终 `route_decision` 写为
`chat_direct`，继续保留相同的 `context_message_ids` 作为历史约束。

`interaction_mode` 固定为 `task`、`conversation`、`feedback`、`control`、
`clarification_answer` 和 `mixed`。一级意图固定为 `conversation`、`knowledge_query`、
`investigation`、`data_query`、`action_request` 和 `content_task`。二级语义由最多 3 个
`task_item` 的 `target_type + action` 表达，避免创建不可维护的意图笛卡尔积；任务依赖只允许映射到
预定义执行组合，不接受任意 DAG。

槽位必须区分本轮显式值和历史继承值，使用 `set/inherit/correct/clear` 操作并记录来源
`message_id`。纠正 Tool 等上游槽位时返回需要失效的旧 Chamber、Recipe 和证据引用；取消操作必须
携带当前生成、当前任务、指定任务或待补充澄清等明确范围。情绪字段只影响表达和审计，不直接决定路由。

复杂输入仍只允许一次结构化 LLM 理解。使用 Provider 支持的 JSON Schema/工具调用约束和
`temperature=0`，再经 Pydantic 校验；最多一次结构化修复，失败后确定性降级。服务端不能把 Schema
约束视为权限、过滤表达式或工具参数校验的替代品。

兼容路由枚举仍为 `history_direct`、`chat_direct`、`reuse_evidence`、`internal_rag`、`tool_only`、
`rag_and_tool`、`rag_and_web`、`clarify` 和 `refuse`。LLM 只返回 `suggested_route`；服务端
`RoutePolicy` 根据权限、风险、槽位、上下文和证据有效性作最终决定。历史回顾和普通聊天不调用 T5；
查询最新版本或实时状态不得直接复用旧证据。

T9-4.3.3c 已完成：新历史请求统一产生 `chat_direct`，并以 `context_message_ids` 和稳定 reason code
区分普通聊天、历史回顾和历史变换。服务端先验证消息属于当前用户与线程，再把选中的精确消息和有界
最近历史交给生成器；生成器不得自行选择历史目标。模型失败时使用确定性精确回退。旧
`history_direct` 不立即删除，避免旧 MongoDB 记录、评估快照和 Trace 反序列化失败。

一期不新增独立向量意图路由、不增加第二次意图 LLM 调用、不引入自由任务 DAG。T9-4.3.3b 只允许
复用现有 Embedding 网关离线检索独立意图示例，和固定 Few-shot 做影子对照；动态示例未通过冻结
`semikb-intent-v3` 门禁并再次取得用户确认前，不进入线上请求链路。置信度阈值按半导体意图
评估集和路由风险校准，不写死通用 `0.8/0.6/0.4`；语义低置信度进入澄清，Provider 熔断只由超时、
429、5xx 或无效响应触发。

T9-4.3.3a 已实现：Prompt 固定加载指定 `catalog_version` 中全部 `status=active` 的意图卡，不新增
候选卡字段、预筛 API 或前置 Embedding 调用。`agent_message_requests.understanding_audit` 保存
`intent_catalog_version`、`intent_catalog_hash`、活动卡数、实际注入卡数、输入 Token 来源、调用来源、
模型和理解时延；这些白名单字段不包含完整 Prompt、卡片正文、用户正文或密钥。动态 Few-shot 只追加
经过隔离的案例，不改变 LLM 可见的全量合法意图集合。

`SendMessageResponse` 和 SSE `completed.data.result` 已以可选、向后兼容字段返回
`interaction_mode`、`route_decision`、`route_confidence`、`task_items`、`task_decisions` 和
`retrieval_skipped_reason`。T9-4.3.3d 继续增加可选 `task_results`，逐项保存 `task_id`、终态
`completed/clarify/refused/deferred/failed`、实际路由、稳定原因码、用户可见状态文案和按路由隔离的
内部证据、工具事实、外部证据 ID。`task_decisions` 是执行前策略决定，`task_results` 是执行后终态，
两者不得混用。`agent_message_requests` 保存同一组路由审计信息，以及上下文消息 ID、
独立查询、槽位操作、继承值、失效引用、取消范围、有限情绪枚举和脱敏理解审计。T9-4.3.3c 新增可选
`direct_reply_audit`，仅保存 `reply_kind`、Provider/模型、生成模式、时延、验证单元数、上下文消息数、
告警码和 Token 用量；不保存 Prompt、完整意图卡、原始历史正文或密钥。MongoDB 终态更新和失败重试
都会显式写入或清空该字段，避免旧尝试的审计信息污染新尝试。`stage` 只报告实际执行的上下文、
检索或工具步骤，不输出隐藏推理。SSE 新增 `task_status`，只发送 `queued/running` 和上述终态；
`completed.data.result.task_results`、请求台账及助手消息 `presentation.task_results` 必须一致。前端按消息
展示服务端决定的最终路由和逐任务状态，不从文案猜测路由。详细规则见
`docs/T9-4.3通用会话记忆与按需路由设计.md`。

## 长期记忆

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `POST` | `/memories` | 显式确认并写入当前用户记忆 |
| `GET` | `/memories` | 仅列出当前用户已批准且未过期的记忆 |
| `DELETE` | `/memories/{memory_id}` | 删除当前用户的一条记忆 |

工程师只能保存 `preference`；`case_summary` 与 `stable_rule` 需要 `knowledge_admin` 或 `admin`。
记忆不自动从对话生成，也不能作为回答事实来源；图只读取已批准偏好用于展示方式。

`POST /retrieval/search` 可选接收 `constraints`：`fab`、`product`、`process_layer`、
`tool_id`、`chamber`、`recipe_id`、`recipe_version`、`as_of`、`use_hyde`。这些字段经过
Pydantic 校验后才生成 Milvus 表达式，客户端不能提交原始过滤表达式。响应 Trace 记录每条候选的
Dense/Sparse/HyDE 分数、路由排名、RRF、Rerank、选择原因、组件版本和安全降级告警。

## 知识与运维接口

### T9-4.4 文档解析与受控入库内部契约

独立包 `semikb_ingest` 已实现内部契约 `semikb-ingest-v1`。受控入库层只允许通过
`IngestDispatcher` 获得 `ParsedDocument`，其中包含 `ChunkDraft`、`ImageAssetDraft`、
`TableAssetDraft`、Parse Provenance、警告和指标；该包不接收审批、权限、Fab/Tool/Recipe、对象路径、
Embedding 或索引字段。扩展名、MIME、Magic Bytes 和 OOXML 容器成员必须一致，未知格式或未注册
Parser/Provider 明确失败；只有 PDF 路由声明 MinerU。

T9-4.4.4 已把该契约接入现有 API、Worker 和受控发布服务。Markdown/TXT/HTML、DOCX、XLSX/CSV、
PPTX 分别使用专用本地适配器，PDF 只走显式 MinerU Provider，JPG/JPEG/PNG 只走显式 Qwen VLM
Provider；未知格式、类型冲突或 Provider 未配置均使用稳定错误码失败，不存在“所有二进制统一交给
MinerU”的隐藏回退。此结论是本地代码与自动测试结果，不替代 T9-4.4.5 的 ECS/浏览器真实闭环。

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `POST` | `/retrieval/search` | 受权限/版本过滤的混合检索与可解释 Trace |
| `POST` | `/ingestion-jobs` | 创建 Markdown/模拟文档入库任务 |
| `POST` | `/ingestion-jobs/upload` | 上传 multipart 原件；`metadata` 为 JSON 字符串 |
| `GET` | `/ingestion-jobs` | 读取任务及阶段事件 |
| `GET` | `/ingestion-jobs/{id}` | 读取单个任务、计数、失败阶段和完整事件 |
| `POST` | `/ingestion-jobs/{id}/retry` | 对可重放失败任务发起幂等重试 |
| `GET` | `/assets/{id}/access` | 重鉴权后返回短时图片访问描述 |
| `GET` | `/retrieval-traces` | 仅查看当前用户 Trace；管理员可全局查看 |
| `GET` | `/evaluation-datasets` | 读取已冻结的数据集版本、Hash 和 Case 快照 |
| `POST` | `/evaluation-runs` | 创建 `queued` 运行并投递 Celery；支持四种固定检索档位和 baseline |
| `GET` | `/evaluation-runs` | 读取 MongoDB 持久化的运行历史、指标和状态 |
| `GET` | `/evaluation-runs/{id}` | 读取配置/组件版本、Case、Trace 和基线差异 |
| `GET` | `/evaluation-runs/{id}/cases/{case_id}/trace` | 校验 Trace 属于该运行后供评估管理员下钻 |
| `POST` | `/evaluation-runs/{id}/retry` | 重新投递失败的评估运行 |

## 关键模型

- `DocumentRevision`：版本、审批状态、有效期、权限范围、MinIO 原件/解析件引用，以及 Parser、Provider、
  上游 Commit、警告和解析指标。
- `Chunk`：正文/表格/图文检索单元，保留图片/表格稳定引用和解析来源；MongoDB 保存正文，Milvus 只
  保存向量及过滤字段。
- `ImageAsset`：真实图片对象引用、图注、OCR、检测摘要、源位置、解析来源及 Case 关联。
- `TableAsset`：表格 Markdown/HTML、表头、行列数、源位置和 MinIO `table.json` 引用。
- `IngestionJob` + `IngestionEvent`：任务状态及不可抵赖阶段时间线。
- `RetrievalTrace`：查询、过滤器、Dense/Sparse/RRF/Rerank 分数、动态截断原因与最终证据；
  包含 `actor_user_id`，避免跨用户泄露检索轨迹。
- `EvidenceLedgerEntry`：统一记录内部 Chunk、模拟工具和外部资料的来源等级、版本、分数与引用 ID。
- `AgentAnswer`：分离已知事实、待验证假设、不确定项、下一步和置信度；事实必须引用账本 ID。
- `MemoryRecord`：显式批准的用户偏好、Case 摘要或稳定规则，带来源、Scope、置信度与有效期。
- `EvaluationDataset`：不可变黄金集快照；同一 `dataset_version` 不允许出现不同 Hash。
- `EvaluationRun`：冻结数据集 Hash、`dense/hybrid/reranked/full` 配置、组件版本、任务所有权、
  Recall@5/MRR/nDCG@5/负例准确率/图片召回/延迟、Case 结果和 baseline 差异。

上传接口暂支持最大 200 MiB。上传格式由 `semikb_ingest` 的扩展名、MIME、Magic Bytes 和 OOXML
成员联合识别，并交给对应专用适配器；只有 PDF 使用 MinerU Precision API，图片使用 Qwen VLM。
`DEMO_MODE=true` 时在请求内同步处理，便于无外部服务演示；
`DEMO_MODE=false` 时 API 只保存原件和可重放元数据，然后向 Celery 投递 Job，由 Worker 执行
解析、分块、Embedding、暂存和发布。重复请求命中相同幂等键时复用原 Job，不重复发布。

如果 Redis/Celery 投递失败，API 返回 `503`，同时把 Job 置为 `failed` 并记录
`QUEUE_SUBMISSION_FAILED`，文档保持未发布，管理员可在队列恢复后调用重试接口。API 响应和
事件只保存安全错误摘要，不返回密钥、连接串或第三方原始异常。

生产模式的评估接口始终异步返回 `202`。Celery task id 是运行认领所有权；同一任务重投可以
恢复，其他任务不能并发认领同一运行。每个 Case 的 `trace_id` 对应 MongoDB `retrieval_traces`，
`thread_id` 使用 `evaluation:<run_id>:<case_id>`，便于区分评估流量和用户流量。
## T9-4.3.1 通用会话上下文契约

`ChatMessage` 新增可选 `turn_seq`。新消息由线程级写租约分配严格递增序号；迁移后的历史消息也具有
稳定序号。`agent_message_requests` 记录 `user_turn_seq` 与 `assistant_turn_seq`，幂等重试复用原用户消息
序号，不重复追加消息。

`ThreadRecord` 新增：

- `summary_upto_message_id`：摘要覆盖到的最后一条精确消息；
- `context_version`：活动上下文契约版本，当前为 `1`；
- `active_context`：带 `source_message_id`、依赖关系与有效标记的槽位和证据引用；
- `next_turn_seq` / `last_turn_seq`：持久化消息顺序游标；
- `active_request_id` / `active_request_started_at`：同一线程的短时写租约。

`ContextAssembler` 在 ACL 验证后装配当前线程：最近 12 轮精确消息、较旧历史的有界抽取式摘要、
活动上下文和显式批准偏好。当前用户消息不重复放入 prior history。LangGraph 只允许有
`source_message_id` 且 `valid=true` 的槽位用于标识继承；摘要不作为受控事实来源。

同一线程已有请求租约时，流式和兼容非流式接口均返回 HTTP `409`，客户端应等待前一请求完成或取消，
不得自动并发重试。不同线程不共享精确消息或活动上下文。

## T9-4.3.3c 消息展示契约

`ChatMessage.presentation` 为可选的向后兼容字段，包含：

- `mode`：`bubble` 或 `structured_card`；
- `route_decision`：产生该消息的最终服务端路由；
- `status`、`answer`、`trace_id`、`verification_warnings`：仅供该消息对应的结构化卡片使用；
- `image_asset_ids`：当前结构化回答最终授权证据中的有序图片 ID；气泡路由固定为空列表。

服务端固定把 `reuse_evidence/internal_rag/tool_only/rag_and_tool/rag_and_web` 映射为
`structured_card`，把 `chat_direct/clarify/refuse` 及旧 `history_direct` 映射为 `bubble`。模型不能输出
或覆盖该决策；前端只按消息字段渲染，不能使用全局最新结果控制历史消息。旧消息缺少展示字段时，服务端
按 `request_id` 从已完成的 `agent_message_requests.result_payload` 补齐，MongoDB 原消息不做破坏性迁移。

T9-4.3.4 起，SSE `completed`、幂等重放和 `GET thread` 必须返回相同的 `image_asset_ids`。前端自动选择
第一项，并通过 `/api/assets/{image_id}/access` 重新校验当前用户、文档版本、有效期和 Scope 后取得短时
链接；不得从 Retrieval Candidate、自行拼接的 MinIO 路径或上一轮全局状态展示图片。多图手动选择只在
当前结果内保留，新结果无图时必须清空旧预览。
