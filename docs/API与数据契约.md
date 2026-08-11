# API 与数据契约

全部 API 使用 `/api/v1` 前缀，所有 JSON 时间字段为带时区的 ISO 8601 格式。请求体中
提交的 `actor_scope` 不作为授权依据；网关只使用 JWT 中的 Scope。

## 鉴权

一期用 `Authorization: Bearer <JWT>`。`POST /api/v1/auth/demo-token` 仅供 Demo 生成预置
工程师/知识管理员令牌；生产环境应由 OIDC 适配器替换。写入入库任务和启动评估均要求
`knowledge_admin` 或 `admin` 角色。

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
| `POST` | `/evaluation-runs` | 在冻结黄金集上运行离线评估 |

## 关键模型

- `DocumentRevision`：版本、审批状态、有效期、权限范围、MinIO 原件引用。
- `Chunk`：正文/表格/图文检索单元，MongoDB 保存正文，Milvus 只保存向量及过滤字段。
- `ImageAsset`：真实图片对象引用、图注、OCR、检测摘要及 Case 关联。
- `IngestionJob` + `IngestionEvent`：任务状态及不可抵赖阶段时间线。
- `RetrievalTrace`：查询、过滤器、Dense/Sparse/RRF/Rerank 分数、动态截断原因与最终证据；
  包含 `actor_user_id`，避免跨用户泄露检索轨迹。
- `EvidenceLedgerEntry`：统一记录内部 Chunk、模拟工具和外部资料的来源等级、版本、分数与引用 ID。
- `AgentAnswer`：分离已知事实、待验证假设、不确定项、下一步和置信度；事实必须引用账本 ID。
- `MemoryRecord`：显式批准的用户偏好、Case 摘要或稳定规则，带来源、Scope、置信度与有效期。

上传接口暂支持最大 200 MiB。`.md`/`.txt` 直接归一化；其他受 MinerU 支持的二进制格式交给
MinerU Precision API。`DEMO_MODE=true` 时在请求内同步处理，便于无外部服务演示；
`DEMO_MODE=false` 时 API 只保存原件和可重放元数据，然后向 Celery 投递 Job，由 Worker 执行
解析、分块、Embedding、暂存和发布。重复请求命中相同幂等键时复用原 Job，不重复发布。

如果 Redis/Celery 投递失败，API 返回 `503`，同时把 Job 置为 `failed` 并记录
`QUEUE_SUBMISSION_FAILED`，文档保持未发布，管理员可在队列恢复后调用重试接口。API 响应和
事件只保存安全错误摘要，不返回密钥、连接串或第三方原始异常。
