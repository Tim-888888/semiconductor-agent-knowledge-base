# Qwen 原生 Sparse 迁移验收

## 结论

`qwen3.7-text-embedding` 在请求参数指定 `output_type=dense&sparse` 时可同时返回 Dense 与
Sparse。项目已停止使用过渡期 `lexical-hash-v1` 生产路径，活动索引从 v3 受控迁移到
`semikb_chunks_v4`。API、Worker 和 Web 仍不加载 Torch、GPU 或任何本地模型权重。

本次是 T9 的部署前置迁移，不代表 T9 已完成。T9 仍需阿里云容器部署、实际镜像兼容、网络出口、
容量、备份恢复、可观测性和端到端复现验收。

## Provider 契约

项目通过 DashScope 原生 Endpoint 发送批量文本、`dimension=1024` 和
`output_type=dense&sparse`。客户端执行以下防御性校验：

- 返回数量与输入数量一致，`text_index` 连续且无重复；
- Dense 为 1024 维有限值，非零范数，并在客户端做 L2 归一化；
- Sparse 非空，索引为非负整数、值为有限正数，且同一向量无重复索引；
- Provider 异常只返回脱敏错误，不记录 API Key 或原始响应正文。

`EMBEDDING_OUTPUT_TYPE=dense` 仅保留为 v3 紧急回滚兼容模式；v4 必须使用
`dense&sparse` 和 `qwen3.7-text-embedding-sparse-v1`。

## 受控迁移

1. Dry-run 确认活动源为 v3、目标 v4 尚不存在，范围为 3 个 Document、8 个 Chunk。
2. `--build` 创建 v4 的 Dense HNSW 与 Sparse Inverted Index，重编码并强一致读回 8 行；活动 Alias 仍指向 v3。
3. 在 `semikb_chunks_v4` 物理 Collection 上执行 T5/T7 影子评估。T5 Reranked/Full Recall@5 与无证据准确率均为 1.0；T7 Full Run `eval_6ee9f0272ebb4cf980f9735dfee8ed6c` 的质量指标全部为 1.0。
4. `--publish` 仅发布已记录的候选，切换 Alias 并同步 3 个 Document、8 个 Chunk 的 `index_version=v4`。
5. v3 Collection 和发布元数据保留为直接回滚投影，v2 继续保留历史投影；迁移未删除旧 Collection。

发布命令被拆成 Build 与 Publish，避免“构建完成即切流”。目标候选必须先通过物理 Collection
影子评估，发布失败时恢复源 Alias；普通检索继续只读 `semikb_chunks_active`。

## 发布后回归

| 范围 | 验收证据 | 结果 |
| --- | --- | --- |
| T4 新文档 | Job `ing_8b5542426cd247af979ad37e765760e8` | Worker 发布 1 个 v4 Chunk，Sparse 485 个非零项；事件记录实际 Qwen/Sparse 版本；随后精确清理 10 条 MongoDB、1 条 Milvus、2 个 MinIO 对象 |
| T5 检索 | `t5-live-v1` Reranked/Full | Recall@5=1.0，无证据准确率=1.0，图片 Case 命中 |
| T6 Agent | Thread `thread_34f34958078d4deea86208c34af3ab27`，Trace `trace_9a38ed7832d040519c84a1da5b63217e` | 14 个 Checkpoint、5 个 Chunk、1 张图片、3 个只读工具、4 条审计；Luna 主模型完成且引用闭合 |
| T7 评估 | Full Run `eval_8d3ed2f15ad046deafef8dd4f25d7e69` | Recall@5=1.0、MRR=0.9、nDCG@5=0.9302、无证据准确率=1.0、图片 Recall@5=1.0、通过率=1.0 |
| T8 浏览器 | `docs/evidence/qwen-sparse-v4/` | 追问、连续调查、真实图片、Trace、任务中心、离线评估和 390 x 844 移动端均完成真实点击与截图 |

T6 发布后首次重复调用出现一次未选择内部证据，随后相同查询直接检索和完整 Agent 复跑均正常。
这说明在线 Reranker 分数或服务响应仍可能存在单次波动；当前证据只能确认稳定复跑通过，不能把一次
结果包装为确定性 SLA。T9 应增加重复运行、超时、限流和远程 Provider 波动门禁。

## 下游缺陷修复

- 生产 Repository 可显式搜索活动 Alias，也可在迁移影子验收中搜索指定物理 Collection。
- PyMilvus Alias 响应归一化，发布记录会移除源版本残留 Alias 标记。
- Agent 兼容 LLM 返回字符串或结构化对象形式的 `next_actions`，不再把对象显示成字典文本。
- Trace 详情长组件键可换行，不与值重叠；移动端页面无横向溢出。
- 入库阶段事件从实际 Encoder 读取 Dense/Sparse 模型版本，不再硬编码 BGE 或“model-free Sparse”。
- API 单元测试显式隔离根目录生产 `.env` 并清理 Settings/Container 缓存，避免测试误投递真实 Worker；首次发现的临时 Document、Chunk、Job、Trace、Evaluation、MinIO 对象和 Redis Result 已按精确 ID 清理，恢复为 3 个 Document、8 个 Chunk。

## 最终工程检查

- 后端：`87 passed`，仅保留 1 条 Starlette TestClient 第三方弃用告警。
- Python 静态检查：`ruff check src tests scripts` 通过。
- 前端：补齐仓库既有 `npm run lint` 所缺的 ESLint 配置和依赖；ESLint、TypeScript/Vite 生产构建均通过，npm 审计 0 个漏洞。
- 部署配置：开发与生产两份 Docker Compose `config -q` 均通过。
- 在线资源：MongoDB 全部目标索引、MinIO 私有 Bucket、Milvus v4 Schema/HNSW/Sparse/Alias、Redis 鉴权/AOF 全部通过 verifier。
- 最终只读迁移计划确认 Alias/MongoDB 均为 v4，现有 3 个 published Document、8 个 Chunk，下一候选 v5 不存在。

## 当前本地算力

| 能力 | 当前生产实现 | 本地模型/GPU |
| --- | --- | --- |
| Dense/Sparse Embedding | `qwen3.7-text-embedding` 在线 API | 否 |
| Reranker | `qwen3-rerank` 在线 API | 否 |
| 主/备 LLM | Luna / Qwen 在线 API | 否 |
| PDF 解析 | MinerU API | 否 |
| Web 检索 | 阿里云 Web Search MCP | 否 |
| Demo/单元测试 | 确定性哈希 Encoder | 否，不是生产模型 |

部署无需 GPU、CUDA、Torch 或模型目录挂载。代价是 Embedding、Reranker、LLM、MinerU 和 Web
检索均依赖外部网络与 Provider；上线前必须冻结地域、数据留存、并发、限流、费用、超时和 SLA。
