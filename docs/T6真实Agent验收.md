# T6 真实 Agent 验收

## 验收范围

T6 验证从 T5 受控检索到连续调查对话的真实闭环：结构化约束、两轮追问、LangGraph
`interrupt/resume`、MongoDB Checkpoint、证据账本、模拟只读工具、Luna/Qwen 主备答案、引用校验、
审计和显式批准长期记忆。知识和制造数据均为合成或模拟内容。

## 实现边界

- `agent_threads` 保存线程目录和展示消息；完整图状态只由官方 `MongoDBSaver` 管理。
- 约束抽取模型产生的标识符只有能在原问题中找到文本依据时才可进入 T5 过滤，防止模型推断字段造成过度过滤。
- T5 Service 是内部证据的唯一入口；T6 不直接访问 Milvus，也不绕过 T5 权限、版本和有效期过滤。
- 所有回答事实必须引用证据账本。外部资料标记为 `external`，存在内部证据时，外部独占支持的事实会被移除。
- Yield/FDC/Recipe 工具为严格参数、只读、模拟实现；不支持任意 SQL、文件访问或设备控制。
- 长期记忆只能经 `/memories` 显式批准；对话、模型假设和未确认工艺经验不会自动沉淀。

## 真实执行结果

执行命令：

```powershell
conda run -n dl python -m semikb.storage.verifier
conda run -n dl python scripts/verify_t6_agent.py
```

2026-08-12 验收结果：

| 检查 | 结果 |
| --- | --- |
| 首轮模糊问题 | 缺少 Product、时间、Tool，`interrupt()` 第 1 轮暂停 |
| 跨实例恢复 | 重建 ConversationService 后使用相同 `thread_id` 成功恢复 |
| MongoDB Checkpoint | 完成时该临时线程共 14 条 Checkpoint |
| 内部证据 | 4 个 `T4-CASE-ETCH-03-R1` Chunk |
| 图文证据 | `T4-IMG-ETCH-03-EDGE-RING` |
| 模拟工具 | Yield、FDC Alarm、Recipe History 共 3 个只读调用 |
| 模型 | 约束抽取和答案均由 `gpt-5.6-luna-2026-07-09` 完成，未触发降级 |
| 引用 | 所有返回引用均存在于本次证据账本 |
| 审计与记忆 | Agent 运行、工具调用和显式批准偏好均持久化 |
| 清理 | 临时线程、Checkpoint、Trace、审计、记忆均清理为 0 |

自动化回归共 66 项通过，Ruff 通过，仅保留 Starlette TestClient 的第三方弃用告警。

## 存储迁移

`semikb.storage.t6_mongo_migration` 仅处理空的 `checkpoints`、`checkpoint_writes`、
`long_term_memories`、`audit_events`。本次迁移删除 4 个早期占位索引，创建 4 个官方
Checkpointer/Store/审计所需索引；执行前保留快照，执行后四类存储 verifier 全部通过。

## 剩余生产差距

- 当前 MongoDB 是单机实例；阿里云生产需副本集、备份恢复、故障转移和最小权限验收。
- 当前鉴权是 Demo JWT；正式部署需 OIDC 与企业 RBAC/ABAC。
- 制造数据工具尚未连接真实 YMS/DMS/MES/FDC/SPC，只能展示参数治理、编排和审计方式。
- 真实工厂数据发送第三方模型前，必须完成地域、留存、脱敏、合同和 SLA 审查。
