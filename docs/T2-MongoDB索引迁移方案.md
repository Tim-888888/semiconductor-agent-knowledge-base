# T2 MongoDB 索引迁移方案

状态：**已批准、已执行并通过复验（2026-08-11）**。

## 迁移边界

- 只处理 MongoDB 数据库 `semikb` 中本项目的 Collection 索引。
- 不修改 MongoDB 容器、镜像、端口、账号、网络或服务配置。
- 不访问或修改其他数据库。
- 不修改 Milvus、MinIO、Attu、Redis 的任何资源。

2026-08-11 只读检查确认 14 个目标 Collection 的文档数量均为 `0`。即使当前无数据，
仍需在用户批准后才允许执行索引删除或重建。

## 发现的差异

| Collection | 当前问题 | 目标 |
| --- | --- | --- |
| `document_catalog` | `(document_id, revision)` 不是唯一索引 | 重建为唯一版本键 |
| `chunk_catalog` | 缺少 `(document_id, revision)` 辅助索引 | 保留唯一 `chunk_id` 并增加版本查询索引 |
| `image_assets` | 缺少 `(document_id, revision)` 辅助索引 | 保留唯一 `image_id` 并增加版本查询索引 |
| `ingestion_jobs` | `job_id` 与 `idempotency_key` 被合并为一个复合唯一索引 | 分别建立唯一索引，并增加状态/时间索引 |
| `retrieval_traces` | `trace_id`、用户、时间被合并，不能支持用户时间线查询 | 唯一 `trace_id` 与 `(actor_user_id, created_at)` 分离 |
| `evaluation_datasets` | `dataset_version` 非唯一 | 重建为唯一索引 |
| `evaluation_runs` | 运行 ID 与创建时间被错误合并 | 唯一运行 ID 与时间索引分离 |
| `agent_threads` | 线程 ID、用户、更新时间被合并为唯一索引 | 唯一线程 ID 与用户时间线索引分离 |
| `long_term_memories` | 记忆 ID 与用户被合并 | 唯一记忆 ID 与用户索引分离 |
| `audit_events` | 字段顺序不适合按用户查询时间线 | 使用 `(actor_user_id, created_at)` |

`ingestion_job_events`、`index_releases`、`checkpoints`、`checkpoint_writes` 当前索引与一期
目标一致，不需要迁移。

## 已执行步骤

1. 再次运行只读 verifier，并导出 `semikb` 全部索引定义和 Collection 文档数量。
2. 确认所有目标 Collection 仍为空；若已有数据，先执行重复键检查并停止自动迁移。
3. 仅删除 verifier 判定为错误且由本项目创建的旧索引。
4. 按 `src/semikb/storage/mongo_schema.py` 创建目标索引。
5. 重新运行 verifier，要求 MongoDB 所有检查通过。
6. 运行 T2 专项测试及完整测试。

## 执行结果

- 新增 `python -m semikb.storage.mongo_index_migration` 专用命令；无 `--apply` 时只生成计划和快照。
- 迁移前确认 14 个目标 Collection 均为 0 条文档，且不存在未知索引。
- 在 10 个有差异的 Collection 上执行 8 个旧索引删除和 16 个目标索引创建；另外 4 个
  已符合契约的 Collection 未修改。
- 执行前快照保存在被 Git 忽略的 `data/runtime/migrations/`，不包含业务文档或连接密钥。
- 迁移后完整 verifier 全部通过，二次 dry-run 返回 `actions: []`，14 个 Collection 仍均为 0 条文档。
- 自动验收为 32 项后端测试通过、Ruff 通过、前端生产构建通过；其中 T2 存储契约专项测试为 16 项。
- 没有修改 MongoDB 容器配置，也没有修改或访问其他数据库和其他基础设施服务。

```powershell
# 默认 dry-run；当前应返回 actions: []
conda run -n dl python -m semikb.storage.mongo_index_migration

# 仅用于经过审批的目标环境
conda run -n dl python -m semikb.storage.mongo_index_migration --apply
```

## 回滚

迁移前保存的旧索引定义是回滚依据。专用命令在新索引创建或迁移后校验失败时，会删除本次变更并
按快照自动恢复旧索引；若迁移和回滚同时失败，命令会保留两类异常并立即失败。由于不修改文档，
回滚只涉及索引元数据。任何发现非空数据或未知索引的情况都会在写操作前中止。
