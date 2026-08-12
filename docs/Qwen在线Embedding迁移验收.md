# Qwen 在线 Embedding 迁移验收

## 变更结论

目标阿里云服务器没有 GPU，因此当前生产检索不再使用本地 BGE-M3。Dense 改为在线
`qwen3.7-text-embedding`，Sparse 改为 CPU 可运行且无模型权重的 `lexical-hash-v1`；RRF、
`qwen3-rerank`、动态截断、权限与版本过滤、图文召回均保留。

模型调用参考 [qwen3.7-text-embedding 页面](https://www.qianwenai.com/models/qwen3.7-text-embedding)。
项目实际请求 DashScope 原生 Embedding Endpoint，并使用与 Qwen Reranker 相同的 Key；
`EMBEDDING_API_KEY` 留空时由配置层复用 `RERANK_API_KEY`，日志和异常均不输出密钥。

## API 与编码验证

| 检查 | 结果 |
| --- | --- |
| HTTP 调用 | 200，返回有效 request id |
| 批量顺序 | 输入 2 条，返回 2 条并按索引恢复原顺序 |
| Dense 维度 | 每条 1024 维 |
| 数值质量 | 全部有限值，客户端 L2 归一化 |
| 语义烟雾 | 两条半导体相关文本余弦相似度约 0.414786 |
| Sparse | Provider 不返回；由 `lexical-hash-v1` 本地生成 |

词法 Sparse 使用稳定 SHA-256 哈希索引、log-TF 权重和 L2 归一化，覆盖英文/数字术语及中文
一元、二元片段。它不是机器学习模型，不依赖 Torch、GPU 或模型文件。

## 索引迁移

1. dry-run 确认源为 `semikb_chunks_v2`、目标为 `semikb_chunks_v3`，范围为 3 个 Document、8 个 Chunk。
2. 创建 v3 的 1024 维 Dense HNSW 与 Sparse Inverted Index，不覆盖 v2。
3. 从 MongoDB 权威正文批量在线重编码，写入 v3 并按 Chunk ID 强一致读回。
4. 使用首个 Chunk 执行 Dense 自检索，确认目标行可检索。
5. 切换 `semikb_chunks_active` Alias，再同步 MongoDB Catalog 与 `index_releases`。
6. 任一步异常恢复源 Alias；本次迁移后 v2 仍完整保留。

最终状态：Alias 指向 `semikb_chunks_v3`，v3 有 8 行 published 数据，3 个 Document 和 8 个
Chunk 的 `index_version` 均为 v3，版本标识为
`qwen3.7-text-embedding+lexical-hash-v1`。

## 检索与评估

T5 完整链路在 `t5-live-v1` 上达到 Recall@5=1.0、无证据准确率=1.0，并命中真实图片 Case。
四档 T7 评估由真实 Redis/Celery Worker 执行并持久化到 MongoDB：

| Profile | Run ID | Recall@5 | 无证据准确率 | 图片 Recall@5 | 通过率 |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense | `eval_5e2b77bc939b4de6ad3fe531fca1fecd` | 1.0 | 0.5 | 1.0 | 0.8571 |
| Hybrid | `eval_5b94df1956504f98946b0776005c6734` | 1.0 | 0.5 | 1.0 | 0.8571 |
| Reranked | `eval_9a46e4b236d14ba895abe9ad102e25f7` | 1.0 | 1.0 | 1.0 | 1.0 |
| Full | `eval_822491ba04ac4145862b5edd320f5ac0` | 1.0 | 1.0 | 1.0 | 1.0 |

Full 平均延迟约 1873.67 ms、P95 约 5127.58 ms；与 Reranked 相比质量未提升而延迟增加，
因此 HyDE 继续按问题条件启用，不能无条件调用。

最低 Rerank 分数从 0.35 校准到 0.40。依据不是迁移后调绿：同一 CMP 负例在 BGE/v2 和
Qwen/v3 下进入 Reranker 后的分数都为 0.35609，而当前正向期望证据最低分为 0.746104。
阈值调整保留了正向安全间隔，并修复已记录的 T8 入库后负例回归。

## 本地算力审计

| 能力 | 当前实现 | 本地模型/GPU |
| --- | --- | --- |
| Dense Embedding | `qwen3.7-text-embedding` 在线 API | 否 |
| Sparse | `lexical-hash-v1` 算法 | 否 |
| Reranker | `qwen3-rerank` 在线 API | 否 |
| 主/备 LLM | Luna / Qwen 在线 API | 否 |
| PDF 解析 | MinerU API | 否 |
| Web 检索 | 阿里云 Web Search MCP | 否 |
| Demo/单元测试 | 确定性哈希 Encoder | 否，不是生产模型 |

当前代码已移除 FlagEmbedding 依赖和未启用的本地 BGE Reranker 分支。部署镜像无需 GPU、CUDA、
Torch 或模型目录挂载。未来若加入视觉 Embedding、OCR/VLM 本地推理或异常检测模型，必须重新做
本地算力审计并优先提供可治理的在线 Provider。

## 部署边界

- 生产 `.env` 必须设置 `DEMO_MODE=false`、`MILVUS_INDEX_VERSION=v3` 和在线 Provider 配置。
- 上线前确认 API 地域、数据留存、并发/限流、超时、费用、SLA 和工厂资料出域政策。
- 在线 Embedding 不可用时，新入库应失败并保持未发布；检索请求应返回可诊断降级，不能把错误向量写入 Milvus。
- Embedding 模型、维度或 Sparse 版本再次变化时必须创建新物理 Collection，重复 dry-run、黄金集和 Alias 门禁。
