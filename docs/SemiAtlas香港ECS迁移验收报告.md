# SemiAtlas 香港 ECS 迁移验收报告

## 1. 当前结论

截至 2026-08-27，SemiAtlas 的应用、MongoDB、Milvus、MinIO、Redis 和历史会话已迁移到香港 ECS，
HTTP 公网业务、桌面/移动浏览器和 24 小时稳定性观察均通过。HTTPS 因 Let’s Encrypt 二次 DNS 校验
超时尚未签发证书，因此结论为 **迁移完成、HTTP 稳定可用、HTTPS Go-Live 待完成**，不是完整生产上线。

## 2. 验收结果

| 门禁 | 结果 | 证据 |
| --- | --- | --- |
| 目标主机预检 | 通过 | 4 vCPU/8 GiB/80 GiB，Docker/Compose 与磁盘检查通过 |
| 冷备完整性 | 通过 | 归档及同期环境副本 SHA-256 通过 |
| 独立恢复 | 通过 | 四类存储内容级比较无差异，检索烟雾通过 |
| 生产 Compose | 通过 | 9 个服务健康，Worker ping 和 API health 通过 |
| 历史连续性 | 通过 | 迁移后的线程、Checkpoint、消息请求、Trace 在工作台可读取 |
| 九场景 Demo | 通过 | 自动验收 `passed=true`，运行前后存储不变量一致 |
| 最终聚合门禁 | 通过 | Demo、安全、离线包、存储、Worker、Health 六类全部为真 |
| 图片闭环 | 通过 | 文本检索命中并自动打开 320 x 320 真实资产 |
| 运营页面 | 通过 | Trace、入库任务中心、离线评估均读取真实记录 |
| 桌面/移动 | 通过 | 1440 x 900、390 x 844 无横向溢出，Console 干净 |
| HTTPS | 未通过 | CA 二次 DNS 校验超时，未签发证书，443 未发布 |
| 本机离线副本 | 通过 | 3.81 GB 离线包四项 SHA-256 全部匹配 |
| 24 小时观察 | 通过 | 覆盖 86,589 秒、286 个样本，Health/重启/OOM/容器状态门禁全部通过 |

## 3. 数据冻结值

- MongoDB：`document_catalog=44`、`chunk_catalog=154`、`image_assets=15`、`table_assets=19`、
  `ingestion_jobs=45`、`evaluation_runs=17`；其余会话和 Trace 集合按内容 Hash 对账。
- Milvus：`semikb_chunks_v4`，Strong 行数 149。
- MinIO：raw 56 个对象、derived 91 个对象。
- Redis：迁移时任务队列为 0。

这些值只描述 2026-08-26 冷备点。公网验收产生新的会话和 Trace 属于可解释的新写入，不用于否定迁移
前的独立恢复一致性结论。

## 4. 浏览器证据

- `docs/evidence/hk-migration-20260826/browser-acceptance.json`
- `docs/evidence/hk-migration-20260826/desktop-workbench-image.png`
- `docs/evidence/hk-migration-20260826/desktop-offline-evaluation.png`
- `docs/evidence/hk-migration-20260826/mobile-workbench-390x844.png`
- `docs/evidence/hk-migration-20260826/mobile-offline-evaluation-390x844.png`
- `docs/evidence/hk-migration-20260826/automated-current/final-verdict.json`
- `docs/evidence/hk-migration-20260826/automated-current/final-demo.json`
- `docs/evidence/hk-migration-20260826/automated-current/security.json`
- `docs/evidence/hk-migration-20260826/automated-current/offline.json`
- `docs/evidence/hk-migration-20260826/observation-24h/README.md`
- `docs/evidence/hk-migration-20260826/observation-24h/semikb-observation-24h-sanitized/summary.json`

## 5. 24 小时稳定性结论

- 观察窗口：`2026-08-26T23:13:50+08:00` 至 `2026-08-27T23:17:03+08:00`，共 86,589 秒、
  286 个五分钟样本。
- 容器：本机/域名 HTTP Health 失败 0 次，异常状态 0 次，不健康状态 0 次，OOM 0 次，所有容器
  最大重启次数均为 0。
- 主机：根盘使用峰值 28%，最低可用内存 5,586,501,632 bytes，内存使用峰值
  2,232,422,400 bytes，1/5/15 分钟负载峰值为 0.83/0.38/0.24。
- 端口：全部样本均保持公网 `22/80` 与回环 `34099`；443 未监听，和 HTTPS 尚未完成一致。
- 结论：24 小时 HTTP 稳定性门禁为 **Go**；HTTPS 门禁仍为 **No-Go**，两者不得合并表述。

## 6. 待完成项

1. 等 DNS 委派和递归解析稳定后，重新执行受控 Certbot HTTP-01 签发。
2. 验证 apex/`www` 双域名 SAN、HTTP 到 HTTPS、`www` 到主域名、TLS 1.2/1.3、HSTS 和 SSE。
3. 深圳 ECS 仍由用户自行决定何时释放；本次验收未对其执行释放或变更。

最终离线包已固定到源码 `ef2d31ce3b7e660f65ad166a9c7218dc70694ef6`，包含 10 个生产服务镜像；
`images.tar` 为 3,807,525,888 bytes，SHA-256 为
`9b862d9222f8e9ecd075d9e01e05f90ec47700f2b2a0ab9d3e645b5b32c79dfd`。深圳回滚机上的异机副本已
独立通过 `SHA256SUMS` 全量校验。迁移临时 SSH 凭据已按精确标签从两台服务器和本机清理。

本机第三副本位于 `data/runtime/hk-migration-20260826/offline-bundle/`，五个文件共
3,813,715,689 bytes，四项内容 Hash 再次通过。24 小时观察已经完成并通过，脱敏归档 SHA-256 为
`b5f266eaace3874b0709e5449ee0fd9507acd8b127543780ad9ecb5e48410727`。

只要 HTTPS 的第 1~2 项尚未全部完成，本报告不得改写为“全部 Go”或“完整生产上线”。
