# SemiAtlas 香港 ECS 迁移验收报告

## 1. 当前结论

截至 2026-08-26，SemiAtlas 的应用、MongoDB、Milvus、MinIO、Redis 和历史会话已迁移到香港 ECS，
HTTP 公网业务和桌面/移动浏览器验收通过。HTTPS 因 Let’s Encrypt 二次 DNS 校验超时尚未签发证书，
因此结论为 **迁移完成、HTTP 可用、HTTPS Go-Live 待完成**，不是完整生产上线。

## 2. 验收结果

| 门禁 | 结果 | 证据 |
| --- | --- | --- |
| 目标主机预检 | 通过 | 4 vCPU/8 GiB/80 GiB，Docker/Compose 与磁盘检查通过 |
| 冷备完整性 | 通过 | 归档及同期环境副本 SHA-256 通过 |
| 独立恢复 | 通过 | 四类存储内容级比较无差异，检索烟雾通过 |
| 生产 Compose | 通过 | 9 个服务健康，Worker ping 和 API health 通过 |
| 历史连续性 | 通过 | 迁移后的线程、Checkpoint、消息请求、Trace 在工作台可读取 |
| 九场景 Demo | 通过 | 自动验收 `passed=true`，运行前后存储不变量一致 |
| 图片闭环 | 通过 | 文本检索命中并自动打开 320 x 320 真实资产 |
| 运营页面 | 通过 | Trace、入库任务中心、离线评估均读取真实记录 |
| 桌面/移动 | 通过 | 1440 x 900、390 x 844 无横向溢出，Console 干净 |
| HTTPS | 未通过 | CA 二次 DNS 校验超时，未签发证书，443 未发布 |
| 24 小时观察 | 未开始 | HTTPS 完成后开始计时 |

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

## 5. 待完成项

1. 等 DNS 委派和递归解析稳定后，重新执行受控 Certbot HTTP-01 签发。
2. 验证 apex/`www` 双域名 SAN、HTTP 到 HTTPS、`www` 到主域名、TLS 1.2/1.3、HSTS 和 SSE。
3. 生成并核验包含当前源码和 10 个固定镜像的最终离线包，并复制到香港实例外。
4. 删除两台服务器及本机迁移临时 SSH 密钥。
5. HTTPS 全绿后开始至少 24 小时观察；深圳 ECS 仍由用户自行决定何时释放。

只要第 1~4 项尚未全部完成，本报告不得改写为“全部 Go”。
