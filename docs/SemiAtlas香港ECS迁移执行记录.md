# SemiAtlas 香港 ECS 迁移执行记录

## 1. 执行范围

- 执行日期：2026-08-26。
- 源端：阿里云深圳测试 ECS，保留作为人工回滚源，不执行释放。
- 目标端：阿里云中国香港 D，4 vCPU、8 GiB、80 GiB，EIP `47.76.218.205`。
- 域名：`semiatlas.cn`、`www.semiatlas.cn` 均解析到香港 EIP。
- 变更边界：等版本迁移、域名入口、HTTPS 部署能力和迁移验收；不开发业务功能。

## 2. 源端冻结与冷备

1. 确认运行中的入库和评估任务均为 0，Redis 队列为 0。
2. 冻结并停止深圳 Web/API/Worker，MongoDB、Milvus、MinIO、Redis、etcd 保留。
3. 生成冷备 `semikb-20260826T102823Z.tar.gz`：
   - 文件大小：248,876,503 bytes。
   - SHA-256：`5a288faae16abc6185e20ef53810eb98a05689c65eaa345eeae726fb2327b9fc`。
   - Manifest SHA-256：`ba9a4fece4714cb5cf39dff75481fee9ffaba0b885984eacbb781f4e3f224c52`。
   - 共 527 个归档文件，解包后 778,777,076 bytes。
4. 备份前后业务指纹一致，源端没有在冷备期间产生新的业务写入。

## 3. 目标运行时与恢复

1. 安装 Moby/Docker 28.3.3、Docker Compose 2.26.1，并通过 CPU、内存、磁盘、AVX 和端口预检。
2. 将固定源码、离线基础镜像和当前应用镜像传入目标主机；生产根目录固定为
   `/opt/semiconductor-agent-knowledge-base/`，真实 `.env` 权限为 `0600`。
3. 先恢复到独立数据根和 Compose Project，验证通过后停止独立 Project，再提升为生产数据根。
4. 独立恢复和生产提升后的比较结果一致：
   - MongoDB 受控业务集合内容 Hash 无差异。
   - Milvus Active Collection `semikb_chunks_v4`，Strong 行数 149。
   - MinIO `semikb-raw` 56 个对象，`semikb-derived` 91 个对象。
   - Redis 队列为 0，当前检索烟雾通过。
5. 生产 Project `semikb` 的 API、Worker、Web、MongoDB、Milvus、Milvus MinIO、MinIO、Redis、etcd
   均启动成功并保持健康。

## 4. 公网业务验收

1. Google DoH 和权威 DNS 均返回 `47.76.218.205`；本机 TUN 出现的 `198.18.x.x` 为代理 Fake-IP，
   不作为真实 DNS 结论。
2. `http://semiatlas.cn/healthz` 与 `http://www.semiatlas.cn/healthz` 返回 200。
3. 公网浏览器实际登录后确认历史线程已恢复，并完成：
   - `chat_direct` 自然回答与 SSE 中间状态。
   - `internal_rag` 混合检索和结构化调查卡。
   - `IMG-FA-ETCH-03-2026-004` 首图自动预览，天然尺寸 320 x 320。
   - Trace、入库任务中心、离线评估真实记录读取。
4. 桌面 1440 x 900 与移动 390 x 844 均无横向溢出；移动输入器按钮不重叠；Console 0 error/warning。
5. 服务器自动验收的 9 场景全部通过，运行前后 MongoDB、Milvus、MinIO、Redis 指纹一致，Worker 和
   健康检查通过。

## 5. HTTPS 当前状态

HTTPS 代码和部署产物已经准备：ACME Webroot、TLS Nginx 配置、Docker Certbot v5.6.0、续期脚本和
systemd timer。HTTP challenge 路径在目标主机可读取，但 Let’s Encrypt 二次验证连续报告 A/AAAA/CAA
或 `.cn` DNSSEC 查询超时，因此证书尚未签发，443 尚未发布。

当前状态属于 DNS/CA 外部依赖阻塞，不是 Nginx 80 端口或应用服务故障。恢复重试必须留出间隔，不能
高频请求 CA。证书成功前，对外入口保持 `http://semiatlas.cn/`，迁移不得标记为完整 Go-Live。

## 6. 保留与清理

- 深圳 ECS 未释放，数据容器保持可回滚状态；旧应用当前停写。
- 冷备和同期环境副本已复制到香港实例外；真实凭据不进入 Git 或验收报告。
- 迁移临时 SSH 公钥只用于本次跨机传输，全部操作结束后从两台服务器和本机删除。
- 香港云盘随抢占实例释放且无自动快照；重建必须依赖实例外离线包、冷备和固定 EIP。
