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
6. 最终运行源码固定为 `ef2d31ce3b7e660f65ad166a9c7218dc70694ef6`；部署时只重建 API、Worker、
   Web，六个数据容器 ID 保持不变。HTTPS/ACME 切换也使用 `--no-deps` 强制重建 Web，不联动重启数据服务。

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
6. 最终提交自动化总门禁再次执行，`final_demo`、`security`、`offline_bundle`、`storage_invariants`、
   `worker`、`health` 六类结果全部通过；脱敏证据位于
   `docs/evidence/hk-migration-20260826/automated-current/`。

## 5. HTTPS 当前状态

HTTPS 代码和部署产物已经准备：ACME Webroot、TLS Nginx 配置、Docker Certbot v5.6.0、续期脚本和
systemd timer。HTTP challenge 路径在目标主机可读取，但 Let’s Encrypt 二次验证连续报告 A/AAAA/CAA
或 `.cn` DNSSEC 查询超时，因此证书尚未签发，443 尚未发布。

当前状态属于 DNS/CA 外部依赖阻塞，不是 Nginx 80 端口或应用服务故障。恢复重试必须留出间隔，不能
高频请求 CA。证书成功前，对外入口保持 `http://semiatlas.cn/`，迁移不得标记为完整 Go-Live。

2026-08-26 23:08（DNS 修改已超过 6 小时）执行了一次新的受控重试。ACME Webroot 自检、Web 重建和
Certbot 启动均成功，但 Let’s Encrypt 二次验证仍报告：查询 `semiatlas.cn` 的 CAA 记录超时。公网权威
DNS、Google Resolver 以及多个 DNSPod Anycast 地址对 CAA 的 UDP/TCP 查询均能返回权威 NODATA，说明
故障仍集中在 CA 二次验证视角。下一步优先在 DNSPod 增加显式 `letsencrypt.org` CAA 授权，或改用受控
证书签发路径；不继续高频重试 Let’s Encrypt。

## 6. 保留与清理

- 深圳 ECS 未释放，数据容器保持可回滚状态；旧应用当前停写。
- 冷备和同期环境副本已复制到香港实例外；真实凭据不进入 Git 或验收报告。
- 固定提交 `ef2d31c` 的最终离线包包含 10 个生产服务镜像，`images.tar` 为 3,807,525,888 bytes，
  SHA-256 为 `9b862d9222f8e9ecd075d9e01e05f90ec47700f2b2a0ab9d3e645b5b32c79dfd`。
  包已复制到深圳回滚机 `/opt/semiconductor-agent-knowledge-base-offline/hk-ef2d31c/`，深圳端独立
  `sha256sum -c SHA256SUMS` 四项全部通过。
- 2026-08-26 23:12 已把同一离线包下载到本机
  `data/runtime/hk-migration-20260826/offline-bundle/`；五个文件共 3,813,715,689 bytes，
  `images.tar`、`source.tar.gz`、`manifest.json` 和 `README.txt` 均按包内 `SHA256SUMS` 再次通过。
- 迁移临时 SSH 公钥只用于本次跨机传输，全部操作结束后从两台服务器和本机删除。
- 香港云盘随抢占实例释放且无自动快照；重建必须依赖实例外离线包、冷备和固定 EIP。

## 7. 24 小时观察

- 观察服务：`semikb-hk-observation.service`，已正常完成并以 `0/SUCCESS` 退出；未保持无意义常驻。
- 开始时间：`2026-08-26T23:13:50+08:00`；完成时间：`2026-08-27T23:17:03+08:00`；
  实际覆盖 86,589 秒。
- 采样频率：每 300 秒，共取得 286 个样本；首个样本为 `20260826T151350Z`，末个样本为
  `20260827T151659Z`。
- 采样内容：容器状态/健康/重启次数、容器 CPU/内存、主机内存/磁盘/负载、监听端口、本机 Health 和
  域名 Host Health。
- 稳定性结果：本机 HTTP Health 和域名 HTTP Health 失败均为 0；异常容器状态、不健康状态、OOM
  事件均为 0；所有服务最大重启次数均为 0。最终 9 个长运行服务仍在运行，8 个带 Healthcheck 的
  服务均为 `healthy`，一次性 `milvus-init` 任务保持 `exited(0)`。
- 资源结果：根盘使用率峰值 28%；主机最低可用内存 5,586,501,632 bytes，内存使用峰值
  2,232,422,400 bytes；1/5/15 分钟负载峰值分别为 0.83/0.38/0.24。单容器 CPU 峰值最高为
  MongoDB 51.74%，单容器内存峰值最高为 MongoDB 360,290,713 bytes，均未触发资源或健康异常。
- 端口结果：286 个样本均只出现公网 `22/80` 和本机回环 `34099`；443 在全部样本中均未监听，与
  HTTPS 证书尚未签发的已知状态一致，不把它误判为 HTTP 迁移不稳定。
- 远端原始证据保留在
  `/opt/semiconductor-agent-knowledge-base-backups/hk-migration/observation-24h-current/`；脱敏汇总位于
  `docs/evidence/hk-migration-20260826/observation-24h/`，归档 SHA-256 为
  `b5f266eaace3874b0709e5449ee0fd9507acd8b127543780ad9ecb5e48410727`。
- 结论：**香港 ECS 的 24 小时 HTTP 稳定性门禁通过**。HTTPS 仍因外部 DNS/CA 校验阻塞而保持
  No-Go，迁移仍不得描述为完整 HTTPS Go-Live；深圳 ECS 未执行释放或变更。
