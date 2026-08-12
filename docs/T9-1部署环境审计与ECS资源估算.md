# T9-1 部署环境审计与 ECS 资源估算

## 1. 范围与结论

本阶段只完成部署前只读审计、兼容性盘点和阿里云 ECS 资源估算，不安装、停止、重启或修改
MongoDB、Milvus、MinIO、Attu、Redis，也不执行应用部署。审计前后的 7 个容器 ID、镜像、状态和
端口快照完全一致。

结论：当前 CentOS 虚拟机可以继续作为内网开发验证环境，但不适合作为公网 Demo 的部署基线。
阿里云全栈单机 Demo 的最低可行规格为 **4 vCPU / 8 GiB / 80 GiB ESSD / 5 Mbps**；更稳妥的
面试演示规格为 **4 vCPU / 16 GiB / 100 GiB ESSD / 5-10 Mbps**。项目全部模型调用均已在线化，
两档规格都不需要 GPU。

## 2. 当前虚拟机快照

审计时间：2026-08-12。目标：`192.168.10.100`，仅通过已有 SSH 凭据执行只读命令。

| 项目 | 实际值 | 判断 |
| --- | --- | --- |
| 虚拟化 / OS | VMware；CentOS Linux 7；Kernel 3.10 | CentOS 7 已 EOL，不用于公网部署 |
| CPU | x86_64，4 vCPU，支持 SSE/AVX/AVX2 | 满足 Milvus SIMD 与建议核心数 |
| 内存 | 约 3.68 GiB；审计时可用约 2.16 GiB | 低于 Milvus Standalone 8 GiB 下限 |
| Swap | 约 3.87 GiB，审计时未使用 | 只能防止突发 OOM，不能替代内存 |
| 根磁盘 | 约 35.1 GiB；可用约 25.0 GiB | 仅适合当前小型开发数据 |
| Docker | Engine 26.1.4；Compose 2.27.1 | 满足项目与 Milvus 基础要求 |
| 容器运行时 | overlay2；cgroup v1；SELinux Enforcing | 新 ECS 改用受支持 OS 与 cgroup v2 |
| 防火墙 | firewalld 未运行 | 不能直接照搬到公网 ECS |
| Docker 占用 | 8 个镜像约 3.77 GB；7 个运行容器约 1.08 GiB 空闲态内存 | 空闲快照不能代替峰值容量验证 |

Milvus 官方对 Standalone 的要求是 8 GiB 内存，建议 16 GiB 和 4 核以上；磁盘至少为 SATA SSD，
etcd 理想指标为大于 500 IOPS 且 P99 fsync 小于 10 ms。参见
[Milvus Standalone prerequisites](https://milvus.io/docs/prerequisite-docker.md)。

## 3. 服务与镜像盘点

| 服务 | 当前镜像 / 版本 | 端口与网络 | 主要结论 |
| --- | --- | --- | --- |
| MongoDB | `mongo:latest`，服务端 8.2.6 | `0.0.0.0:27017`，bridge | 功能可用；必须固定不可变版本，公网禁止直出 |
| Milvus | `milvusdb/milvus:v2.5.5` | `0.0.0.0:19530/9091`，milvus | 当前 v4 Dense/Sparse 已验收；公网禁止直出 |
| Milvus etcd | `quay.io/coreos/etcd:v3.5.18` | 仅 Docker 网络 | 当前健康 |
| Milvus MinIO | `minio/minio:RELEASE.2023-03-20T20-16-18Z` | `0.0.0.0:9002/9003` | 旧于当前 Milvus 文档基线，T9-2 需兼容验证 |
| 业务 MinIO | `quay.io/minio/minio:RELEASE.2024-12-18T13-15-44Z` | `0.0.0.0:9000/9001`，bridge | 与当前 Milvus 文档版本一致；公网禁止直出 |
| Attu | `zilliz/attu:v2.5.10` | `0.0.0.0:7000` | 只作管理工具；公网默认不开放 |
| Redis | `redis:7.4-alpine`，服务端 7.4.10 | 仅 `192.168.10.100:6379`，semikb-network | 已启用密码、保护模式、AOF 与健康检查 |

7 个容器均为 `restart=unless-stopped`，但没有显式 CPU/内存限制。当前还没有部署 API、Worker、Web
容器，因此 1.08 GiB 只是存储服务空闲值，不能据此推导 `4 GiB` 足够全栈运行。

## 4. 公网部署风险

1. **OS 生命周期**：CentOS 7 已于 2024-06-30 EOL，阿里云有限安全更新也已于
   2025-06-30 结束，不能继续作为新公网节点基线。参见
   [阿里云 CentOS 7 安全更新说明](https://help.aliyun.com/zh/ecs/user-guide/centos-7-security-update-subscription)。
2. **内存不足**：当前约 4 GiB 低于 Milvus 单机最低要求，叠加 MongoDB、双 MinIO、API 和 Worker
   后没有可靠余量。
3. **端口暴露**：MongoDB、Milvus、两个 MinIO、Attu 当前监听全部网卡；迁移到 ECS 后不能照搬。
4. **版本不可重现**：MongoDB 使用 `latest`，未来重新拉取可能得到不同版本。
5. **资源无上限**：所有现有容器均未设置 CPU/内存限制，异常任务可能挤占整机。
6. **应用 Compose 待加固**：现有 `docker-compose.prod.yml` 的 Redis 没有密码，Worker 固定并发 2，
   API 缺少容器健康检查，服务没有日志轮转、资源限制和独立内部网络。
7. **公网入口待建设**：Nginx 当前只有 HTTP，尚未配置 TLS、安全响应头、上传大小、代理超时和限流。
8. **本地构建阻塞**：Windows 当前 Docker Desktop Linux Engine 未运行，暂时不能在本机完成镜像构建。

## 5. 阿里云 ECS 规格建议

### 5.1 最低可行 Demo

| 项目 | 建议 |
| --- | --- |
| 实例 | `ecs.u1-c1m2.xlarge` 或同级 x86_64：4 vCPU / 8 GiB |
| 系统盘 | 80 GiB ESSD，T9-2 实测 etcd IOPS 与 fsync 延迟 |
| 公网带宽 | 5 Mbps；只用于单人面试演示和少量 PDF/图片上传 |
| OS | 首选 Alibaba Cloud Linux 4 LTS x86 容器优化版；回退 3.2104 LTS x86 |
| GPU | 不需要 |
| 并发约束 | 1 个交互用户；Celery `concurrency=1`；入库和离线评估串行 |
| 可用性 | 单机、无 HA；Demo 前做云盘快照和 MinIO/MongoDB 备份 |

`ecs.u1-c1m2.xlarge` 官方规格为 4 vCPU / 8 GiB；购买时仍以目标地域实际库存为准。参见
[U1 实例规格](https://help.aliyun.com/zh/ecs/user-guide/general-work-force)。8 GiB 是整个单机 Demo 的
容量下限，不是舒适配置；运行批量入库或评估时需要控制 Worker 并发，并监控 OOM、Swap 和磁盘延迟。

### 5.2 稳妥演示档

| 项目 | 建议 |
| --- | --- |
| 实例 | `ecs.u1-c1m4.xlarge` 或同级 x86_64：4 vCPU / 16 GiB |
| 系统盘 | 100 GiB ESSD |
| 公网带宽 | 5-10 Mbps |
| 使用场景 | 浏览器问答、图片回传、单个入库任务和离线评估可较从容切换 |

16 GiB 与 Milvus Standalone 官方建议值一致，也为 MongoDB 缓存、Worker 解压/分块和 Docker 镜像构建
留出更合理余量。正式购买前还需确定地域、计费方式和演示持续时间，才能给出准确价格；本报告不使用
跨地域、随时间变化的临时报价。

### 5.3 8 GiB 档建议资源上限

以下是 T9-2 的初始限制预算，不是本阶段已经应用的配置：

| 容器组 | 建议内存上限 |
| --- | ---: |
| Milvus | 2.5 GiB |
| etcd | 256 MiB |
| Milvus MinIO | 384 MiB |
| MongoDB | 768 MiB |
| 业务 MinIO | 384 MiB |
| Redis | 128 MiB |
| API | 512 MiB |
| Worker | 768 MiB |
| Web/Nginx | 64 MiB |
| 合计 | 约 5.75 GiB |

剩余内存留给 OS、Docker、页缓存和短时峰值。若真实验收持续出现 Swap、OOM、Milvus 加载失败或 P95
明显抖动，不能继续压缩，应升级到 16 GiB。

## 6. 网络与安全组基线

- 公网只开放 `80/443`；完成 TLS 后可将 `80` 仅用于跳转。
- `22` 只允许用户的固定公网 IP，不长期开放 `0.0.0.0/0`。
- `27017`、`19530`、`9091`、`9000-9003`、`6379`、`7000` 均不得出现在公网安全组。
- API、Worker、Web、Redis、MongoDB、Milvus 和 MinIO 通过 Docker 内部网络或 ECS 私网通信。
- Attu 默认不部署公网入口；临时管理使用 SSH 隧道。

该规则与阿里云将安全组作为虚拟防火墙、SSH 仅授权可信 IP 的建议一致。参见
[阿里云安全组使用说明](https://help.aliyun.com/zh/ecs/user-guide/start-using-security-groups)。

## 7. 可重复审计工具

新增 `scripts/audit_docker_host.sh`。脚本只读取 OS、CPU、内存、磁盘、Docker、容器版本、资源策略、
网络、卷和监听端口；不读取容器环境变量，不输出 API Key 或密码。脚本会比较执行前后的运行容器
快照，若发生外部变化则返回退出码 3。

在目标 Linux 根目录执行：

```bash
bash scripts/audit_docker_host.sh | tee t9-host-audit.txt
```

`tee` 只保存审计输出，不修改服务。当前 CentOS 7 没有 Python 3，因此该工具使用 Bash 和系统命令，
不引入项目运行时依赖。

## 8. T9-2 前置门禁

购买 ECS 并继续前，需由用户确认：

1. 选择 8 GiB 最低档还是 16 GiB 稳妥档，以及地域、计费方式、公网带宽。
2. 是否在一台 ECS 上部署全部存储，还是保留/迁移现有存储节点。
3. 允许修订生产 Compose：固定 MongoDB 镜像、Redis 鉴权、Worker 并发、资源上限、内部网络、
   健康检查、日志轮转和 TLS 入口。
4. 新 ECS 上的磁盘性能、镜像构建、容器启动和端到端演示验收均通过后，才能把 T9 标记完成。
