# SemiAtlas 香港 ECS 24 小时稳定性观察证据

## 观察范围

- 时间：`2026-08-26T23:13:50+08:00` 至 `2026-08-27T23:17:03+08:00`。
- 时长：86,589 秒。
- 样本：286 个，预期间隔 300 秒。
- 检查项：容器运行/健康/重启/OOM、容器 CPU/内存、主机内存/磁盘/负载、监听端口、本机 HTTP
  Health 和域名 HTTP Health。

## 结论

- 本机 HTTP Health 失败：0。
- 域名 HTTP Health 失败：0。
- 异常容器状态、不健康状态、OOM：均为 0。
- 所有容器最大重启次数：0。
- 根盘使用率峰值：28%。
- 主机最低可用内存：5,586,501,632 bytes。
- 主机内存使用峰值：2,232,422,400 bytes。
- 主机 1/5/15 分钟负载峰值：0.83/0.38/0.24。
- 286 个样本均监听公网 `22/80` 与回环 `34099`，没有样本监听 443。

因此 24 小时 **HTTP 稳定性门禁通过**。HTTPS 证书仍未签发，443 未发布，HTTPS 门禁仍为 No-Go。

## 文件说明

- `semikb-observation-24h-sanitized/summary.json`：整个窗口的脱敏聚合指标。
- `semikb-observation-24h-sanitized/latest-sample.json`：最后一个样本的脱敏状态。
- `started-at`、`completed-at`、`elapsed-seconds`、`latest-sample`、`completed`：观察边界标记。
- `semikb-observation-24h-sanitized.tar.gz`：上述脱敏证据的归档，SHA-256：
  `b5f266eaace3874b0709e5449ee0fd9507acd8b127543780ad9ecb5e48410727`。

原始样本保留在香港 ECS：
`/opt/semiconductor-agent-knowledge-base-backups/hk-migration/observation-24h-current/`。
原始容器 Inspect 可能包含运行时配置，因此没有下载或提交 Git；本目录不包含密码、Token、API Key、
Prompt、原始业务内容或容器环境变量。
