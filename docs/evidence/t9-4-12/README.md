# T9-4.12 验收证据

本目录只保存脱敏后的 T9-4.12 本地质量、香港 ECS 运行态和真实浏览器验收证据。访问码、JWT、
API Key、Prompt、原始上下文、服务器密码和 Web Provider 原始响应均未写入 Git。

## 文件

- `local-quality.json`：后端测试、静态检查、前端构建和冻结校准集摘要。
- `hk-runtime.json`：香港 ECS 容器、资源、存储不变量、公开 Web Case 与内部 SOP Case 摘要。
- `browser-acceptance.json`：桌面、Trace、内部 SOP、移动端、刷新与 Console 验收摘要。
- `browser-desktop-public-web.png`：公开通用知识使用自然回答并显示 5 个 Web 来源。
- `browser-desktop-trace.png`：低价值 Chunk 保留在 Trace，但最终证据为 0。
- `browser-desktop-trace-web-audit.png`：Web 查询、Provider 结果和标准化计数审计。
- `browser-desktop-internal-sop.png`：内部 SOP 保持结构化调查卡，未调用 Web。
- `browser-mobile-public-web.png`：精确 `390 x 844` 移动端自然回答，无横向溢出。

## 证据边界

`t9412-public-answerability-calibration-v1` 只用于校准默认阈值，包含 5 个未使用用户报告原句的通用
Case，不是独立人工盲测集，不能把通过结果描述为生产准确率。低价值 Chunk 仍保留在 Trace 供审计，
但不会进入答案证据账本。Web 结果不会写入受控知识库。

香港入口当前仍是 HTTP；443 未监听和证书未签发属于既有 HTTPS 阻塞，不能从本次功能验收中推导为
HTTPS 已完成。
