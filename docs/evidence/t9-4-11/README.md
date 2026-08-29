# T9-4.11 验收证据

本目录只保存脱敏后的 T9-4.11 质量、香港 ECS 运行态和真实浏览器验收证据。访问码、JWT、API Key、
Prompt、原始上下文和服务器密码均未写入 Git。

## 文件

- `local-quality.json`：本地测试、静态检查、前端构建和冻结评测摘要。
- `hk-runtime.json`：香港 ECS 应用容器、受保护数据容器、运行配置、存储不变量和 API/SSE 验收摘要。
- `browser-acceptance.json`：桌面、移动、刷新恢复、路由标签和 Console 验收摘要。
- `browser-desktop-knowledge.png`：一般知识问题完成后的桌面工作台。
- `browser-desktop-trace.png`：包含 `evidence_sufficiency` 与 `web_fallback` 路由的 Trace 页面。
- `browser-desktop-clarification.png`：动态良率问题只等待时间范围的澄清状态。
- `browser-mobile-clarification-resume.png`：补齐时间并恢复工具任务后的移动工作台。
- `browser-desktop-internal-sop.png`：内部受控 SOP 仅走内部知识检索的桌面工作台。

## 证据边界

`semikb-intent-v4` 是由生成脚本构造并冻结的 132 条评审数据，不是独立人工盲测集，不能把其满分结果
描述为真实生产准确率。公开知识 Case 已触发阿里云 Web MCP，Provider 调用成功，但经过网关处理后没有
留下可用外部证据；系统因此保留内部证据边界和未知项，没有把外部搜索冒充成有效引用。
