# T9-4.4.5 多格式黄金文件

该目录保存完全合成、可公开展示的半导体文档，用于验证九个首期解析类别：Markdown、TXT、HTML、
PDF、DOCX、XLSX、CSV、PPTX 和图片。全部内容采用 `CC0-1.0`，不包含真实工厂、客户或设备数据。

`manifest.json` 冻结每个文件的 SHA-256、媒体类型、Parser、最小 Chunk/图片/表格数和检索问题。
二进制文件由以下命令确定性生成：

```powershell
uv run python -m scripts.generate_t9445_golden_files
```

黄金文件只证明测试输入和预期已冻结。是否真正通过必须以 API/Worker、MongoDB、MinIO、Milvus、
检索 Trace 和浏览器验收的联合证据为准。
