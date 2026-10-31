# Chunk Metadata 设计规范

每个 chunk 保留以下 metadata 字段，方便回答时引用来源、支撑检索评估和管理侧查询。

```json
{
  "chunk_id": "battery_safety_0001",
  "doc_id": "battery_safety",
  "title": "battery safety",
  "section": "电池鼓包处理",
  "source": "manual",
  "source_path": "data/raw_docs/battery_safety.md",
  "updated_at": "2026-07-01",
  "content": "如果设备出现电池鼓包，应立即停止使用设备，断开电源，并避免继续充电。"
}
```

## 字段说明

| 字段 | 说明 |
|---|---|
| `chunk_id` | `{doc_id}_{序号:04d}`，全局唯一 |
| `doc_id` | 文档文件名（不含扩展名），如 `battery_safety` |
| `title` | 文档标题，来自 loader 阶段（`doc_id` 中 `_` 替换为空格），后续可在 chunker 阶段升级为 Markdown 一级标题 |
| `section` | 该 chunk 所属的 Markdown 二级标题（`##`），用于精细化溯源；实现阶段2时在 chunker 中解析 |
| `source` | 文档来源类型，目前统一为 `manual`（人工构造），后续可扩展 `official_pdf`、`crawled` 等 |
| `source_path` | 原始文档相对路径，供管理侧点击查看原文 |
| `updated_at` | 文档更新日期，用于第 30.3 节的版本控制（优先使用最新文档） |
| `content` | 实际 chunk 文本内容 |

## 与 plan.md 第 6 节的差异说明

第 6 节示例中的 `doc_id` 使用了 `_001` 后缀（如 `battery_safety_001`），本项目实际实现中 `doc_id` 直接使用文件名（不加数字后缀），因为一个文件名已经唯一标识一篇文档，`chunk_id` 通过序号区分同文档内的不同 chunk，避免冗余编号。后续阶段2的 `loader.py`/`chunker.py` 实现将遵循本规范。
