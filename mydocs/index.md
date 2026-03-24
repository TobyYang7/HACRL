# HACRL Knowledge Base 索引

本目录是仓库知识库（knowledge base），用于帮助快速定位实现、流程与协作规范。

## 使用说明

1. 先阅读 `docs/training_pipeline.md`（canonical train/eval pipeline）。
2. 再阅读本目录内容做补充理解。
3. 若两者冲突，以 `docs/training_pipeline.md` 为准，并回补 `mydocs`。

## 文档导航

- `mydocs/repo_map.md`
  - 仓库结构与关键目录职责
  - HACPO 相关核心代码路径索引
- `mydocs/hacpo_impl.md`
  - HACPO 相对通用 PPO 的实现扩展
  - dual-tokenizer、dual-rollout、MAPO、`mapo_clip` 的实现解构
- `mydocs/data_and_runbook.md`
  - 环境、数据准备、路径约定
  - Git/分支/推送工作流操作手册

## 维护规则

- 本目录使用中文为主，代码与配置键名保持英文原文。
- 当训练流程、数据流程、Git 协作约束变化时，必须同步更新本目录。
- commit message 必须遵循严格格式：`xxx(xxx): content`。
