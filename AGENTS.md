# HACRL Agent 协作规范

本文件用于约束在本仓库内协作的 agent 行为。若与其他文档描述冲突，以本文件和用户最新明确指令为准。

## 1. 工作优先级

1. 优先保证可复现和可维护。
2. 先更新规范文档，再实现代码改动。
3. 独立任务优先并行，优先使用 subagent。

## 2. 知识源优先级

1. `mydocs/training_pipeline.md` 是当前 train/eval pipeline 的 canonical 文档。
2. `mydocs/` 是仓库知识库，用于解释与导航，不替代 canonical 文档。

## 3. Subagent 使用策略

当任务可以拆成相互独立的子问题时，默认优先使用 subagent 并行处理，例如：

- 文档起草与代码路径核对
- 不同子模块的实现梳理
- 独立验证任务

必须避免：

- 多个 subagent 同时编辑同一文件
- 为了“形式上并行”而拆分高度耦合任务
- 未经主 agent 汇总就直接采用单个 subagent 的结论

## 4. Git 与分支协作规范

### 4.1 Remote 约定

- `origin`: `https://github.com/TobyYang7/HACRL`
- `upstream`: `https://github.com/Fred990807/HACRL`

同步上游推荐流程：

1. `git fetch upstream`
2. 将当前分支 rebase 或 merge 到 `upstream/main`
3. `git push origin <branch>`

### 4.2 Commit message 规范

Commit message 必须严格遵循：

`xxx(xxx): content`

要求：

- 必须包含 type 和 scope
- `:` 后必须有一个空格
- `content` 必须具体，不允许模糊描述

示例：

- `docs(agent): define repo collaboration rules`
- `docs(mydocs): add data runbook and repo map`
- `chore(git): align origin upstream workflow`

### 4.3 Git 身份规范

提交前必须确认仓库本地身份（repo-local）为：

- `user.name=TobyYang7`
- `user.email=tobyyang7@outlook.com`

禁止使用 `git config --global` 修改该身份。

### 4.4 Push 认证规范

- Push 时认证参考仓库根目录 `.env` 中的 `GITHUB_TOKEN`。
- 禁止在文档、日志、命令回显或 remote URL 中暴露 token 明文。
- 禁止把 token 写入 git config。

## 5. 环境与数据规范

默认执行环境：

- `conda activate verl`

数据路径约定：

- 训练集：`~/data/math/train.parquet`
- 验证集：`~/data/math/math500_test.parquet`

若脚本中存在占位路径（如 `Your own path`），应替换为上述路径或在文档中明确说明实际路径。

## 6. 文档维护规范

以下文档为长期维护对象：

- `docs/training_pipeline.md`（canonical）
- `mydocs/index.md`
- `mydocs/repo_map.md`
- `mydocs/data_and_runbook.md`

每次涉及训练流程、数据流程、Git 协作流程变更时，必须同步更新文档，避免知识漂移。
