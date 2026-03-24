# HACRL 数据与运行手册（Data and Runbook）

本手册覆盖环境、数据准备、路径约定、Git 工作流和安全推送规范。

## 1. 运行环境

默认环境为：

```bash
conda activate verl
```

若未激活 `verl` 环境，不应执行训练与数据处理命令。

## 2. 数据准备（官方预处理流程）

目标数据路径：

- `data/train.parquet`
- `data/math500_test.parquet`

推荐流程：

```bash
conda activate verl
PYTHONPATH=. python examples/data_preprocess/math_dataset.py --local_dir ./data
PYTHONPATH=. python examples/data_preprocess/math500.py --local_dir ./data
```

说明：

- `math_dataset.py` 产出训练集 `train.parquet`。
- `math500.py` 产出验证集 `math500_test.parquet`。
- 当前仓库里直接运行 `python examples/data_preprocess/...py` 可能因为本地包导入路径缺少仓库根而失败；使用 `PYTHONPATH=.` 可以稳定运行。
- `math_dataset.py` 还会同时生成 `test.parquet`，但 HACPO 当前启动脚本应优先绑定 `math500_test.parquet` 作为验证集。

## 3. HACPO 脚本路径绑定

`recipe/hacpo/run_qwen3-1.7b_qwen3-4b.sh` 中若使用占位路径（如 `Your own path`），应替换为：

- `math_train_path=${REPO_ROOT}/data/train.parquet`
- `math500_test_path=${REPO_ROOT}/data/math500_test.parquet`

## 4. 文档与知识源约定

- `docs/training_pipeline.md` 是 train/eval pipeline 的 canonical 文档。
- `mydocs/` 是知识库（knowledge base），用于补充说明与快速导航。
- 若出现不一致，以 canonical 文档为准并回补知识库。

## 5. Git 工作流约定

### 5.1 Remote

- `origin`: `https://github.com/TobyYang7/HACRL`
- `upstream`: `https://github.com/Fred990807/HACRL`

同步上游建议：

```bash
git fetch upstream
# rebase 或 merge upstream/main 到当前分支
git push origin <branch>
```

### 5.2 提交身份（repo-local）

提交前确认：

```bash
git config user.name TobyYang7
git config user.email tobyyang7@outlook.com
```

禁止使用 `git config --global` 设置该身份。

### 5.3 Commit message 强约束

必须严格使用：

`xxx(xxx): content`

示例：

- `docs(agent): add collaboration rules`
- `docs(mydocs): add data runbook`
- `chore(git): align remote workflow`

## 6. Push 认证与安全

- push 认证参考仓库根目录 `.env` 中的 `GITHUB_TOKEN`。
- 禁止在任何文档、日志、命令回显中暴露 token 明文。
- 禁止把 token 写入 remote URL 或 git config。
- 推荐做法是临时加载 `.env` 后，以一次性认证方式执行 `git push`，而不是把 token 持久写入任何 Git 配置。

## 7. Subagent 协作策略

对于独立任务，优先使用 subagent 并行推进，例如：

- 文档起草与实现核对
- 不同模块的事实梳理
- 独立验证步骤

同时保持以下边界：

- 不并发编辑同一文件
- 主 agent 负责最终汇总与一致性校验
