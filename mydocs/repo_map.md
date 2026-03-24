# HACRL 仓库结构图（Repo Map）

本文件用于快速建立“代码在哪里、职责是什么”的全局认知。

## 一、顶层目录职责

- `README.md`
  - 项目简介与 HACPO 快速启动说明。
- `docs/`
  - 文档体系。
  - 其中 `docs/training_pipeline.md` 被定义为 train/eval pipeline 的 canonical 描述。
- `mydocs/`
  - 本地知识库，用于实现导读、运行手册、协作规范落地。
- `recipe/`
  - 各算法/场景脚本入口。
  - `recipe/hacpo/` 当前主要是启动脚本入口，不是核心训练逻辑承载点。
- `verl/`
  - 核心训练框架实现（trainer、worker、dataset、rollout、reward 等）。
- `examples/`
  - 数据预处理脚本与样例训练脚本。
- `tests/`
  - 测试与校验用例。

## 二、HACPO 相关关键路径

以下路径是理解 HACPO 的高优先级入口：

- `recipe/hacpo/run_qwen3-1.7b_qwen3-4b.sh`
  - HACPO 启动脚本与参数覆盖入口。
- `verl/trainer/main_ppo.py`
  - PPO/HACPO 主入口（配置解析、worker 组织、dataset 构建、trainer 启动）。
- `verl/trainer/ppo/ray_trainer.py`
  - 训练主循环，包含 dual-model rollout/update 流程与 batch 管理逻辑。
- `verl/trainer/ppo/core_algos.py`
  - `MAPO` advantage 与 `mapo_clip` policy loss 等核心算法逻辑。
- `verl/utils/dataset/rl_dataset.py`
  - `RLHFDataset` 数据读取与 tokenization 流程（含 aux tokenizer 支持）。
- `verl/workers/actor/dp_actor.py`
  - actor 更新逻辑与 policy loss 调用路径。
- `verl/workers/rollout/rollout_worker.py`
  - rollout 生成入口及 `aux_gen` 相关 meta 处理。

## 三、数据与脚本相关路径

- `examples/data_preprocess/math_dataset.py`
  - 训练数据预处理脚本。
- `examples/data_preprocess/math500.py`
  - 验证数据预处理脚本。
- `docs/training_pipeline.md`
  - 当前 train/eval pipeline 的 canonical 描述，优先级高于本目录其他文档。
- `mydocs/hacpo_impl.md`
  - HACPO 的实现解构、关键不变量和排查顺序。
- 目标数据产物路径约定：
  - `~/data/math/train.parquet`
  - `~/data/math/math500_test.parquet`

## 四、协作与提交约束（简版）

- 优先使用 subagent 并行处理独立任务。
- canonical 文档优先级：`docs/training_pipeline.md` > `mydocs/*`。
- commit message 必须严格为：`xxx(xxx): content`。
- Git 远端约定：
  - `origin` 指向个人 fork。
  - `upstream` 指向上游仓库。
- 提交身份必须使用 repo-local 配置（`TobyYang7` / `tobyyang7@outlook.com`）。
