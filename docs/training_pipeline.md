# HACPO 训练与评估流水线（Canonical）

> 状态：基于当前仓库实现路径整理。  
> 本文件是当前 HACPO train/eval pipeline 的**唯一 canonical 描述**。  
> 训练逻辑发生变化时，应先更新本文件，再更新 `mydocs/` 中的解释性文档。

## 1. 作用域与真实实现位置

HACPO 不是在 `recipe/hacpo` 下单独实现的一套 trainer。  
`recipe/hacpo/run_qwen3-1.7b_qwen3-4b.sh` 只是启动脚本，负责把 Hydra 覆盖参数传给：

- `python3 -m verl.trainer.main_ppo`

真正决定行为的代码在以下路径：

- `verl/trainer/main_ppo.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/ppo/core_algos.py`
- `verl/utils/dataset/rl_dataset.py`
- `verl/workers/actor/dp_actor.py`
- `verl/workers/rollout/rollout_worker.py`
- `verl/workers/fsdp_workers.py`
- `verl/workers/megatron_workers.py`
- `verl/trainer/ppo/utils.py`（`Role.AuxModel` 定义）

## 2. 启动脚本层（`recipe/hacpo`）

`recipe/hacpo/run_qwen3-1.7b_qwen3-4b.sh` 的职责是提供配置覆盖，核心包括：

- 启用异构双模型：
  - `aux_model.enable=True`
  - `aux_model.model.path=...`
- 切到 HACPO 对应算法分支：
  - `algorithm.adv_estimator=mapo`
  - `actor_rollout_ref.actor.policy_loss.loss_mode=mapo_clip`
- 辅助模型权重与裁剪超参：
  - `actor_rollout_ref.actor.alpha`
  - `actor_rollout_ref.actor.accuracy_window_size`
  - `actor_rollout_ref.actor.aux_clip_ratio_low`
  - `actor_rollout_ref.actor.aux_clip_ratio_step`

因此该脚本是配置层，不是实现层。

## 3. 入口与初始化（`verl/trainer/main_ppo.py`）

## 3.1 角色注册与资源池

`TaskRunner.run()` 会注册并初始化：

- actor/rollout
- critic
- reward model（可选）
- reference policy（当 KL loss 或 KL-in-reward 打开）
- aux model（`aux_model.enable=True` 时）

同时将角色映射到资源池，aux 通过 `Role.AuxModel` 进入调度图。

## 3.2 主/辅 tokenizer 初始化

入口会分别加载主模型与辅助模型路径，并创建：

- `tokenizer`（主模型）
- `aux_tokenizer`（辅助模型）

随后把两者都传给 dataset 与 trainer，保证同一批数据可同时保留两套 tokenizer 视图。

## 3.3 数据与 trainer 构建

- `create_rl_dataset(...)` 默认使用 `RLHFDataset`，并传入 `aux_tokenizer`。
- `RayPPOTrainer` 同时持有 main/aux tokenizer、worker 映射、reward 函数、dataloader 等组件。
- 真正训练从 `trainer.fit()` 开始。

## 4. 数据契约（`RLHFDataset` 双视图）

`RLHFDataset` 在 `aux_tokenizer` 存在时，会为每个样本同时构造：

- 主视图字段：
  - `input_ids`, `attention_mask`, `position_ids`, `raw_prompt_ids`
- 辅视图字段：
  - `aux_input_ids`, `aux_attention_mask`, `aux_position_ids`, `aux_raw_prompt_ids`

这一步是 HACPO 的基础不变量：后续 rollout、reward、update 都依赖这两套字段同步存在。

## 5. Worker 组装与 Aux 独立进程（`ray_trainer.init_workers`）

`RayPPOTrainer.init_workers()` 会构建：

- main actor/rollout worker group
- critic worker group（可选）
- ref worker group（可选）
- aux_ref（当 ref 与 aux 同时启用）
- aux_model worker group

关键实现细节：

- 当 aux 启用时，`aux_model` 会被拆成独立 worker group 进程（但仍在同一资源池中）。
- 代码注释给出的动机是适配 vLLM sleep-mode 的进程级约束。

后端支持：

- FSDP worker 的 role 集合包含 `"aux_model"`，并把它视作 actor+rollout 可执行角色。
- Megatron worker 的 role 集合也包含 `"aux_model"`，在 rollout 语义下参与生成/评估路径。

## 6. 训练主循环（`RayPPOTrainer.fit`）

以下顺序是当前 HACPO 的实际执行顺序。

## 6.1 双 rollout 生成

每个 dataloader batch 会形成两条生成路径：

1. 主路径：
   - 构建 `gen_batch`
   - `actor_rollout_wg.generate_sequences(gen_batch)`
2. 辅路径：
   - 构建 `aux_batch`
   - `_get_aux_gen_batch(aux_batch)` 将 aux prompt 字段切换到 `input_ids/...`
   - 写入 `meta_info["aux_gen"]` + aux eos/pad token id
   - `aux_model_wg.generate_sequences(aux_gen_batch)`

生成后标记来源：

- main 样本：`model_source = 0`
- aux 样本：`model_source = 1`

`rollout_worker.generate_sequences()` 会读取 `aux_gen` 决定使用主还是辅 tokenizer 的 eos/pad 配置。

## 6.2 合并与 cross-encode

主/辅 rollout 结果 union 后，执行双向 cross-encode：

- main 响应：`actor tokenizer` 解码 -> `aux tokenizer` 重新编码，写入 `aux_*`
- aux 响应：`aux tokenizer` 解码 -> `actor tokenizer` 重新编码，写入 `aux_*`

然后 `DataProto.concat([batch, aux_batch])` 合并成一个训练批。

这样单个批次内就同时存在两套 tokenizer 对齐后的序列张量。

## 6.3 reward / old log prob / ref log prob / value

### reward

aux 场景下，reward 前会对 aux 子集执行 `swap(batch, mask=aux_mask)`，reward 后再 swap 回来。  
目的是让每个样本在 reward 函数里使用与其来源一致的 tokenizer 视图。

### old log prob

按 `model_source` 拆 main/aux 两个子批，分别调用对应 worker 的 `compute_log_prob`，再合并写回。

### ref log prob

同样按 `model_source` 分支：

- main 样本走 main ref
- aux 样本走 aux_ref（或 aux worker 内 ref）

最后统一回填到一个 `ref_log_prob` 张量。

### value

critic 在合并批次上计算 values（若启用 critic）。

## 6.4 MAPO advantage 与 performance 信号

adv 之前会先计算 main/aux 的滑动窗口准确率，得到 `performance_ratio`，并写入：

- `performance`

然后调用：

- `compute_advantage(..., adv_estimator=MAPO)`

并传递 `model_source`, `performance`, `old_seq_ratio`。

## 6.5 actor 更新 + aux 更新

aux 开启时当前实现顺序为：

1. actor 更新阶段：
   - `metric_prefix="actor"`
   - 对 aux 子集做 `swap(batch, mask=aux_mask)`，再 `update_actor`
2. aux 更新阶段：
   - `swap(batch)` 全量切换视图
   - `model_source = 1 - model_source`
   - `performance = reciprocal(performance)`
   - MAPO 下重算 advantage
   - 调整序列顺序后调用 `aux_model_wg.update_actor(batch)`

两条更新链路分别记录 actor/aux 指标。

## 7. 算法层细节（`core_algos.py` + `dp_actor.py`）

## 7.1 MAPO 优势函数

`compute_mapo_outcome_advantage` 当前实现要点：

- 以序列 outcome score（`token_level_rewards.sum(-1)`）为基础；
- 按 `uid/index` 分组；
- main 样本分数直接入组；
- aux 样本分数按 `performance` 倒数缩放；
- 组均值结合 `old_seq_ratio` 做加权；
- 最后经 `response_mask` 展开到 token 级。

## 7.2 `mapo_clip` policy loss

`dp_actor.update_policy()` 在 `loss_mode="mapo_clip"` 时会把 `model_source/performance/old_log_prob_mask` 传入 loss。

`compute_policy_loss_mapo_clip` 中：

- main 分支：PPO 风格 clip；
- aux 分支：按 stepwise 下界裁剪 importance ratio，再乘 `performance` 与 `alpha` 项；
- 返回 main/aux 分离指标（如 `main_ppo_kl`, `aux_ppo_kl`, `aux_clipfrac_*`）。

## 8. 验证流水线（`_validate`）

`_validate(worker="actor"|"aux")` 有两条路径：

- actor 路径：主 tokenizer 直接生成 + 评分；
- aux 路径：
  - 先把测试批切成 aux prompt 视图并打 `aux_gen` 标记；
  - 用 aux worker 生成；
  - 生成后 cross-encode 回 actor 视图并 `swap(...)`；
  - 统一进入 reward 函数评估。

因此 aux 验证不是“复用 actor 同一路径”，而是独立生成后再对齐评估。

## 9. 保存与恢复

aux 启用时 checkpoint 会额外保存：

- `.../aux_model`

恢复时也会加载 aux checkpoint。  
这意味着 HACPO 持久化是双模型完整态，不是单 actor 态。

## 10. 维护建议（强约束）

- 排查 HACPO 行为时，不要只改 `recipe/hacpo`。
- 先检查这 5 个不变量：
  - `model_source` 标记是否正确
  - 主/辅双字段是否齐全
  - `swap` 是否成对出现且顺序正确
  - old/ref log prob 是否按来源拆分再回填
  - aux 更新前是否执行 performance 取倒数与 MAPO 重算

只要这些不变量破坏，训练与评估会快速出现不一致。
