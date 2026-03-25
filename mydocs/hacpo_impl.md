# HACPO 实现拆解（实现视角）

> 这份文档是 HACPO 的实现解构笔记，面向开发者排错和二次开发。
> Pipeline 的规范性定义以 `docs/training_pipeline.md` 为准；本文件是补充说明。

## 1. 先说结论：HACPO 改动不在 `recipe/hacpo` 里

- `recipe/hacpo/run_qwen3-1.7b_qwen3-4b.sh` 只做参数覆盖与入口调用。
- 真正实现散布在通用 PPO 路径：
  - `verl/trainer/main_ppo.py`
  - `verl/trainer/ppo/ray_trainer.py`
  - `verl/trainer/ppo/core_algos.py`
  - `verl/utils/dataset/rl_dataset.py`
  - `verl/workers/actor/dp_actor.py`
  - `verl/workers/rollout/rollout_worker.py`
  - `verl/workers/fsdp_workers.py`
  - `verl/workers/megatron_workers.py`
  - `verl/trainer/ppo/utils.py`

核心事实：HACPO 是在单一 trainer 框架内扩展出双模型协同训练，而不是新增一套平行 trainer。

## 2. 关键对象和字段约定

### 2.1 角色层（Role）

`verl/trainer/ppo/utils.py` 中新增/使用了 `Role.AuxModel`，使 trainer 可以显式管理辅助模型 worker 组。

### 2.2 数据层（DataProto）

HACPO 是否正常，最看重这些字段是否一致：

- 来源标签：
  - `model_source` (`0=main`, `1=aux`)
- 主视图：
  - `input_ids`, `responses`, `response_mask`, `attention_mask`, `position_ids`
- 辅助视图：
  - `aux_input_ids`, `aux_responses`, `aux_response_mask`, `aux_attention_mask`, `aux_position_ids`
- MAPO 相关：
  - `performance`
  - `old_seq_ratio`

### 2.3 tokenizer 双视图约定

`RLHFDataset` 在 `aux_tokenizer` 存在时，样本一开始就同时产出主/辅 prompt 编码（含 `aux_raw_prompt_ids`）。

这意味着：

- 后续 rollout/reward/update 可以在同一批数据上切换视图；
- 不需要在每一步重新从原始文本构建 prompt。

## 3. 训练主路径（为什么它复杂）

下面是 `RayPPOTrainer.fit()` 中 HACPO 分支的逻辑主线。

### 3.1 双 rollout 并行产生

同一个 dataloader batch 会拆成：

- `gen_batch`（主模型）
- `aux_gen_batch`（辅助模型，通过 `_get_aux_gen_batch` 替换为 aux prompt 视图并标记 `aux_gen`）

然后分别调：

- `actor_rollout_wg.generate_sequences(...)`
- `aux_model_wg.generate_sequences(...)`

并写入 `model_source` 区分样本来源。

### 3.2 cross-encode 的作用

HACPO 不是把两份生成结果直接拼起来就结束。

`cross_encode_with_tokenizers(...)` 会做“decode -> re-encode”：

- main 响应用 main tokenizer 解码，再用 aux tokenizer 编码到 `aux_*` 字段
- aux 响应用 aux tokenizer 解码，再用 main tokenizer 编码到 `aux_*` 字段

这样合并后的 batch 才同时具备两个 tokenizer 语义下可更新的序列张量。

### 3.3 `swap(...)` 是关键操作

`swap` 本质是互换主字段和 `aux_*` 字段。

它被用于：

- reward 计算时让 aux 样本以正确 tokenizer 视图参与打分，再切回；
- actor 更新前只对 aux 样本 swap；
- aux 更新前对整批 swap 并翻转 `model_source`。

如果 swap 顺序错，最常见后果是“reward 正常但更新发散”或“update 正常但 eval 异常”。

## 4. MAPO 与 `mapo_clip`：HACPO 的算法层耦合点

## 4.1 MAPO advantage（`core_algos.compute_mapo_outcome_advantage`）

当前实现不是简单 GRPO 复用：

- 先按序列求 outcome score；
- 按 `uid/index` 分组；
- 主模型分数直接入组；
- aux 分数按 performance 倒数缩放；
- 组内均值还结合 `old_seq_ratio` 做加权；
- 最后再按 `response_mask` 扩成 token 级 advantage。

这决定了辅助模型样本在组基线中的影响方式。

## 4.2 `mapo_clip` loss（`core_algos.compute_policy_loss_mapo_clip`）

该 loss 明确分 main/aux 两支：

- main：PPO 风格 clip
- aux：序列重要性比率按 stepwise 下界裁剪，再乘 performance 权重和 `alpha` 项

对应地，`dp_actor.update_policy` 在该 mode 下会上报 main/aux 分离指标（如 `main_ppo_kl`, `aux_ppo_kl`, `aux_clipfrac_*`）。

## 5. 验证路径（actor 与 aux 分开）

`_validate(worker="actor"|"aux")` 里：

- actor 路径直接主视图生成+打分；
- aux 路径先把 test batch 切成 aux prompt 视图并加 `aux_gen` 元信息；
- 生成后再 cross-encode + swap 回 actor 视图，统一进 reward 函数评估。

所以 aux 验证不是“直接套 actor 逻辑”，而是独立 tokenizer 路径后再统一评分视图。

## 6. 后端层差异（FSDP vs Megatron）

两套 worker 都支持 `aux_model` role，但行为重点不同：

- FSDP worker 把 `aux_model` 视为 actor+rollout 可执行角色；
- Megatron worker 中 `aux_model` 主要落在 rollout 语义分支；
- 两者都在角色合法集合中显式包含 `aux_model`，避免 trainer 层调度失败。

这也是为什么 HACPO 改动不是只改 trainer：worker 角色语义必须联动。

## 7. 常见误区和排查顺序

### 7.1 常见误区

- 误区 1：以为改 `recipe/hacpo` 就能改算法行为。
- 误区 2：忽略 `model_source`，把 batch 当单模型处理。
- 误区 3：只看 main tokenizer，忽略 `aux_*` 字段一致性。
- 误区 4：在 reward/ref/logprob 处少做或多做一次 swap。

### 7.2 建议排查顺序

1. 看 `model_source` 分布是否正确（main/aux 样本数）。
2. 看 merge 后是否存在主/辅两套完整字段。
3. 看 reward 前后 swap 是否成对出现。
4. 看 old/ref logprob 是否按 main/aux 分支分别计算并正确写回。
5. 看 aux update 前是否执行了 performance 取倒数和 MAPO 优势重算。

## 8. 最短阅读路径（给新接手者）

若要快速定位 HACPO 训练逻辑，建议按下面顺序阅读：

1. `recipe/hacpo/run_qwen3-1.7b_qwen3-4b.sh`（看覆盖参数）
2. `verl/trainer/main_ppo.py`（看角色、tokenizer、dataset、trainer 构建）
3. `verl/utils/dataset/rl_dataset.py`（看 dual-tokenizer 样本结构）
4. `verl/trainer/ppo/ray_trainer.py`（看 fit 主流程和 validate 分支）
5. `verl/trainer/ppo/core_algos.py`（看 MAPO + mapo_clip）
6. `verl/workers/actor/dp_actor.py`（看 loss mode 进入点）
7. `verl/workers/rollout/rollout_worker.py`（看 `aux_gen` 元信息生效）

这条路径能覆盖 90% 的 HACPO 行为问题定位。
