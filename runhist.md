# mlAgent 实验记录与优化时间线

## 竞赛奖牌线

| 竞赛 | Gold | Silver | Bronze | Median | 指标 |
|------|------|--------|--------|--------|------|
| spooky-author-identification | 0.165 | 0.270 | 0.294 | 0.419 | Log-loss (越低越好) |
| random-acts-of-pizza | 0.979 | 0.765 | 0.692 | - | AUC (越高越好) |
| jigsaw-toxic-comment | 0.987 | 0.987 | 0.986 | 0.981 | AUC (越高越好) |

---

## 全部实验

### 第一批 — 初始代码 (03/26)

| Run | 策略 | 竞赛 | 轮数 | Best | 奖牌 | 花费 | DL |
|-----|------|------|------|------|------|------|----|
| 0326 | baseline | pizza | 40 | 0.695 | BRONZE | $12 | 失败 |
| 0326 | board | pizza | 28 | 0.675 | - | $24 | 无 |
| 0326 | baseline | spooky | 30 | 0.292 | BRONZE | $7 | 失败 |
| 0326 | board | spooky | 27 | 0.317 | - | $19 | 失败 |

**使用的方法：** TF-IDF (word/char n-grams)、Logistic Regression、NB-SVM、SVM、SGD、LightGBM、XGBoost、stacking (LR/HGB meta-learner)、calibration (isotonic/sigmoid)。全部是传统 ML，没有成功跑过任何深度学习。

**发现的问题：**

1. **Kernel 用错了 Python** — `jupyter_executor.py` 里写死 `"python"`，解析到系统 python (`/usr0/.../miniconda3/bin/python`)，里面没有 torch。`mlagent` 环境有 torch 但 kernel 没用到。所有 DL 尝试都报 `ModuleNotFoundError: No module named 'torch'` 然后放弃。

2. **Pizza board 训练集回代 bug** — coding agent 写了 `model.fit(Xtr, y)` 然后 `model.predict_proba(Xtr)` 当验证分数。Isotonic regression 在训练集上能完美拟合，假 CV 从 0.70 飙到 0.97，但真实分数从 0.70 暴跌到 0.577。Agent 完全信任假分数，再也没回退。

3. **Spooky board 微调陷阱** — Board planner 在 R4 把 "Transformer Fine-Tuning" 标记为 "dead" 分支（因为缺包失败），之后 22 轮全在做 5 个模型之间的权重微调，从 0.315 到 0.315，零提升。

4. **Planner 没有退化感知** — 分数连续下降时 planner 完全不知道，继续推进已经失败的方向。

5. **Spooky baseline 0.292 是 continuation run** — 继承了前一轮 11 个 round 的 artifacts（包括 NB-SVM、calibrated SVM、多层 stacking），不是从零开始，所以起点比 board 高。

### 第二批 — 修复环境 + 核心 Prompt (03/30)

| Run | 策略 | 竞赛 | 轮数 | Best | 奖牌 | 花费 | DL |
|-----|------|------|------|------|------|------|----|
| 0330 | baseline | spooky | 30 | 0.259 | SILVER | $8 | DistilBERT |
| 0330 | board | spooky | 27 | 0.315 | - | $24 | frozen MiniLM embedding |

**Baseline 迭代过程：**
- R1-R6: TF-IDF + LR/NB-SVM 各种变体 stacking，到 0.31
- R7-R14: 卡在 0.31 做微调，planner 从 R12 开始说"换方向"但 coding agent 不听
- **R15: coding agent 终于尝试 fine-tune DistilBERT (3-fold CV)，分数跳到 0.28**
- R16-R21: 把 transformer OOF 和传统模型 stacking，多个 transformer 变体叠加，到 0.259
- R22-R30: 停滞，反复微调无果

**Board 迭代过程：**
- R1-R4: TF-IDF + LR 建立基础
- R5: 用了 frozen SentenceTransformer MiniLM embedding (OOF=0.67)，太弱没帮助
- R6-R27: 22 轮在 5 个模型之间做权重微调，始终 0.315

**Board 为什么还是没突破：** 这批还没加 anti-stagnation 规则。STAGNATION 警告虽然触发了，但 board planner 里没有相应的约束，继续规划 "ultra-local blend search"。

### 第三批 — 修复 Board 策略 + Plan 执行 (03/31)

| Run | 策略 | 竞赛 | 轮数 | Best | 奖牌 | 花费 | DL |
|-----|------|------|------|------|------|------|----|
| 0331 | baseline | spooky | 30 | **0.246** | **SILVER** | $8 | DistilBERT, DistilRoBERTa, RoBERTa |
| 0331 | board | spooky | 27 | **0.275** | **BRONZE** | $27 | DistilBERT, DistilRoBERTa, DeBERTa |
| 0331 | baseline | pizza | 40 | **0.796** | **SILVER** | $16 | DistilBERT |
| 0331 | board | pizza | 30 | 0.695 | BRONZE | $27 | DistilBERT, RoBERTa, DeBERTa |
| 0331 | baseline | jigsaw | 3 | 0.982 | above-median | $1 | 无 |
| 0331 | board | jigsaw | 3 | 0.977 | above-median | $1 | DistilBERT, RoBERTa, DeBERTa |

**Spooky baseline 0.246（最佳成绩）迭代过程：**
- R1-R2: TF-IDF word/char + LR/NB 基线 stacking，0.34 → 0.32
- **R3: 首次成功 fine-tune DistilBERT (3-fold CV)，加入 stacking → 0.278**
- R4-R5: 加入更多 sparse 模型 (char SVM, sparse alt)，stacking 提升到 0.272
- **R6: 训练 char-CNN (5-fold CV) 作为新 diversity source → 0.267**
- R7-R10: 尝试 DistilRoBERTa、更多 stack 变体，逐步到 0.265
- **R11: 训练 `transformer_strong` (DistilBERT 更仔细的 fine-tune) → 0.263**
- **R12-R13: 用 HGB meta-stack 把 `transformer_strong` + `transformer_better` + `transformer2` 等 5+ 个 transformer OOF 一起 stack → 0.246**
- R14-R30: 停滞在 0.246-0.248，尝试 RoBERTa 等新模型但 OOF 太弱(0.49)无法帮助

**关键发现：最终 0.246 不是靠某个很强的单模型，而是靠 6+ 个不同训练配置的 transformer OOF（每个单独都只有 0.40-0.54）stacking 在一起。多样性 > 单模型质量。**

**Spooky board 0.275 迭代过程：**
- R1-R4: TF-IDF + LR 基线，0.38 → 0.33
- **R5: fine-tune DistilBERT (2-fold, 1 epoch) + 6 源 stacking → 0.299**
- R6-R8: 加入 DistilRoBERTa、char NB 等 → 0.278
- R9-R11: 开始停滞，STAGNATION 警告触发
- **R12: Anti-stagnation 规则生效 — planner 说"per the rules, we must not plan blend-only"，改去训练新 transformer checkpoint**
- R13-R16: 新 transformer 变体 + stacking → 0.275
- R17-R27: 再次停滞

**Board 0.275 vs Baseline 0.246 差距原因：** Board 的 transformer 训练太浅（2-fold, 1 epoch, max_len=64），而 baseline 用了 3-5 fold、更长的 max_len、多次不同配置训练出了 6+ 个 diverse 的 transformer OOF。不是模型种类的问题，是训练深度和多样性的问题。

**Pizza baseline 0.796 迭代过程：**
- R1-R4: text TF-IDF + tabular features + LR/LightGBM stacking → 0.793
- R5-R10: 加入更多 text 变体、honest tabular 特征 → 0.796
- R11-R40: 围绕 0.79 波动，偶尔退化到 0.69（不稳定）

**Jigsaw（刚开始跑）：** 只跑了 3 轮。Baseline 用 NB-SVM + TF-IDF stacking 就到了 0.982（刚过 median 0.981）。奖牌线极其紧凑（bronze 0.986 到 gold 0.987），纯传统 ML 很难够到，可能需要 fine-tuned BERT/DeBERTa ensemble。

---

## 优化时间线

### Day 1 (03/26): 初始实验，发现问题

跑了 4 个 run（spooky + pizza 各一对 baseline/board），发现所有 DL 尝试都因为环境问题失败，board 策略有系统性缺陷。

### Day 2 (03/30): 修复环境和核心 prompt

| 文件 | 改动 | 解决什么问题 |
|------|------|-------------|
| `jupyter_executor.py` | `"python"` → `sys.executable` | Kernel 用 mlagent 环境的 python（有 torch+CUDA） |
| mlagent conda env | `torch 2.11.0+cu130` → `2.6.0+cu124` | CUDA 驱动兼容性（驱动只支持到 12.4） |
| `coding_agent.py` | 加 `## Package Installation` | 缺包时 `!pip install` 而不是放弃 |
| `coding_agent.py` | 加 `## Validation Rules (CRITICAL)` | 禁止 `model.predict(Xtr)` 当验证分数 |
| `utils.py` | `format_parallel_summary` 加警告 | 3 轮不提升 → STAGNATION，分数退化 → DEGRADATION |
| `orchestrator.py` | 追踪 `best_round`/`score_history` | 给 planner 提供分数趋势信息 |
| `planning_agent.py` | 加 `## Score Monitoring & Rollback` | Planner 知道何时回退 |

**效果：** Baseline spooky 从 BRONZE (0.292) 升到 SILVER (0.259)。Board 没变（还没加 anti-stagnation）。

### Day 3 (03/31): 修复 Board 策略 + Plan 执行合规

| 文件 | 改动 | 解决什么问题 |
|------|------|-------------|
| `board_planner.py` | 加 `## Anti-Stagnation Rules (CRITICAL)` | 3+ 轮不提升必须换模型族，不许继续 blend |
| `coding_agent.py` | 加 `## Following the Plan` | Coding agent 必须执行 plan 的核心目标 |
| `orchestrator_strategies.py` | Board 追踪 `best_round`/`score_history` | Board 也能触发 STAGNATION/DEGRADATION 警告 |

**效果：** Board spooky 从无牌 (0.315) 升到 BRONZE (0.275)。新开 pizza 和 jigsaw 实验。

---

## 分数进步对比

### Spooky (log-loss, 越低越好)

```
                第一批 (03/26)     第二批 (03/30)     第三批 (03/31)
Baseline:       0.292 BRONZE       0.259 SILVER        0.246 SILVER
Board:          0.317 -            0.315 -             0.275 BRONZE
─────────────────────────────────────────────────────────────────────
Gold: 0.165    Silver: 0.270    Bronze: 0.294
```

### Pizza (AUC, 越高越好)

```
                第一批 (03/26)     第三批 (03/31)
Baseline:       0.695 BRONZE       0.796 SILVER
Board:          0.675 -            0.695 BRONZE
─────────────────────────────────────────────────
Gold: 0.979    Silver: 0.765    Bronze: 0.692
```

### Jigsaw (AUC, 越高越好)

```
                第三批 (03/31)
Baseline:       0.982 above-median (3R, 还在跑)
Board:          0.977 above-median (3R, 已停)
─────────────────────────────────────────────────
Gold: 0.987    Silver: 0.987    Bronze: 0.986
```

---

## Baseline vs Board 总结

| 维度 | Baseline | Board |
|------|----------|-------|
| 分数 | 始终更好 | 始终更差 |
| 成本 | ~$8-16/run | ~$20-27/run (2-3x) |
| 收敛速度 | 快 (R4-R15) | 慢 |
| DL 使用 | 自由探索，容易误打误撞突破 | 分支标记为 "dead" 后不回头 |
| 停滞后表现 | Coding agent 有自由度尝试新东西 | 需要 anti-stagnation 规则强制转向 |

**结论：** Board 策略在当前实现下是净负面 — 增加 token 开销，分支标记机制容易变成陷阱。Anti-stagnation 规则缩小了差距但没消除。Baseline 的非结构化方式反而给了 coding agent 更大的自由度。

### Batch 4 — Jigsaw 专场 + cell_timeout/data_profile 改进 (03/31 下午)

| Run | 策略 | 竞赛 | 轮数 | Best | 奖牌 | 花费 | DL |
|-----|------|------|------|------|------|------|----|
| 0331_154838 | baseline | jigsaw | 15 | **0.9856** | - (差 bronze 0.001) | $5 | DistilBERT, CNN |
| 0331_154838 | board | jigsaw | 6 | 0.9853 | - | $4 | DistilBERT |

**What changed (Batch 3 → 4):**
1. `cell_timeout`: 900s → 2400s（三个 yaml 全改）。之前 jigsaw 的 DistilBERT 2-fold 训练在 900s 超时（fold 0 完成但 fold 1 没跑完），现在有足够时间。
2. 新增 `_data_profile()`: 自动扫描 `./input/` 生成数据文件概况（行数、列类型），注入 coding agent system prompt。模型能看到 "train.csv: 159571 rows" + "GPU: A6000 48GB" + "Cell timeout: 2400s"，自行调整超参。

**Jigsaw baseline 0.9856 迭代过程：**
- R1: TF-IDF char/word LR + NB + SGD 基线，5-fold OOF stacking → 0.981
- R2-R3: 加入 char CNN (PyTorch, GPU, 53 秒完成 3-fold)，stacking 改善 → 0.981
- **R4: 首次成功 fine-tune DistilBERT (3-fold CV on GPU) → 加入 stacking → 0.983**
- R5-R9: 持续改善 stacking，尝试不同 blend 权重 → 0.984
- **R10: 重新训练 DistilBERT (2-fold, clean pipeline)，label-wise 优化 stacking → 0.9855**
- R11-R15: 微调 + 尝试更多 transformer 但 API 兼容问题，停在 0.9856

**DistilBERT 单 fold AUC = 0.988**，已经超过 gold 线 (0.987)！但 2-fold OOF 后 stacking 到 0.9856，没能完全保留单 fold 的强度。

**Jigsaw board 0.9853：** 只跑了 6 轮就达到 `total_time_limit` (14400s=240分钟)。Board 的 token 开销导致每轮更慢（56 分钟/轮 vs baseline 的 17 分钟/轮）。R6 训练了 full-size DistilBERT 2-fold OOF，跳到 0.9853。

**关键问题：**
1. **离 bronze (0.986) 只差 0.001** — 非常接近但没过线。主要受限于：
   - Transformer 只用了 DistilBERT，没尝试更大的 RoBERTa-base/DeBERTa
   - HuggingFace API 兼容性问题反复出错（`AdamW` import 失败、`evaluation_strategy` vs `eval_strategy`、label dtype 不对），浪费了很多 step
   - 只训练了 2-3 个 transformer OOF 变体，远少于 spooky 的 6+
2. **Board 受 time limit 限制严重** — 160k 数据 + board token 开销 = 每轮近 1 小时，6 轮就到时间上限

---

## 第四次优化 (03/31 下午)

| 文件 | 改动 | 解决什么问题 |
|------|------|-------------|
| `configs/*.yaml` (全部 3 个) | `cell_timeout`: 900s → 2400s | 大数据集的 transformer/tree 训练不再超时 |
| `coding_agent.py` | 新增 `_data_profile()` 函数 | 自动扫描数据文件，报告行数、列类型，注入 system prompt |
| `coding_agent.py` | `_coding_system` 整合 env 信息 | GPU + 数据规模 + cell timeout 一起呈现，让模型自行调整超参 |

---

## 分数进步对比（更新版）

### Spooky (log-loss, 越低越好)

```
                第一批 (03/26)     第二批 (03/30)     第三批 (03/31)
Baseline:       0.292 BRONZE       0.259 SILVER        0.246 SILVER
Board:          0.317 -            0.315 -             0.275 BRONZE
─────────────────────────────────────────────────────────────────────
Gold: 0.165    Silver: 0.270    Bronze: 0.294
```

### Pizza (AUC, 越高越好)

```
                第一批 (03/26)     第三批 (03/31)
Baseline:       0.695 BRONZE       0.796 SILVER
Board:          0.675 -            0.695 BRONZE
─────────────────────────────────────────────────
Gold: 0.979    Silver: 0.765    Bronze: 0.692
```

### Jigsaw (AUC, 越高越好)

```
                第四批 (03/31)
Baseline:       0.9856 (差 bronze 0.001)
Board:          0.9853 (差 bronze 0.001)
─────────────────────────────────────────────────
Gold: 0.987    Silver: 0.987    Bronze: 0.986
```

---

## Baseline vs Board 总结

| 维度 | Baseline | Board |
|------|----------|-------|
| 分数 | 始终更好 | 始终更差 |
| 成本 | ~$5-16/run | ~$4-27/run (通常 2-3x) |
| 收敛速度 | 快 (R4-R15) | 慢 |
| DL 使用 | 自由探索，容易误打误撞突破 | 分支标记为 "dead" 后不回头（改善后好很多） |
| 停滞后表现 | Coding agent 有自由度尝试新东西 | 需要 anti-stagnation 规则强制转向 |
| 大数据集 | 15 轮/4小时 | 6 轮/5.6 小时（token 开销重） |

---

## 当前瓶颈与下一步

| 竞赛 | 当前最佳 | 目标 | 差距 | 瓶颈 |
|------|---------|------|------|------|
| Spooky | 0.246 SILVER | 0.165 GOLD | 0.081 | 单个 transformer OOF ~0.40 太弱；需要更大模型 (DeBERTa-v3-base)、5-fold、3-4 epoch、max_len=256 |
| Pizza | 0.796 SILVER | 0.979 GOLD | 0.183 | 数据太少 (2.9k)；transformer OOF 只有 0.60 远弱于传统 stacking 0.90；需要更好的特征工程和正则 |
| Jigsaw | 0.9856 | 0.986 BRONZE | 0.001 | 极接近！DistilBERT 单 fold 已达 0.988 (超 gold)，但 OOF stacking 后降到 0.986。需要：(1) 修复 HuggingFace API 兼容问题减少浪费 (2) 更大模型 RoBERTa/DeBERTa (3) 更多 transformer 变体多样性 |