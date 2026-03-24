# mlAgent 架构详解

## 总览：双 Agent 循环

mlAgent 由 **Planning Agent** 和 **Coding Agent** 两个 LLM 角色组成，围绕一个**共享的 Jupyter kernel** 进行多轮迭代。入口是 `scripts/run.py` → `mlagent/orchestrator.py:run_experiment()`。

```
                    ┌─────────────────────────────┐
                    │    orchestrator.run_experiment│
                    └────────┬────────────────────-┘
                             │  for rnd in 1..max_rounds
                             ▼
                    ┌────────────────┐
                    │ PlanningAgent  │  (带跨轮历史)
                    │   .plan(rnd)   │
                    └───────┬────────┘
                            │ plan (str)
                            ▼
                    ┌────────────────┐
                    │  CodingAgent   │  (每轮新建, 无跨轮历史)
                    │  .run_round()  │
                    │  ↕ tool calls  │
                    └───────┬────────┘
                            │ coding_summary (str)
                            ▼
                    ┌────────────────┐
                    │ competition    │
                    │  .grade()     │  mle-bench 对 submission.csv 打分
                    └───────┬────────┘
                            │ GradingReport
                            ▼
                     format_summary → 传给下一轮 Planning
```

---

## 1. 入口：`scripts/run.py`

- 文件：`scripts/run.py`
- 解析 CLI 参数（`--config`、`--competition`、OmegaConf overrides）
- 调用 `mlagent/config.py:load_config()` 合并 YAML 与 dataclass 默认值
- 调用 `mlagent/orchestrator.py:run_experiment(config)`
- `--check-only` 模式仅加载赛题、不跑 LLM

---

## 2. 配置系统：`mlagent/config.py`

三个 dataclass 嵌套：

- **`LLMConfig`** — 模型名、temperature、max_tokens、timeout、keep_history、native_function_calling
- **`JupyterConfig`** — cell_timeout(600s)、monitor_interval(300s)、execution_timeout(7200s)、GPU/CPU 配额
- **`AgentConfig`** — 顶层，包含三个 LLMConfig（planning/coding/monitor）、JupyterConfig、max_rounds、max_steps_per_round、total_time_limit

`load_config()` 流程：`OmegaConf.structured(AgentConfig)` → 合并 YAML → 合并 CLI dotlist overrides → `to_object()`。

---

## 3. 赛题加载：`mlagent/competition.py` + `competition_loader.py`

- `Competition.from_mlebench(competition_id)` 做三件事：
  1. `from mlebench.registry import registry` 拿到赛题对象
  2. `is_dataset_prepared()` / `download_and_prepare_dataset()` 自动下载
  3. 读 `description.md`，判断 `is_lower_better`
- `competition.grade(submission_path)` 调用 `mlebench.grade.grade_csv()`，返回 `GradingReport`（score, medal, valid_submission 等）
- `competition.get_data_preview()` → `mlagent/data_preview.py:generate_preview()`，生成文件树 + CSV schema + 小文件内容，给 Planning Agent 做上下文

---

## 4. 编排器：`mlagent/orchestrator.py:run_experiment()`

这是核心循环，关键行为：

### 4.1 初始化（整个实验只做一次）

```python
run_dir = workspace/run_{timestamp}_{competition_id}/
jupyter = JupyterExecutor(...)   # 启动一个 Jupyter kernel，整个实验共享
planner = PlanningAgent(...)     # 初始化一个 PlanningLLM，整个实验共享
```

### 4.2 每轮循环

```python
for rnd in range(1, max_rounds + 1):
    plan = planner.plan(rnd, last_summary)          # Planning Agent 出计划
    coder = CodingAgent(config.coding_llm, ...)     # ← 每轮新建！
    coding_summary = coder.run_round(plan, ..., jupyter, ...)  # Coding Agent 跑
    grade = competition.grade(run_dir / submission_file)        # 打分
    last_summary = format_summary(...)              # 组装文字摘要
```

### 关键设计决策

| 组件 | 跨轮共享？ | 原因 |
|------|-----------|------|
| **PlanningAgent / PlanningLLM** | **是** — 同一个 `self.llm.messages` 列表累积 | Planning 需要知道所有历史轮次的结果来做长期规划 |
| **CodingAgent / ToolCallingLLM** | **否** — 每轮 `CodingAgent(...)` 新建 | 每轮的 LLM 对话从头开始，只携带当前 plan，不带前几轮的 tool call 历史 |
| **JupyterExecutor** | **是** — 同一个 kernel 进程 | **Jupyter kernel 的 Python 状态（变量、import、模型）跨轮保留** |
| **run_dir** | **是** — 同一个目录 | submission.csv 会被覆盖，所有轮共用同一目录 |

**重要含义**：虽然 Coding Agent 的 **LLM 对话历史**每轮清零，但 Jupyter kernel 里的 **Python 变量/已 import 的包/已训练的模型** 全部保留。Round 2 的代码可以直接用 Round 1 里定义的 DataFrame、模型对象等。只是 Coding LLM "不记得"自己前一轮写过什么代码——它只从 Planning Agent 传来的 plan 文字描述中了解之前做过什么。

---

## 5. Planning Agent：`mlagent/planning_agent.py`

### 5.1 初始化

```python
self.llm = PlanningLLM(llm_cfg)
self.llm.messages = [{"role": "system", "content": sys_prompt}]
```

`sys_prompt` 包含：赛题 description（全文）、data_preview（文件树+CSV信息）、metric_hint。

### 5.2 每轮调用 `plan(round_num, last_summary)`

- Round 1：`append_user("Round 1: no prior coding results. Propose the first plan.")`
- Round N：`append_user("Results from previous round(s):\n{last_summary}\n\nPropose the plan for round N.")`
- 调用 `self.llm.chat()` → 整份 `messages`（system + 所有历史 user/assistant）一起发给 API
- 返回的 plan 字符串直接传给 CodingAgent

### 5.3 Planning 的"记忆"

`PlanningLLM` 的 `messages` 列表**从不清空**，每轮追加一对 user+assistant。到 Round 3 时 API 请求实际包含：

```
[system, user_r1, assistant_r1, user_r2, assistant_r2, user_r3]
```

→ API 返回 assistant_r3，追加后变成 6 条 + system = 7 条。

### 5.4 debug_prompts 只记当轮

`tracer.write("planning_r3", str(self.llm.messages[-2:]), reply)` — 只截取最后两条（本轮的 user + assistant），**不是完整请求**。如果想看完整上下文，需要把所有 `planning_r*` 文件拼起来。

---

## 6. Coding Agent：`mlagent/coding_agent.py`

### 6.1 每轮初始化（无跨轮记忆）

```python
llm = ToolCallingLLM(self.coding_cfg)    # 全新的 messages = []
llm.set_system(_coding_system(plan, ...)) # system prompt 包含赛题描述 + 本轮 plan
llm.append_user("Start the round: call execute_cell or other tools. End with finish_round.")
```

`_coding_system()` 的 system prompt 包含：
- 工作目录路径
- 数据在 `./input/`（symlink）
- 赛题 description 前 12000 字符
- **本轮 plan 全文**（来自 Planning Agent）
- 工具使用规则

### 6.2 Tool-calling 循环

```python
for step in range(max_steps):
    step_result = llm.complete_with_tools(tools)
    # 若无 tool call → nudge 模型
    # 若有 tool call → dispatch 每个 tool → 结果写回 messages
    # 若 tool 是 finish_round → return summary
```

每一步：
1. LLM 返回一个或多个 tool_calls
2. 对每个 tool_call 调用 `_dispatch_tool()`
3. 结果以 `{"role": "tool", "tool_call_id": ..., "content": ...}` 写回 `messages`
4. 进入下一步，LLM 看到所有之前的 tool 调用和结果

**step limit**：如果跑完 `max_steps` 步仍未调 `finish_round`，返回 `"Round ended without finish_round; step limit reached."`。

### 6.3 四个 Tool

| Tool | 功能 | 实现位置 |
|------|------|---------|
| `execute_cell` | 在 Jupyter kernel 里执行 Python 代码 | → `jupyter.execute_cell(code, goal)` |
| `check_submission` | 检查 `submission.csv` 是否存在、shape、head | 直接 `pd.read_csv` |
| `read_file` | 读工作目录下的文件（限制不能逃逸 work_dir） | `Path.read_text` |
| `finish_round` | 结束本轮，返回 summary 给 Planning | 直接返回 `{"ok": True, "summary": ...}` |

### 6.4 Tool 结果截断

- `execute_cell` 的 output 截取前 **20000 字符**，error 截取前 **8000 字符**
- `read_file` 默认最多 **120 行**
- tracer 写 debug_prompts 时，tool 结果截取前 **8000 字符**

---

## 7. Jupyter 执行器：`mlagent/jupyter_executor.py`

### 7.1 初始化（整个实验一次）

```python
JupyterExecutor(work_dir, data_dir, jupyter_cfg, monitor_llm)
```

做的事：
1. 创建 `work_dir/input` → symlink 到赛题数据目录
2. 创建空 `experiment.ipynb`
3. 自动选 GPU（nvidia-smi 选最空闲）和 CPU 核（psutil 选最空闲）
4. 安装临时 Jupyter kernel spec（带 CUDA_VISIBLE_DEVICES 和 taskset CPU 亲和性）
5. 启动 `KernelManager` → `BlockingKernelClient` → `wait_for_ready`

### 7.2 `execute_cell(code, goal)` 的完整流程

```
kc.execute(code)  →  msg_id
     │
     ▼
_collect_iopub(msg_id, code, goal, cell_timeout)
     │
     ├─ 循环读 iopub 消息（stdout/stderr/error/status=idle）
     │
     ├─ 每隔 monitor_interval 秒：调 Monitor LLM 判断是否该中断
     │
     ├─ 超过 cell_timeout → interrupt_kernel
     │
     └─ 返回 ExecutionResult(success, output, error, execution_time, ...)
         │
         ▼
追加 cell 到 self.nb (notebook)  →  保存 experiment.ipynb
```

### 7.3 Monitor LLM（第三个 LLM）

当一个 cell 跑太久（默认每 **300 秒** 检查一次），会调用 `_monitor_llm_decision()`：
- 把 code、goal、当前 output、已用时间发给 monitor LLM
- monitor LLM 回复 `<action>CONTINUE</action>` 或 `<action>STOP</action>`
- 如果 STOP → `interrupt_kernel()` + 标记 `llm_terminated=True`

这是**第三个独立的 LLM 调用**（用 `monitor_llm` 配置），不带历史、每次独立判断。

### 7.4 Kernel 生命周期

| 事件 | Kernel 状态 |
|------|------------|
| `JupyterExecutor.__init__` | 启动一个新 kernel 进程 |
| Round 1 execute_cell × N | 同一个 kernel，变量累积 |
| Round 2 execute_cell × N | **同一个 kernel**，Round 1 的变量仍在 |
| Round 3 execute_cell × N | **同一个 kernel**，所有之前的状态都在 |
| 实验结束 `jupyter.shutdown()` | 关闭 kernel |

**所以后面的 coding 轮可以直接用前面轮在 Jupyter 里定义的变量、import 的库、训练好的模型**。但是 Coding LLM 的对话历史是每轮重置的，它不知道前面写过什么代码——只能通过 Planning Agent 的 plan 描述来间接了解。

### 7.5 Notebook 保存

每次 `execute_cell` 都会：
1. 创建 `nbformat.v4.new_code_cell(source=code)`
2. 附上所有 stdout/stderr/error 输出
3. `self.nb.cells.append(cell)`
4. `nbformat.write(self.nb, self.notebook_path)`

所以 **`experiment.ipynb` 记录了所有轮次的所有 cell**，可以事后在 Jupyter 里打开查看完整执行历史。

---

## 8. LLM 客户端：`mlagent/llm.py`

两个类：

### 8.1 `ToolCallingLLM`（Coding Agent 用）

- 支持 OpenAI function calling（`tools` + `tool_choice`）
- `complete_with_tools()` → 解析 `tool_calls` → 返回 `LLMStepResult`
- 通过 `litellm.completion()` 调用（支持任何 LiteLLM 兼容模型）
- `append_tool_result()` 把 tool 执行结果写回 messages
- 每次 `complete_with_tools` 发送**整份 `self.messages`**（system + 所有之前的 user/assistant/tool 消息）

### 8.2 `PlanningLLM`（Planning Agent 用）

- 纯文本对话，不用 tools
- `chat()` 发送整份 `self.messages`，返回文本
- `keep_history=True` 时 messages 不清空（默认行为）

两个类都使用 `litellm.completion()`，底层通过 `OPENAI_API_KEY` 环境变量鉴权。

---

## 9. 信息在轮次间的传递方式

```
Round 1:
  Planning LLM ──plan──→ Coding LLM ──tool calls──→ Jupyter kernel
                                                        │
  <── coding_summary ←── finish_round ──────────────────┘
       │
       ▼
  competition.grade(submission.csv) → GradingReport
       │
       ▼
  format_summary(coding_summary, grade, best_score, round=1) = last_summary
       │
       ▼
Round 2:
  Planning LLM (带 Round 1 历史) + last_summary ──plan──→ 新 Coding LLM
                                                              │
                                                   同一个 Jupyter kernel
                                                   （Round 1 的变量还在）
```

### 信息流表格

| 信息 | 从哪来 | 到哪去 | 怎么传 |
|------|--------|--------|--------|
| 赛题描述 | mle-bench description.md | Planning system prompt + Coding system prompt | 全文/截断嵌入 |
| 数据预览 | `data_preview.py` | Planning system prompt | 文件树 + CSV schema |
| 上一轮结果 | `format_summary()` | Planning 的下一轮 user message | 文本拼接 |
| Plan | Planning reply | Coding system prompt | 原文嵌入 |
| Jupyter 变量 | Kernel 内存 | 下一轮 execute_cell | 同一个 kernel 进程 |
| Coding 历史 | 本轮 LLM messages | **不传到下一轮** | 每轮新建 ToolCallingLLM |

---

## 10. Debug / Tracing 系统

### 10.1 `PromptTracer`（`mlagent/utils.py`）

- 写到 `run_dir/debug_prompts/NNNN_phase.txt`
- Planning 阶段：`planning_r1`, `planning_r2`, ...（只记当轮的 user+assistant）
- Coding 阶段：`coding_tool_execute_cell_s0`, `coding_tool_check_submission_s3`, ...
- 格式：`phase= / timestamp= / --- PROMPT --- / --- RESPONSE ---`

### 10.2 `ExperimentLog`（`mlagent/utils.py`）

- 写到 `run_dir/experiment_log.json`
- 每轮一条记录：plan、coding_summary、grade（score/medal/valid_submission）、notebook 路径

### 10.3 `experiment.ipynb`

- 所有 cell 按执行顺序记录，含完整 stdout/stderr/error
- 可以用 Jupyter 打开直接查看

---

## 11. 资源管理

### GPU

1. `orchestrator._ensure_cuda_visible()` → `auto_select_gpu()` 用 nvidia-smi 选最空闲 GPU
2. `JupyterExecutor._select_optimal_resources()` 再次查询，选 `max_gpu_count` 个最空闲的
3. 写入 kernel spec 的 `env.CUDA_VISIBLE_DEVICES`

### CPU

1. `JupyterExecutor._cpu_per_core()` 用 psutil 查每核利用率
2. 选 `max_cpu_cores` 个最空闲的核
3. Linux 上用 `taskset -c` 绑核启动 kernel

### 超时层级

| 层级 | 默认值 | 含义 |
|------|--------|------|
| `cell_timeout` | 600s | 单个 cell 最大运行时间 |
| `execution_timeout` | 7200s | cell 的绝对上限 |
| `monitor_interval` | 300s | 每 N 秒让 Monitor LLM 检查一次 |
| `total_time_limit` | 14400s (4h) | 整个实验的总时限 |
| LLM `timeout` | 600s | 单次 API 调用超时 |

---

## 12. 完整时序（以 3 轮为例）

```
t=0    run_experiment() 开始
       ├── 创建 run_dir
       ├── 启动 Jupyter kernel（进程 A）
       ├── 初始化 PlanningAgent（PlanningLLM, messages=[system])
       │
t=1    Round 1
       ├── planner.plan(1, None)
       │   ├── messages: [system, user_r1] → API → [system, user_r1, assistant_r1]
       │   └── return plan_1
       ├── coder = CodingAgent(...)           ← 新建
       ├── coder.run_round(plan_1, ..., jupyter, ...)
       │   ├── llm = ToolCallingLLM(...)      ← 新建, messages=[system_with_plan_1, user_start]
       │   ├── step 0: execute_cell("import pandas...") → Jupyter kernel A → output
       │   ├── step 1: execute_cell("train model...")   → Jupyter kernel A → output
       │   ├── ...
       │   ├── step 9: finish_round("summary: trained logistic, CV=0.56")
       │   └── return "summary: trained logistic, CV=0.56"
       ├── competition.grade(submission.csv) → score=0.53
       ├── last_summary = format_summary(...)
       │
t=60   Round 2
       ├── planner.plan(2, last_summary)
       │   ├── messages: [system, user_r1, asst_r1, user_r2] → API
       │   │   （Planning LLM 看到 Round 1 的完整对话）
       │   └── return plan_2
       ├── coder = CodingAgent(...)           ← 又新建（Round 1 的 LLM 对话丢弃）
       ├── coder.run_round(plan_2, ..., jupyter, ...)
       │   ├── llm = ToolCallingLLM(...)      ← 新建, 只有 plan_2 在 system prompt 里
       │   ├── step 0: execute_cell("# 直接用 Round 1 的 df 变量")
       │   │   → Jupyter kernel A（Round 1 的 df 仍在内存）→ OK
       │   ├── ...
       │   └── finish_round / step limit
       ├── competition.grade(...)
       │
t=120  Round 3 ...（同上）
       │
t=end  finally: jupyter.shutdown() → 关闭 kernel 进程 A
```
