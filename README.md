# mlAgent — Planning + Coding (MLE-bench)

Planning agent（长期记忆）与 Coding agent（Jupyter kernel + tool calling）交替跑实验；评分走 **myAgent** 的 `Competition`（mle-bench）。默认模型 **`gpt-5.1-codex-mini`**（OpenAI Chat Completions + function calling，见 [模型页](https://developers.openai.com/api/docs/models/gpt-5.1-codex-mini)）。

环境与 **myAgent** 相同即可：本机 venv/conda；服务器上 **`actmyagent`**，先 `pip install -e ../myAgent` 再 `pip install -e .`（多 `jupyter_client` / `ipykernel` / `nbformat` / `psutil` 等）。

## 用 GitHub 同步本机 ↔ 服务器

仓库名可自定（例如 `mlAgent`）。**你在本机执行 push**，服务器用 **clone / pull** 更新代码（`workspace/` 已在 `.gitignore`，不会进仓库）。

**本机（首次）**

```bash
cd ~/Desktop/ai4mle-research/mlAgent
git init
git add .
git commit -m "Initial mlAgent"
# 在 GitHub 新建空仓库后：
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git branch -M main
git push -u origin main
```

**服务器（首次）**

```bash
cd /usr1/data/weiwei/jerry/ai4mle-research
git clone https://github.com/<你的用户名>/<仓库名>.git mlAgent
cd mlAgent
# 依赖与 myAgent
pip install -e ../myAgent && pip install -e .
```

**之后日常**

- 本机：`git add` / `git commit` / `git push`
- 服务器：`cd .../mlAgent && git pull`

可选：仍可用下面 **rsync** 做无 Git 的快速同步。

---

## run code

```bash
## run
ssh weiweis@pitt.lti.cs.cmu.edu
tmux new -s mlagent
cd /usr1/data/weiwei/jerry/ai4mle-research/mlAgent
actmyagent

export OPENAI_API_KEY=...   # 或写在 configs/default.yaml 的 api_key

nohup python scripts/run.py --config configs/default.yaml --competition spooky-author-identification > mlagent_run.log 2>&1 &

tail -f mlagent_run.log

# kill
pkill -f "python scripts/run.py"
ps aux | grep run.py
```

## clean things

```bash
ssh weiweis@pitt.lti.cs.cmu.edu 'cd /usr1/data/weiwei/jerry/ai4mle-research/mlAgent && \
  rm -rf workspace/* && \
  rm -f mlagent_*.log && \
  echo "Cleared."'
```

## pull data

```bash
# 用法
bash scripts/pull_run.sh pull [RUN_TIMESTAMPS] [--clear]   # 拉取（可加 --clear 拉完后清空远端）
bash scripts/pull_run.sh clear                             # 仅清空远端 workspace

# 每次 run：workspace/run_{timestamp}_{competition_id}/
# 内含 experiment.ipynb、experiment_log.json、debug_prompts/、submission.csv

# 示例
bash scripts/pull_run.sh pull                              # 拉当前远端全部 run_*
bash scripts/pull_run.sh pull "20260312_021700" --clear    # 拉指定时间戳后清空远端
bash scripts/pull_run.sh clear                             # 只清空远端
```

## Quick Start

```bash
# 依赖 myAgent（mle-bench 加载与评分）
cd ../myAgent && pip install -e .

# 本包
cd ../mlAgent && pip install -e .

# 建议：与跑代码同一套 ML 环境（与 myAgent 一致）
cd ../myAgent && pip install -e ".[ml]"
```

## Running

```bash
python scripts/run.py --config configs/default.yaml --competition spooky-author-identification
```

OmegaConf 覆盖示例：

```bash
python scripts/run.py max_rounds=5 max_steps_per_round=15 coding_llm.model_name=gpt-5.1-codex-mini
```

## Running on Server

```bash
# 1. 同步代码（二选一）
# A) Git（推荐）
cd /usr1/data/weiwei/jerry/ai4mle-research/mlAgent && git pull

# B) rsync（无 Git 时）
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.egg-info' \
    --exclude 'workspace' \
    ~/Desktop/ai4mle-research/mlAgent/ \
    weiweis@pitt.lti.cs.cmu.edu:/usr1/data/weiwei/jerry/ai4mle-research/mlAgent/

# 2. SSH 后
cd /usr1/data/weiwei/jerry/ai4mle-research/mlAgent
actmyagent
pip install -e ../myAgent && pip install -e .

python scripts/run.py --config configs/default.yaml --competition spooky-author-identification

# 断线重连
tmux attach -t mlagent
```

## CLI Reference

### `scripts/run.py`

| Arg | Default | Description |
|-----|---------|-------------|
| `--config` | `configs/default.yaml` | Config YAML |
| `--competition` | from YAML | Override `competition_id` |
| `overrides` | — | OmegaConf dotlist，如 `max_rounds=3` |

## Configs

| 文件 | 说明 |
|------|------|
| `configs/default.yaml` | 默认：`gpt-5.1-codex-mini`，Jupyter，`max_rounds` / `max_steps_per_round` |

## Project Structure

```
mlagent/
  orchestrator.py      Planning <-> Coding 主循环
  planning_agent.py    规划（跨 round 记忆）
  coding_agent.py      编码（tool calling）
  jupyter_executor.py  Jupyter kernel + notebook
  llm.py               LiteLLM + tools
  competition_loader.py 加载 myAgent Competition
  config.py            配置
  utils.py             trace / experiment_log
scripts/
  run.py               入口
  pull_run.sh          从服务器拉回 workspace
configs/
  default.yaml
workspace/             运行输出（gitignore）
```

## Requirements

- Python 3.10+
- `~/.kaggle/kaggle.json`（mle-bench 数据）
- LLM：`OPENAI_API_KEY` 或写在 YAML
- GPU 推荐（与 myAgent 服务器用法一致）
- 需先安装 **myAgent**（同级目录 `../myAgent`）
