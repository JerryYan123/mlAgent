# mlAgent — Planning + Coding (MLE-bench)

Planning + Coding 双智能体，数据与评分走 **mle-bench**（同级 `../mle-bench`）。用法和 **myAgent** 一样：**进目录 → 激活环境 → `python` / `nohup python ...`**。

---

## 一次跑通（本机 + 云上通用）

**没有 `scripts/setup_env.sh`**（已删）。照下面做即可，**顺序不能跳**。

### A. 本机（Mac，在 `mlAgent` 根目录）

```bash
cd /path/to/ai4mle-research/mlAgent

# 1) 若从未建过 mlagent 环境（只需一次）
conda env create -f environment.yml

# 2) 每次开终端
conda activate mlagent
pip install -e . && pip install -e ../mle-bench

# 3) API Key（或写进 ~/.zshrc）
export OPENAI_API_KEY=sk-...

# 4) 自检（不跑 LLM）
python scripts/run.py --check-only

# 5) 真跑（可先改 configs/minimal.yaml 测通）
python -u scripts/run.py --config configs/minimal.yaml
```

### B. 云上（Pitt 等，出现 `EnvironmentNameNotFound: mlagent` 时）

说明 **还没创建过** `mlagent` 环境。SSH 上去在 **`mlAgent` 根目录**执行（**只需一次**）：

```bash
cd /usr1/data/weiwei/jerry/ai4mle-research/mlAgent
# conda 初始化（路径按服务器实际改，常见：~/miniconda3 或 ~/anaconda3）
source ~/miniconda3/etc/profile.d/conda.sh

conda env create -f environment.yml
conda activate mlagent
pip install -e .
pip install -e ../mle-bench
```

之后每次：`conda activate mlagent`（或你配的 `actmlagent`），再 `nohup python -u scripts/run.py ...`。

**找不到 `conda.sh`？** 在服务器执行：`which conda`，若得到 `/home/xxx/miniconda3/bin/conda`，则 `source /home/xxx/miniconda3/etc/profile.d/conda.sh`。

---

## Current（云上）

与 myAgent README 同结构：

```bash
## run
ssh weiweis@pitt.lti.cs.cmu.edu
tmux new -s mlagent
cd /usr1/data/weiwei/jerry/ai4mle-research/mlAgent
actmlagent

export OPENAI_API_KEY=...   # 或已写在 ~/.bashrc

nohup python -u scripts/run.py --config configs/default.yaml --competition spooky-author-identification > mlagent_run.log 2>&1 &

tail -f mlagent_run.log

# kill
pkill -f "scripts/run.py"
ps aux | grep run.py
```

**`actmlagent`**：和 **`actmyagent`** 一样，是你在服务器 `~/.bashrc` 里自己配的一行 alias（见下）。**不要**用 `bash scripts/xxx.sh` 去启动主程序。

---

## 云上第一次：先有 conda 环境 + `actmlagent`

若出现 **`EnvironmentNameNotFound: mlagent`**，在服务器 **mlAgent 根目录**执行一次（装环境 + 依赖）：

```bash
cd /usr1/data/weiwei/jerry/ai4mle-research/mlAgent
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate mlagent
pip install -e .
pip install -e ../mle-bench
```

然后在 **`~/.bashrc`** 加一行（路径按你机器上 conda 实际位置改，和配 `actmyagent` 一样）：

```bash
alias actmlagent='source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlagent'
```

`source ~/.bashrc` 之后，`actmlagent` 就能用。

> 若你之前已经建过 **`actagent`** 环境，也可以：`conda activate actagent`，不必重复建 `mlagent`；此时 alias 写成 `conda activate actagent` 即可。

---

## 本地第一次：隔离环境

```bash
cd ai4mle-research/mlAgent
conda env create -f environment.yml
conda activate mlagent
pip install -e .
pip install -e ../mle-bench
```

以后：

```bash
conda activate mlagent
cd /path/to/mlAgent
export OPENAI_API_KEY=...
python -u scripts/run.py --config configs/default.yaml
```

自检：

```bash
python scripts/run.py --check-only
```

---

## clean（远端）

```bash
ssh weiweis@pitt.lti.cs.cmu.edu 'cd /usr1/data/weiwei/jerry/ai4mle-research/mlAgent && rm -rf workspace/* && rm -f mlagent_run.log && echo OK'
```

---

## pull data（本机执行，与 myAgent 相同用法）

```bash
bash scripts/pull_run.sh pull
bash scripts/pull_run.sh pull "20260321_010447" --clear
bash scripts/pull_run.sh clear
```

---

## git pull

若 **`cache/cache.db` 未跟踪导致冲突**：

```bash
rm -f cache/cache.db && git pull
```

---

## 配置说明

- `configs/default.yaml`：默认轮数/步数、模型名。
- `configs/minimal.yaml`：最小跑通。
- 覆盖示例：`python -u scripts/run.py max_rounds=1 max_steps_per_round=6`

---

## 只停本目录的 `run.py`（避免误杀其它任务）

```bash
for pid in $(pgrep -f "scripts/run.py"); do
  c=$(readlink -f /proc/$pid/cwd 2>/dev/null)
  echo "$c" | grep -q 'ai4mle-research/mlAgent$' && kill "$pid" && echo killed "$pid"
done
```

---

## 项目结构（简要）

```
mlagent/
configs/
scripts/run.py          # 唯一入口（主流程）
scripts/smoke_test.py
scripts/pull_run.sh
environment.yml           # conda 环境名 mlagent
workspace/                # gitignore
```
