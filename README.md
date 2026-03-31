# mlAgent — Planning + Coding (MLE-bench)

双智能体（Planning + Coding），支持 1\~N 个并行 Coding Agent。数据与评分走 **mle-bench**。

主入口：**`scripts/run.py`**（`--config` 默认 `configs/default.yaml`，可省略）。

---

## 日常运行

本地和云命令一样，进目录 → 激活 conda → nohup 跑。

**本地：**

```bash
conda activate mlagent
cd ~/Desktop/ai4mle-research/mlAgent
nohup python -u scripts/run.py > mlagent_run.log 2>&1 &
tail -f mlagent_run.log
```

**云上：**

```bash
ssh weiweis@pitt.lti.cs.cmu.edu
tmux new -s mlagent
cd /usr1/data/weiwei/jerry/ai4mle-research/mlAgent
actmlagent

nohup python -u scripts/run.py --competition spooky-author-identification > mlagent_run.log 2>&1 &
tail -f mlagent_run.log
```
nohup python -u scripts/run.py max_rounds=30 > spooky_baseline_run.log 2>&1 &

nohup python -u scripts/run.py --config configs/board_replan.yaml max_rounds=30 > spooky_board_run.log 2>&1 &


nohup python -u scripts/run.py --competition jigsaw-toxic-comment-classification-challenge max_rounds=30 > jigsaw_baseline_run.log 2>&1 &

nohup python -u scripts/run.py --config configs/board_replan.yaml --competition jigsaw-toxic-comment-classification-challenge max_rounds=30 > jigsaw_board_run.log 2>&1 &


nohup python -u scripts/run.py --competition random-acts-of-pizza max_rounds=40 > pizza_baseline_run.log 2>&1 &

nohup python -u scripts/run.py --config configs/board_replan.yaml --competition random-acts-of-pizza max_rounds=40 > pizza_board_run.log 2>&1 &

nohup python -u scripts/run.py --config configs/codex_todo.yaml --competition random-acts-of-pizza max_rounds=40 > pizza_todo_run.log 2>&1 &

**三种 planning 策略：**

默认跑 `baseline`（`configs/default.yaml`）。用 `--config` 切换：

```bash
# baseline（默认，可省略 --config）
nohup python -u scripts/run.py > mlagent_run.log 2>&1 &

# Experiment Board：planner 维护 branch 实验板，coder 用 log_to_board 记录实验
nohup python -u scripts/run.py --config configs/board_replan.yaml > board_run.log 2>&1 &

# Codex Todo：planner 生成 todo list，coder 按序执行，定期 revise
nohup python -u scripts/run.py --config configs/codex_todo.yaml > todo_run.log 2>&1 &
```

三个同时跑（日志分开）即可对比。

**覆盖参数示例：**

```bash
nohup python -u scripts/run.py max_rounds=1 max_steps_per_round=6 > mlagent_run.log 2>&1 &
nohup python -u scripts/run.py --config configs/board_replan.yaml max_rounds=5 > map_run.log 2>&1 &
```

**停止：**

```bash
pkill -f "scripts/run.py"
ps aux | grep run.py
```

---

## 第一次：环境（本机 / 服务器各执行一次）

一行只敲一条命令。目录需为 `ai4mle-research/mlAgent`，上一级有 `mle-bench`。

```bash
cd /path/to/ai4mle-research/mlAgent
source ~/miniconda3/etc/profile.d/conda.sh

conda env create -f environment.yml
conda activate mlagent
pip install -e .
pip install -e ../mle-bench
```

**OPENAI\_API\_KEY 一次性配好（之后不用再 export）：**

- 本地 Mac (zsh)：在 `~/.zshrc` 末尾加 `export OPENAI_API_KEY="sk-..."`，然后 `source ~/.zshrc`
- 云上 (bash)：在 `~/.bashrc` 末尾加 `export OPENAI_API_KEY="sk-..."`，然后 `source ~/.bashrc`

**`actmlagent` alias（云上 `~/.bashrc`）：**

```bash
alias actmlagent='source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlagent'
```

**常见问题：**

- 没有 `environment.yml`：先 `git pull`。
- `git pull` 被 `cache/cache.db` 挡住：`git restore cache/cache.db` 再 `git pull`。

---

## clean（远端）

```bash
ssh weiweis@pitt.lti.cs.cmu.edu 'cd /usr1/data/weiwei/jerry/ai4mle-research/mlAgent && rm -rf workspace/* && rm -f mlagent_run.log && echo OK'
```

---

## pull data

```bash
bash scripts/pull_run.sh pull
bash scripts/pull_run.sh pull "20260321_010447" --clear
bash scripts/pull_run.sh clear
```

---

## 项目结构

```
mlagent/              # Python 包
configs/              # yaml 配置
  default.yaml        #   baseline 策略（默认）
  board_replan.yaml   #   Experiment Board 策略
  codex_todo.yaml     #   Codex Todo 策略
scripts/run.py        # 唯一入口
environment.yml       # conda 环境名 mlagent
workspace/            # 运行产物，gitignore
```
