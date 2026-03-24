# mlAgent — Planning + Coding (MLE-bench)

双智能体（Planning + Coding），数据与评分走 **mle-bench**。用法与 **myAgent** 相同：**进目录 → 激活 conda → `python` / `nohup python ...`**，主入口只有 **`scripts/run.py`**。

**已验证可跑通：** 本机（Mac）与 Pitt 云上均可完成 `conda` 环境 + `pip install -e .`；自检命令 **`python scripts/run.py --check-only`** 在本机已通过（云上装好 `../mle-bench` 后同样执行即可）。

---

## Current（云上）

与 myAgent README 同结构：

```bash
## run
ssh weiweis@pitt.lti.cs.cmu.edu
tmux new -s mlagent
cd /usr1/data/weiwei/jerry/ai4mle-research/mlAgent
actmlagent

export OPENAI_API_KEY=...   # 或写在 ~/.bashrc

nohup python -u scripts/run.py --config configs/default.yaml --competition spooky-author-identification > mlagent_run.log 2>&1 &

tail -f mlagent_run.log

# kill
pkill -f "scripts/run.py"
ps aux | grep run.py
```

**`actmlagent`**：与 **`actmyagent`** 一样，在 `~/.bashrc` 里配 alias，例如：

`alias actmlagent='source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlagent'`

（conda 路径按机器改。）

---

## 第一次：环境（本机 / 服务器各执行一次）

**一行只敲一条命令**，不要把多行和中文注释粘成一行。

目录需为 **`ai4mle-research/mlAgent`**，且上一级有 **`mle-bench`**（与 myAgent 同级）。

```bash
cd /path/to/ai4mle-research/mlAgent
source ~/miniconda3/etc/profile.d/conda.sh

conda env create -f environment.yml
conda activate mlagent
pip install -e .
pip install -e ../mle-bench

python scripts/run.py --check-only
export OPENAI_API_KEY=sk-...
```

- 没有 `environment.yml`：先 **`git pull`**（单独仓库则拉 `mlAgent` 的 `main`）。
- **`git pull` 被 `cache/cache.db` 挡住**：`git restore cache/cache.db` 或删掉该文件再 `git pull`。

---

## 本地日常

```bash
conda activate mlagent
cd ~/Desktop/ai4mle-research/mlAgent
export OPENAI_API_KEY=...
python -u scripts/run.py --config configs/default.yaml
# 或轻量：configs/minimal.yaml
```

覆盖示例：`python -u scripts/run.py max_rounds=1 max_steps_per_round=6`

---

## clean（远端）

```bash
ssh weiweis@pitt.lti.cs.cmu.edu 'cd /usr1/data/weiwei/jerry/ai4mle-research/mlAgent && rm -rf workspace/* && rm -f mlagent_run.log && echo OK'
```

---

## pull data（与 myAgent 相同）

```bash
bash scripts/pull_run.sh pull
bash scripts/pull_run.sh pull "20260321_010447" --clear
bash scripts/pull_run.sh clear
```

---

## 项目结构（简要）

```
mlagent/              # Python 包
configs/              # default.yaml, minimal.yaml
scripts/run.py        # 唯一入口
environment.yml       # conda 环境名 mlagent
workspace/            # 运行产物，gitignore
```
