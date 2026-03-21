#!/bin/bash
# Pull workspace runs from remote server.
# Usage:
#   bash scripts/pull_run.sh pull              # pull all run_* dirs
#   bash scripts/pull_run.sh pull "TS1 TS2"  # pull specific timestamp prefixes
#   bash scripts/pull_run.sh clear           # clear remote workspace

REMOTE="${REMOTE:-weiweis@pitt.lti.cs.cmu.edu}"
BASE="${BASE:-/usr1/data/weiwei/jerry/ai4mle-research/mlAgent}"
LOCAL_WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)/workspace"

DO_PULL=""
DO_CLEAR=""
RUN_TIMESTAMPS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    pull) DO_PULL=1; shift ;;
    clear) DO_CLEAR=1; shift ;;
    --clear) DO_CLEAR=1; shift ;;
    *) RUN_TIMESTAMPS="$1"; shift ;;
  esac
done

if [[ -n "$DO_CLEAR" && -z "$DO_PULL" ]]; then
  ssh "$REMOTE" "cd $BASE && rm -rf workspace/*"
  echo "Cleared remote $BASE/workspace"
  exit 0
fi

if [[ -z "$DO_PULL" && -z "$DO_CLEAR" ]]; then
  DO_PULL=1
fi

mkdir -p "$LOCAL_WORKSPACE"

if [[ -z "$RUN_TIMESTAMPS" ]]; then
  RUN=$(ssh "$REMOTE" "cd $BASE && TS=\$(date +%Y%m%d_%H%M%S) && \
    mkdir -p workspace/pack_\${TS} && \
    for d in workspace/run_*; do \
      [ -d \"\$d\" ] && rsync -a \"\$d/\" \"workspace/pack_\${TS}/\$(basename \$d)/\"; \
    done && \
    tar -czf workspace/pack_\${TS}.tgz -C workspace pack_\${TS} && echo pack_\${TS}")
else
  RUN="runs_$(date +%Y%m%d_%H%M%S)"
  TS_LIST="$RUN_TIMESTAMPS"
  RUN=$(ssh "$REMOTE" "cd $BASE && mkdir -p workspace/${RUN} && \
    for ts in $TS_LIST; do \
      for d in workspace/run_\${ts}_*; do \
        [ -d \"\$d\" ] && rsync -a \"\$d/\" \"workspace/${RUN}/\$(basename \$d)/\"; \
      done; \
    done && \
    tar -czf workspace/${RUN}.tgz -C workspace ${RUN} && echo ${RUN}")
fi

echo "Pulling ${RUN}.tgz ..."
scp "$REMOTE:$BASE/workspace/${RUN}.tgz" "$LOCAL_WORKSPACE/"
cd "$LOCAL_WORKSPACE" && tar -xzf "${RUN}.tgz"

if [[ -n "$DO_CLEAR" ]]; then
  ssh "$REMOTE" "cd $BASE && rm -rf workspace/*"
  echo "Cleared remote."
fi

echo "Pulled ${RUN}"
