#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Optional per-rank stdout/stderr files, also streamed to the console.
LAUNCH_LOG_ARGS=()
if [[ -n "${DIST_LOG_DIR:-}" ]]; then
    LAUNCH_LOG_ARGS=(--log_dir "$DIST_LOG_DIR" --tee 3)
fi

PYTHONPATH="${SCRIPT_DIR}/..${PYTHONPATH:+:$PYTHONPATH}" \
PYTHONUNBUFFERED=1 \
python -u -m torch.distributed.launch \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    "${LAUNCH_LOG_ARGS[@]}" \
    "$SCRIPT_DIR/train.py" \
    "$CONFIG" \
    --seed 0 \
    --launcher pytorch "${@:3}"
