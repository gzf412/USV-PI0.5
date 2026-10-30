#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_python="${USV_MODEL_PYTHON:-python}"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export USV_TOKENIZER_PATH="${USV_TOKENIZER_PATH:-$project_root/assets/tokenizer/paligemma_tokenizer.model}"
cache_root="${USV_CACHE_DIR:-$project_root/outputs/cache}"
export OPENPI_DATA_HOME="$cache_root/openpi"
export HF_HOME="$cache_root/huggingface"
export HF_LEROBOT_HOME="${USV_DATA_HOME:-$project_root/data/lerobot}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
cd "$project_root"
subcommand="${1:-help}"
case "$subcommand" in
  check)
    shift
    export JAX_PLATFORMS=cpu
    export PYTHONWARNINGS=ignore
    exec "$model_python" -m usv_vla.check_openpi "$@"
    ;;
  batch-check)
    shift
    export JAX_PLATFORMS=cpu
    exec "$model_python" -m usv_vla.batch_check "$@"
    ;;
  inspect-weights)
    shift
    export JAX_PLATFORMS=cpu
    exec "$model_python" -m usv_vla.inspect_weights "$@"
    ;;
  export)
    shift
    export JAX_PLATFORMS=cpu
    exec "$model_python" -m usv_vla.export_lerobot "$@"
    ;;
  train)
    shift
    # Training keeps the caller's accelerator selection; checks alone force CPU.
    exec "$model_python" "$project_root/scripts/train.py" "${USV_TRAIN_CONFIG:-pi05_usv}" "$@"
    ;;
  serve)
    shift
    exec "$model_python" "$project_root/scripts/serve.py" "$@"
    ;;
  *)
    echo '用法：bash scripts/model.sh check|batch-check [参数]'
    echo '      bash scripts/model.sh inspect-weights 本地NPZ --check-model [校验参数]'
    echo '      bash scripts/model.sh export --output 新目录 回合目录...'
    echo '      bash scripts/model.sh train --exp-name 名称 [训练参数]'
    echo '      bash scripts/model.sh serve --checkpoint 已训练检查点目录'
    echo '使用当前python或USV_MODEL_PYTHON；数据根目录可用USV_DATA_HOME指定。'
    ;;
esac
