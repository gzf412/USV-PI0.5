#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$project_root/configs/paths.local.env" ]]; then
  source "$project_root/configs/paths.local.env"
fi
sim_python="${USV_SIM_PYTHON:-/usr/bin/python3}"
ros_setup="${USV_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
if [[ -f "$ros_setup" ]]; then
  set +u
  source "$ros_setup"
  set -u
fi
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export GZ_SIM_RESOURCE_PATH="$project_root/assets/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$project_root/sim_runtime/lib${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"
export LD_LIBRARY_PATH="$project_root/sim_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$project_root"
subcommand="${1:-help}"
case "$subcommand" in
  prepare)
    shift
    exec "$sim_python" -m usv_vla.world "$@"
    ;;
  record)
    shift
    exec "$sim_python" -m usv_vla.record "$@"
    ;;
  audit)
    shift
    exec "$sim_python" -m usv_vla.audit "$@"
    ;;
  video)
    shift
    exec "$sim_python" -m usv_vla.video "$@"
    ;;
  collect)
    shift
    exec "$sim_python" -m usv_vla.batch_collect "$@"
    ;;
  recollect)
    shift
    exec "$sim_python" -m usv_vla.recollect_standard "$@"
    ;;
  test)
    shift
    exec "$sim_python" -m unittest discover -s "$project_root/tests" -p 'test_*.py' "$@"
    ;;
  *)
    echo '用法：bash scripts/sim.sh prepare [布局参数]'
    echo '      bash scripts/sim.sh record --controller scripted|manual|zero|policy [--output 新目录] [--gui]'
    echo '      bash scripts/sim.sh audit 回合目录...'
    echo '      bash scripts/sim.sh video 回合目录 [--allow-missing-aerial]'
    echo '      bash scripts/sim.sh collect [--limit 6] [--resume] [--plan-only]'
    echo '      bash scripts/sim.sh test'
    ;;
esac
