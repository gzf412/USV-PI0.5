#!/usr/bin/env bash
set -euo pipefail
runtime_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ros_setup="${USV_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
if [[ ! -f "$ros_setup" ]]; then
  printf 'ROS/Gazebo setup script missing: %s\n' "$ros_setup" >&2
  exit 1
fi
set +u
source "$ros_setup"
set -u
cmake -S "$runtime_root" -B "$runtime_root/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$runtime_root"
cmake --build "$runtime_root/build" -j "${USV_BUILD_JOBS:-2}"
cmake --install "$runtime_root/build"
