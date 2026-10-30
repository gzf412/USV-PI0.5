# Minimal Gazebo simulation runtime

Only four VRX libraries are retained: `Waves` (shared wave-field math),
`PolyhedraBuoyancyDrag` (boat buoyancy and drag), `PublisherPlugin` (zero-wave
parameter publication), and `WaveVisual` (water rendering). The 12 upstream
source/header files retain their copyright headers; `LICENSE` is the original
VRX Apache 2.0 license. `CMakeLists.txt` and `build.sh` are project-specific
minimal build files. This is not the complete VRX repository.

The libraries in `lib/` were rebuilt locally from these retained sources on
Ubuntu 24.04 x86_64, ROS 2 Jazzy, Gazebo Harmonic (gz-sim 8). They are architecture
and ABI dependent, not portable to arbitrary operating systems. Rebuild on a
compatible destination with:

```bash
bash sim_runtime/build.sh
```

`USV_ROS_SETUP` can select a different compatible ROS environment; the default
is `/opt/ros/jazzy/setup.bash`. The script requires CMake, a C++17 compiler,
Gazebo Harmonic development packages, Eigen3, and the dependencies found by
those packages. It does not fetch dependencies or require the old workspace.
The build directory is generated and may be removed after installation.

At run time, set `GZ_SIM_SYSTEM_PLUGIN_PATH` and `LD_LIBRARY_PATH` to this
folder's `lib/`. The project launcher does this. Built libraries use `$ORIGIN`
for their internal shared-library search path. System ROS/Gazebo libraries and
rendering/physics plugins remain external dependencies and are not copied.
