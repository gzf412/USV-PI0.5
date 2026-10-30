# 仿真资产整理交接

已完成资产归拢；原工作区资源没有修改。

## 新路径

- 场景模板：`assets/worlds/simple_waterway.sdf`。
- 动态艇：`assets/models/small_cora_float/model.sdf`，船壳为同目录 `meshes/hull.stl`。
- 仅三项传感器系统模板：`assets/worlds/sensor_systems.sdf`（Imu、NavSat、Contact）。
- 模型查找根目录：`assets/models/`。
- 浮力、水面渲染及参数发布库：`sim_runtime/lib/`。
- 最小插件源码：`sim_runtime/src/`，配套 CMakeLists 与 `build.sh`。

`src/usv_vla/world.py` 如果沿用该布局，应使用项目根目录
`Path(__file__).resolve().parents[2]` 定位上述资源；不再读取旧的
sim_vlm、small_cora 或 vrx_ws。艇的网格已改成 model://，不要再次通过旧目录绝对化。
场景中的全部外部工作区/home资源路径也已改为 model://。

## 启动环境

仍需系统 ROS 2 Jazzy / Gazebo Harmonic，默认 `/opt/ros/jazzy/setup.bash`。
启动设置：

```text
GZ_SIM_RESOURCE_PATH = 项目/assets/models + 原环境
GZ_SIM_SYSTEM_PLUGIN_PATH = 项目/sim_runtime/lib + 原环境
LD_LIBRARY_PATH = 项目/sim_runtime/lib + 原环境
```

不应 source 旧 vrx_ws/install/setup.bash。常规 Gazebo 物理/相机/推进器插件、
ROS 运行时和系统动态库是系统依赖，并未复制到项目里。当前 .so 为本机
Ubuntu 24.04 x86_64 的构建产物；其他系统需执行 `bash sim_runtime/build.sh`
重新编译。该脚本无需整个 VRX 仓库。本次已完成一次独立编译并安装。

## 精简范围与大小

仅保留两种浮标网格、对应贴图、CoRa 船壳、当前 Sydney 场景实际引用的
地形/建筑/树木网格与贴图、水面网格与着色器。未复制缩略图、完整 Fuel
缓存、VRX 仓库、ROS 系统库、原始数据、模型权重或构建缓存。

最终运行资产、最小源码及四个本地编译插件共 53 个文件，162,399,203 字节
（约 154.9 MiB；含世界与许可证，不含说明/manifest）。大部分空间是现有
地形贴图。完整原始来源及最终 SHA256 分别记录在
`assets/SOURCE_MANIFEST.json` 和 `assets/PORTABILITY_MANIFEST.json`；来源文件
中的旧路径是审计字段，不是运行依赖。

## 已做验证

- `python3 assets/check_resources.py`：55 项 SDF/COLLADA 资源引用全部可解析，
  无绝对路径、HTTP资源或缺失资源。
- `gz sdf -k assets/worlds/simple_waterway.sdf`：Valid。此独立命令需额外设置
  `SDF_PATH=项目/assets/models`，因为 sdf CLI 不自动读取 Gazebo 的路径回调。
- 四项最小插件均独立编译成功，`ldd` 无 not found，无旧 vrx_ws 路径依赖；
  内部 libWaves 使用本项目路径，RPATH 为 $ORIGIN。
- 未启动 GUI 或仿真。动态世界整体验证请在新的 world.py 适配后执行。

## 来源与许可

保留 VORC/CoRa 与 VRX 的原 Apache 2.0 文件及源码版权头。
Fuel 中的 Sydney 与浮标保留原 model.config 作者信息；本机缓存未附许可
声明，因此没有将其错误标注成项目代码许可。对外分发前需确认这些上游
资产条款。更详细来源见 `assets/README.md`。
