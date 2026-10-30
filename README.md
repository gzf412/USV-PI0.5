# 无人艇VLA精简工程

从现有工程中整理出的无人艇专用版本。模型为PaliGemma底座的π0.5，状态33维、动作2维、前向RGB，另外预留两路图像。只保留当前模型、训练/推理、仿真、采集与任务验收所需文件。

本目录是当前唯一的无人艇VLA源码工程。旧usv_vla、sim_vlm和原openpi仓库已删除。新工程运行不导入旧工程源码，仿真模型与必要插件也随项目提供。当前动作示范保存在本项目`outputs/episodes/`，PaliGemma初始化权重在外置`../models/paligemma/`。

## 文件放在哪里

| 目录 | 内容 |
|---|---|
| `src/usv_vla/` | 动态世界、遥控、策略执行、同步记录、任务验收、数据审计与导出 |
| `src/openpi/` | 精简的π0.5/PaliGemma、USV适配器和JAX训练/推理组件 |
| `src/openpi_client/` | 推理通信与图像工具 |
| `scripts/` | 统一仿真入口、模型检查、训练和策略服务入口 |
| `configs/` | 33维输入协议和可选环境路径示例 |
| `assets/` | 实际用到的河道、艇、浮标、水面资源和4MB分词器 |
| `sim_runtime/` | 必要浮力/水面插件、最小源码和独立编译脚本 |
| `tests/`、`reports/` | 核心测试与本次独立验证结果 |
| `docs/`、`licenses/` | 当前协议、运行依赖、来源及许可证 |
| `data/` | 数据放置说明；默认LeRobot根目录为`data/lerobot/` |

已移除机械臂/ALOHA/DROID/LIBERO样例与配置、RLDS/TensorFlow链路、FAST动作模型、PyTorch模型实现和Transformers替换补丁、Docker、旧方案文档、历史实验缓存。Torch仍是LeRobot的数据读取依赖；本工程模型训练/推理统一使用已验证接口的JAX路线。

## 仿真和采集

系统需要ROS 2 Jazzy / Gazebo Harmonic、渲染驱动、系统Python的Numpy/Pillow/Gazebo绑定。本机现有系统环境可直接使用，不再source旧`vrx_ws/install`。自带插件当前为Ubuntu 24.04 x86_64构建，换系统可用`bash sim_runtime/build.sh`从随附最小源码重编。

在本目录执行：

```bash
# 核心测试和生成8浮标世界
bash scripts/sim.sh test
bash scripts/sim.sh prepare --buoys 8

# 人工驾驶并记录一个回合，输出目录必须尚不存在
bash scripts/sim.sh record --controller manual --gui \
  --output outputs/episodes/manual_001

# 用规则控制器检查采集与验收链路
bash scripts/sim.sh record --controller scripted --distance-m 4 \
  --output outputs/episodes/check_001
```

默认接近红色圆形浮标，在中心距1.9～2.3米、速度≤0.2m/s持续2秒时完成。W/S调整油门，A/D调整转向，空格推力归零；面板显示距离、速度、保持时间和任务阶段。松开键后保持设定，推力归零仍可能惯性滑行。

可用`--buoys 1..8`、`--target blue-pointed`、`--bearing-deg 120`、`--seed 1`改变数量、目标、镜头外方位与布局。指定侧通过用`--task pass --side left`；组合任务用`--task pass_then_approach --buoys 2 --target red-round --destination blue-pointed --side right`。每个回合独立启动/停止自己的Gazebo，不会接管其他仿真。

不指定`--output`时，新回合自动存到`outputs/episodes/本地时间戳/`，不会覆盖已有回合；可继续显式指定新目录。相对输出路径以本项目根目录为准；脚本也可从其他工作目录用绝对路径调用。所有生成数据和缓存默认进入`outputs/`，与源码分开。

每次采集默认同时记录艇载第一视角和独立空中俯视相机，回合结束后自动生成`回合目录/videa/first_person.mp4`与`aerial_global.mp4`。两路视频按相同仿真时间以10帧/秒播放；空中相机仅用于检查回放，不进入VLA观测或训练。原图、动作和时间戳仍单独保留。视频导出需要FFmpeg，本机已配置现有模型环境提供的编码器；可用`--no-videos`关闭。目录结构、重新导出及迁移配置见[视频说明](docs/videos.md)。

红、绿、蓝目标各有200条已验收的仿真回合，另外各有10条沿岸畅通补充回合，分别保存在`outputs/episodes/`对应批次中。红色原始批次有40条多浮标场景，覆盖绕行、穿越、混合障碍以及目标可见或初始画面外的直达任务；另有10条多浮标沿岸补充回合。三批当前均将所有已验收回合用于训练，不预留验证集；原采集分组保留在清单中用于溯源。采集与质量标准见[批采集说明](docs/batch_collection.md)和[轨迹平顺性验收](docs/trajectory_quality.md)。

全部200条通过任务、平顺性、时间因果、图像及双视频检查。20条沿岸任务实际航程23.13～27.25米，记录时艇中心距岸3.06～3.90米。40条复杂障碍任务的初始角度、航程与逐门验证见[复杂任务实测](outputs/episodes/red_round_200_20261027/COMPLEX_REVIEW.md)；沿岸数据见[沿岸实测](outputs/episodes/red_round_200_20261027/SHORE_REVIEW.md)。训练导出会校验原始提示，不把评分阶段和冻结航向送给模型；原始记录保留这些信息供审计。策略推理输入同样使用固定指令。

其中20条示范增加了事后训练文本标注：10条的路线无需避障，10条的原示范连续绕过2～3个圆筒。文本有简有详，均不指定绕行侧别；原来的轨迹、教师路线和逐帧 `prompt` 保持不变。这些标注不是当时采集时的指令，原示范只是可行路线之一。具体回合和文字见[指令清单](outputs/episodes/red_round_200_20261027/instruction_overrides.json)与[自由导航标注说明](outputs/episodes/red_round_200_20261027/FREE_NAVIGATION_REVIEW.md)。地图仍有1～4个圆筒，直达示范只表示航线上没有必须绕行的障碍。

按下方批次导出命令操作时，导出器自动读取批次根目录的 `instruction_overrides.json`，先验证原始数据和清单所绑定的回合、分组及文件哈希，再对所列回合应用训练文本。其他回合仍使用原固定指令。复制数据到服务器时需复制整个批次文件夹，包括该清单、`manifest.json`、场景计划和 `splits/`；如果遗漏清单，这20条不会应用新标注。原教师的隐藏通过门只用于审核旧示范，未来评估自由导航需独立按安全避障、到达目标和停车判断，不应要求模型经过原教师选定的侧别。

## 模型环境与数据

模型依赖按Python 3.11配置，清单在`pyproject.toml`；系统仿真Python可保持3.12。已有openpi Conda环境可继续用于检查，但代码从本项目`src`加载。在本机可先激活该环境，或显式设置`USV_MODEL_PYTHON`到该环境Python。脚本不再通过某个固定home路径猜环境。

新服务器可在Python 3.11环境中用`uv sync --locked`安装依赖；CUDA版JAX使用`uv sync --locked --extra cuda`。依赖安装需要网络，本次未重新安装整套依赖；已验证的是现有环境中的独立源码、数据与仿真路径。

`pyproject.toml`固定关键依赖版本，`uv.lock`已锁定195个包并通过`uv lock --check --offline`。锁文件校验不等于在新服务器安装和运行成功，仍需执行服务器首跑检查。

```bash
# 使用当前已激活的模型Python环境；检查不加载完整模型权重
bash scripts/model.sh check

# 读取外置数据；该根目录下应有 local/usv_v1/meta 和 data
USV_DATA_HOME=/你的/lerobot根目录 bash scripts/model.sh batch-check

# 也支持直接指定其他数据集及报告路径
bash scripts/model.sh batch-check --data-home /你的/lerobot根目录 \
  --repo-id local/usv_new --report outputs/checks/usv_new.json

# 审计和导出成功回合，不覆盖、不上传
bash scripts/sim.sh audit /你的/episode目录
bash scripts/model.sh export --repo-id local/usv_new \
  --output data/lerobot/local/usv_new /你的/episode目录
```

红、绿、蓝三批各有200条原始回合和10条沿岸畅通补充回合，尚未整体导出为LeRobot。每批 `splits/train_episodes.txt` 包含全部210条已验收回合，`validation_episodes.txt` 为空；原始清单中的采集分组是历史信息。不要把失败尝试混入数据。以下命令将三批全部训练回合合并导出，输出目录必须尚不存在：

```bash
episodes=()
for batch_root in "$PWD/outputs/episodes/red_round_200_20261027" "$PWD/outputs/episodes/green_round_200_20261028" "$PWD/outputs/episodes/blue_round_200_20261028"; do
  while IFS= read -r episode; do
    episodes+=("$batch_root/$episode")
  done < "$batch_root/splits/train_episodes.txt"
done
bash scripts/model.sh export --repo-id local/usv_v1 \
  --output data/lerobot/local/usv_v1 --image-format original-jpeg "${episodes[@]}"

# 导出成功后，数据根目录下才有 local/usv_v1/meta 和 data
export USV_DATA_HOME="$PWD/data/lerobot"
bash scripts/model.sh batch-check
```

默认将每个决策所引用的原始 JPEG 原字节嵌入 LeRobot Parquet，避免转成 PNG；也可显式使用 `--image-format original-jpeg`。已用三条真实回合检查图像、状态、动作和时间顺序，并用蓝色回合通过 pi0.5 数据加载检查；仍应在正式训练前对完整导出做批次检查。如需旧方式，可指定 `--image-format png`。

`USV_DATA_HOME`指向导出后的LeRobot根目录，不能直接填写原始批次路径。迁往服务器时复制本项目、所需动作数据和PaliGemma初始化权重，再设置对应路径即可。工作区不再提供旧大禹、旧仿真图文或旧动作数据目录；本次清理记录见`reports/cleanup_20261026_vlm_assets.json`。

## 训练与推理入口

唯一训练配置名为`pi05_usv`，固定状态33维、动作2维、动作块10步和文本上限384。保持原选定初始化：`PaliGemmaWeightLoader`加载通用PaliGemma，动作专家新初始化；没有自动改用官方`pi05_base`。完整初始化权重已保存在`../models/paligemma/paligemma-3b-pt-224.npz`，通过官方大小/SHA256及当前模型参数形状检查。

兼容上游的默认入口仍为官方GCS；2026-10-26实际检查该入口返回403。现已通过Google的[官方Hugging Face JAX仓库](https://huggingface.co/google/paligemma-3b-pt-224-jax)授权并完成下载。训练时用`--weight-loader.weights-path`指定本地文件，避免临时下载；来源、核验及重下步骤见`docs/weights.md`。

默认GCS缓存位置为`outputs/cache/openpi/vertex-model-garden-paligemma-us/paligemma/pt_224.npz`，设置`USV_CACHE_DIR`后对应其`openpi/`子目录；HF离线变量不拦截GCS下载。此前下载的官方π0.5基础检查点已清理，当前PaliGemma初始化方式保持不变。

以下是后续服务器入口，本次没有执行训练或加载完整权重：

```bash
USV_DATA_HOME=/你的/lerobot根目录 bash scripts/model.sh train \
  --exp-name pilot --data.repo-id local/usv_v1 \
  --weight-loader.weights-path /你的/完整PaliGemma初始化.npz

# 完成训练后，服务只接受本配置训练出的检查点
bash scripts/model.sh serve --checkpoint checkpoints/pi05_usv/pilot/步骤目录
```

检查入口强制JAX用CPU；训练和服务保留调用者的加速设备选择。`USV_CACHE_DIR`可改变大模型/HF缓存目录，`USV_OUTPUT_DIR`可改变仿真默认输出目录。完整变量示例见`configs/paths.env.example`。

策略服务默认只监听`127.0.0.1:8000`。服务启动后，用系统仿真Python运行以下命令，模型通信由指定的Python 3.11子进程完成：

```bash
# 新回合独立验收；成功由位置、速度、通过侧和停止条件判断
USV_MODEL_PYTHON=/你的/模型环境/bin/python bash scripts/sim.sh record \
  --controller policy --policy-host 127.0.0.1 --policy-port 8000 \
  --output outputs/evaluations/pilot_001
```

每次传入33维状态、前向RGB和任务文本，只执行返回动作块的第1步，单位为左右推力N。推理等待期间推力为零，仿真持续运行。推理超时、观测过期、断连、维度或单位错误会终止回合并归零。默认墙钟推理预算0.5秒、观测年龄预算0.75秒、非零命令保持上限0.25秒；可通过`--policy-timeout-s`、`--policy-max-source-age-s`、`--policy-command-lease-s`调整。首次JAX编译或较慢服务器可能超出预算，应先预热并实测延迟，再确定可用控制频率。

当前采用单请求执行：收到结果后执行一步，后续决策再提交新观测。记录频率10Hz不表示模型持续以10Hz更新动作；本次快速假服务测试约每两个决策执行一次结果。训练后的控制效果和推理延迟需一起评估。

服务metadata和检查点来源会记入回合；连接成功本身不表示策略已经训练或能够完成任务。

按小时服务器的安装、GPU检查、10步真实训练及重新加载模板在`docs/server_first_run.md`。参数已在本机解析核对，尚未验证新服务器安装或执行真实训练。

## 当前验证范围

历史检查通过核心/策略桥接测试、7项模型检查（分词器不跳过，含离线NPZ加载与保留新动作参数）、55项资源引用检查，并曾读取895帧旧数据、重新导出并读取60帧数据；这些旧数据现已不在工作区，报告仅记录当时的验证。所有模型模块路径均在本项目`src`。自带资产和独立编译插件已用规则控制器完成一次4米接近停车，60个决策帧，5.9秒成功。详见`reports/verification.md`与`reports/cleanup_and_policy_bridge.md`。

这些是源码、接口和采集链路验证。代码已提供采集→导出→训练→推理服务→仿真执行及验收的完整入口，但尚未用完整权重执行梯度训练，也没有训练好的无人艇VLA性能结果。本机6GB显存尚未验证能运行完整π0.5训练或推理；初始化权重已备齐，下一步是在合适服务器验证真实加载、梯度更新、保存和重载。位姿和速度目前是明确标注的仿真调试源，真实定位融合仍待实现。

上游模型来源为Physical-Intelligence/openpi commit `215abfb217dbac7d5f1273282331b9b1866c0479`及本地USV适配；许可证和来源散列在`licenses/`。仿真资源作者信息和来源在`assets/README.md`与manifest中保留。
