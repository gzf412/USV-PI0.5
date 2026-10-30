# 采集回合的两路视频

`scripts/sim.sh record`默认记录艇载第一视角和真实Gazebo空中俯视相机。先停止推进器和Gazebo、写完原始数据，再编码视频。人工、规则控制器和策略控制器均使用同一流程。

每个回合的文件结构如下，目录名按需求使用`videa`：

```text
回合目录/
  images/                    艇载RGB原图，仍用于训练
  frames.jsonl               艇载图像时间索引
  aerial_images/             空中相机RGB原图，仅供回看
  aerial_frames.jsonl         空中图像时间索引
  steps.jsonl                状态、指令、动作与验收结果
  episode.json               相机位置、参数与回合配置
  summary.json               回合结果及videos导出状态
  videa/
    first_person.mp4         艇载第一视角
    aerial_global.mp4        空中全局俯视
    metadata.json            两路视频时间映射、分辨率和覆盖统计
```

空中相机为没有可见模型、没有碰撞的固定观察相机，分辨率1280×720；单回合默认10Hz，当前200条批次以5Hz采集空中原图、10Hz采集艇载原图。高度与位置根据艇起点、全部浮标、圆筒和任务区域计算并留边距。俯视相机不会被当作模型的第二或第三路输入；训练仍使用艇载前向图像。

两路MP4统一为H.264、1280×720、10帧/秒，按首个至最后一个决策的仿真时间播放。Gazebo加速运行不会使视频加速。每个输出时刻选取不晚于该时刻的最近相机原图，拒绝超过0.3秒的缺口。最后一个采样帧也显示0.1秒，因此5.9秒的决策跨度会生成6.0秒视频；确切时间、重复帧及原图映射记在`metadata.json`中。编码不会改变训练图像或动作记录。

新回合默认保存在本项目`outputs/episodes/本地时间戳/`，每个回合的两路视频仍放在它自己的`videa/`。不指定`--output`即可使用该默认位置，也可指定尚不存在的目录。正常采集示例：

```bash
bash scripts/sim.sh record --controller scripted --distance-m 4 \
  --output outputs/episodes/new_001
```

如果编码失败，原始数据保留，`summary.json`中的`videos.status`为`failed`且命令返回非零。修复编码器后可重新导出：

```bash
bash scripts/sim.sh video outputs/episodes/new_001
# 已存在的视频只有显式指定时才覆盖
bash scripts/sim.sh video outputs/episodes/new_001 --overwrite
```

历史回合没有采集空中相机原图，无法恢复当时的空中视频；可单独生成艇载视频：

```bash
bash scripts/sim.sh video /你的/历史回合 --allow-missing-aerial
```

`record --no-videos`会关闭空中相机及自动视频导出，保留原有前向原图与训练记录。

编码器按`--ffmpeg`/`USV_FFMPEG`、系统PATH、环境内`imageio_ffmpeg`的顺序查找。需要跨Python环境时设置`USV_VIDEO_PYTHON`或`USV_MODEL_PYTHON`。本机`configs/paths.local.env`已指向现有openpi环境，不需要另装编码器；迁移电脑时修改该本地文件或设置环境变量。路径配置可参照`configs/paths.env.example`，仿真仍由系统Python运行。

当前保留[重新采集的200个已验收回合](/home/gzf/robot_ws/usv_vla_project/outputs/episodes/red_round_200_20261027/README.md)，共400份两路视频、60,505个决策样本，仿真跨度合计6,030.5秒。单筒绕行的提前回正和连续绕行的逐门检查见[采集说明](batch_collection.md)及[复杂任务复查](../outputs/episodes/red_round_200_20261027/COMPLEX_REVIEW.md)。打开[批次manifest](/home/gzf/robot_ws/usv_vla_project/outputs/episodes/red_round_200_20261027/manifest.json)，按状态为`accepted`的条目中`accepted_episode`定位回合；该路径相对于批次根目录，两路视频分别位于其下的`videa/first_person.mp4`和`videa/aerial_global.mp4`。

例如[左绕圆筒的空中视频](/home/gzf/robot_ws/usv_vla_project/outputs/episodes/red_round_200_20261027/episode_0003_around_left/attempt_01/videa/aerial_global.mp4)与[第一视角](/home/gzf/robot_ws/usv_vla_project/outputs/episodes/red_round_200_20261027/episode_0003_around_left/attempt_01/videa/first_person.mp4)。20条沿岸任务重新采集，回看示例及实测见[沿岸复查](../outputs/episodes/red_round_200_20261027/SHORE_REVIEW.md)。整批通过任务、平顺性和提前回正检查后，旧批次、旧沿岸回合、速度试验以及本批失败尝试的原始图片与视频已删除，只归档小型元数据供追溯；后续回看和训练均以当前manifest及有效清单为准。
