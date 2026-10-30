# GPS 状态 v2（39 维）

旧版 v1 仍为 33 维，输出左右两路连续推力。GPS 版使用训练配置 `pi05_usv_gps`，动作仍为 2 维。两版的检查点和导出数据不能混用。

39 维前 33 维与旧版相同；最后 6 维依次为：相对于本回合首个有效 GPS 测量位置的东向、北向位置（米），GPS 有效标记，GPS 目标相对于船头的前向、左向位置（米），目标有效标记。原始 GPS 经纬度仍保存在 `raw_sensors.jsonl`，不直接转为 float32。位置计算使用 WGS84 局部切平面近似，适用于当前短距离仿真，不作为长距离大地测量实现。

GPS 目标距离上限为 150 米。目标相对位置采用带符号对数压缩，靠近 0 时档距约 0.14 米，远处更粗；编码精度不代表 GPS 测量或控制精度。船相对本回合起点的位置支持东向和北向各 ±256 米，超范围报错。目标和定位均有有效标记，不存在时数值填零。方位角使用同一 ENU 坐标系下的船体航向；当前仿真使用仿真器位姿，并非真实 GNSS/IMU 融合。

新回合的 `steps.jsonl` 同时保存旧 `state` 和 `state_gps_v2`，不更改旧版验收字段。旧记录可通过 `--state-version 2` 导出为 GPS 版：导出器使用每条决策记录中的 GPS 来源时间及到达时间，匹配 `raw_sensors.jsonl` 中完全相同的 GPS 测量。缺少测量会报错，原数据不改写。旧数据没有 GPS 目的地，目标标记为 0，不能用来证明学会到 GPS 目的地。

```bash
# 在项目目录中运行；输出目录必须不存在。
bash scripts/model.sh export --state-version 2 --repo-id local/usv_v2 \
  --output /path/to/new-dataset /path/to/episode

# 将 USV_DATA_HOME 指向包含导出数据的 LeRobot 根目录；路径和训练参数按实际修改。
USV_TRAIN_CONFIG=pi05_usv_gps bash scripts/model.sh train --exp-name gps_experiment

# 训练完成后显式选择相同配置。
bash scripts/model.sh serve --config pi05_usv_gps --checkpoint /path/to/checkpoint
```

推理桥可根据模型元数据接受 33 或 39 维，现有浮标仿真推理会向 GPS 版模型提供实时 GPS 位置，但 GPS 目标标记仍为 0。目前尚无 GPS 目的地任务的场景生成、语言目标输入、成功判定、专家示范及训练数据；`extend_state` 已提供目标经纬度和 ENU 航向的转换接口，不代表已能自主导航到 GPS 目标。模拟 GPS 数据有有效性和延迟检查，真实定位融合、GNSS 失效处理与实艇标定尚未实现。
