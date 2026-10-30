# 无人艇 GPS 状态（35 维）

当前 GPS 配置 `pi05_usv_gps35` 在原有 33 维后增加两维：无人艇当前 GPS 纬度、经度，单位为 WGS84 度。没有 GPS 目标，也没有目标有效标记；动作仍是左右两路连续推力。当前 GPS 数据源是仿真 GPS，船的其他状态仍包含仿真器位姿；实艇定位尚未实现。

GPS 经纬度在原始记录中保留双精度。为避免 LeRobot 默认把浮点序列转换成 float32 导致精度损失，训练数据将原 33 维存为 `observation.state`，另将纬度、经度以 1e-7 度为单位存入两个整数 `observation.gps_e7`。数据适配器还原为 35 维，再把 GPS 以七位小数的数字文本编码进 pi0.5 的语言前缀，原 33 维仍使用已有离散编码。七位小数是输入表示精度，不代表 GPS 传感器或控制精度。缺失 GPS 时拒绝构造状态，不使用虚假的零坐标。

历史数据使用 `--state-version 3` 导出；导出器按每个决策记录中的 GPS 来源和到达时间，匹配 `raw_sensors.jsonl` 中相同的测量，不改写原始数据。新采集会保留旧 `state` 并额外保存 35 维 `state_gps_v3`。旧 33 维、之前的 39 维版本仅为兼容而保留，其检查点和导出数据不能与新版本混用。

```bash
bash scripts/model.sh export --state-version 3 --repo-id local/usv_v3 \
  --output /path/to/new-dataset /path/to/episode
USV_TRAIN_CONFIG=pi05_usv_gps35 bash scripts/model.sh train --exp-name gps_experiment
bash scripts/model.sh serve --config pi05_usv_gps35 --checkpoint /path/to/checkpoint
```

GPS 本身不包含目的地。目前没有 GPS 目的地任务的输入、示范数据或成功判定。未来要完成此类任务，需要再设计如何向系统提供目标，并采集对应示范。真实 GPS 丢失、延迟和融合策略也尚未实现。
