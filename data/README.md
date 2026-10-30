# 数据说明

当前原始动作数据在 [red_round_200_20261027](../outputs/episodes/red_round_200_20261027/README.md)，共200条完成任务及独立平顺性验收的回合、52,190个决策和400份回看视频。有效回合路径以该批次的 `splits/accepted_episodes.txt` 为准；训练/验证清单为 `train_episodes.txt` 和 `validation_episodes.txt`（160/40回合），路径相对于批根目录。不要将已替换或失败尝试混入训练。

其中20条沿岸回合在2026-10-28重新采集，详细路程、实际岸距及视频见[沿岸实测](../outputs/episodes/red_round_200_20261027/SHORE_REVIEW.md)。数据为Gazebo中规则控制器驾驶的仿真示范，位姿使用仿真调试真值。

当前回合尚未导出为LeRobot。按有效清单导出，示例见[项目说明](../README.md)；旧大禹、旧仿真图文、旧动作数据及其895帧LeRobot导出不再位于工作区。
