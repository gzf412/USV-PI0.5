# 按小时服务器的首次训练试跑

目标是用已有3回合、895帧验证完整权重加载、真实梯度训练、保存和重新加载。10步试跑不代表模型已学会驾驶。以下为服务器模板：CLI和数据接口在本机已有环境核查，新机器安装与GPU训练尚未验证，也没有显存或耗时承诺。

## 复制与安装

先把精简工程、已有LeRobot数据和完整初始化文件复制到服务器，以下命令假定布局为：

```text
/work/usv_vla_project/                         工程
/data/usv_actions/lerobot/local/usv_v1/        meta/、data/
/models/paligemma/pt_224.npz                   完整PaliGemma初始化权重
```

本机完整初始化权重已在`/home/gzf/robot_ws/models/paligemma/paligemma-3b-pt-224.npz`，官方SHA256与参数结构核验通过，报告见`reports/paligemma_initialization.json`。复制到服务器时可以重命名为模板中的`pt_224.npz`；格式仍须为JAX NPZ，不能用PyTorch权重或`pi05_base`替代。

这里只需动作数据；大禹和旧仿真图文不会混入本次训练。服务器需有可用NVIDIA GPU驱动、Git、uv和Python 3.11。

```bash
cd /work/usv_vla_project
uv sync --locked --python 3.11 --extra cuda
export USV_MODEL_PYTHON="$PWD/.venv/bin/python"
export USV_DATA_HOME=/data/usv_actions/lerobot
export USV_CACHE_DIR=/work/usv_cache
export HF_LEROBOT_HOME="$USV_DATA_HOME"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

安装需要网络；现有`uv.lock`已锁定195个包并通过离线一致性检查，首次安装仍须确认实际依赖安装与GPU检查成功。模型Python 3.11与Gazebo系统Python分开使用；仅训练的服务器无需安装Gazebo。

## 先验证环境与数据

```bash
nvidia-smi
JAX_PLATFORMS=cuda "$USV_MODEL_PYTHON" - <<'PY'
import sys
import jax
assert sys.version_info[:2] == (3, 11), sys.version
print("JAX", jax.__version__, "devices", jax.devices())
assert any(device.platform == "gpu" for device in jax.devices())
PY
bash scripts/model.sh check
bash scripts/model.sh batch-check --report outputs/checks/server_batch.json
bash scripts/model.sh inspect-weights /models/paligemma/pt_224.npz \
  --expected-bytes 11693352432 \
  --expected-sha256 eb6fa0d4ad1d551589cddd9d4ea18a9fa41bef3149fbcdccf8ef89e26adb46d3 \
  --check-model --report outputs/checks/server_initial_weights.json
bash scripts/model.sh train --help
```

数据检查应读取895帧、3回合，状态为`[2,33]`、动作为`[2,10,2]`。`check`和`batch-check`主动用CPU，不能代替上面的GPU检查。任一步失败先修复，再开始训练。

权重检查以流式SHA256和NPY头核对官方文件及实际模型的参数形状，不会分配完整模型；通过后仍需下面的真实加载与训练验证。

## 10步真实训练

选择单张卡，使用独立实验名；重复执行会产生新目录。没有指定覆盖或恢复已有实验。显存不足时应先评估机器与训练配置，不要把进程改成CPU继续等待。

```bash
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
usv_run_name="first_real_$(date -u +%Y%m%dT%H%M%SZ)"
export USV_RUN_NAME="$usv_run_name"
bash scripts/model.sh train \
  --exp-name "$USV_RUN_NAME" \
  --data.repo-id local/usv_v1 \
  --weight-loader.weights-path /models/paligemma/pt_224.npz \
  --checkpoint-base-dir /work/usv_checkpoints \
  --batch-size 1 --num-workers 0 --fsdp-devices 1 \
  --num-train-steps 10 --log-interval 1 --save-interval 5 \
  --no-wandb-enabled --no-overwrite --no-resume
```

这是当前配置的全参数训练，动作专家从新初始化。保留默认学习率与预热；10步只检查训练链路，不据此评价损失收敛或任务能力。

验收时保存完整日志：权重加载成功，`loss`与`grad_norm`有限，完成10次更新，程序正常等候检查点写完并退出。出现NaN、OOM或保存失败不算完成。当前保存目录按循环编号命名，10步成功结束后的最终目录为：

```bash
usv_checkpoint="/work/usv_checkpoints/pi05_usv/$USV_RUN_NAME/9"
test -d "$usv_checkpoint/params"
bash scripts/model.sh serve --checkpoint "$usv_checkpoint" --host 127.0.0.1 --port 8000
```

`9`是保存标签，训练状态已执行10次更新；不要填成`10/`或直接传`9/params/`。服务真正加载并启动后，还要发送一次真实观测，检查返回有限的`[10,2]`左右推力，再在独立Gazebo回合中验收。服务使用新检查点，不会重新读取上述PaliGemma初始化文件。

下一次正式训练应另建实验名，并按完整回合划分训练与测试数据。895帧只用于首轮流程验证，不能据此宣称无人艇VLA已训练成功。
