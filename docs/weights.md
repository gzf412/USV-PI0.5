# 准备PaliGemma初始化权重

当前配置使用第一代PaliGemma的3B、224分辨率、JAX NPZ，动作专家新初始化。完整文件已下载并归档至`/home/gzf/robot_ws/models/paligemma/paligemma-3b-pt-224.npz`，大小11693352432字节（约11.7GB）。官方SHA256完全匹配，32个主干参数张量与当前配置形状匹配，19个动作相关张量按计划新初始化。核验报告见`../reports/paligemma_initialization.json`；这不是完整模型数值运行或训练结果。

官方入口：[google/paligemma-3b-pt-224-jax](https://huggingface.co/google/paligemma-3b-pt-224-jax)。Google要求登录并确认模型条款。请本人在网页完成确认，然后在本机终端登录；不要在聊天中发送密码或Token。参考[Google官方说明](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md)。

2026-10-26检查：上游代码的GCS直链返回403、元数据API返回401。随后通过浏览器完成官方授权、下载和核验。本机命令行环境没有保存Hugging Face登录凭据；已有文件可直接离线使用，以下命令用于其他机器重新下载。

官方公开元数据记录的文件大小为`11693352432`字节，SHA256为`eb6fa0d4ad1d551589cddd9d4ea18a9fa41bef3149fbcdccf8ef89e26adb46d3`，仓库revision为`b9d98e0ad9b7e5119a6344b3451555dce1e6ab8b`。来源记录在`/home/gzf/robot_ws/models/paligemma/source.json`。

本机已有模型环境提供`huggingface-cli`。授权后在终端执行：

```bash
/home/gzf/miniconda3/envs/openpi/bin/huggingface-cli login
/home/gzf/miniconda3/envs/openpi/bin/huggingface-cli download \
  google/paligemma-3b-pt-224-jax paligemma-3b-pt-224.npz \
  --local-dir /home/gzf/robot_ws/models/paligemma
```

这是命令行重下模板；本机现有权重由已授权浏览器取得。下载后保留官方文件名也可以，训练直接指定该文件的绝对路径：

```bash
USV_MODEL_PYTHON=/home/gzf/miniconda3/envs/openpi/bin/python \
USV_DATA_HOME=/home/gzf/robot_ws/data/usv_actions/lerobot \
bash /home/gzf/robot_ws/usv_vla_project/scripts/model.sh train \
  --exp-name pilot \
  --weight-loader.weights-path /home/gzf/robot_ws/models/paligemma/paligemma-3b-pt-224.npz
```

以上训练参数示意路径，不建议在本机6GB显存直接启动全参数训练。实际服务器试跑见[server_first_run.md](server_first_run.md)，该模板将文件重命名为`pt_224.npz`，复制时统一文件名和参数即可。

拿到文件后先核对官方摘要、文件大小及参数结构：

```bash
USV_MODEL_PYTHON=/home/gzf/miniconda3/envs/openpi/bin/python \
bash /home/gzf/robot_ws/usv_vla_project/scripts/model.sh inspect-weights \
  /home/gzf/robot_ws/models/paligemma/paligemma-3b-pt-224.npz \
  --expected-bytes 11693352432 \
  --expected-sha256 eb6fa0d4ad1d551589cddd9d4ea18a9fa41bef3149fbcdccf8ef89e26adb46d3 \
  --check-model
```

检查器只读取参数头，并流式计算文件摘要；模型结构通过CPU抽象执行生成，不分配完整模型参数。摘要与官方一致才记录为官方完整性校验通过；结构匹配只说明能够复用参数，不代表已训练或能够驾驶。

格式必须是JAX NPZ；PyTorch safetensors、PaliGemma 2和官方π0.5检查点都不是当前加载器的直接输入。
