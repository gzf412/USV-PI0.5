# 模型子集与依赖交接

这是无人艇专用的 openpi JAX 子集，源自 Physical-Intelligence/openpi commit `215abfb217dbac7d5f1273282331b9b1866c0479` 的本地已适配版本。源文件追踪与 SHA256 见 `licenses/OPENPI_SOURCE_MANIFEST.json`。Openpi 与 Gemma 许可分别在 `licenses/OPENPI_LICENSE`、`licenses/LICENSE_GEMMA.txt`，Gemma、SigLIP 文件原有版权头保留。

## 保留

- `src/openpi/models`：PaliGemma 所需 Gemma、SigLIP、LoRA 支持、π0/π0.5 flow action expert、配置和本地 SentencePiece 分词器。
- `src/openpi/policies`：USV 33维状态/两路推力适配器、JAX 推理包装及训练检查点加载。
- `src/openpi/training`：唯一 `pi05_usv` 配置、LeRobot 数据读取、分片、优化器、权重加载、检查点及训练状态。
- `src/openpi/shared`：类型、图像、归一化、下载与 NNX 工具。
- `src/openpi/serving` 与 `src/openpi_client`：WebSocket 推理协议、Numpy 消息封装、图像工具及基础策略接口。
- `scripts/train.py`：官方真实 JAX 训练循环；直接执行会把本项目 `src` 插入搜索路径首位。不是空壳训练脚本。

## 删除

机械臂/ALOHA/DROID/LIBERO 适配器和配置、RLDS/TensorFlow 数据链、FAST/FSQ/自动回归动作模型及分词器、PyTorch 模型和 transformers 替换目录、样例、Docker、无关配置及上游测试集合。PyTorch 仍为 LeRobot 数据读取依赖，保留的模型训练/推理实现均为 JAX。

## 固定模型约定

唯一训练配置 `pi05_usv`：π0.5、state_dim=33、action_dim=2、action_horizon=10、max_token_len=384、strict_tokenization=True、固定尺度的 USV 输入/输出归一化。仍使用官方 `PaliGemmaWeightLoader` 初始化通用 PaliGemma，动作专家新初始化；没有切换为 `pi05_base`，也没有在本次整理中加载大权重、训练或下载。

分词器默认只读本项目 `assets/tokenizer/paligemma_tokenizer.model`；可用 `USV_TOKENIZER_PATH` 指定另一已有文件。不存在会明确报错，不回落旧工程缓存、不自动下载。该4MB分词器与模型权重分开。

## 依赖

根目录 pyproject 由主代理整合。模型直接使用：JAX/JAXLIB 0.5.3、Flax 0.10.2、Orbax-checkpoint 0.11.13、Optax 0.2.4、Numpy<2、PyTorch 2.7.1、旧版 LeRobot（git revision `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`）、etils、augmax、einops、jaxtyping 0.2.36、sentencepiece、Pillow、fsspec/gcsfs、tqdm-loggable、typing-extensions、tyro、wandb、filelock、beartype 0.19.0、chex 0.1.90、treescope、websockets>=14、msgpack。LeRobot 的图像/视频与数据依赖由其自身安装。没有保留顶层 gym-aloha、transformers 替换或 TensorFlow/RLDS 安装要求。

CPU 检查可用已存在 `/home/gzf/miniconda3/envs/openpi/bin/python`；服务器训练需根据 GPU/CUDA 安装 JAX 对应插件。不要因 CPU 检查通过而把训练也强制设置为 CPU。

## 验证

1. 独立导入10个关键模块；所有 `__file__` 在新目录 `src`，未导入旧工程 openpi，也未导入 models_pytorch。
2. `scripts/train.py pi05_usv --help` 成功，只显示无人艇配置。
3. 最终新版6项适配检查全部通过且0跳过，使用本项目分词器验证33个状态、384上限和16token超长拒绝。
4. Full-size π0.5 抽象 loss/sampling 检查通过，状态输入(1,33)、动作输入输出(1,10,2)，不实际分配/加载3B权重。
5. 从外置旧 LeRobot 数据读取3回合895帧，检查6个回合边界，token长度194～197；实际批次state=(2,33)、action=(2,10,2)，3路224×224图像仅前向mask=true。所有 openpi 模块仍来自新项目，HF缓存指向新项目outputs/cache/hf。

限制：以上是整理后的接口、抽象计算和数据通路验证，不是实际参数更新、预训练权重推理或 VLA 闭环成功证明。
