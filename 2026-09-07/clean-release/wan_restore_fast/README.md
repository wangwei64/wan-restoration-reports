# Wan Restore Fast · 整理版

这份文件夹只保留当前已验证最快的「先恢复，再自适应细化」路线：**β=2.0，τ=0.010**。输入一个 prompt 即可生成视频；主体、区域覆盖、逐 token 刷新时机、种子和输出目录均有自动处理。Wan 源码和权重放在包外。


## GitHub 源码版本

此目录上传了整理后的推理源码、配置、说明文档、VBench 样例和验证记录。
**模型二进制权重、生成 latent 和视频未放进这个源码目录。**

运行前，将完整交付包中的以下三个文件复制到本目录的 `weights/`，文件名保持不变：

```text
weights/restorer.safetensors
weights/features.safetensors
weights/subject.safetensors
```

完整交付包为 `wan_restore_fast_20260907.zip`；项目服务器上的原始文件在
`/root/autodl-tmp/wan_restore_fast/weights/`。这是现有交付位置，不是公共下载链接。
`weights/manifest.json` 提供哈希核对，配置和误差标定文件已经包含在本目录。
外部 Wan 的准备方式见安装与接口文档。补齐已有权重和外部 Wan 后，仍然只需输入 prompt 即可运行。

GitHub 中的 `code_manifest.json` 记录源码文件哈希，生产 Python 代码与已验证的最快版逐字节一致。
下方“目前服务器直接运行”针对服务器上已准备好权重的完整目录。

## 在目前服务器直接运行

```bash
cd /root/autodl-tmp/wan_restore_fast
./run.sh "A cute happy Corgi playing in park, sunset, zoom out"
```

不需要输入主体名称、YOLO 类别、mask、刷新间隔或阈值。默认 seed 是 `20260828`；相同 prompt 和 seed 可复现同一结果，换 seed 是可选操作。

脚本打印视频路径，结果保存为 `outputs/自动生成目录/final.mp4`。同目录还有 `run.json`（完整参数、时间、选区和刷新统计）与 `outputs.safetensors`（latent、置信度、区域权重及刷新记录）。已有输出不会被覆盖。

```bash
./run.sh --doctor
# 以下参数仅用于复现或自定义保存位置，普通生成不需要：
./run.sh "A person is riding a bike" --seed 20260828 --out outputs/my_first_run
```

## 在另一台机器复现

已验证环境为 Linux、Python 3.12、CUDA 12.4、PyTorch 2.5.1、FlashAttention 2.7.4.post1、Triton 3.1.0，GPU 为 A800 80GB。本包保留原有 FP32 权重及 BF16 autocast，未加入量化或 CPU offload；其他显存容量尚未验证。Windows 上建议使用 WSL2/Linux CUDA 环境运行，文件夹和 HTML 可直接在 Windows 阅读。

外部目录按下面放置，程序就会自动找到 Wan：

```text
your_workspace/
├── wan_restore_fast/            # 本源码目录，另补齐我们自己的约 292 MB 权重
├── Wan2.1/                     # 外部官方源码
└── models/Wan2.1-T2V-1.3B/      # 外部原版权重，含 T5、VAE、tokenizer
```

一次性安装依赖和准备外部模型见 [环境安装](docs/INSTALL.md)。安装完成后仍然只需传 prompt。已有 Wan 放在别处时，可在环境中设置 `WAN_REPO`、`WAN_CHECKPOINT`，也可使用同名 CLI 路径参数，不需修改任何源码。

## 保留的主结构

![整体流程](docs/method.svg)

1. 原版 Wan 完整运行前 10 步，取得第 10 步的 **clean prediction** `z10`，并保留真实含噪状态供后续积分使用。
2. 原有小网络直接从 `z10` 提取 `F`（46 通道特征）和 `H`（1 通道细节强度）；恢复网络输出恢复端点 `R`、置信度 `C` 和误差/风险预测。
3. 使用 prompt 的 Step10 注意力与已有 latent 主体头自动定位。按恢复误差、细节、变化量形成互斥的主体细化区、低置信度细化区、纯恢复区。
4. 在原来的 Step11–50 时间线上，每步用轻量张量运算判断刷新需求。选中的 token 执行 Wan 全部层；跳过 token 使用自身真实历史积分。纯恢复区从不再做 Wan 刷新。
5. 利用固定恢复端点构建 K/V 参照，融合最近实测残差。Step50 刷新所有细化候选 token，按互斥权重组装并执行一次最终 VAE 解码。没有额外尾部恢复步骤。

主体也可以跳步。`H` 不是置信度；`C` 由恢复网络对恢复误差的预测得到。误差与风险是学习得到的估计，不是严格概率或视觉误差上界。

## 文件结构

| 文件/目录 | 职责 |
|---|---|
| `run.py` / `run.sh` | 单 prompt 入口、自动环境与输出路径 |
| `restore_fast/pipeline.py` | 唯一主流程，包含对照用原版 GT50 接口 |
| `restore_fast/models/` | F/H 提取、基础与细节恢复、置信度、主体头 |
| `restore_fast/subject.py` / `routing.py` | prompt 自动对应与互斥区域划分 |
| `restore_fast/refresh.py` / `integrator.py` | 恢复相对刷新需求、逐 token UniPC 历史 |
| `restore_fast/wan/adapter.py` / `controller.py` | 外部 Wan 装载与可撤销的稀疏执行接口 |
| `restore_fast/kernels/` | 固定恢复参照 K/V、选中 token 算术融合 |
| `weights/` | 3 个 safetensors、模型配置、误差标定及来源哈希 |
| `benchmarks/` | 固定 VBench 样例、复测脚本、结果与 HTML 报告 |
| `tests/` | 核对脚本、历史数值回归证据、实测环境记录 |

清理掉了旧调度器的多层继承、固定间隔/分档策略、手动主体与在线检测入口、旧校准分支、训练脚本以及未参与生成的第二份 Diffusers DiT。恢复主干内未使用的旧质量头已删除，共 123,073 个参数；有效恢复网络及权重没有重新训练。

## 复测与阅读顺序

```bash
# 使用本次预先固定的三个 VBench prompt，逐一生成加速版和原版 GT50
../envs/wan-downscaler-full/bin/python benchmarks/run_vbench.py
# 对已生成结果计算指标、导出对比视频和 HTML
../envs/wan-downscaler-full/bin/python benchmarks/build_report.py
```

在自行安装的环境中用 `python` 替换上面的解释器路径即可。复测脚本自动跳过完整且匹配的结果；要重新计时，使用 `--output benchmarks/rerun_01`，报告脚本传入同一个 `--results` 路径。

先看 [在线测试报告](https://wangwei64.github.io/wan-restoration-reports/2026-09-07/clean-release/)，再看 [方法和网络说明](docs/METHOD.md)、[Wan 接口](docs/WAN_INTERFACE.md)、[复现与测试定义](docs/REPRODUCIBILITY.md)。当前报告使用 3 个 VBench prompt 样例，**不是完整 VBench 排名或总分**。

最快参数允许局部纹理、边缘和运动细节与 GT50 有差异。本次整理的目标是忠实保留已选最快版，不把代码清理描述成无损恢复或新的质量提升。
