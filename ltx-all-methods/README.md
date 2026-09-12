# LTX acceleration benchmark — fixed beta release

用于已安装 **LTX-Video 2B 0.9.6** 的 Linux/CUDA 服务器，统一生成、测速、断点续跑与 VBench 导出。本目录包含加速器、我们的恢复器加载代码、固定配置和运行工具；外部 LTX 源码、基础模型及文本编码器另行安装。自有恢复器权重由本次 Release 提供并自动校验。

**只保留以下 13 个入口。** 我们的方法直接使用 LTX β，不做 WAN 质量或速度映射。所有方法的流速与生成设置维持已验证配置。

| 方法 | 运行入口 | 固定参数 |
|---|---|---|
| 原版对照 | `native` | 50 步全量 |
| TeaCache | `teacache_slow` / `teacache_fast` | threshold=0.03 / 0.05 |
| NaviCache | `navicache_slow` / `navicache_fast` | threshold=0.04 / 0.07 |
| SenCache | `sencache_slow` / `sencache_fast` | ε=0.5,K=2 / ε=0.7,K=4 |
| ProfilingDiT | `profiling_official` | 上游默认 step_start=6、step_interval=6；缓存20/28层；末步不强制全量 |
| 我们的方法 | `ours_beta_0p1` | β=0.1 |
| 我们的方法 | `ours_beta_1p0` | β=1.0 |
| 我们的方法 | `ours_beta_2p0` | β=2.0 |
| 我们的方法 | `ours_beta_3p0` | β=3.0 |
| 我们的方法 | `ours_beta_4p0` | β=4.0 |

## 安装后快速运行

先按 [安装说明](docs/INSTALL.md) 准备 Python/CUDA、固定外部 LTX 版本、模型和依赖，再进入本目录：

```bash
export LTX_REPO="/your/path/LTX-Video"
export LTX_CHECKPOINT="/your/models/ltxv-2b-0.9.6-dev-04-25.safetensors"
export LTX_TEXT_ENCODER="/your/models/PixArt-XL-2-1024-MS"
python download_weights.py
python run.py --doctor --verify-weights
CUDA_VISIBLE_DEVICES=0 python run.py --method ours_beta_1p0 \
  --prompt "A small boat drifts slowly on calm water." --seed 20260912 --out outputs/boat
```

输出为 `outputs/boat/video.mp4`。仅查看参数可运行 `python run.py --describe`。也可通过 `--paths configs/paths.local.json` 指定自己的路径；见 [配置和参数说明](docs/CONFIGURATION.md)。安装后不需要旧服务器目录、训练数据或再次离线 profiling。

## 全部方法的 VBench

```bash
# 检查默认工作量：13 个配置 × 6,220 = 80,860 条
python benchmark.py plan --methods all
# 先做 13 条冒烟测试：第一条提示词的一个 seed，覆盖每个入口
CUDA_VISIBLE_DEVICES=0 python benchmark.py run --methods all --samples 1 --flicker-samples 1 \
  --limit 13 --output outputs/smoke
# 确认环境后运行完整采样，必须另用目录
CUDA_VISIBLE_DEVICES=0 python benchmark.py run --methods all --output outputs/full
python benchmark.py export --methods all --output outputs/full --export-dir outputs/vbench_standard
python benchmark.py summarize --methods all --output outputs/full
```

支持 `--methods` 子集、分卡/跨机器分片和断点续跑。完整规则、官方 16 维评估与 SSIM/LPIPS 配对评分见 [BENCHMARK.md](docs/BENCHMARK.md)。本次发布验证不是完整 VBench。

## 参数依据与结果

ProfilingDiT 只有一个配置：直接调用固定上游提交中**未修改的 Wan 缓存实现**，将28/40层比例映射为LTX的20/28层，采用上游默认固定6步间隔。该移植没有按本次速度或质量结果选档。上游没有 LTX 实现；离线前景使用 RGB 分割，与论文流程有差异。[完整协议和来源](docs/PROFILING_PROTOCOL.md) 保留这些差异，不能称为作者官方 LTX 实现。

[固定 β 测速质量与点线图](docs/RESULTS.md) 提供五档4条视频结果及全方法单样本比较。SSIM是对原版的相似度，不是官方VBench分数。我们的恢复器训练提示词覆盖 VBench；ProfilingDiT 离线统计也使用 VBench，因此应披露训练/校准提示词重合。其它方法保持先前适配和配置，具体来源及许可证见 [THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md)。
