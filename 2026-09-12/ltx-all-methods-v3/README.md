# LTX All Methods

统一运行 LTX-Video 2B 0.9.6 原版、TeaCache、NaviCache、SenCache、ProfilingDiT，以及本项目的恢复与自适应细化方法。官方 LTX 源码、基础权重和文本编码器放在包外；本项目恢复器通过 GitHub Release 下载。

本版本用于正式横向比较，仅提供 `profiling_official` 一个ProfilingDiT入口。直接调用固定提交的未修改官方Wan缓存代码，固定6步预热、每6步刷新，缓存比例28/40映射为LTX20/28层。不按速度或SSIM/LPIPS选档，不追求恰好2×。其它10档与生成配置沿用已验证版本。这是官方代码的LTX适配，不是作者提供的LTX实现，也不是论文表2的严格复现；详细差异见docs/PROFILING_PROTOCOL.md。

## 准备环境与单条运行

按 [INSTALL.md](docs/INSTALL.md) 安装固定版本的外部 LTX 和依赖，设置 `LTX_REPO`、`LTX_CHECKPOINT`、`LTX_TEXT_ENCODER`，然后：

```bash
python download_weights.py
python run.py --doctor --verify-weights
CUDA_VISIBLE_DEVICES=0 python run.py --method ours_fast \
  --prompt 'A small boat drifts slowly on calm water.' --seed 20260912 --out outputs/boat
```

也可复制 `configs/paths.example.json` 为自己的路径配置，通过 `--paths` 使用。所有运行共用 `configs/generation.json`；方法独立参数在 `configs/methods.json`。`python run.py --describe` 无需 GPU 即可查看全部参数。

| 方法 | 入口名称 | 参数含义 |
|---|---|---|
| 原版 | `native` | 外部 LTX 完整 50 步 |
| TeaCache | `teacache_slow`, `teacache_fast` | 阈值 0.03 / 0.05 |
| NaviCache | `navicache_slow`, `navicache_fast` | 阈值 0.04 / 0.07 |
| SenCache | `sencache_slow`, `sencache_fast` | ε=0.5,K=2 / ε=0.7,K=4 |
| ProfilingDiT | `profiling_official` | 唯一版本：官方代码默认6/6节奏，缓存20/28层，无速度或质量选档 |
| 我们的方法 | `ours_quality`, `ours_balanced`, `ours_fast` | 实际 LTX β=0.262 / 1.732 / 3.184 |

我们的三个名字对应此前匹配 WAN 质量的档位；实际 β 不是 WAN 的 0.1 / 1 / 2。代码加载既有 LTX 恢复器，不重新训练。快慢档表示方法自己的设置，不表示各方法生成质量已经完全相同。

## 完整 VBench

```bash
# 先查看工作量，不生成视频
python benchmark.py plan --methods all
# 单 GPU，自动核验并跳过已完成且配置一致的输出
CUDA_VISIBLE_DEVICES=0 python benchmark.py run --methods all --output outputs/full
# 全部生成完后，整理官方命名的目录
python benchmark.py export --methods all --output outputs/full --export-dir outputs/vbench_standard
python benchmark.py summarize --methods all --output outputs/full
```

默认覆盖官方 946 条元数据记录，完全重复的文本合并为 944 条，保留原始记录编号和所有维度。普通提示词生成 5 个种子；75 条 temporal_flickering 提示词生成 25 个种子供官方静态筛选。每个方法 6,220 条视频，11 个方法/档位共 68,420 条。种子根据基础 seed、提示词和重复编号确定，与方法、执行顺序及机器分片无关。

只想每条提示词先跑一条，显式添加 `--samples 1 --flicker-samples 1`；这覆盖全部提示词，但不满足官方完整采样要求。快速验证入口可用 `--limit 2` 和单独的 `--output outputs/smoke`，不要与完整输出目录混用。

两张卡或两台机器分别运行：

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark.py run --methods all --shards 2 --shard 0 --output outputs/full
CUDA_VISIBLE_DEVICES=1 python benchmark.py run --methods all --shards 2 --shard 1 --output outputs/full
```

不同机器各用本地 GPU 编号，生成后按相同相对目录合并输出，再从一台机器执行 export。两台机器必须使用同一发布代码、配置、基础 seed 和分片总数。每个进程只允许一张可见 GPU；测速仅监测该卡，不会因为其他 GPU 的工作而停止。发现同卡并发占用会作废该次计时并等待后重试。

完整评测的官方安装、16 个维度、静态筛选和相似度对比见 [BENCHMARK.md](docs/BENCHMARK.md)。`run.json` 保留耗时、完整配置、种子、缓存/细化轨迹和哈希；`completed.json` 是断点完成标记。批量默认仅保存视频和元数据，使用 `--keep-latents` 才保留最终潜变量。已有不同配置或损坏的完成记录会报错，未完成输出移到同目录下的 `partial_runs` 留档。

## 方法与结果边界

NaviCache 是从官方 Wan 算法移植到 LTX；SenCache 迁移官方 LTX 0.9.1 灵敏度到当前 0.9.6；ProfilingDiT 直接复用固定官方缓存代码并适配 LTX，离线前景使用 Grounding DINO + SAM2 RGB 分割，与原论文 PCA 流程不同。运行固定档位无需再次分割或 profiling。来源及许可证见 [THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md)。

我们的恢复器已经训练过全部这批 VBench 短序列提示词，故全套 VBench 不能标为该恢复器的未见提示词测试集。我们的方法也包含自己的融合算子与 STG 前缀复用；其他缓存方法没有套用这些优化。应报告整套方法的速度与质量，不能将所有差异只归因于缓存。当前 8 条对照的 SSIM/LPIPS 是相对原版相似度，不是 VBench 官方总分。

[速度与质量报告](docs/RESULTS.md)包含当前官方代码适配版的同轮11档比较图和固定8条逐视频结果。协议、来源与离线差异见[ProfilingDiT说明](docs/PROFILING_PROTOCOL.md)。

本发布包只提供一个 ProfilingDiT 版本：`profiling_official`。其它10个原版/方法档位保持原配置。验证/测试提示词与旧实验相同，不能声称首次未见；我们的方法训练提示词覆盖VBench。
