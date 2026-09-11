# LTX All Methods

统一运行 LTX-Video 2B 0.9.6 原版、TeaCache、NaviCache、SenCache、ProfilingDiT，以及本项目的恢复与自适应细化方法。官方 LTX 源码、基础权重和文本编码器放在包外；本项目恢复器通过 GitHub Release 下载。

ProfilingDiT v2 已按独立验证集选定快慢档。全部 14 个方法/档位已逐项实际生成，潜变量、采样时间表、原始 RGB 与冻结实现重放一致，同一固定编码下 MP4 也逐位一致；原版在所有方法切换后再次生成一致。发布包采用独立 CPU 编码进程，固定单线程、帧时间戳和颜色转换，保留 x264 fast / CRF18，修复了生成进程内对同一 RGB 编码不稳定的问题。历史 MP4 不作为字节一致的目标，详见 tests/gpu_smoke.json 和 tests/encoder_fixture.json。

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
| ProfilingDiT 旧统计 | `profiling_v1_slow`, `profiling_v1_fast` | 六个校准样本的固定层与刷新日程 |
| ProfilingDiT VBench 统计 | `profiling_v2_slow`, `profiling_v2_fast` | 慢档缓存层15/16，快档15/16/17/19；刷新间隔2→1 |
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

默认覆盖官方 946 条元数据记录，完全重复的文本合并为 944 条，保留原始记录编号和所有维度。普通提示词生成 5 个种子；75 条 temporal_flickering 提示词生成 25 个种子供官方静态筛选。每个方法 6,220 条视频，14 个方法/档位共 87,080 条。种子根据基础 seed、提示词和重复编号确定，与方法、执行顺序及机器分片无关。

只想每条提示词先跑一条，显式添加 `--samples 1 --flicker-samples 1`；这覆盖全部提示词，但不满足官方完整采样要求。快速验证入口可用 `--limit 2` 和单独的 `--output outputs/smoke`，不要与完整输出目录混用。

两张卡或两台机器分别运行：

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark.py run --methods all --shards 2 --shard 0 --output outputs/full
CUDA_VISIBLE_DEVICES=1 python benchmark.py run --methods all --shards 2 --shard 1 --output outputs/full
```

不同机器各用本地 GPU 编号，生成后按相同相对目录合并输出，再从一台机器执行 export。两台机器必须使用同一发布代码、配置、基础 seed 和分片总数。每个进程只允许一张可见 GPU；测速仅监测该卡，不会因为其他 GPU 的工作而停止。发现同卡并发占用会作废该次计时并等待后重试。

完整评测的官方安装、16 个维度、静态筛选和相似度对比见 [BENCHMARK.md](docs/BENCHMARK.md)。`run.json` 保留耗时、完整配置、种子、缓存/细化轨迹和哈希；`completed.json` 是断点完成标记。批量默认仅保存视频和元数据，使用 `--keep-latents` 才保留最终潜变量。已有不同配置或损坏的完成记录会报错，未完成输出移到同目录下的 `partial_runs` 留档。

## 方法与结果边界

NaviCache 是从官方 Wan 算法移植到 LTX；SenCache 迁移官方 LTX 0.9.1 灵敏度到当前 0.9.6；ProfilingDiT 是论文方法的 LTX 适配，离线前景使用 Grounding DINO + SAM2 RGB 分割，与原论文 PCA 流程不同。运行固定档位无需再次分割或 profiling。来源及许可证见 [THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md)。

我们的恢复器已经训练过全部这批 VBench 短序列提示词，故全套 VBench 不能标为该恢复器的未见提示词测试集。我们的方法也包含自己的融合算子与 STG 前缀复用；其他缓存方法没有套用这些优化。应报告整套方法的速度与质量，不能将所有差异只归因于缓存。当前 8 条对照的 SSIM/LPIPS 是相对原版相似度，不是 VBench 官方总分。

[已完成的速度与质量参考](docs/RESULTS.md)包含原有与新增独立样本的分组结果、我们三档的历史数据及统计边界。
