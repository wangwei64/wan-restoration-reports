# 生成、测速与官方质量评估

生成参数集中在 `configs/generation.json`：1216×704、121 帧、30 fps、50 步、dynamic flow 0.1、CFG 3、STG 1、rescaling 0.7、AttentionValues / block 19、decode timestep 0.05、noise 0.025。所有方法共用相同提示词、种子、初始噪声与解码设置。变更参数时应另建输出目录；冻结的快慢档质量证据仅适用于已验证配置。

`benchmark.py plan` 只显示工作量。完整采样每方法 6,220 条视频；每条普通提示词 5 个 seed，temporal_flickering 25 个 seed，完全重复文本共享结果并保留所有官方元数据维度。[官方采样规则](https://github.com/Vchitect/VBench/tree/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/prompts)。基础 seed 固定为 20260912，可通过 `--seed` 统一修改；记录实际 31 位派生种子。

输出先采用不受长文件名影响的 `method/vbench_id/sample_index/`，每条包含 `video.mp4`、`run.json`、`completed.json` 和步数状态。`benchmark.py export` 在全量完整后转为官方要求的 `prompt-index.mp4`；硬链接可用时不额外复制视频。946 条官方记录合并为 944 条完全相同文本分组不会丢弃维度。

## 官方 VBench 16 维评估

评估工具放在包外，建议另建评估环境。按 [VBench 官方安装说明](https://github.com/Vchitect/VBench/tree/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490) 安装依赖和视觉评估权重，固定提交 `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`。这些评估模型不是 LTX 恢复器权重，不包含在本 Release。

```bash
git clone https://github.com/Vchitect/VBench.git /your/path/VBench
git -C /your/path/VBench checkout fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490
# 在该工具独立的 Python 环境安装其官方依赖并准备 pretrained 模型。
python evaluate_vbench.py --vbench-repo /your/path/VBench \
  --python /your/vbench-env/bin/python --videos outputs/vbench_standard \
  --output outputs/vbench_scores --dry-run
# 查看将执行的命令后，去掉 --dry-run 执行全部方法和维度。
```

包装入口调用官方 `static_filter.py` 筛选 temporal_flickering，再使用筛选后的视频目录评估该维度。其他 15 维直接使用原始导出目录。若某条提示词未得到 5 个合格静态样本，会停止并保存不足情况，不能将有缺失的结果称作完整官方得分。官方筛选会保留筛选通过的静态视频，并重编号为 0–4。

`--methods native teacache_fast ours_beta_4p0` 可选择子集。`--dimensions` 可选择维度。每个方法、维度保存独立日志与完成记录；失败不写成功标记。评估所用 Python 环境与生成环境可以完全分开。

## 相对原版的 SSIM / LPIPS

```bash
CUDA_VISIBLE_DEVICES=0 python score_pairs.py --results outputs/full \
  --method teacache_fast --output outputs/pair_metrics/teacache_fast.json
```

读取同一个 id / sample_index 的 native 视频，以全部 121 帧评分：SSIM 使用 832×480 area resize 和 FP64，LPIPS-Alex 使用 416×240 area resize。这与当前横向对照的定义一致。LPIPS 首次使用遵循 torchvision 的模型缓存/下载规则，可通过 `TORCH_HOME` 指定缓存。独立评分不计入生成速度。

相对原版相似度与官方 VBench 16 维分数分开报告。我们的恢复器使用过完整 VBench 短序列提示词，必须披露训练与评估提示词重合；新增 seed 或更长视频不会自动使这些提示词变成未见训练提示词。ProfilingDiT 也使用 VBench 提示词作离线层统计，应披露其校准数据。

## 时间口径与复现证据

在线耗时包含文本编码、采样、缓存/恢复、解码与视频编码；排除装载、模型校验、离线 profiling、潜变量保存、评分。代码在计时前后及期间每秒检查同卡 GPU 进程；发现争用后保留无效输出，下次自动重跑该条。

GPU 型号、UUID、软件版本、源码/权重哈希、全部请求配置与方法配置保存为运行元数据。原版和缓存方法未启用我们的方法专有融合；`ours_*` 使用发布配置中的全部优化。建议在同型号 GPU 上比较配对种子的均值，并报告视频数与各方法失败/未完成条数。`timing_summary.json` 在未完成时会明确标记 complete=false。


## 固定视频编码

本发布统一在独立 CPU 进程中采用 x264 fast、CRF18、单线程编码（含 lookahead，关闭 sliced threads），明确指定每帧时间戳及 BT.601 全范围RGB到限幅YUV转换。编码采用已验证的固定隔离进程实现。生成参数和方法档位保持一致；`run.json` 保存原始 RGB 哈希与编码参数。在线时间包含传输、进程启动及固定编码成本，不能直接把旧编码的历史时间当作本版本速度。历史数据与本版实际核验见 [RESULTS.md](RESULTS.md)。
