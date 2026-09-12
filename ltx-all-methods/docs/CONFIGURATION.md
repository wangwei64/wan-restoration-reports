# 统一配置与参数

## 路径和命令行

`configs/paths.example.json` 是路径模板；复制为 `configs/paths.local.json` 后填自己的目录，并在 run.py/benchmark.py 中传 `--paths`。环境变量优先于JSON。路径可包含空格，shell中使用引号。

| JSON键 | 环境变量 | 用途 |
|---|---|---|
| ltx_repo | LTX_REPO | 固定提交的外部 LTX 源码根目录 |
| checkpoint | LTX_CHECKPOINT | LTX 2B 0.9.6 safetensors 文件 |
| text_encoder | LTX_TEXT_ENCODER | 含 text_encoder/ 和 tokenizer/ 的父目录 |
| restorer（可选） | LTX_RESTORER | 自有恢复器；默认本包 weights/restorer.pt |

`run.py --method` 选择 methods.json 中一个入口；`--prompt`、`--seed`、`--out` 设置提示词、随机种子和输出目录。默认seed=20260912。`--config` 可指向完整生成配置副本，必须包含与 generation.json 相同的所有字段。`--describe` 不加载GPU；`--doctor --verify-weights` 检查环境、源码和全部模型。

## 所有方法共用 generation.json

| 字段 | 固定值 | 含义 |
|---|---|---|
| width / height | 1216 / 704 | 视频宽/高；须为32的倍数 |
| frames | 121 | 总帧数；须满足8n+1 |
| fps | 30 | 视频编码帧率 |
| steps | 50 | 去噪步数；缓存日程固定50步 |
| dynamic_flow | true | 保留动态流速设置 |
| flow_scale | 0.1 | 当前流速系数 |
| guidance_scale | 3.0 | CFG强度 |
| stg_scale | 1.0 | STG强度；当前适配器要求CFG和STG合批 |
| rescaling_scale | 0.7 | 引导重缩放 |
| skip_block_list | [19] | STG作用层列表（0基） |
| decode_timestep | 0.05 | VAE解码时间步 |
| decode_noise_scale | 0.025 | 解码噪声系数 |
| negative_prompt | worst quality, inconsistent motion, blurry, jittery, distorted | 固定负面提示词 |

配置变更会改变请求哈希，需另建输出目录；已有测试结果只对应这组参数。

## 方法独立参数 methods.json

我们的五档仅 `beta` 不同：0.1、1.0、2.0、3.0、4.0，直接写入 `controller.budget`。`controller_overrides={}` 表示不额外覆盖共同控制器。β控制恢复与稀疏细化的预算阈值，不是SSIM目标或固定缓存间隔；同一个β在不同模型/提示词上不保证相同速度或质量。

共同控制器在 `configs/base_profile.json`：warmup_ratio=0.2、tolerance=0.01、history_integration=native_euler；fused_context、fused_arithmetic、fused_rms、deduplicate、share_warmup、continuous_history、wan_evidence、rms_after_warmup_fast均为true。这些为已测试实现设置，本次不改变。`configs/calibration.json` 和恢复器权重保持固定，无需重新训练。

TeaCache threshold=0.03/0.05；NaviCache threshold=0.04/0.07；均沿用前10步和最后一步全量策略。SenCache threshold=0.5,K=2 / threshold=0.7,K=4，前10步和末步全量，灵敏度表为随包 `data/sensitivity_ltx.npz`。这些是本次横向比较沿用的档位，不代表跨方法质量已对齐。

ProfilingDiT唯一入口为 `profiling_official`。warmup_steps=6、refresh_interval=6，background_blocks=[4,7,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]；final_step_full=false。实际全量步（0基）为0–6、12、18、24、30、36、42、48。缓存代码直接来自固定上游；层比例和离线处理差异见 [PROFILING_PROTOCOL.md](PROFILING_PROTOCOL.md)。层表已冻结，在线运行无需 DINO/SAM 或再次 profiling。

## 批量参数与时间口径

`benchmark.py --methods all` 只包含本版13个入口。默认普通提示词5个seed、temporal_flickering 25个seed，每入口6220条，总80860条。`--samples`、`--flicker-samples` 用于显式改变采样量；`--limit` 限制本次分片执行条数，适合冒烟，不代表完整评估。`--shards N --shard K` 分片，K从0开始；分片总数、方法列表、基础seed和配置必须相同。不同机器使用各自GPU编号，合并相同相对输出目录后统一export。

```bash
# 两张卡/两台机器分别执行，各自只暴露一张GPU
CUDA_VISIBLE_DEVICES=0 python benchmark.py run --methods all --shards 2 --shard 0 --output outputs/full
CUDA_VISIBLE_DEVICES=0 python benchmark.py run --methods all --shards 2 --shard 1 --output outputs/full
```

计时包括文本编码、采样、VAE、CPU单线程x264 fast/CRF18视频编码，排除加载/校验/离线profiling/评分。每条run.json保存实际参数、种子、源码/权重哈希、硬件、耗时和轨迹；completed.json含请求与输出哈希。相同配置可断点续跑，发现同GPU其它计算进程会作废计时并等待；不同GPU任务不受影响。
