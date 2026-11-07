# SenCache：LTX 0.9.6 / 8 条 MixKit 校准迁移包

本包用于在另一台 Linux/CUDA 服务器上复用已完成的 LTX 0.9.6 灵敏度校准，执行单条生成或完整 VBench 测试。无需在测试服务器重新生成校准视频。它是本项目的 native LTX 适配，不是 SenCache 作者发布的 0.9.6 校准。

默认保持本项目论文的生成设置：CFG **3.0**、STG **1.0**、guidance rescaling **0.7**、STG block **19**（从零开始），1216×704、121 帧、50 步、30 fps、dynamic flow=true / flow scale=0.1，VAE decode timestep=0.05 / noise=0.025。完整配置在 `configs/generation.json`，包括负面提示词。

SenCache 固定 start/main 阈值 **0.01/0.7**、前 10 步使用 start 阈值、首尾强制完整计算、最多连续跳过 **4** 步、阈值缩放 **1015.9685034488028**。前 10 步不是额外强制 dense warmup。新旧校准表使用相同的调度。

## 另一台服务器需要什么

下载并解压整个小型发布包，保持目录结构。已有原实验环境时只需设置三个模型路径；本包不需要恢复网络权重，也不包含基础模型或 MixKit 原视频。

- `data/sensitivity_ltx_096.npz`：新的 0.9.6 校准表，SHA256 `c79f1cb81b506f946c3ed6e0f68199ad58f58ddc5639c7f1ac40f92d6c3d6a9b`。
- `accelerators/official_adapters.py` + `accelerators/cache_adapters.py`：与完成实验相同的 native 缓存适配器。
- `runtime.py`、`restore/native.py`、`restore/video.py`：固定的模型调用、表选择、身份记录和视频编码。
- `configs/`：生成条件、模型及外部源码的校验信息。
- `benchmark.py`、`data/VBench_full_info.json`、`evaluate_vbench.py`：批量计划、断点续跑和官方评估入口。
- `evidence/`：8 条视频来源、随机种子、校准协议、有限验证结果与原始校准脚本存档。**推理不读取原视频。**

只把新 NPZ 改名覆盖旧文件，会丢失新旧表身份，而且不能保证调用端调度正确；请使用本包的明确方法名。

## 下载与快速检查

```bash
curl -fL -o sencache-ltx096-mixkit8-v1.zip \
  https://github.com/wangwei64/wan-restoration-reports/releases/download/sencache-ltx096-mixkit8-v1-20261107/sencache-ltx096-mixkit8-v1.zip
unzip sencache-ltx096-mixkit8-v1.zip
cd sencache_ltx096
python verify_package.py
python -m unittest discover -s tests -v

export LTX_REPO=/your/path/LTX-Video
export LTX_CHECKPOINT=/your/models/ltxv-2b-0.9.6-dev-04-25.safetensors
# 此目录下应有 text_encoder/ 和 tokenizer/
export LTX_TEXT_ENCODER=/your/models/pixart_text_reference
export CUDA_VISIBLE_DEVICES=0
python run.py --doctor --verify-weights
python run.py --describe

python run.py --method sencache_ltx096 \
  --prompt 'A dog runs across a grassy field.' --seed 20260912 \
  --out outputs/one_video
```

`--verify-weights` 在本包中只检查 LTX 和文本编码器；不需要 `restorer.pt`。也可复制 `configs/paths.example.json` 为 `configs/paths.local.json`，填写本机路径后对生成/检查命令传 `--paths configs/paths.local.json`。不要把包含本机路径的文件提交到仓库。

全新环境见 [安装说明](docs/INSTALL.md)。原服务器环境不等于任何最新 LTX 安装；检查器会拒绝不匹配的模型、源码和关键软件版本。

## 按之前协议做全量测试

先用目标服务器保存的旧 `benchmark_protocol.json` 或 `plan_*.json` 核对。默认继承原发布包的 **base seed=20260912，普通维度 5 个样本，temporal flickering 25 个候选样本**，合并重复提示词后每种方法 **6220 条视频**。这个默认计划不代表已证实你在另一台服务器当时实际运行了该计划。

```bash
python check_previous_plan.py /path/to/previous/benchmark_protocol.json
python benchmark.py plan --methods sencache_ltx096
# 先做一条入口检查，使用独立输出目录
python benchmark.py run --methods sencache_ltx096 --limit 1 --output outputs/entry_smoke
# 正式全量；同一条命令可断点续跑
python benchmark.py run --methods sencache_ltx096 --output outputs/vbench096
python benchmark.py summarize --methods sencache_ltx096 --output outputs/vbench096
python benchmark.py export --methods sencache_ltx096 \
  --output outputs/vbench096 --export-dir outputs/vbench096_standard
```

若旧计划使用不同的 `seed/samples/flicker_samples`，在核对、计划、生成、汇总和导出命令中一致传入 `--seed`、`--samples`、`--flicker-samples`。若之前使用自定义 prompt 列表或种子清单，需要先对齐该清单；不要把默认 6220 条结果称为同一批实验。脚本不会修改旧测试目录。

种子规则继承原包：`SHA256(base_seed + '\0' + prompt + '\0' + sample_index)` 的前 8 字节按大端整数取模 `2**31-1`。新旧方法逐 prompt、逐 seed 对齐，默认测试种子与校准种子 20260828/20260829/20260830 不重叠。

方法名分别是 `native`、`sencache_original091`、`sencache_ltx096`。若要重跑三组，在上述各命令中一致使用 `--methods native sencache_original091 sencache_ltx096`，并给一个新的输出目录。本包默认只运行新表；`--methods all` 会运行全部三组。多 GPU 用一进程一张卡，并传相同 `--shards N`、不同 `--shard 0..N-1`，全量导出时恢复 `--shards 1`。

如旧 native 视频、提示词、种子、生成设置及评估流程均一致，可用于质量参照；速度比较需同硬件和计时口径。不要混用不同 CFG 生成的原版参照视频。

官方 VBench 评分在生成全部完成后进行。安装与评分命令见 [评估说明](docs/EVALUATION.md)。校准结果中 12 条验证不等于全量 VBench。

## CFG 等参数能不能修改

本次发布使用上面的论文原设置。代码支持用 `--config your_config.json` 显式修改 CFG/STG/rescaling，并在每个 `run.json` 的 `calibration_scope` 中记录偏离；必须保持 CFG>1、STG>0，以保留 native negative/positive/STG 三分支布局。

灵敏度估计测的是正条件分支的 Transformer 输出，CFG 不直接作为这个函数的输入。但更改 guidance 会改变采样轨迹及缓存误差的放大，因此原来的质量验证不能自动延续到新配置。若修改，请重新做独立验证，并让原版与所有比较方法使用相同新参数。关闭 CFG/STG 需要另行适配分支逻辑。模型、分辨率、帧数、步数、时间表等结构条件的变化不在本表已验证范围内，本包会拒绝相应的 SenCache 配置；正式对比应重新校准。

## 如何描述这次校准

建议表述：**“参照 SenCache 的官方校准方法，使用 8 条 MixKit 视频在本实验的 LTX 0.9.6 生成设置下重新估计灵敏度，并固定原有缓存阈值进行测试。”**

- [SenCache 论文 §5](https://arxiv.org/html/2602.24208#S5) 使用 8 条 MixKit 视频；我们沿用数据来源与样本数。作者具体 8 条视频的编号未找到公开记录，本项目的 8 条视频及标题在测量前固定。
- 阈值 0.01/0.7、K=4 对齐 [固定提交的官方 LTX 代码](https://github.com/vita-epfl/SenCache/blob/3fb96a41511ede1e38f3555c863b1cb7cddfe567/LTX-Video/sencache.py) 与本项目论文附录。SenCache 论文文本中的 LTX 阈值/K 与代码默认值有差异；不要写成完全复现其论文的全部设置。
- 新旧表同时改变了 checkpoint、视频及生成设置，不能把对比解释为严格的“仅版本变化”因果消融。
- `evidence/results_report.json`：12 条固定阈值验证中，新表均时 44.45s、旧表 47.29s；新表 LPIPS 0.03668、旧表 0.02946。新表更快、平均重建指标有所下降。这不是同速度/NFE 对比，也未据此调整阈值。
- 原始脚本保存在 `evidence/calibrate_original.py.txt` 供审计，保留原环境路径，**不是跨服务器入口**。此发布包只迁移已完成的校准与测试工具，不会自动重新校准。

`tests/gpu_portability.json` 记录本包在原 A800 上的完整样本重放。结果只证明迁移入口与已验证实现一致，不保证不同硬件上逐位相同。第三方来源见 [说明](docs/THIRD_PARTY_NOTICES.md)。
