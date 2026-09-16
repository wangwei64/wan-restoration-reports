# Encoder / Decoder 统一消融代码

统一入口为 `run.py`，配置为 `config.json`。两部分各8行，共享一份Ours，因此每个样本15个独立逻辑配置。4个prompt×2个生成种子=120个逻辑条件。encoder的六种随机/覆盖策略各3个调度重复，ROI偏移404固定一次，Ours一次；decoder其他7组各一次，总计216条原始运行记录。三次调度先在各样本内平均，再对8样本平均，不当作额外内容样本。

## 运行环境

代码使用当前服务器已冻结的Wan加速实现、预训练权重及原生参考数据。未打包大模型权重。默认依赖路径：

- `/root/autodl-tmp/wan_restore_fast`：模型及恢复网络实现。
- `/root/autodl-tmp/Wan2.1`：官方DiT。
- `/root/autodl-tmp/experiments/wan_paper_experiments_20260912/common.py`：冻结校验与GPU独占检查。
- Python：`/root/autodl-tmp/envs/wan-downscaler-full/bin/python`，已有torch、flash-attn、triton、safetensors、lpips、opencv-python、scikit-image。
- 原生参考、Full和旧结果路径集中列在config.json；复用来源登记在registration.json，不使用模糊文件搜索自动挑结果。

若迁移服务器，须调整runtime.py/decoder_controls.py的依赖路径及config.json中的数据路径，重新注册。实验开始后不直接修改冻结配置和源文件。

## 一个命令运行

在实验目录执行：

```bash
/root/autodl-tmp/envs/wan-downscaler-full/bin/python run.py all
```

分阶段运行和状态查询：

```bash
python run.py register
python run.py generate
python run.py evaluate
python run.py summarize
python run.py status
```

使用上述GPU环境的python运行。`all`按顺序启用独立子进程，生成结束释放模型显存后才评测。TORCH_HOME使用已存在的AlexNet缓存路径。

## 代码结构

- `register.py`：登记源哈希、参考视频哈希、复用结果与缺失项；逐样本验证encoder与decoder的Ours输出哈希和指标一致。
- `generate.py`：统一样本循环、共享Ours、同一参考预算、生成/复用与实际输出校验。
- `encoder_controls.py`：分发已冻结的选择策略；random/roi/region/cue/neutral_controls保留原算法。
- `decoder_controls.py`：状态填充与注意力参照内容开关，以及实际K/V噪声插值开关。
- `evaluate.py`：全部组统一PSNR/SSIM/LPIPS；仅匹配同一样本和视频SHA时复用指标。
- `summarize.py`：每部分8行，先调度平均再样本平均；从同一canonical Ours记录汇总两张表，断言完全相同。
- `runtime.py`：共同配置、任务清单、哈希、原子保存及状态。
- `status_remote.py`、`export_remote.py`、`launch_remote.py`：现有安全SSH工具调用的管理入口；无凭据。

## 开关与控制变量

Encoder仅替换刷新位置选择，保持每一步token更新数量，恢复网络、候选区域与对齐保留。Decoder S/C关闭时采用固定历史外推内容，A关闭时将K/V插值系数置0；真实K/V覆盖、历史残差规则保留，公式（4）的潜变量对齐始终开启。Decoder保持原Full逐步更新位置完全一致。固定10步预热，生成至50步。

## 复用和续跑

所有已有decoder结果及匹配的encoder结果直接复用；只生成registration.json的missing项。Ours在runs/<sample>/ours只有一个记录，两部分共享。native参考也只登记一次。

完成标记是每组measurement.json和视频哈希。重启generate会验证并跳过完整组；未完成目录移入incomplete/保留后重跑，不覆盖原数据。evaluate将已完成评分当作精确SHA缓存。任何冻结校验、掩码或入口检查失败都会停止，必须保留原日志后排查。并行GPU任务不自动抢占。

## 输出

`results.json`含两张表、逐样本数据和216条原始记录；`report_cn.md`为简要表；`final_audit.json`为完整性检查；`evidence.zip`包含代码/配置/日志/结果，不包含大视频或权重。所有结果与失败记录保留；不依据Ours排名删改实验。旧双种子decoder结果不变，本次补齐encoder后按同一个八样本集合统一汇总。阈值实验不在本任务范围内。
