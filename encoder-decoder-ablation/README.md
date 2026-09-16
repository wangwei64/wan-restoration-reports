# Wan Encoder / Decoder 统一消融

本目录整理 2026-09-16 完成的统一实验：**4 个 prompt × 2 个生成种子，encoder 和 decoder 各 8 个配置，共享同一份 Ours**。216 条原始运行记录，180 条复用、36 条补跑；先平均同一样本内的调度重复，再平均 8 个样本，共 120 个独立样本/配置条件。

## 查看结果

- [两部分完整结果](results/report_cn.md)：各 8 行，显示三位小数。
- [结论和逐样本差异](results/conclusions_cn.md)。
- [原始记录及逐样本数值](results/results.json)、[汇总 CSV](results/summary.csv)。
- [完整性校验](results/final_audit.json)、[本地独立复核](results/local_validation.json)。
- [实验代码](code/)、[配置与 prompt](code/config.json)、[运行环境说明](code/README.md)。

Ours 在两部分的平均指标完全一致：**LPIPS 0.097、SSIM 0.793、PSNR 24.813 dB**，在本次两部分平均比较中均最好。逐样本并非全部严格最优：小船第二个种子的单独视觉引导略好；decoder 去掉状态恢复有 4 个样本持平。所有数值保留。

## 实验与方法的对应关系

Encoder 保持每一步 token 更新数量一致，比较位置选择策略；它不是删除整个编码网络。Decoder 保持 Full 的逐步更新位置完全一致，比较三个开关的全部八种组合。

| 部分 | 配置 | 含义 |
|---|---|---|
| Encoder | random / roi / uniform_coverage | 全局随机 / 固定 ROI 随机 / 历史覆盖 |
| Encoder | detail_random / detail_tiebreak | 视觉敏感性引导下的随机 / 覆盖选择 |
| Encoder | subject_random / subject_tiebreak | 语义相关性引导下的随机 / 覆盖选择 |
| Encoder | ours | 联合感知语义引导 |
| Decoder | S | 恢复结果用于状态填充 |
| Decoder | C | 恢复结果用于注意力参考内容 |
| Decoder | A | 实际 K/V 噪声插值对齐 |

Decoder 的 0/1 表示关闭/开启；`ours` 对应 `S1_C1_A1`。S/C 关闭时使用固定历史外推端点。A 关闭时把插值系数设为 0，保留真实 K/V 更新与历史残差规则。**公式 4 的潜变量变换在所有配置中始终开启，这不是该公式的独立消融。**

## 固定参数

| 参数 | 设置 |
|---|---|
| 分辨率、帧数、帧率 | 832 × 480、41 帧、16 FPS |
| 总采样步数、预热 | 50 步、前 10 步完整计算 |
| CFG、noise shift | 5.0、2.0 |
| 加速预算 β、质量阈值 τ | 2.0、0.01 |
| 生成种子 | 2026091401、2026091402 |
| 场景 | 蓝鸟、水面小船、行走的狗、飘动织物 |

六个 encoder 随机/覆盖策略各做 3 次调度重复（101/202/303），ROI 使用偏移 404，Ours 只保留一次。指标以相同种子原生 Wan 输出为参考；具体分辨率及逐帧计算见 [metric_functions.py](code/metric_functions.py)。结果限于这组场景，不声明普遍最优或统计显著性。

## 无 GPU 复核已发表的数值

Python 3.9+，仅需标准库，在任意工作目录执行：

```bash
python encoder-decoder-ablation/verify_results.py
```

脚本检查代码和配置哈希、216 条记录、120 个逻辑条件、8 份共享 Ours，并从原始记录重新计算逐样本及总体均值。它复核已保存的数值，不重新解码视频计算指标。

## 重新生成实验

本目录保留当时运行的原始代码与哈希。它依赖已有 Wan 源码、恢复模型及权重、原生参考视频和历史结果清单；**当前不是下载后即可从零生成的独立模型发行包**。这些大文件与环境不包含在此仓库。路径与依赖详见 [code/README.md](code/README.md)。

在原实验服务器目录中，使用已配置的 GPU Python：

```bash
/root/autodl-tmp/envs/wan-downscaler-full/bin/python run.py all
```

迁移时调整 `runtime.py`、`decoder_controls.py`、`config.json` 的路径并提供对应资产，在新实验目录重新登记，勿覆盖本次冻结记录。`all` 依次登记、生成、评测与汇总，已有完整输出可校验后复用。

## 文件边界

`code/` 是冻结代码；`results/` 是原始结果、来源登记及审计；`verify_results.py` 是用于本发布目录的独立只读复核脚本。模型权重、视频、SSH 工具和凭据不随本次提交上传。
