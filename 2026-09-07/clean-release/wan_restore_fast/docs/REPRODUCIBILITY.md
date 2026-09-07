# 复现记录与测量边界

## 锁定的生成设置

`restore_fast/config.py` 是唯一默认参数来源，`configs/fastest.json` 是导出的记录。832×480、41帧、16fps、50步、前10步完整 Wan、CFG5、shift2、β2.0、τ.010、默认seed20260828；negative prompt 同样写入结果 JSON。

`weights/manifest.json` 记录三份模型的原检查点哈希、转换后的 safetensors 哈希、删除的无用参数键、每个配置/标定文件哈希。包不依赖转换前的 `.pt` 或任何旧实验目录，权重保留 FP32，转换过程中没有训练或量化。

`tests/external_weights.json` 只记录外部 Wan 权重和 tokenizer 的大小与 SHA256，不包含文件内容。需要严格核对外部权重时可运行 `python tests/verify_external_weights.py`；它会读取较大的模型文件，因此没有放入每次在线生成。

## 清理等价性

`tests/model_regression.json`：把9个旧案例的相同 Step10 clean latent 输入清理后的网络，比较 R/C/e，全部最大绝对误差为0。

`tests/full_regression.json`：0625/20260828 从新接口重新生成；所有共有输出张量（含 z9/z10、选区、刷新事件/分数、最终 latent）最大差为0，最终 MP4 SHA256一致。它证明该历史案例的全链路相同，不能外推为所有未来输入绝对一致。

历史原版运行是 β2.0 的冻结最快版。等价性核对只读取历史结果用于验证；生产推理不读取历史数据或 GT。

## VBench 样例复测

样例源为 [官方 VBench](https://github.com/Vchitect/VBench) 的 `vbench/VBench_full_info.json`，本地快照含946项，SHA256在 `benchmarks/protocol.json` 中。预先固定0-based索引157/500/625及seed20260828，对应人物动作、水彩海滩、柯基镜头运动。保留两个历史 prompt/seed，并增加一个数据集中的场景 prompt。

每例重新跑加速版与同配置原版 GT50。每种方法测一次，没有在这3例上筛选参数或择优种子。模型装载耗时单独记录；加速比使用 prompt 编码、前10步、特征/恢复/路由、端点观察、后续细化、最终 VAE 和 MP4 编码的总和。排除指标、诊断预览、报告生成与 latent 存盘。Triton 编译缓存可能已由历史回归预热，单 prompt 的模型装载与生成合计耗时另在 CLI smoke 中报告；计时从 Python 模块导入后开始，不包含解释器启动和模块导入。

GT50 不复用加速版的编码结果、噪声历史、恢复输出或 K/V；两者共用已载入的同一套外部权重，不把第二次模型加载算进 GT50。

LPIPS 使用 AlexNet、416×240逐帧计算；PSNR和SSIM在保存的832×480视频上计算。GT50 只是相同模型50步的对照，不是真实拍摄真值；这些差异指标不能代替视频视觉判断。报告提供等长、同步拼接视频，以及分区/恢复/Step10辅助展示。

这是小规模 prompt 复测，不是完整16维 VBench 评测。仅此3个样例或9个历史案例不能保证任何 prompt 的质量或加速比。

## 命令

```bash
./run.sh --doctor
python tests/check_release.py
python benchmarks/run_vbench.py --output benchmarks/rerun_01
python benchmarks/build_report.py --results benchmarks/rerun_01 --out benchmarks/report_rerun_01
```

使用当前 GPU Python 环境执行 `python`。`check_release.py` 核对文件完整性、默认参数、网络加载与数量；不重新跑 Wan。报告脚本的额外 VAE 解码全部发生在生成计时之后。
