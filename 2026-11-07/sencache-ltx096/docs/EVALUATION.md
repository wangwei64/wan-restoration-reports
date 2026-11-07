# 评估

VBench 建议使用单独环境，避免更改已验证的生成依赖。官方源码固定为 `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`；使用与原测试服务器一致的官方安装及评估模型。

```bash
git clone https://github.com/Vchitect/VBench.git /your/path/VBench
git -C /your/path/VBench checkout fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490
# 在独立 VBench 环境依照该版本 README 安装依赖和评估模型
python evaluate_vbench.py --vbench-repo /your/path/VBench \
  --python /path/to/vbench_env/bin/python \
  --videos outputs/vbench096_standard --output outputs/vbench096_scores \
  --methods sencache_ltx096 --dry-run
# 检查后去掉 --dry-run 实际评分
```

评估入口验证完整导出、官方元数据与固定源码。temporal flickering 先执行官方 static filter，要求每个静态 prompt 至少 5 条合格视频；缺失时会报错，不将不足的结果冒充全量完成。

如同一批结果目录中已包含 `native` 和新 SenCache（两者 prompt、seed、generation 必须一致），可用原项目冻结的 LPIPS/SSIM 口径逐帧对比：

```bash
python score_pairs.py --results outputs/paired_vbench \
  --method sencache_ltx096 --output outputs/paired_scores.json
```

LPIPS 使用 AlexNet、416×240；SSIM 使用 832×480、double，覆盖全部帧。此工具输出 LPIPS/SSIM，不替代 VBench。历史校准报告的 PSNR 为原分辨率逐帧计算的平均值。要报告论文整套指标，请沿用目标服务器之前的完整评估脚本及版本；不要把不同尺度/抽帧规则的结果合并比较。

旧实验的 native 结果可在请求字段、视频哈希及硬件计时口径核对后复用；不要仅凭相同文件名拼接。如果目标服务器使用自定义数据布局，先按其原清单接入同一组 prompt/seed。
