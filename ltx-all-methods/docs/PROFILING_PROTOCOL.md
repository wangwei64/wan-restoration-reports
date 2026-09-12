# ProfilingDiT：官方代码的 LTX 适配

推荐表名：**ProfilingDiT (official-code LTX port, mapped defaults)**。不使用“官方LTX复现”或“作者参数最优”等表述。

固定来源：[官方仓库](https://github.com/GeekGuru123/ProfilingDiT/tree/fa9d1983418b481cc37eb69e7f6c5a3ec6e182a1)。

参数默认值对应固定提交[generate.py的72–77行](https://github.com/GeekGuru123/ProfilingDiT/blob/fa9d1983418b481cc37eb69e7f6c5a3ec6e182a1/Wan2.1/generate.py#L72)；在线执行分支对应[cache.py](https://github.com/GeekGuru123/ProfilingDiT/blob/fa9d1983418b481cc37eb69e7f6c5a3ec6e182a1/Wan2.1/wan/cache.py)。冻结协议中的official_defaults.source指向缓存实现，实际默认数值请以同协议列出并校验的generate.py为依据。

| 项目 | 本版决定 |
|---|---|
| 在线缓存 | 未修改原始Wan DitCache.forward_single_original，通过薄包装调用 |
| 预热/间隔 | step_start=6，step_interval=6；最后一步不强制全量 |
| 层比例 | round(28/40×28)=20，不参考速度或质量结果 |
| LTX层列表 | [4, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] |
| 选层 | 既有离线foreground attention从低到高排序的前20层，完整统计见data/layer_profile.json |
| 在线全量步（0基） | 0–6、12、18、24、30、36、42、48 |
| 计算调用检查 | 680/1400个block calls；不能当成实测加速比 |

原始代码硬编码0–39的剩余层集合；LTX仅调用0–27，额外编号不执行。当前连续缓存组10–27有前驱9，符合原始实现的残差边界。包装拒绝原代码不支持的从0开始的连续组。原始代码的备用递减/前景分支保留为源文件内容，但没有第二个公开可选方法。

## 必须披露的差异

- Official authors provide no LTX code or preset. This is a declared cross-model port.
- Official generator scans layer lists; repository defaults are not authenticated paper-table parameters.
- Offline foreground comes from Grounding DINO + SAM2 RGB masks, not the paper noise-PCA pipeline.
- Fixed existing ranking aggregates valid records; 683/946 short and 24/32 long records are valid.
- Calibration and restoration training involve VBench prompts; full VBench is not wholly unseen.
- Previously inspected 8 cases are used without parameter selection, not a new independent test set.
- Other accelerators keep their existing adapters and settings; this is a complete-system comparison.

论文提出按前景注意力阈值分组与递减间隔；公开Wan默认执行路径使用固定间隔，主生成器又扫描层组合。无法从仓库确认论文表2唯一最终参数。本版选择可追溯的代码默认分支，并明确披露跨模型映射。

若严格要求作者认可的同模型基线，仍需作者提供或确认LTX移植；本版透明、固定、可审计，但不保证审稿人不会质疑跨模型和离线处理差异。
