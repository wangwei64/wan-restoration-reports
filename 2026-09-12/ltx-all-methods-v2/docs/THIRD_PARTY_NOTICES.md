# 来源与实现边界

- 外部 LTX-Video：Lightricks/LTX-Video，提交 `4b2d053057623ddd4d0a1d3e9cd28890e9ef487f`。不再分发官方模型源码或基础权重，遵循原作者对应许可证与模型使用条件。`restore/native.py` 是本项目的调用适配器。
- TeaCache：ali-vilab/TeaCache 的 LTX 信号、多项式与残差边界，适配当前 native CFG/STG 合批结构，加入本实验固定的前 10 步全量与最后一步全量。保留 Apache-2.0 许可证。
- NaviCache：HelloZicky/NaviCache，提交 `8dcfffa35e3efb555d8dd28344e575e439da0246`。其 Wan 原始输入/输出状态估计缓存被移植到 LTX 三分支合批，官方无该 LTX 实现。保留 Apache-2.0 许可证。
- SenCache：vita-epfl/SenCache，提交 `3fb96a41511ede1e38f3555c863b1cb7cddfe567`。基于官方 LTX 算法适配 native 0.9.6；灵敏度数据来自 [Yassaman/SenCache](https://huggingface.co/datasets/Yassaman/SenCache)，`sensitivity_ltx.npz` 的 SHA256 为 `42cd631394ab1174f595bdfa3052312b65482720b367c7fbb5767742acd36b14`。原表针对 LTX 0.9.1，本包未训练或重新估计它，不能称为当前版本最优校准。
- ProfilingDiT：GeekGuru123/ProfilingDiT，提交 `fa9d1983418b481cc37eb69e7f6c5a3ec6e182a1`。本地代码按论文公式实现连续背景层分组残差复用与刷新调度。LTX 离线前景来自 Grounding DINO + SAM2 的 RGB 分割，区别于原论文前景处理；不是作者官方 LTX 代码。固定层列表的在线使用不需要分割模型。
- VBench：Vchitect/VBench，提交 `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`。保留完整元数据、采样说明和 Apache-2.0 许可证。官方评估器在包外安装，未重新实现它的评分模型。
- `restore/models`、稀疏细化与融合内核源于本项目自己的 WAN 恢复路线和已经训练、验证的 LTX 适配。发布模型没有新训练、量化或裁剪。

随包提供上述获取到的许可证原文。来源参考不表示各上游作者认可此跨模型移植或本次测得的性能。缓存方法的新增集成改动与固定配置在本包保留，源码清单、训练来源、软件和模型哈希均可核对。
