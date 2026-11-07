# 来源与实现边界

- LTX-Video：Lightricks/LTX-Video，固定提交 `4b2d053057623ddd4d0a1d3e9cd28890e9ef487f`。本包不分发其模型源码或基础权重；外部下载遵守上游许可证。`restore/native.py` 为本项目调用适配器。
- SenCache：vita-epfl/SenCache，固定提交 `3fb96a41511ede1e38f3555c863b1cb7cddfe567`。实现遵循其 LTX 灵敏度门控、残差边界与双阈值调度；本项目适配 native LTX 0.9.6 三分支。原始 0.9.1 校准表来自 [Yassaman/SenCache](https://huggingface.co/datasets/Yassaman/SenCache)，新 0.9.6 表由本项目测量。固定提交的根目录 LICENSE 端点未提供许可证文件；此说明不授予上游额外权利。
- MixKit：原视频不随包分发，只保留来源、标题、链接、许可链接及哈希。采集时对应 Free Stock Video License，详见 `evidence/download_manifest.json`。作者实际使用的 8 条视频编号未知，不声称是同一组素材。
- VBench：Vchitect/VBench，固定提交 `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`。保留官方元数据与 Apache-2.0 许可证；评估器从上游安装。
- TeaCache / NaviCache：共享适配器文件保留原实现以保持已验证代码一致，当前发布入口不选择这两种方法；对应 Apache-2.0 许可文本保留在 `licenses/`。

上游引用不表示作者认可本次移植或实验。新旧表与有限验证结果均保留，不根据结果选择阈值。
