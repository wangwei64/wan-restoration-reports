# 外部来源

Wan2.1 源码与权重由用户在包外提供，来源为 [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) 和 [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)。使用和分发这些资源时遵循它们各自的许可。

VBench prompt 快照来自 [Vchitect/VBench](https://github.com/Vchitect/VBench)，用于复现实验样例；这份样例报告不代表官方 VBench 评测结果。NAF 风格恢复块沿用本项目已经使用和训练的实现；没有在本次整理中重新下载或训练公开 NAFNet 权重。

FlashAttention、Triton、PyTorch、Diffusers、Transformers、LPIPS 等通过依赖安装，不在此文件夹内复制源码。LPIPS 的预训练 AlexNet 只用于生成后的评估，可在首次评估时下载；不会参与在线生成。

本次上传用于项目代码查阅与复现，未为项目自身代码或训练权重新设开源许可。外部依赖仍遵循各自许可。
