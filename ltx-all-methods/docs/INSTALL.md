# 安装与迁移运行

目标环境为 Linux x86_64、Python 3.12、支持 CUDA 12.4 的 NVIDIA 驱动。已验证的 GPU 为 A800 80GB；其他显卡的耗时、显存和数值需以当地实测为准。代码使用 Triton 和 Linux 文件锁，不支持直接在原生 Windows 上运行。

## 1. 下载代码与安装依赖

从本次 [Release](https://github.com/wangwei64/wan-restoration-reports/releases/tag/ltx-fixed-betas-v5-20260912) 下载 `ltx-all-methods-fixed.zip`，解压并进入 `ltx_all_methods/`。也可只取得仓库中的 `ltx-all-methods/` 目录。

```bash
conda create -n ltx-fixed python=3.12 -y
conda activate ltx-fixed
python -m pip install -r requirements.txt
```

`requirements.lock` 固定生成及配对指标依赖，PyTorch为2.5.1+cu124、torchvision为0.20.1+cu124。安装需能访问 PyPI 和 PyTorch CUDA wheel 源。独立的 VBench 官方评分工具使用另一个环境。

## 2. 安装固定外部 LTX

如果已装有此提交的 LTX，可使用已有目录；否则：

```bash
git clone https://github.com/Lightricks/LTX-Video.git ../LTX-Video
git -C ../LTX-Video checkout 4b2d053057623ddd4d0a1d3e9cd28890e9ef487f
python -m pip install -e ../LTX-Video --no-deps
```

对应 `ltxv-2b-0.9.6-dev-04-25.safetensors`，不是 LTX-2、蒸馏版或另一个 Diffusers pipeline。运行检查器核对29个外部源文件与关键依赖版本。LTX官方源码、基础权重和文本编码器未放入此代码包。

## 3. 准备模型和路径

已有相同模型可直接指定；没有时，运行以下工具按官方固定 revision 下载并逐项校验：

```bash
python download_external_models.py --directory /your/models/ltx
```

工具只下载 LTX 2B checkpoint 和 PixArt-alpha/PixArt-XL-2-1024-MS 的 `text_encoder/`、`tokenizer/`。输出 JSON 给出实际目录；将其用于下列变量。约需6.34GB基础模型、19.05GB文本编码器，另预留295MB恢复器和生成视频空间。下载来源、文件名和SHA256均在 `configs/external_models.json` 与 `configs/external_download_sources.json`。

```bash
export LTX_REPO="/your/path/LTX-Video"
export LTX_CHECKPOINT="/your/models/ltxv-2b-0.9.6-dev-04-25.safetensors"
export LTX_TEXT_ENCODER="/your/models/PixArt-XL-2-1024-MS"
python download_weights.py
python run.py --doctor --verify-weights
python -m unittest discover -s tests
```

`LTX_TEXT_ENCODER` 必须是同时包含 `text_encoder`、`tokenizer` 的父目录。恢复器默认下载到本包 `weights/restorer.pt`，通过本次Release的公开链接取得，不需要GitHub账号；也可用 `LTX_RESTORER` 指向同哈希已有文件。原服务器的T5 .bin存储与官方PixArt分片已验证张量等价，检查器支持两种形式，新机器建议官方safetensors分片。

## 4. 冒烟验证与完整运行

按 README 运行 `--limit 13 --samples 1 --flicker-samples 1` 的冒烟命令，成功后另用 `outputs/full` 跑完整VBench。每个进程通过 `CUDA_VISIBLE_DEVICES` 只选择一张GPU。首次源文件和大模型哈希验证在计时前完成；生成日志每5步更新，首轮加载与Triton编译可能较慢。

环境检查失败时按报出的文件或版本修正；不要关闭检查器或复用不同配置的完成记录。所有运行路径均由环境变量或自己的 `--paths` 文件提供，本包不依赖原服务器绝对路径。参数、分片、恢复和评估说明见 [CONFIGURATION.md](CONFIGURATION.md) 与 [BENCHMARK.md](BENCHMARK.md)。
