# Linux / CUDA 安装

已验证的目标是 A800 80GB、Python 3.12、CUDA 12.4、PyTorch 2.5.1。其他硬件/驱动需要重新验证速度和数值。原始 LTX 基础 checkpoint 约 6.34 GB，指定文本编码器约 19.05 GB，本项目恢复器约 295 MB；本包没有再次分发前两项。

```bash
conda create -n ltx-all python=3.12 -y
conda activate ltx-all
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
git clone https://github.com/Lightricks/LTX-Video.git ../LTX-Video
git -C ../LTX-Video checkout 4b2d053057623ddd4d0a1d3e9cd28890e9ef487f
pip install -e ../LTX-Video --no-deps
```

这份外部官方提交的 29 个运行源码文件已逐一与原服务器比对一致。不要直接安装最新 LTX 2、蒸馏模型或 Diffusers 另一条 pipeline 来替代。检查器会核对源文件哈希、版本及模型文件。

通过官方 [LTX-Video 模型](https://huggingface.co/Lightricks/LTX-Video) 下载 `ltxv-2b-0.9.6-dev-04-25.safetensors`。文本编码器使用 [PixArt-alpha/PixArt-XL-2-1024-MS](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS) 的 `text_encoder/` 和 `tokenizer/`。`python download_external_models.py --directory /your/models/ltx` 会按固定官方 revision 下载必要文件并逐项校验 SHA256，不下载额外的 PixArt 生成模型。文件名、大小、完整 SHA256 在 `configs/external_models.json`；其中 LTX 模型 SHA256 为 `e6fb3fe995e23c1f2457bc26c554789a8d74718c17f1853baab33939bd8dfb56`。

原服务器文本编码器使用 DeepFloyd/T5 的 .bin 缓存；其全部 tensor bytes 已按官方 safetensors 头重新计算并与 PixArt 两个官方 shard 的完整 SHA256 相同。检查器支持这两种已验证的等价存储形式。新服务器建议直接使用脚本下载官方 PixArt safetensors，无需复制旧服务器缓存路径。

```bash
export LTX_REPO=/your/path/LTX-Video
export LTX_CHECKPOINT=/your/models/ltxv-2b-0.9.6-dev-04-25.safetensors
export LTX_TEXT_ENCODER=/your/models/PixArt-XL-2-1024-MS
python download_weights.py
python run.py --doctor --verify-weights
python -m unittest discover -s tests
```

路径可以包含空格，但 shell 中需要正确引用。`LTX_TEXT_ENCODER` 指向同时包含 `text_encoder`、`tokenizer` 的父目录。若恢复器另放目录，用 `LTX_RESTORER=/path/restorer.pt`。没有任何原服务器绝对路径是推理必需配置。

首次加载前会核对基础模型和文本编码器的完整哈希，额外 I/O 不计入在线耗时。请预留磁盘和显存。计时包括文本编码、采样、恢复/缓存、VAE 解码和视频编码；模型装载、离线 profiling、潜变量序列化和评估不计入。

安装官方 VBench 评估工具建议使用独立环境，避免其视觉模型依赖改变已验证生成环境。见 BENCHMARK.md。


## 固定视频编码

本发布统一在独立 CPU 进程中采用 x264 fast、CRF18、单线程编码（含 lookahead，关闭 sliced threads），明确指定每帧时间戳及 BT.601 全范围RGB到限幅YUV转换。旧实验在生成进程内对相同 RGB 编码曾不稳定，仅限制线程未能解决；隔离后完整视频五次重复检查通过。生成参数和方法档位保持一致；`run.json` 保存原始 RGB 哈希与编码参数。在线时间包含传输、进程启动及固定编码成本，不能直接把旧编码的历史时间当作本版本速度。历史数据与本版实际核验见 [RESULTS.md](RESULTS.md)。
