# Linux / CUDA 环境

已验证 A800 80GB、Python 3.12、CUDA 12.4、PyTorch 2.5.1。已有原实验环境可直接使用并运行 doctor。

```bash
conda create -n sencache096 python=3.12 -y
conda activate sencache096
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
git clone https://github.com/Lightricks/LTX-Video.git ../LTX-Video
git -C ../LTX-Video checkout 4b2d053057623ddd4d0a1d3e9cd28890e9ef487f
pip install -e ../LTX-Video --no-deps
```

基础 checkpoint 为 `ltxv-2b-0.9.6-dev-04-25.safetensors`，约 6.34 GB。文本编码器和 tokenizer 约 19.05 GB，来自 PixArt-alpha/PixArt-XL-2-1024-MS；只下载其文本组件，无需 PixArt 生成模型。可以复用原服务器的已验证文件。

```bash
python download_external_models.py --directory /your/models/ltx
```

脚本固定上游 revision 并校验下载内容；按其实际输出填写 `LTX_CHECKPOINT` 与 `LTX_TEXT_ENCODER`。完整清单在 `configs/external_models.json`。检查器支持原实验的 DeepFloyd T5 binary 存储及逐 tensor 验证等价的 PixArt safetensors 存储。不要用 LTX 2、另一 checkpoint 或另一 pipeline 替代。

运行前设置 `LTX_REPO`、`LTX_CHECKPOINT`、`LTX_TEXT_ENCODER`，然后执行 `python run.py --doctor --verify-weights`。`configs/training_provenance.json` 从原包保留，当前使用其中的软件和 29 个外部源码文件校验；本包不加载恢复网络。

显式选择单张 GPU：`export CUDA_VISIBLE_DEVICES=0`。检查器拒绝其他进程占用同卡时的有效性能计时。计时包括文本编码、采样、VAE 解码和视频编码，排除模型加载、文件哈希校验、潜变量序列化和评估。不同服务器的速度应在其本机与 native 对照，不与原 A800 秒数直接混算加速比。
