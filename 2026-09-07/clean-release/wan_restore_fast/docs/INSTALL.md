# 一次性环境安装

下载源码后，按以下步骤准备 Linux CUDA 环境和外部 Wan。这是一次性安装；之后每次生成只需要 prompt。

```bash
# 在下载的 wan_restore_fast 源码目录中执行
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

FlashAttention 若从源码构建，需要匹配的 CUDA toolkit、编译器和 Ninja。实测版本完整记录在 `tests/environment.json`。依赖安装会访问软件包源；生成过程不需要登录服务。

从 [Wan 官方仓库](https://github.com/Wan-Video/Wan2.1) 准备外部源码。验证使用的 commit：

```text
9737cba9c1c3c4d04b33fcad41c111989865d315
```

从官方提供的 [Wan2.1-T2V-1.3B 权重](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) 准备完整原版目录。不要改用 Diffusers 转换版 checkpoint。目录关系按 README 设置。

如果按 README 的相邻目录布局，Wan 路径会自动识别。若用 GitHub 克隆后的嵌套目录，或已有 Wan 在其他位置，可以在终端一次性指定：

```bash
export WAN_REPO=/absolute/path/to/Wan2.1
export WAN_CHECKPOINT=/absolute/path/to/models/Wan2.1-T2V-1.3B
```

本方法的三个训练权重已经公开，无需从项目服务器取文件或使用 GitHub 登录。
普通生成会自动获取缺失的权重；也可以在检查环境前先单独下载：

```bash
python run.py --download-weights
./run.sh --doctor
./run.sh "A person is riding a bike"
```

`--doctor` 核对 GPU、依赖是否存在、外部 Wan 核心文件 SHA256、内置模型/配置 SHA256。它不会下载权重、加载大模型或开始生成；缺少本方法权重时会给出预下载命令。它不保证其他 GPU/软件版本与实测环境完全一致，复现时应采用锁定版本。

本次本地整理没有从零重装整个 CUDA 环境；验证是在服务器现有环境中、使用新包和新接口完成。路径探测不依赖旧实验文件夹。

下载地址见 `weights/sources.json` 中的公开 Release。常规下载地址不可达时，程序会自动尝试公开 GitHub 资产 API；两条路径均不发送认证凭据。需要联网下载一次，或从 Release 手动下载同名文件放到 `weights/` 后离线使用。网络下载与哈希校验时间独立记录，不用于计算加速比。

下载遵循当前环境的 HTTP/HTTPS 代理设置；如果提示 `network/proxy unavailable`，请检查代理是否仍在运行或移除失效配置。程序不会关闭 TLS 证书校验或自行覆盖代理设置。
