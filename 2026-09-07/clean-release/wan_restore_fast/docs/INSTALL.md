# 一次性环境安装

当前服务器已经有可用环境，直接执行根目录 `run.sh` 即可。以下用于新机器。

```bash
cd wan_restore_fast
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

```bash
./run.sh --doctor
./run.sh "A person is riding a bike"
```

`--doctor` 核对 GPU、依赖是否存在、外部 Wan 核心文件 SHA256、内置模型/配置 SHA256。它不会加载大模型或开始生成。它不保证其他 GPU/软件版本与实测环境完全一致，复现时应采用锁定版本。

本次本地整理没有从零重装整个 CUDA 环境；验证是在服务器现有环境中、使用新包和新接口完成。路径探测不依赖旧实验文件夹。
