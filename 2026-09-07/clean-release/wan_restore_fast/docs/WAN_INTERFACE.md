# 外部 Wan 接口

## 文件约定

包内不包含 Wan 源码、DiT 权重、T5 权重、VAE 权重或 tokenizer。`OfficialWanAdapter` 通过参数或自动发现的目录读取这些文件：

```text
Wan2.1/
  wan/modules/model.py
  wan/modules/t5.py
  wan/modules/vae.py
  wan/utils/fm_solvers_unipc.py
Wan2.1-T2V-1.3B/
  config.json
  diffusion_pytorch_model.safetensors
  models_t5_umt5-xxl-enc-bf16.pth
  Wan2.1_VAE.pth
  google/umt5-xxl/...
```

## 调用示例

```python
from restore_fast.pipeline import RestorationPipeline

pipeline = RestorationPipeline()   # 自动发现外部 Wan，装载一次
try:
    result = pipeline.generate("A person is riding a bike")
    print(result["video"])
    # 可继续调用 generate，不用重新装载模型；每次清空生成历史和缓存。
finally:
    pipeline.close()               # 恢复 Wan 原来的 forward / unpatchify 方法
```

`RestorationPipeline(repo="...", checkpoint="...")` 可显式指定外部路径。`generate(prompt, seed=None, out=None)` 只有 prompt 必填；显式 out 必须是尚不存在的目录，以保护已有结果。

构造 `RestorationPipeline` 时会先校验本方法的三个权重，缺失时从公开 Release 自动下载，随后才装载 Wan。`weights=` 可指定包含完整配置、`manifest.json` 与 `sources.json` 的独立权重目录；设置 `WAN_RESTORE_OFFLINE=1` 可强制只使用已校验的本地文件。

## 适配层的职责

| 成员 | 输入 → 输出 / 约束 |
|---|---|
| `native_model` | 外部 `WanModel`，1.3B、30 层、dim1536、12 heads、patch=(1,2,2) |
| `official_text_encoder` | 外部 T5；tokenizer 和 context 保留原版结构供自动词组定位 |
| `device` | CUDA device |
| `set_request(prompt, seed)` | 设置本次生成描述和噪声种子 |
| `encode_prompt()` | 生成正/负 `native_context`，返回耗时 |
| `initial_latents()` | FP32 `[1,16,11,60,104]`，固定形状、显式 GPU generator |
| `scheduler()` | 原版 FlowUniPC，50 步、shift2、order2/bh2 |
| `predict(latent, idx, scheduler, controller=None)` | 返回 CFG 后的 FP32 flow；idx 为 0-based |
| `clean_estimate(latent, flow, sigma)` | `latent - sigma * flow`，得到 clean prediction |
| `step(scheduler, flow, timestep, latent)` | 原生 UniPC step，处于 BF16 autocast 范围 |
| `decode(latent)` | 原版 FP32 VAE → RGB uint8 帧列表、耗时 |

`SparseWanController` 需要直接访问 Wan block 内的 attention、norm、FFN、RoPE 和 unpatchify，因此这个接口是 **针对指定 Wan2.1 版本的白盒适配层**，不是随便替换一个黑盒视频 API 就能接入。若换 Wan2.2、14B、I2V 或其他 token 布局，需要新的兼容性验证，不能只改模型路径。

## 缓存和原版对照

Step10 捕获左右 CFG 的 K/V 与 prompt 对应图。恢复后在 Step50 噪声位置做一次全 token 的端点观察，提供另一组固定 K/V；这是一次模型观察，不是额外去噪步骤，时间计入加速版。

Step11–50 的条件与无条件分支合批，仍同时计算两支；选中 token 每次经过全部 30 层。没选中的 token 提供恢复参照加最近实测残差的 K/V。此近似依赖我们已有的恢复端点。

`native_reference()` 切回原版方法、清空缓存，以相同 prompt/seed 重跑完整 50 步，不使用加速版选区、恢复结果或生成历史。两条路线公用同一套外部权重、原生 scheduler 与 VAE。

## 精度约定

DiT 参数、初始 latent 和 VAE 保留 FP32；DiT/恢复网络在 BF16 autocast 中执行。逐 token 积分器保留旧版 CPU 标量倒数与 BF16 contraction 的舍入方式。不能直接把全部权重 `.bfloat16()` 或把积分移出 autocast，否则会改变轨迹。
