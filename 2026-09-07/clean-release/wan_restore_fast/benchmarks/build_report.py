"""Post-generation diagnostics and a self-contained local HTML/video report."""

import argparse, html, json, math, shutil, sys
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from skimage.metrics import structural_similarity

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from restore_fast.config import external_paths
from restore_fast.video import write_video


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) != 41:
        raise ValueError(f"Expected 41 frames, got {len(frames)}: {path}")
    return frames


def labeled(frame, label):
    canvas = Image.new("RGB", (frame.shape[1], frame.shape[0] + 48), "#132236")
    canvas.paste(Image.fromarray(frame), (0, 48))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 23)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(canvas).text((16, 6), label, font=font, fill="white")
    return np.array(canvas)


@torch.inference_mode()
def metrics(model, pred, target):
    lpips = []
    for start in range(0, len(pred), 4):
        arrays = [
            np.stack(
                [
                    cv2.resize(f, (416, 240), interpolation=cv2.INTER_AREA)
                    for f in frames[start : start + 4]
                ]
            )
            for frames in (pred, target)
        ]
        inputs = [
            torch.from_numpy(a).permute(0, 3, 1, 2).cuda().float() / 127.5 - 1
            for a in arrays
        ]
        lpips.extend(model(*inputs).flatten().cpu().tolist())
    psnr, ssim = [], []
    for a, b in zip(pred, target):
        mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
        psnr.append(10 * math.log10(255**2 / max(mse, 1e-12)))
        ssim.append(structural_similarity(a, b, channel_axis=-1, data_range=255))
    return {
        "lpips_alex_416x240": float(np.mean(lpips)),
        "psnr_db": float(np.mean(psnr)),
        "ssim": float(np.mean(ssim)),
        "frame_count": len(pred),
        "per_frame_lpips": lpips,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=PACKAGE / "benchmarks/results")
    parser.add_argument("--out", type=Path, default=PACKAGE / "benchmarks/report")
    args = parser.parse_args()
    summary = json.loads((args.results / "summary.json").read_text())
    if len(summary["results"]) != len(summary["protocol"]["indices"]):
        raise RuntimeError("Generate all fixed cases before building the report")
    assets = args.out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for filename in ["method.svg", "restorer.svg"]:
        shutil.copyfile(PACKAGE / "docs" / filename, assets / filename)
    repo, checkpoint = external_paths()
    sys.path.insert(0, str(repo))
    from wan.modules.vae import WanVAE
    import lpips

    vae = WanVAE(
        z_dim=16,
        vae_pth=str(checkpoint / "Wan2.1_VAE.pth"),
        dtype=torch.float32,
        device="cuda",
    )
    metric = lpips.LPIPS(net="alex").cuda().eval()
    rows, cards = [], []
    visual_notes = {
        157: "首、中、尾帧的构图和动作方向相近；车把、篮筐边缘和衣服褶皱存在局部偏移，后段差异更明显。",
        500: "海岸线、配色和整体构图接近；波浪边界、泡沫及水面纹理存在差异。",
        625: "柯基姿态和镜头构图接近；嘴部、毛发和草地有细小差异。",
    }
    with torch.inference_mode():
        for case in summary["results"]:
            key = f'{case["index"]:04d}_{case["seed"]}'
            folder = args.results / key
            pred = read_video(folder / "fastest/final.mp4")
            gt = read_video(folder / "gt50/final.mp4")
            case["metrics"] = metrics(metric, pred, gt)
            case["visual_note"] = visual_notes.get(
                case["index"], "请查看完整联合视频。"
            )
            tensors = load_file(
                str(folder / "fastest/outputs.safetensors"), device="cuda"
            )
            decoded = {}
            for field in ["x0_step10", "restored"]:
                output = vae.decode([tensors[field].float()])[0]
                decoded[field] = list(
                    np.uint8(
                        np.clip(
                            (output.float().cpu().permute(1, 2, 3, 0).numpy() + 1)
                            * 127.5,
                            0,
                            255,
                        )
                    )
                )
            joint, workflow, region_frames = [], [], []
            s, l, r = [
                tensors[field][0].cpu().numpy()
                for field in [
                    "subject_weight",
                    "low_confidence_weight",
                    "restored_weight",
                ]
            ]
            for i, (a, b) in enumerate(zip(pred, gt)):
                joint.append(
                    np.concatenate(
                        [
                            labeled(a, "Ours | fastest beta=2.0"),
                            labeled(b, "GT50 | original Wan"),
                        ],
                        axis=1,
                    )
                )
                t = min((i + 3) // 4, s.shape[0] - 1)
                colors = (
                    s[t, ..., None] * np.array([245, 90, 105])
                    + l[t, ..., None] * np.array([246, 182, 64])
                    + r[t, ..., None] * np.array([79, 169, 210])
                )
                colors = cv2.resize(
                    colors.astype(np.uint8), (832, 480), interpolation=cv2.INTER_NEAREST
                )
                region = np.uint8(0.60 * decoded["restored"][i] + 0.40 * colors)
                region_frames.append(labeled(region, "S: pink | L: amber | R: blue"))
                parts = [decoded["x0_step10"][i], decoded["restored"][i], region, a, b]
                labels = ["Step10 clean", "Restoration R", "S / L / R", "Ours", "GT50"]
                workflow.append(
                    np.concatenate(
                        [
                            labeled(
                                cv2.resize(x, (416, 240), interpolation=cv2.INTER_AREA),
                                label,
                            )
                            for x, label in zip(parts, labels)
                        ],
                        axis=1,
                    )
                )
            write_video(assets / f"{key}_compare.mp4", joint, 16)
            write_video(assets / f"{key}_workflow.mp4", workflow, 16)
            write_video(assets / f"{key}_regions.mp4", region_frames, 16)
            Image.fromarray(joint[20]).save(assets / f"{key}_poster.jpg", quality=94)
            Image.fromarray(
                np.concatenate([joint[i] for i in [0, 20, 40]], axis=0)
            ).save(assets / f"{key}_contact.jpg", quality=95)
            Image.fromarray(workflow[20]).save(
                assets / f"{key}_workflow.jpg", quality=95
            )
            m = case["metrics"]
            rows.append(
                f'<tr><td>{case["index"]:04d}</td><td>{case["fast_seconds"]:.2f}s</td><td>{case["gt50_seconds"]:.2f}s</td><td><b>{case["speedup"]:.2f}×</b></td><td>{m["lpips_alex_416x240"]:.4f}</td><td>{m["psnr_db"]:.2f}</td><td>{m["ssim"]:.4f}</td></tr>'
            )
            cards.append(
                f"""<article id="case-{case['index']}"><div class="eyebrow">VBENCH {case['index']:04d} · SEED {case['seed']} · {case['speedup']:.2f}×</div><h2>{html.escape(case['prompt_en'])}</h2><p>{html.escape(', '.join(case['dimension']))} · 在线平均刷新覆盖 {case['active_ratio']:.1%} · 初始候选覆盖 {case['initial_active_ratio']:.1%}</p><video controls playsinline loop preload="metadata" poster="assets/{key}_poster.jpg" src="assets/{key}_compare.mp4"></video><p>同步联合视频：左为整理后的最快版，右为本次重新生成的原版 GT50。</p><p><b>抽帧检查：</b>{html.escape(case["visual_note"])} 完整动态差异请播放视频查看。</p><details><summary>查看 Step10、恢复结果、三种区域与最终结果</summary><video controls playsinline loop preload="none" src="assets/{key}_workflow.mp4"></video><p>顺序：Step10 clean → 恢复 R → 互斥选区 → 最终 → GT50。诊断解码只在测试后执行，不计入在线耗时。</p><video controls playsinline loop preload="none" src="assets/{key}_regions.mp4"></video><p>粉色 S：主体细化候选；黄色 L：非主体低置信度细化候选；蓝色 R：完全由恢复负责。这是区域归属图，S/L 内并非每步都刷新。</p></details><a href="assets/{key}_contact.jpg">首帧 / 中帧 / 尾帧原尺寸比较</a></article>"""
            )
            print(json.dumps({"case": key, "metrics": m}), flush=True)
    summary["quality_note"] = (
        "Fastest profile is approximate; inspect local edge, texture and motion differences. This is a VBench prompt sample, not the full VBench score."
    )
    (args.out / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total_fast = sum(x["fast_seconds"] for x in summary["results"])
    total_gt = sum(x["gt50_seconds"] for x in summary["results"])
    mean_lpips = np.mean(
        [x["metrics"]["lpips_alex_416x240"] for x in summary["results"]]
    )
    cli_file = PACKAGE / "tests/cli_smoke.json"
    cli_note = ""
    if cli_file.exists():
        cli = json.loads(cli_file.read_text())
        cli_note = f"只传 prompt 的独立 CLI 复核：在线 {cli['online_seconds']:.2f}s；装载模型 {cli['model_load_seconds']:.2f}s；含装载与产物保存的墙钟时间 {cli['wall_seconds']:.2f}s。此项不参与上表加速比。"
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Wan Restore Fast · 整理与 VBench 复测</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#f1f4f8;color:#17263b;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1260px;margin:auto;padding:44px 24px}}header,article,section{{background:white;border:1px solid #dce3ec;border-radius:16px;padding:30px;margin-bottom:26px}}h1{{font-size:36px;line-height:1.25;margin:12px 0}}h2{{font-size:24px;line-height:1.4}}.eyebrow{{font-size:13px;letter-spacing:2px;color:#357590;font-weight:700}}.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:25px 0}}.stat{{background:#eef4f8;padding:18px;border-radius:10px}}.stat b{{font-size:32px;display:block}}p{{color:#50617a}}a{{color:#187599}}video,img{{width:100%;height:auto;background:#e8edf4;border-radius:8px}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #dce3ec}}.scroll{{overflow:auto}}pre{{background:#122239;color:#d7ebff;padding:18px;overflow:auto;border-radius:9px}}details{{margin:18px 0;padding:16px;background:#f5f8fb;border-radius:8px}}summary{{cursor:pointer;color:#287b96}}details video{{margin-top:18px}}.badge{{display:inline-block;background:#e4f1e8;color:#2b6740;padding:5px 12px;border-radius:20px}}@media(max-width:700px){{main{{padding:20px 12px}}header,article,section{{padding:19px}}.stats{{grid-template-columns:1fr}}h1{{font-size:28px}}}}</style><main>
    <header><div class="eyebrow">REPRODUCIBLE RELEASE · 2026-09-07</div><h1>先恢复，再自适应细化<br>最快版本的独立整理与复测</h1><p>保留已选 β=2.0 / τ=0.010，清理冗余组件，输入一个 prompt 即可运行。外部 Wan 通过明确接口接入。</p><div class="stats"><div class="stat"><b>{total_gt/total_fast:.2f}×</b>三例总耗时比</div><div class="stat"><b>{total_fast/3:.2f}s</b>加速版平均在线耗时</div><div class="stat"><b>{mean_lpips:.4f}</b>平均 LPIPS-Alex</div></div><span class="badge">历史恢复 9/9 数值一致 · 完整案例视频 SHA256 一致</span><p>加速与 GT50 的视觉比较是近似方法的比较；清理版与旧最快版的一致性另行核对。不能将后者理解为与 GT50 无差异。</p></header>
    <section><h2>本次重新测试</h2><div class="scroll"><table><thead><tr><th>VBench ID</th><th>最快版</th><th>原版 GT50</th><th>加速比</th><th>LPIPS ↓</th><th>PSNR ↑</th><th>SSIM ↑</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p>A800 80GB · 832×480 · 41帧 · 16fps · 50步 · seed20260828。三个 prompt 预先固定，每例每方法测量一次。LPIPS 在416×240逐帧评估，PSNR/SSIM在保存的视频上计算。这是 VBench prompt 样例复测，不是完整 VBench 总分。</p><p>总耗时：最快版 {total_fast:.2f}s，GT50 {total_gt:.2f}s。在线计时包含文本编码、10步预热、恢复/选区、端点观察、细化、最终 VAE 与视频编码；不含模型装载、指标和报告。模型装载单独实测 {summary['model_load_seconds']:.2f}s。Triton 缓存可以被前面的回归检查预热。</p><p>{cli_note}</p><a href="results.json">下载完整数据与测试协议</a></section>
    {''.join(cards)}
    <section><h2>整个方法</h2><a href="assets/method.svg"><img src="assets/method.svg" alt="完整生成方法框图"></a><p>前10步原版 Wan → latent 中提取 F/H → 恢复 R 与质量预测 → prompt 自动主体定位和互斥 S/L/R → 沿11–50步逐 token 自适应细化 → 归一化组装 → 最终 VAE。</p><p>纯恢复区完全不刷新；主体与低置信度区根据真实 Wan 对恢复 R 的修正、新修正的意外量、稳定证据与重要性决定是否刷新。每步做轻量需求判断，不设固定刷新档位。最后一步刷新所有细化候选，不增加尾部步骤。</p><p>固定 R 提供端点 K/V 参照，与 Step10 K/V 和最近实测修正共同服务跳过 token。融合算子减少参照构建和残差更新时的中间张量，不再增加额外近似。原版没有预先恢复的 R，因此不能直接使用这一固定端点计算流。</p></section>
    <section><h2>恢复网络：一张完整结构图</h2><a href="assets/restorer.svg"><img src="assets/restorer.svg" alt="完整恢复网络层级和输入输出"></a><p>F/H 小网络直接接收16通道 z10，输出46通道 F 和1通道 H；H表示细节强度。F与基础/细节残差共同引导 R，质量头同时读取隐藏特征与 |R−z10|，得到 C、误差 e 和风险 u。主体头是另一个已有的轻量 latent 头；线上不运行 YOLO，训练时曾用离线 YOLOE 标签。</p><p>保留恢复器72,911,811参数、F/H提取器95,567参数、主体头88,481参数。删除未使用的旧质量头123,073参数，删除未参与生成的第二份 Diffusers DiT；有效权重没有重新训练。</p></section>
    <section><h2>一条命令复现</h2><pre>cd /root/autodl-tmp/wan_restore_fast\n./run.sh "A person is riding a bike"</pre><p>只有 prompt 必填。seed默认20260828，路径自动生成，主体和调度全自动。文件夹内含 README、安装、方法、接口与复现说明；Wan源码/权重保持包外。其他机器按文档一次性准备依赖与外部模型后使用相同入口。</p><p>质量请以完整联合视频判断，特别留意细小纹理、轮廓、人物动作和镜头运动；本报告不宣称所有 prompt 无可见差异或稳定达到同一加速比。</p><p>来源：<a href="https://github.com/Wan-Video/Wan2.1">官方 Wan2.1</a> · <a href="https://github.com/Vchitect/VBench">官方 VBench prompt 集</a>。没有使用额外训练、在线图像编码器或 GT 辅助生成。</p></section></main></html>"""
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.out / "index.html"),
                "aggregate_speedup": total_gt / total_fast,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
