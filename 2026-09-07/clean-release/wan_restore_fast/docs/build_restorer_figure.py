"""Single connected, cuboid-style figure of the existing restoration network."""

from pathlib import Path
from html import escape
import json

OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)
W, H = 3540, 1640
C = {
    "blue": ("#d4e2f5", "#ecf3fc", "#a5bfdc", "#50749c"),
    "green": ("#d6eee7", "#eef9f4", "#a5cdbf", "#467866"),
    "amber": ("#f4dfc8", "#fff1e2", "#d4ae89", "#a17549"),
    "purple": ("#e7ddf2", "#f5f0fa", "#c2afd8", "#8263a3"),
    "rose": ("#f5dbe0", "#fff0f3", "#dba7b2", "#a46777"),
    "out": ("#d9eacb", "#f0f8e9", "#adc994", "#66844d"),
    "gray": ("#e4e9ef", "#f4f7fa", "#bac6d2", "#697d91"),
}
wire = []
node = []
txt = []


def text(x, y, s, size=24, anchor="middle", color="#26364a", bold=False):
    txt.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{650 if bold else 400}" paint-order="stroke" stroke="white" stroke-width="3" stroke-linejoin="round">{escape(s)}</text>'
    )


def lines(x, y, ss, size=22, anchor="middle", color="#52667c", gap=31):
    for i, s in enumerate(ss):
        text(x, y + i * gap, s, size, anchor, color)


def route(ps, color="#607185", dash=None, width=2.3, arrow=True):
    d = "M " + " L ".join(f"{x},{y}" for x, y in ps)
    attrs = f'd="{d}" fill="none" stroke-linejoin="round" stroke-linecap="round"'
    wire.append(f'<path {attrs} stroke="white" stroke-width="{width+5}"/>')
    wire.append(
        f'<path {attrs} stroke="{color}" stroke-width="{width}"'
        + (f' stroke-dasharray="{dash}"' if dash else "")
        + (' marker-end="url(#arr)"' if arrow else "")
        + "/>"
    )


def box(x, y, w, h, key, label="", count=None, above="", below=None):
    front, top, side, edge = C[key]
    x0 = x - w / 2
    y0 = y - h / 2
    d = 17
    node.append(
        f'<polygon points="{x0},{y0} {x0+d},{y0-d} {x0+w+d},{y0-d} {x0+w},{y0}" fill="{top}" stroke="{edge}" stroke-width="1.8"/>'
    )
    node.append(
        f'<polygon points="{x0+w},{y0} {x0+w+d},{y0-d} {x0+w+d},{y0+h-d} {x0+w},{y0+h}" fill="{side}" stroke="{edge}" stroke-width="1.8"/>'
    )
    node.append(
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="{front}" stroke="{edge}" stroke-width="1.8"/>'
    )
    if count:
        for f in [1 / 3, 2 / 3]:
            node.append(
                f'<path d="M{x0+w*f},{y0}V{y0+h}" stroke="{edge}" opacity="0.3"/>'
            )
    if label:
        text(x, y + 8, label, 22 if w >= 60 else 19, color=edge, bold=True)
    if above:
        text(x, y0 - 31, above, 25, bold=True)
    if count:
        text(x, y0 + h + 35, "×" + str(count), 24, color=edge, bold=True)
    if below:
        lines(x, y0 + h + (71 if count else 39), below, 21)


def join(x, y, s="+"):
    node.append(
        f'<circle cx="{x}" cy="{y}" r="21" fill="white" stroke="#697c91" stroke-width="2"/>'
    )
    text(x, y + 8, s, 29)


def tag(x, y, label, key="gray", w=190):
    front, _, _, edge = C[key]
    node.append(
        f'<rect x="{x-w/2}" y="{y-24}" width="{w}" height="48" rx="5" fill="{front}" stroke="{edge}" stroke-width="1.4"/>'
    )
    text(x, y + 8, label, 23, color=edge)


text(70, 57, "Latent Restoration Network", 43, "start", bold=True)
text(70, 100, "完整恢复网络 · 单张连通结构图", 29, "start", color="#6c7d91")
text(3480, 62, "z₁₀ → F / H → 基础与细节恢复 → 融合 → R、C", 28, "end", color="#50647d")

# Shared input and original-latent residual bus.
box(105, 740, 46, 192, "gray", "z₁₀", above="16C", below=["11×60×104"])
route([(145, 740), (220, 740), (220, 420), (270, 420)], C["blue"][3])
route([(180, 740), (180, 1230), (332, 1230)], C["green"][3])
route([(200, 740), (200, 840), (357, 840)], C["amber"][3])
route([(164, 740), (164, 147), (2240, 147), (2240, 396)], "#9aabba", width=1.9)
text(235, 133, "z₁₀ 原始残差", 22, "start", color="#8090a0")

# U-Net trunk, fully unfolded spatial scales.
text(325, 222, "基础恢复主干", 31, "start", color=C["blue"][3], bold=True)
text(325, 256, "Heat-conditioned U-Net", 22, "start", color="#7e92aa")
box(304, 420, 37, 160, "blue", "Conv", above="64C", below=["1×3×3"])
xs = [425, 560, 695, 830, 1005, 1180, 1315, 1450, 1585]
chs = [64, 128, 256, 512, 1024, 512, 256, 128, 64]
ys = [420, 448, 477, 502, 530, 502, 477, 448, 420]
hs = [160, 136, 112, 88, 66, 88, 112, 136, 160]
ws = [39, 47, 55, 65, 92, 65, 55, 47, 39]
ns = [1, 1, 1, 8, 6, 1, 1, 1, 1]
shapes = ["64×112", "32×56", "16×28", "8×14", "4×7", "8×14", "16×28", "32×56", "64×112"]
route([(340, 420), (391, 420)], C["blue"][3])
for i in range(4):
    y = 275 + i * 17
    route(
        [
            (xs[i] + ws[i] / 2 + 17, ys[i] - hs[i] / 2 - 17),
            (xs[i] + ws[i] / 2 + 17, y),
            (xs[8 - i], y),
            (xs[8 - i], ys[8 - i] - hs[8 - i] / 2 - 24),
        ],
        "#aabbd0",
        width=1.7,
    )
text(1020, 260, "同尺度 skip 相加", 21, color="#8b9eb6")
for i, (x, y, cc, h, w, n, shape) in enumerate(zip(xs, ys, chs, hs, ws, ns, shapes)):
    box(x, y, w, h, "blue", "", n, str(cc) + "C", below=[shape])
    if i < 8:
        route(
            [(x + w / 2 + 20, y), (xs[i + 1] - ws[i + 1] / 2 - 16, ys[i + 1])],
            C["blue"][3],
        )
route([(1620, 420), (1664, 420)], C["blue"][3])
box(1700, 420, 40, 160, "blue", "X", above="64C")
route([(1740, 420), (1820, 420)], C["blue"][3])
box(1856, 420, 40, 115, "blue", "Conv", above="16C", below=["LN + 1×3×3", "× 0.107"])
route([(1895, 420), (1964, 420)], C["blue"][3])
box(2000, 420, 38, 115, "out", "B", above="16C")
route([(2037, 420), (2217, 420)])
join(2240, 420)
route([(2263, 420), (3140, 420), (3140, 724)], "#8d9da8", width=2)
text(2680, 400, "基础恢复端点  z₁₀ + B", 25, color="#7a8c9d")
route([(2040, 420), (2140, 420), (2140, 646)], C["blue"][3])

# H conditioning bus, connected once and distributed by per-scale projections.
route(
    [
        (970, 1150),
        (1010, 1150),
        (1010, 1070),
        (865, 1070),
        (865, 1010),
        (248, 1010),
        (248, 660),
        (1650, 660),
    ],
    C["amber"][3],
    dash="8 7",
    width=1.8,
    arrow=False,
)
for i in [0, 1, 2, 3, 5, 6, 7, 8]:
    x = xs[i] + ws[i] / 2 + 55
    route(
        [(x, 660), (x, ys[i] + 23), (xs[i] + ws[i] / 2 + 20, ys[i] + 23)],
        C["amber"][3],
        dash="7 6",
        width=1.5,
    )
text(735, 648, "H → 按尺度 Resize + 1×1×1 投影", 23, color=C["amber"][3])

# Spatiotemporal detail branch, X and H projections visibly additive.
text(357, 707, "细节恢复分支", 29, "start", color=C["amber"][3], bold=True)
box(420, 840, 90, 122, "amber", "Bands", above="112C", below=["7组时空高频 / 差分"])
route([(482, 840), (612, 840)], C["amber"][3])
box(650, 840, 51, 122, "amber", "Conv", above="192C", below=["1×1×1"])
route([(692, 840), (918, 840)], C["amber"][3])
join(940, 840)
route([(1740, 420), (1780, 420), (1780, 726), (820, 726), (820, 749)], C["blue"][3])
box(820, 784, 47, 65, "blue", "Conv", above="X: 64→192C")
route([(857, 784), (940, 784), (940, 817)], C["blue"][3])
route([(963, 840), (1090, 840)], C["amber"][3])
box(1140, 840, 72, 125, "amber", "NAF", 12, "192C")
route([(1193, 840), (1268, 840)], C["amber"][3])
box(1320, 840, 75, 125, "amber", "TRF", 4, "192C")
route([(1375, 840), (1463, 840)], C["amber"][3])
box(1500, 840, 48, 125, "amber", "Conv", above="16C", below=["LN + 1×3×3"])
route([(1541, 840), (1637, 840)], C["amber"][3])
join(1660, 840, "×")
route([(1683, 840), (1773, 840)], C["amber"][3])
box(1810, 840, 41, 125, "out", "D", above="16C")
tag(1660, 993, "0.195 (0.2 + 0.8H̄)", "amber", w=320)
route([(1660, 968), (1660, 863)], C["amber"][3])
text(1850, 943, "B / D 裁回 60×104", 22, "start", color="#7d90a4")

# The same F/H student as existing checkpoint; H is upper so it feeds detail directly.
text(355, 1096, "F / H 提取器", 31, "start", color=C["green"][3], bold=True)
box(376, 1230, 49, 105, "green", "Conv", above="48C", below=["3×3×3"])
route([(418, 1230), (504, 1230)], C["green"][3])
box(550, 1230, 69, 105, "green", "NAF", 4, "48C")
route([(603, 1230), (692, 1230)], C["green"][3])
box(730, 1230, 49, 105, "green", "Conv", above="47C", below=["LN + 1×1×1"])
route([(772, 1230), (841, 1230), (841, 1150), (891, 1150)], C["amber"][3])
route([(841, 1230), (841, 1300), (891, 1300)], C["green"][3])
box(930, 1150, 39, 69, "amber", "H", above="1C")
text(1010, 1200, "Sigmoid", 21, color=C["amber"][3])
box(930, 1300, 43, 80, "green", "F", above="46C")
route([(970, 1150), (1100, 1150), (1100, 1064)], C["amber"][3])
box(1100, 1030, 84, 63, "amber", "Smooth", above="H̄ · 1C", below=["时间平滑 0.35"])
route([(1056, 1030), (1010, 1030), (1010, 990), (978, 990)], C["amber"][3])
box(940, 990, 41, 63, "amber", "Conv", above="1→192C")
route([(940, 941), (940, 863)], C["amber"][3])
route([(1160, 1030), (1250, 1030), (1250, 993), (1496, 993)], C["amber"][3])
route([(972, 1300), (1050, 1300)], C["green"][3])
box(1090, 1300, 48, 92, "green", "Conv", above="32C", below=["1×1×1 + SiLU"])
route([(1131, 1300), (1204, 1300)], C["green"][3])
box(1245, 1300, 48, 92, "green", "Conv", above="32C", below=["1×3×3 + SiLU"])
route([(1286, 1300), (1378, 1300)], C["green"][3])
box(1417, 1300, 42, 92, "green", "I", above="32C")

# Shared fusion trunk. Incoming tensors enter different Cat faces.
text(2100, 569, "F 引导融合", 31, "start", color=C["purple"][3], bold=True)
route([(1848, 840), (1992, 840), (1992, 785), (2078, 785)], C["amber"][3])
route(
    [(1456, 1300), (1900, 1300), (1900, 1036), (2140, 1036), (2140, 838)], C["green"][3]
)
route(
    [(164, 740), (164, 1060), (2040, 1060), (2040, 713), (2078, 713)],
    "#a4b1bc",
    width=1.8,
)
box(2140, 750, 91, 160, "purple", "Cat", above="80C", below=["z₁₀ | B | D | I"])
route([(2205, 750), (2283, 750)], C["purple"][3])
box(2325, 750, 48, 132, "purple", "Conv", above="64C", below=["1×1×1"])
route([(2366, 750), (2438, 750)], C["purple"][3])
box(2490, 750, 76, 132, "purple", "NAF", 4, "64C")
route([(2545, 750), (2604, 750)], C["purple"][3])
box(2642, 750, 40, 132, "purple", "U", above="64C", below=["LayerNorm"])
route([(2679, 750), (2730, 750), (2730, 637), (2780, 637)], C["purple"][3])
route([(2730, 750), (2730, 891), (2780, 891)], C["purple"][3])
box(2820, 637, 49, 80, "purple", "g", above="1C", below=["Conv + 2σ"])
box(2820, 891, 49, 80, "purple", "Δ", above="16C", below=["Conv × 0.05"])
route([(2862, 637), (2989, 637)], C["purple"][3])
join(3012, 637, "×")
route(
    [(1848, 840), (1941, 840), (1941, 539), (3012, 539), (3012, 614)],
    C["amber"][3],
    width=2,
)
text(2908, 520, "D", 23, color=C["amber"][3])
route([(3035, 637), (3078, 637), (3078, 750), (3117, 750)])
route([(2862, 891), (3140, 891), (3140, 771)], C["purple"][3])
join(3140, 750)
route([(3163, 750), (3269, 750)])
box(3310, 750, 50, 169, "out", "R", above="16C", below=["恢复结果"])
text(3265, 914, "R = z₁₀ + B + g⊙D + Δ", 24, color=C["out"][3])

# Quality: explicit shared-feature and correction inputs, distinct from H.
text(2100, 1250, "恢复质量预测", 31, "start", color=C["rose"][3], bold=True)
route(
    [(2679, 750), (2700, 750), (2700, 1146), (2046, 1146), (2046, 1340), (2080, 1340)],
    C["purple"][3],
    width=1.9,
)
text(2230, 1129, "共享特征 U", 22, color=C["purple"][3])
route(
    [(1456, 1300), (1843, 1300), (1843, 1450), (2080, 1450)], C["green"][3], width=1.9
)
text(1848, 1489, "I", 23, color=C["green"][3])
route(
    [(3353, 750), (3422, 750), (3422, 1210), (1983, 1210), (1983, 1400), (2080, 1400)],
    "#7d9270",
    width=1.9,
)
text(2810, 1192, "|R − z₁₀|：恢复修正幅度", 23, color="#71896c")
box(2140, 1395, 92, 151, "rose", "Cat", above="112C", below=["U | I | |R−z₁₀|"])
route([(2204, 1395), (2290, 1395)], C["rose"][3])
box(2330, 1395, 47, 120, "rose", "Conv", above="48C", below=["1×1×1 + SiLU"])
route([(2371, 1395), (2437, 1395)], C["rose"][3])
box(2490, 1395, 77, 120, "rose", "NAF", 2, "48C")
route([(2545, 1395), (2618, 1395)], C["rose"][3])
box(2660, 1395, 48, 120, "rose", "Conv", above="2C", below=["LN + 1×1×1"])
route([(2702, 1395), (2770, 1395), (2770, 1331), (2832, 1331)], C["rose"][3])
route([(2770, 1395), (2770, 1490), (2832, 1490)], C["rose"][3])
tag(2910, 1331, "q₀ → e", "rose", w=151)
tag(2910, 1490, "q₁ → u", "rose", w=151)
lines(
    3025, 1303, ["e = 0.04 · Softplus(q₀)", "C = 1 − σ((e−0.025)/0.010)"], 23, "start"
)
text(3025, 1498, "u = σ(q₁)", 25, "start")
route(
    [(2910, 1356), (2910, 1410), (3370, 1410), (3370, 1460), (3382, 1460)], C["rose"][3]
)
box(3410, 1460, 30, 85, "rose", "C", above="1C", below=["置信度"])
route([(2910, 1515), (2910, 1540), (3260, 1540), (3260, 1519)], C["rose"][3])
box(3260, 1490, 28, 54, "rose", "u", above="1C")

# Unambiguous conventions, not a separate panel or additional graph.
text(
    70,
    1545,
    "读图：长方体表示特征或运算阶段；×n 为 Temporal NAF / TRF 重复次数；细箭头为数据流，橙色虚线为 H 条件注入。",
    24,
    "start",
    color="#75869a",
)
text(
    70,
    1580,
    "T 始终为 11。U-Net 内空间补齐至 64×112；F/H、融合与质量头保持 60×104。几何尺寸为示意，维度以标签为准。",
    23,
    "start",
    color="#75869a",
)
text(
    70,
    1613,
    "有效恢复路径与权重保持一致；未使用的旧质量支路已移除。σ = Sigmoid；质量头输入 detach。",
    22,
    "start",
    color="#75869a",
)
defs = '<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7.3" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8Z" fill="#64788d"/></marker></defs>'
svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="完整恢复网络长方体结构图">{defs}<style>text{{font-family:Arial,"Microsoft YaHei","Noto Sans CJK SC",sans-serif}}</style><rect width="100%" height="100%" fill="white"/>'
    + "".join(wire + node + txt)
    + "</svg>"
)
(OUT / "restorer.svg").write_text(svg.replace("H̄", "Hₛ"), encoding="utf-8")
(OUT / "connected_figure_manifest.json").write_text(
    json.dumps(
        {
            "single_connected_figure": True,
            "effective_graph_preserved": True,
            "legacy_head_removed": True,
            "canvas": [W, H],
            "source": "verified frozen implementation; same architecture and checkpoint configuration",
            "visual_style": "vector cuboids, shared tensor paths, no subplot panels",
        },
        indent=2,
    )
)
print(OUT / "restorer.svg")
