"""Deterministic SVG diagram of the clean runtime; no external renderer needed."""

from pathlib import Path
from html import escape

out = Path(__file__).resolve().parent / "method.svg"
parts = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900"><defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0L9,4.5L0,9Z" fill="#5b718c"/></marker></defs><style>text{font-family:Arial,"Microsoft YaHei",sans-serif}</style><rect width="1600" height="900" rx="20" fill="#fff"/>'
]


def text(x, y, s, size=22, color="#263d58", anchor="middle"):
    parts.append(
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(s)}</text>'
    )


def box(x, y, w, h, title, lines=(), color="#edf3fa"):
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="13" fill="{color}" stroke="#b6c7d9" stroke-width="2"/>'
    )
    text(x + w / 2, y + 40, title, 25)
    for i, line in enumerate(lines):
        text(x + w / 2, y + 74 + 27 * i, line, 19, "#52677e")


def arrow(points):
    parts.append(
        '<polyline points="'
        + " ".join(f"{x},{y}" for x, y in points)
        + '" fill="none" stroke="#5b718c" stroke-width="2.5" marker-end="url(#a)"/>'
    )


text(60, 60, "Wan Restore Fast · 先恢复，再自适应细化", 34, anchor="start")
text(
    60,
    100,
    "β = 2.0 · τ = 0.010 · 一个 prompt · 外部原版 Wan2.1 T2V-1.3B",
    22,
    "#75859b",
    "start",
)
box(60, 160, 210, 120, "Prompt", ["T5 文字编码", "自动词组对应"])
box(335, 160, 260, 120, "原版 Wan · 1—10步", ["完整 token / CFG", "z10 + 真实含噪状态"])
box(660, 160, 245, 120, "F/H 提取器", ["输入 z10 · 16C", "输出 F46C + H1C"], "#e9f4ed")
box(
    970,
    160,
    290,
    120,
    "恢复网络",
    ["输入 z10 / F / H", "恢复端点 R + C / e / u"],
    "#e9f4ed",
)
arrow([(270, 220), (335, 220)])
arrow([(595, 220), (660, 220)])
arrow([(905, 220), (970, 220)])
box(
    335,
    355,
    315,
    135,
    "自动主体定位",
    ["已有 latent 主体概率", "与 prompt 空间响应相互对应", "无需手动类别 / mask"],
    "#f3eef8",
)
arrow([(465, 280), (465, 355)])
arrow([(782, 280), (782, 315), (620, 315), (620, 355)])
box(
    745,
    355,
    515,
    135,
    "互斥区域 · 恢复误差自适应选区",
    [
        "S：主体细化 · L：低置信度细化",
        "R 区：纯恢复（Wan 刷新次数为 0）",
        "局部阈值参考恢复误差、细节、变化量",
    ],
    "#fff2e5",
)
arrow([(650, 423), (745, 423)])
arrow([(1115, 280), (1115, 355)])
box(
    60,
    585,
    420,
    185,
    "恢复相对刷新需求 · 每步 check",
    [
        "实际 Wan 修正 d = clean − R",
        "修正创新量 / 稳定证据 / 邻域反馈",
        "主体概率与细节决定重要性",
        "需求超过 β 则刷新，无固定档位",
    ],
    "#fff2e5",
)
box(
    560,
    585,
    420,
    185,
    "Wan · 11—50步自适应细化",
    [
        "选中 token：30层完整计算",
        "跳过 token：自己的真实历史积分",
        "Step50 刷新全部细化候选",
        "Q/K/V 参照 + 实测残差融合算子",
    ],
    "#edf3fa",
)
box(
    1080,
    600,
    440,
    155,
    "三种权重组装 → 最终 VAE",
    [
        "S / L / R 权重互斥，总和 = 1",
        "最终 41 帧视频 · 不加额外尾部",
        "在线无图像预览解码",
    ],
    "#e9f4ed",
)
arrow([(875, 490), (875, 535), (270, 535), (270, 585)])
arrow([(480, 680), (560, 680)])
arrow([(980, 680), (1080, 680)])
arrow([(765, 770), (765, 815), (270, 815), (270, 770)])
text(510, 851, "只用实际新观测更新修正证据", 20, "#75859b")
arrow([(1260, 220), (1460, 220), (1460, 535), (945, 535), (945, 585)])
text(1390, 335, "固定恢复端点 R", 21)
text(1390, 368, "捕获 K/V 参照", 20)
text(1390, 401, "一次额外观察", 20)
text(1390, 434, "耗时计入在线", 20)
parts.append("</svg>")
out.write_text("".join(parts), encoding="utf-8")
