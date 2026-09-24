"""Generate the profile's SVG assets in light and dark variants.

Run from the repo root:  python3 scripts/build_assets.py
Writes assets/header-*.svg, assets/numbers-*.svg and assets/cards/*-*.svg.
Standard library only. Text is wrapped with an approximate width model,
so keep copy short and check the rendered result after edits.
"""

import math
import random
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

THEMES = {
    "dark": {
        "bg": "#0d1117", "panel": "#161b22", "line": "#30363d",
        "text": "#e6edf3", "muted": "#8b949e", "accent": "#7ee2b8",
        "accent2": "#79c0ff", "grid": "#21262d",
    },
    "light": {
        "bg": "#ffffff", "panel": "#f6f8fa", "line": "#d0d7de",
        "text": "#1f2328", "muted": "#59636e", "accent": "#0a7f5a",
        "accent2": "#0969da", "grid": "#eaeef2",
    },
}

CARDS = [
    {
        "slug": "numpy-lm", "tag": "01  LANGUAGE MODELS",
        "title": "A language model in pure NumPy",
        "body": "Decoder-only transformer on my own autograd engine: RoPE, grouped-query "
                "attention, SwiGLU, QK-Norm and a KV-cache. Pretrained on DailyDialog, "
                "then tuned on EmpatheticDialogues.",
        "metric": "3.87M", "label": "parameters · 78 unit tests",
    },
    {
        "slug": "qwen-tools", "tag": "02  POST-TRAINING",
        "title": "Teaching Qwen3-0.6B to call tools",
        "body": "LoRA fine-tune with TRL on 1,423 tool and chat examples. Assistant-only "
                "loss and five file and shell tools in Hermes format.",
        "metric": "6 → 11/12", "label": "exact tool calls, greedy eval",
    },
    {
        "slug": "dual-stream", "tag": "03  DEEPFAKE FORENSICS",
        "title": "Dual-Stream deepfake detection",
        "body": "Sobel boundary and multi-band FFT streams, fused by iterative "
                "cross-attention. 227,504 frames; code, data and checkpoint are public.",
        "metric": "96.3%", "label": "AUC on a video-disjoint split",
    },
    {
        "slug": "mance", "tag": "04  NLP · IEEE ICCCMLA 2025",
        "title": "MANCE: morphology-aware embeddings",
        "body": "Words built from nested character sequences, so related word forms "
                "share what they learn. Transformer, CNN and BiLSTM encoders.",
        "metric": "99.0%", "label": "DBpedia · 93.5% AG News",
    },
    {
        "slug": "deepguard", "tag": "05  ON-DEVICE ML",
        "title": "DeepGuard for iOS",
        "body": "EfficientNet-B1 deepfake detector trained on 25k images on an Apple M4, "
                "then converted to Core ML for local inference.",
        "metric": "26 MB", "label": "Core ML model · under 500 ms",
    },
    {
        "slug": "bie", "tag": "06  MODEL COMPRESSION",
        "title": "Bit-index encoding",
        "body": "Stores sparse weights as the positions of nonzero bits, with Numba "
                "sparse matrix multiplication that runs on the compressed form.",
        "metric": "40×", "label": "compression at 95% sparsity",
    },
]

NUMBERS = [
    ("1st", "Global Challenge Lab 2026", "Imperial College London"),
    ("IEEE", "First-author paper", "ICCCMLA 2025"),
    ("19", "AI from Scratch posts", "maths first, then code"),
    ("9.72", "B.Tech CGPA out of 10", "KL University"),
]


def text_width(s, size):
    """Rough rendered width for a proportional sans-serif font."""
    return sum(size * (0.3 if c in " .,:;'il|" else 0.52) for c in s)


def wrap(s, size, max_width):
    lines, line = [], ""
    for word in s.split():
        trial = f"{line} {word}".strip()
        if text_width(trial, size) <= max_width:
            line = trial
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


def svg(width, height, title, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" fill="none" role="img">\n'
        f"<title>{escape(title)}</title>\n{body}</svg>\n"
    )


def header(t):
    w, h = 1200, 400
    parts = [
        f'<rect width="{w}" height="{h}" rx="16" fill="{t["panel"]}"/>',
        f'<rect x=".5" y=".5" width="{w - 1}" height="{h - 1}" rx="15.5" stroke="{t["line"]}"/>',
        f'<g font-family="{FONT}">',
        f'<text x="64" y="84" font-family="{MONO}" font-size="16" letter-spacing="3" fill="{t["accent"]}">GRUHESH SRI SAI KARTHIK KURRA</text>',
        f'<text x="60" y="170" font-size="60" font-weight="700" letter-spacing="-1.5" fill="{t["text"]}">I write the maths,</text>',
        f'<text x="60" y="240" font-size="60" font-weight="700" letter-spacing="-1.5" fill="{t["accent"]}">then I train the model.</text>',
        f'<text x="64" y="304" font-size="21" fill="{t["muted"]}">MSc Computing (AI &amp; ML) · Imperial College London · 2026–27</text>',
        f'<text x="64" y="340" font-family="{MONO}" font-size="15" fill="{t["muted"]}">language models · deepfake forensics · on-device ML</text>',
        "</g>",
    ]

    # A training-loss plot: pretraining, then a fine-tuning phase after the dashed line.
    x0, y0, pw, ph = 820, 70, 320, 250
    parts.append(f'<g transform="translate({x0} {y0})">')
    for i in range(6):
        y = ph * i / 5
        parts.append(f'<path d="M0 {y:.1f}H{pw}" stroke="{t["grid"]}"/>')
    parts.append(f'<path d="M0 0V{ph}H{pw}" stroke="{t["line"]}" stroke-width="1.5"/>')
    rng = random.Random(7)
    split = 0.68
    pts = []
    for i in range(121):
        f = i / 120
        if f < split:
            loss = 0.16 + 0.78 * math.exp(-f * 6.5)
        else:
            loss = 0.12 + 0.20 * math.exp(-(f - split) * 14)
        loss += rng.uniform(-0.018, 0.018) * (1.2 - f)
        pts.append((f * pw, ph * (1 - loss)))
    d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    parts.append(f'<path d="{d}" stroke="{t["accent"]}" stroke-width="2.5" stroke-linejoin="round"/>')
    sx = pw * split
    parts.append(f'<path d="M{sx:.1f} 0V{ph}" stroke="{t["accent2"]}" stroke-dasharray="5 5"/>')
    parts.append(
        f'<g font-family="{MONO}" font-size="13" fill="{t["muted"]}">'
        f'<text x="8" y="-14">loss</text>'
        f'<text x="{sx / 2 - 30:.0f}" y="{ph + 28}">pretrain</text>'
        f'<text x="{sx + 22:.0f}" y="{ph + 28}" fill="{t["accent2"]}">fine-tune</text>'
        f"</g></g>"
    )
    return svg(w, h, "Gruhesh Sri Sai Karthik Kurra. I write the maths, then I train the model.", "\n".join(parts))


def numbers(t):
    w, h = 1200, 170
    gap = 20
    tw = (w - gap * (len(NUMBERS) - 1)) / len(NUMBERS)
    parts = [f'<g font-family="{FONT}">']
    for i, (big, label, sub) in enumerate(NUMBERS):
        x = i * (tw + gap)
        parts += [
            f'<rect x="{x + .5:.1f}" y=".5" width="{tw - 1:.1f}" height="{h - 1}" rx="12" fill="{t["panel"]}" stroke="{t["line"]}"/>',
            f'<text x="{x + 28:.1f}" y="72" font-size="46" font-weight="700" fill="{t["accent"]}">{escape(big)}</text>',
            f'<text x="{x + 28:.1f}" y="110" font-size="19" font-weight="600" fill="{t["text"]}">{escape(label)}</text>',
            f'<text x="{x + 28:.1f}" y="138" font-size="16" fill="{t["muted"]}">{escape(sub)}</text>',
        ]
    parts.append("</g>")
    alt = "; ".join(f"{b} {l}" for b, l, _ in NUMBERS)
    return svg(w, h, alt, "\n".join(parts))


def card(c, t):
    w, h = 600, 300
    pad = 36
    parts = [
        f'<rect x=".5" y=".5" width="{w - 1}" height="{h - 1}" rx="14" fill="{t["panel"]}" stroke="{t["line"]}"/>',
        f'<path d="M1 28V272" stroke="{t["accent"]}" stroke-width="3"/>',
        f'<g font-family="{FONT}">',
        f'<text x="{pad}" y="50" font-family="{MONO}" font-size="14" letter-spacing="1.5" fill="{t["accent"]}">{escape(c["tag"])}</text>',
        f'<text x="{pad}" y="92" font-size="26" font-weight="700" fill="{t["text"]}">{escape(c["title"])}</text>',
    ]
    lines = wrap(c["body"], 18, w - 2 * pad)
    if len(lines) > 4:
        raise ValueError(f'{c["slug"]}: body wraps to {len(lines)} lines; shorten it')
    for i, line in enumerate(lines):
        parts.append(f'<text x="{pad}" y="{128 + i * 26}" font-size="18" fill="{t["muted"]}">{escape(line)}</text>')
    parts += [
        f'<path d="M{pad} 226H{w - pad}" stroke="{t["line"]}"/>',
        f'<text x="{pad}" y="268" font-size="30" font-weight="700" fill="{t["accent"]}">{escape(c["metric"])}</text>',
        f'<text x="{w - pad - 34}" y="265" text-anchor="end" font-size="16" fill="{t["muted"]}">{escape(c["label"])}</text>',
        f'<text x="{w - pad}" y="266" text-anchor="end" font-size="22" fill="{t["accent2"]}">→</text>',
        "</g>",
    ]
    title = f'{c["title"]}. {c["body"]} {c["metric"]} {c["label"]}.'
    return svg(w, h, title, "\n".join(parts))


def main():
    (ASSETS / "cards").mkdir(parents=True, exist_ok=True)
    for name, t in THEMES.items():
        (ASSETS / f"header-{name}.svg").write_text(header(t), encoding="utf-8")
        (ASSETS / f"numbers-{name}.svg").write_text(numbers(t), encoding="utf-8")
        for c in CARDS:
            (ASSETS / "cards" / f'{c["slug"]}-{name}.svg').write_text(card(c, t), encoding="utf-8")
    print(f"Wrote {2 * (2 + len(CARDS))} SVGs to {ASSETS}")


if __name__ == "__main__":
    main()
