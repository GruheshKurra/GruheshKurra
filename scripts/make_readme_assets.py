"""Build the SVG header and pill buttons used in README.md.

Text is converted to outlines, so the images look the same on every machine
(GitHub serves README images through a proxy that blocks web fonts). Each image
comes in a light and a dark version, picked by <picture> in the README.

Fonts (both SIL Open Font License, not committed):
  Newsreader  https://github.com/google/fonts/tree/main/ofl/newsreader -> ~/.cache/readme-fonts/Newsreader.ttf
  Inter       https://rsms.me/inter/                                     -> ~/Library/Fonts/Inter-*.otf

Logo tiles need ImageMagick (`magick`). Run from the repository root:
  uv run --with fonttools --with uharfbuzz python scripts/make_readme_assets.py
"""

import subprocess
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

OUT = Path("assets/readme")
NEWSREADER = Path.home() / ".cache/readme-fonts/Newsreader.ttf"
INTER = Path.home() / "Library/Fonts"

# Colours follow gruheshkurra.com (light) and GitHub's dark theme (dark).
THEMES = {
    "light": dict(card="#fbfaf7", surface="#ffffff", line="#e7e3da", ink="#1b1b1a", soft="#3d3d3a",
                  muted="#6e6c66", accent="#1f4e8c", accent_soft="#eaf0f7", dot="#d9d5cb",
                  ok="#1f6b3a", ok_soft="#e5f1e8", grey="#6e6c66", grey_soft="#efede8"),
    "dark": dict(card="#161b22", surface="#161b22", line="#30363d", ink="#e6edf3", soft="#c9d1d9",
                 muted="#8b949e", accent="#8db4ea", accent_soft="#1b2635", dot="#2a313b",
                 ok="#7fd49b", ok_soft="#15261b", grey="#8b949e", grey_soft="#21262d"),
}


def num(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


class Face:
    def __init__(self, path, variations=None):
        face = hb.Face(hb.Blob.from_file_path(str(path)))
        self.hb = hb.Font(face)
        if variations:
            self.hb.set_variations(variations)
        self.upem = face.upem
        tt = TTFont(str(path))
        self.glyphs = tt.getGlyphSet(location=variations) if variations else tt.getGlyphSet()
        self.order = tt.getGlyphOrder()

    def _shape(self, text):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb, buf, {"kern": True, "liga": True})
        return zip(buf.glyph_infos, buf.glyph_positions)

    def width(self, text, size, tracking=0.0):
        adv = sum(p.x_advance for _, p in self._shape(text))
        return (adv + tracking * self.upem * len(text)) * size / self.upem

    def path(self, text, x, y, size, tracking=0.0):
        """SVG path data for `text` with its baseline starting at (x, y)."""
        s = size / self.upem
        pen = SVGPathPen(self.glyphs, ntos=num)
        cursor = 0
        for info, pos in self._shape(text):
            name = self.order[info.codepoint]
            dx = x + (cursor + pos.x_offset) * s
            dy = y - pos.y_offset * s
            self.glyphs[name].draw(TransformPen(pen, (s, 0, 0, -s, dx, dy)))
            cursor += pos.x_advance + tracking * self.upem
        return pen.getCommands()


SERIF = Face(NEWSREADER, {"wght": 500, "opsz": 48})
SANS = Face(INTER / "Inter-Regular.otf")
SANS_MED = Face(INTER / "Inter-Medium.otf")
SANS_SEMI = Face(INTER / "Inter-SemiBold.otf")


def svg(width, height, title, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{num(width)}" height="{num(height)}" '
            f'viewBox="0 0 {num(width)} {num(height)}" role="img" aria-label="{title}">'
            f"<title>{title}</title>{body}</svg>\n")


def text(face, string, x, y, size, fill, tracking=0.0):
    return f'<path fill="{fill}" d="{face.path(string, x, y, size, tracking)}"/>'


NAME = "Gruhesh Sri Sai Karthik Kurra"
ROLE = "MSc Computing (Artificial Intelligence and Machine Learning) · Imperial College London"
TOPICS = ["Language models and NLP", "Representation learning", "Diverse generation", "Efficient models"]

# A sparse weight matrix: the filled cells are the non-zero weights (a nod to BIE).
SPARSE = [
    "1.....1.",
    "..1.....",
    ".....1..",
    "1..1....",
    "......1.",
    ".1......",
    "....1..1",
]


def header(c):
    w, h = 1000, 270
    out = [f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="14" fill="{c["card"]}" stroke="{c["line"]}"/>']

    x = 52
    out.append(text(SANS_SEMI, "AI RESEARCHER · LONDON", x, 70, 12.5, c["accent"], tracking=0.12))
    out.append(text(SERIF, NAME, x - 2, 124, 46, c["ink"]))
    out.append(text(SANS, ROLE, x, 160, 16, c["muted"]))

    # Topic chips
    cx, cy, ch = x, 192, 32
    for t in TOPICS:
        tw = SANS_MED.width(t, 13.5)
        cw = tw + 28
        out.append(f'<rect x="{num(cx)}" y="{cy}" width="{num(cw)}" height="{ch}" rx="16" fill="{c["accent_soft"]}"/>')
        out.append(text(SANS_MED, t, cx + 14, cy + 21, 13.5, c["accent"]))
        cx += cw + 8

    # Sparse matrix motif, top right
    cell, gap = 13, 7
    gx = w - 52 - len(SPARSE[0]) * (cell + gap) + gap
    gy = 52
    for r, row in enumerate(SPARSE):
        for col, v in enumerate(row):
            fill = c["accent"] if v == "1" else c["dot"]
            out.append(f'<rect x="{gx + col * (cell + gap)}" y="{gy + r * (cell + gap)}" width="{cell}" '
                       f'height="{cell}" rx="3" fill="{fill}"/>')

    name_end = x + SERIF.width(NAME, 46)
    assert name_end < gx - 24, f"name ({name_end:.0f}px) runs into the motif ({gx}px)"
    role_end = x + SANS.width(ROLE, 16)
    assert role_end < gx - 24, f"role ({role_end:.0f}px) runs into the motif ({gx}px)"
    assert cx < w - 40, f"chips end at {cx:.0f}px"
    return svg(w, h, f"{NAME}. {ROLE}.", "".join(out))


def pill(label, fill, ink, stroke=None, size=13.5, height=32, radius=16, pad=16, face=SANS_MED):
    tw = face.width(label, size)
    w = tw + 2 * pad
    border = f' stroke="{stroke}"' if stroke else ""
    inset = 0.5 if stroke else 0
    body = (f'<rect x="{inset}" y="{inset}" width="{num(w - 2 * inset)}" height="{height - 2 * inset}" '
            f'rx="{radius}" fill="{fill}"{border}/>'
            + text(face, label, pad, height / 2 + size * 0.36, size, ink))
    return svg(w, height, label, body)


LINKS = ["Website", "CV", "Google Scholar", "ORCID", "LinkedIn", "Email"]
PUB_LINKS = ["Paper", "Code", "Results", "Dataset"]
STATUS = {"published": ("Published", "ok"), "accepted": ("Accepted", "accent"), "preprint": ("Preprint", "grey")}


def slug(s):
    return s.lower().replace(" ", "-")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for theme, c in THEMES.items():
        files = {f"header-{theme}.svg": header(c)}
        for label in LINKS:
            files[f"link-{slug(label)}-{theme}.svg"] = pill(label, c["surface"], c["soft"], stroke=c["line"])
        for label in PUB_LINKS:
            files[f"pub-{slug(label)}-{theme}.svg"] = pill(label, c["accent_soft"], c["accent"],
                                                          size=12.5, height=26, radius=6, pad=11)
        for key, (label, tone) in STATUS.items():
            files[f"status-{key}-{theme}.svg"] = pill(label, c[f"{tone}_soft"], c[tone], size=11.5, height=22,
                                                     radius=11, pad=9, face=SANS_SEMI)
        for name, content in files.items():
            (OUT / name).write_text(content)
    # Organisation logos on white rounded tiles, so dark logos stay visible in dark mode.
    # The IIIT Hyderabad file is the full wordmark; keep only the banyan-tree emblem so it reads at 48 px.
    crops = {"iiith": ["-crop", "707x190+0+0", "+repage", "-trim", "+repage",
                       "-crop", "275x163+0+0", "+repage", "-trim", "+repage"], "infoajax": []}
    for org, crop in crops.items():
        subprocess.run(["magick", "-size", "192x192", "xc:none", "-fill", "white", "-stroke", "#e7e3da",
                        "-strokewidth", "2", "-draw", "roundrectangle 1,1 190,190 30,30",
                        "(", f"assets/org-logos/{org}.png", *crop, "-resize", "148x148", ")",
                        "-gravity", "center", "-composite", str(OUT / f"logo-{org}.png")], check=True)
    print(f"wrote {len(list(OUT.glob('*.svg')))} files to {OUT}")


if __name__ == "__main__":
    main()
