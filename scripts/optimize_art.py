"""Turn the art masters dropped into static/art/ into web-sized, consistently named assets.

The masters are named per game (e.g. `xenoblade-1_logo.png`, `xenoblade-1_keyart.png`) and run
1-8 MB each, too heavy to serve. This script derives the short-code-named web set the UI loads:

    <code>-logo.png  trimmed to its alpha bbox, downscaled to LOGO_H px tall, PNG-optimized (alpha kept)
    <code>-bg.jpg    downscaled to BG_W px wide, progressive JPEG q82 (the faded key-art wash)

Masters are left untouched (gitignored) so this is re-runnable. Local + free (Pillow only).

    python scripts/optimize_art.py
"""
import os

from PIL import Image

ART = os.path.join("xeno_rag", "web", "static", "art")
LOGO_H = 240   # display ~48px tall; 240 keeps it crisp at up to ~5x DPI, files stay tiny
BG_W = 1920    # wash is blurred at 16% opacity, so 1080p-wide is plenty

# code -> (logo master, keyart master)
SOURCES = {
    "xg":  ("xenogears_logo.png",     "xenogears_keyart.png"),
    "xs1": ("xenosaga-ep1_logo.png",  "xenosaga-ep1_keyart.jpg"),
    "xs2": ("xenosaga-ep2_logo.png",  "xenosaga-ep2_keyart.png"),
    "xs3": ("xenosaga-ep3_logo.png",  "xenosaga-ep3_keyart.png"),
    "xc1": ("xenoblade-1_logo.png",   "xenoblade-1_keyart.png"),
    "xc2": ("xenoblade-2_logo.png",   "xenoblade-2_keyart.png"),
    "xc3": ("xenoblade-3_logo.png",   "xenoblade-3_keyart.png"),
    "xcx": ("xenoblade-x_logo.png",   "xenoblade-x_keyart.png"),
}


def _kb(path):
    return os.path.getsize(path) / 1024


def optimize_logo(src, dst):
    im = Image.open(src).convert("RGBA")
    bbox = im.getchannel("A").getbbox()       # crop the transparent margin away
    if bbox:
        im = im.crop(bbox)
    if im.height > LOGO_H:
        w = round(im.width * LOGO_H / im.height)
        im = im.resize((w, LOGO_H), Image.LANCZOS)
    im.save(dst, "PNG", optimize=True)
    return im.size


def optimize_bg(src, dst):
    im = Image.open(src).convert("RGB")
    if im.width > BG_W:
        h = round(im.height * BG_W / im.width)
        im = im.resize((BG_W, h), Image.LANCZOS)
    im.save(dst, "JPEG", quality=82, progressive=True, optimize=True)
    return im.size


def main():
    for code, (logo_src, bg_src) in SOURCES.items():
        lp, bp = os.path.join(ART, logo_src), os.path.join(ART, bg_src)
        lo, bo = os.path.join(ART, f"{code}-logo.png"), os.path.join(ART, f"{code}-bg.jpg")
        if os.path.exists(lp):
            sz = optimize_logo(lp, lo)
            print(f"{code}-logo.png  {sz[0]}x{sz[1]:<4d} {_kb(lo):6.0f} KB  <- {logo_src}")
        else:
            print(f"!! missing logo master: {logo_src}")
        if os.path.exists(bp):
            sz = optimize_bg(bp, bo)
            print(f"{code}-bg.jpg    {sz[0]}x{sz[1]:<4d} {_kb(bo):6.0f} KB  <- {bg_src}")
        else:
            print(f"!! missing keyart master: {bg_src}")


if __name__ == "__main__":
    main()
