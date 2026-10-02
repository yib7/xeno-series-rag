"""Fetch + optimize per-game brand art for the web UI into xeno_rag/web/static/art/.

Sources are the Xeno Series Wiki and Wikimedia Commons (logos that are free / below the copyright
threshold; box art used fair-use as a faded background wash for a local, non-commercial tool).
Logos are trimmed + capped at 640px wide PNG; box art is downscaled to a 760px JPG. Re-runnable.

Coverage is intentionally partial: games without a clean logo fall back to a styled text wordmark
in the UI (see ART in static/index.html). Drop a "<code>-logo.png" / "<code>-bg.jpg" in to fill gaps.

    python scripts/fetch_art.py
"""
import os
from io import BytesIO
from urllib.parse import urlsplit

import requests
from PIL import Image

UA = "XenoRAG/1.4.0 (+https://github.com/yib7/xeno-series-rag)"
XENO = "https://www.xenoserieswiki.org/w/api.php"
COMM = "https://commons.wikimedia.org/w/api.php"
# The API reply names the image URL; only fetch it from the wiki's own hosts.
TRUSTED_HOSTS = ("xenoserieswiki.org", "wikimedia.org")
OUT = os.path.join("xeno_rag", "web", "static", "art")

# (output base, api, File: page): provenance for each asset.
LOGOS = [
    ("xs1-logo", COMM, "File:Xenosaga logo.png"),
    ("xc1-logo", COMM, "File:Xenoblade Chronicles logo.webp"),
    ("xc3-logo", XENO, "File:Xenoblade Chronicles 3 logo white.png"),
    ("xcx-logo", XENO, "File:Xenoblade Chronicles X Definitive Edition logo alt.png"),
]
BACKGROUNDS = [
    ("xg-bg",  XENO, "File:Xenogears NTSC-U.jpg"),
    ("xc1-bg", XENO, "File:Xenoblade 1 PAL.png"),
    ("xcx-bg", XENO, "File:Xenoblade X PAL.jpg"),
    ("xc2-bg", XENO, "File:Xenoblade 2 PAL.jpg"),
    ("xc3-bg", XENO, "File:XC3 PAL boxart.png"),
]

S = requests.Session()
S.headers["User-Agent"] = UA


def fetch(api, file_title):
    r = S.get(api, params={"action": "query", "format": "json", "titles": file_title,
                           "prop": "imageinfo", "iiprop": "url"}, timeout=30).json()
    pages = r.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    info = page.get("imageinfo")
    if not info:
        raise SystemExit(f"fetch_art: no image found for {file_title!r} on {api} "
                         "(renamed or deleted on the wiki? update the title in LOGOS/BACKGROUNDS).")
    url = info[0]["url"]
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not any(host == h or host.endswith("." + h) for h in TRUSTED_HOSTS):
        raise SystemExit(f"fetch_art: {file_title!r} points at an untrusted image host {host!r}; refusing to download it.")
    resp = S.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def trim_alpha(im):
    if im.mode != "RGBA":
        return im
    bbox = im.split()[-1].getbbox()
    return im.crop(bbox) if bbox else im


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, api, title in LOGOS:
        im = trim_alpha(Image.open(BytesIO(fetch(api, title))).convert("RGBA"))
        if im.width > 640:
            im = im.resize((640, round(im.height * 640 / im.width)), Image.LANCZOS)
        im.save(os.path.join(OUT, name + ".png"), optimize=True)
        print("logo", name, im.size)
    for name, api, title in BACKGROUNDS:
        im = Image.open(BytesIO(fetch(api, title))).convert("RGB")
        scale = 760 / max(im.size)
        if scale < 1:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        im.save(os.path.join(OUT, name + ".jpg"), "JPEG", quality=82, optimize=True, progressive=True)
        print("bg  ", name, im.size)


if __name__ == "__main__":
    main()
