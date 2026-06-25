# Per-game art (not included in this repository)

This directory is intentionally empty in version control. Only `.gitkeep` is tracked.

Official game logos, key art, and box art are copyrighted by their respective owners (Nintendo,
Monolith Soft, Bandai Namco Entertainment, and Square Enix). They are not committed to this repository
and are not redistributed with this project. All rights are reserved to those owners.

The web UI loads optional `<code>-logo.png` and `<code>-bg.jpg` files from this folder at runtime. To
populate it locally, run `python scripts/fetch_art.py` (see that script for sources and licensing
notes). The image files are git-ignored. When the art is absent, the UI falls back to styled text
wordmarks, so the app runs fully without it.

This is an unofficial, non-commercial fan project and is not affiliated with or endorsed by any rights
holder.
