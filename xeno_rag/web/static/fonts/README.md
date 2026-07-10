# Self-hosted web fonts

Latin-subset woff2 faces served from `/static/fonts/` so the UI keeps its typography offline and
no page view touches a third-party CDN (audit finding P2-11). Files were downloaded once from
Google Fonts (fonts.gstatic.com).

| Family | File(s) | Weights | Copyright |
|--------|---------|---------|-----------|
| Cinzel (variable) | `cinzel-latin-wght.woff2` | 400-700 (variable `wght` axis) | Copyright 2020 The Cinzel Project Authors (https://github.com/NDISCOVER/Cinzel) |
| Spectral | `spectral-latin-{400,500,600,700}.woff2` | 400 / 500 / 600 / 700 | Copyright 2017 The Spectral Project Authors (https://github.com/productiontype/Spectral) |

Both families are licensed under the SIL Open Font License, Version 1.1 — full texts in
`OFL-Cinzel.txt` and `OFL-Spectral.txt` (from the Google Fonts repo). The OFL permits bundling
and redistribution with attribution; these notices satisfy that.
