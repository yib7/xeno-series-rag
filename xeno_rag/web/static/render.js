// Minimal, safe Markdown renderer for the Xeno-RAG answer pane.
// Extracted from index.html so it can be unit-tested under Node (`node --test`) AND loaded in the
// browser via <script src="/static/render.js">. UMD-ish: exports for CommonJS, globals otherwise.
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    Object.assign(root, api);
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function inline(s, cite) {
    // Stash code spans and links FIRST as opaque tokens, so emphasis (*, _) can't corrupt
    // URLs that contain those characters (e.g. .../wiki/Infinity_Blade_(XC3)). The placeholder is
    // wrapped in NUL bytes (\x00<idx>\x00) so it can never collide with a literal number in the
    // prose — a bare-digit token would swallow real numbers like "100 HP" on restore.
    const stash = [];
    const keep = (html) => { stash.push(html); return `\x00${stash.length - 1}\x00`; };
    s = s.replace(/`([^`]+)`/g, (_, c) => keep(`<code>${c}</code>`));
    // URLs go into an attribute value, so they need attribute escaping (escapeHtml upstream leaves
    // `"` alone): an unescaped quote would close href early and turn the rest of the URL into live
    // attributes (e.g. an onmouseover handler). Scheme stays restricted to http(s) by the regex.
    s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      (_, t, u) => keep(`<a href="${escapeAttr(u)}" target="_blank" rel="noopener">${t}</a>`));
    s = s.replace(/(https?:\/\/[^\s<]+)/g, (m, url) => {
      let trail = "", mm;
      // Peel trailing sentence punctuation, but keep a ")" that balances a "(" in the URL
      // (Xeno wiki pages legitimately end in "(XC3)", "(XC1)", etc.).
      while ((mm = url.match(/[.,;:!?)]$/))) {
        if (mm[0] === ")" && (url.match(/\)/g) || []).length <= (url.match(/\(/g) || []).length) break;
        trail = mm[0] + trail; url = url.slice(0, -1);
      }
      return keep(`<a href="${escapeAttr(url)}" target="_blank" rel="noopener">${url}</a>`) + trail;
    });
    // Inline citation markers ([1]..[n]) -> superscript links to that turn's source cards. Runs
    // AFTER the link stash (so a markdown link's [text] can never be misread as a marker) and only
    // for numbers the sources list actually has — [9] with 3 sources stays plain prose text.
    // Stashed like links so the emphasis passes below can't corrupt the generated HTML.
    if (cite && cite.count > 0) {
      const tid = escapeAttr(cite.turnId == null ? "" : cite.turnId);
      s = s.replace(/\[(\d{1,3})\]/g, (m, num) => {
        const n = +num;
        if (n < 1 || n > cite.count) return m;
        return keep(`<sup class="cite"><a class="cite-link" href="#src-${tid}-${n}">[${n}]</a></sup>`);
      });
    }
    // Emphasis on the remaining plain text only. Bold is matched non-greedily and may contain a
    // nested italic (LLM answers often emit "**In *Torna*:**"), so the italic pass below still
    // resolves the inner `*...*`; without this the raw ** markers would leak to the reader.
    s = s.replace(/\*\*([^\n]+?)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
    s = s.replace(/(^|[^_])_([^_\n]+)_/g, "$1<em>$2</em>");
    // Restore the stashed tokens (matched by their NUL-wrapped index, never a prose number).
    return s.replace(/\x00(\d+)\x00/g, (_, i) => stash[+i]);
  }

  // ---- GFM-style tables: a header row, a |---|---| separator, then body rows ----
  function splitRow(line) {
    let t = line.trim();
    if (t.startsWith("|")) t = t.slice(1);
    if (t.endsWith("|")) t = t.slice(0, -1);
    return t.split("|").map((c) => c.trim());
  }
  function isTableSep(line) {
    const t = (line || "").trim();
    if (!t.includes("|")) return false; // a bare "---" is a rule, not a table separator
    const cells = splitRow(t);
    return cells.length > 0 && cells.every((c) => /^:?-{1,}:?$/.test(c));
  }
  function buildTable(header, rows, cite) {
    const th = header.map((c) => `<th>${inline(c, cite)}</th>`).join("");
    const trs = rows
      .map((r) => `<tr>${r.map((c) => `<td>${inline(c, cite)}</td>`).join("")}</tr>`)
      .join("");
    return `<table><thead><tr>${th}</tr></thead><tbody>${trs}</tbody></table>`;
  }

  // opts.citations = { count, turnId }: link [1]..[count] markers in the text to that turn's
  // source-card anchors (#src-<turnId>-<n>). Omitted (streaming, or no sources yet) -> markers
  // stay plain text; the page re-renders once the sources event lands with the real count.
  function renderMarkdown(md, opts) {
    const cite = opts && opts.citations ? opts.citations : null;
    const lines = escapeHtml(md).split("\n");
    let html = "", list = null, para = [];
    const flushPara = () => { if (para.length) { html += `<p>${inline(para.join("<br>"), cite)}</p>`; para = []; } };
    const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
    let i = 0;
    while (i < lines.length) {
      const line = lines[i].trimEnd();
      // Table: a piped header row immediately followed by a separator row.
      if (line.includes("|") && isTableSep(lines[i + 1])) {
        flushPara(); closeList();
        const header = splitRow(line);
        i += 2; // consume header + separator
        const rows = [];
        while (i < lines.length && lines[i].includes("|") && lines[i].trim() !== "") {
          rows.push(splitRow(lines[i].trimEnd()));
          i++;
        }
        html += buildTable(header, rows, cite);
        continue;
      }
      const h = line.match(/^(#{1,6})\s+(.*)$/);
      const ul = line.match(/^\s*[-*]\s+(.*)$/);
      const ol = line.match(/^\s*\d+\.\s+(.*)$/);
      if (line.trim() === "") { flushPara(); closeList(); i++; continue; }
      if (h) { flushPara(); closeList(); const lvl = Math.min(h[1].length + 2, 4); html += `<h${lvl}>${inline(h[2], cite)}</h${lvl}>`; i++; continue; }
      if (ul) { flushPara(); if (list !== "ul") { closeList(); list = "ul"; html += "<ul>"; } html += `<li>${inline(ul[1], cite)}</li>`; i++; continue; }
      if (ol) { flushPara(); if (list !== "ol") { closeList(); list = "ol"; html += "<ol>"; } html += `<li>${inline(ol[1], cite)}</li>`; i++; continue; }
      closeList(); para.push(line); i++;
    }
    flushPara(); closeList();
    return html;
  }

  function escapeAttr(s) {
    return escapeHtml(String(s == null ? "" : s)).replace(/"/g, "&quot;");
  }

  // Only http(s) URLs become live links. Source URLs come from the corpus, so a `javascript:` or
  // `data:` value must never reach an href; it degrades to an inert "#".
  function safeHref(url) {
    const u = String(url == null ? "" : url).trim();
    return /^https?:\/\//i.test(u) ? u : "#";
  }

  // The display name for a source: a dict's title, else derive a readable page name from a wiki URL.
  function sourceName(s) {
    if (s && typeof s === "object") return s.title || s.url || "";
    let name = String(s);
    try { name = decodeURIComponent(name.split("/wiki/")[1] || name).replace(/_/g, " "); } catch (e) { /* malformed %-escape: keep the raw name */ }
    return name;
  }

  // Percentage-match label for a source (from the cross-encoder relevance, 0..1). Legacy string
  // sources (or any without a numeric relevance) return null so the % + bar are simply omitted.
  function relPct(s) {
    return s && typeof s === "object" && typeof s.relevance === "number"
      ? Math.round(s.relevance * 100) : null;
  }

  // One source card. `kind` is "top" (the single most-relevant, a hero card), "mid" (a flex card),
  // or "low" (a compact one-line row). `n` is the 1-based citation number; with a turnId the card
  // carries the anchor id (src-<turnId>-<n>) that the answer's inline [n] markers link to.
  function sourceCard(s, n, kind, tid) {
    const isObj = s && typeof s === "object";
    const url = isObj ? s.url : s;
    const anchor = tid == null ? "" : ` id="src-${tid}-${n}"`;
    const link = `href="${escapeAttr(safeHref(url))}" target="_blank" rel="noopener" title="${escapeAttr(url)}"`;
    const game = isObj && s.game ? `<span class="src-game">${escapeHtml(s.game)}</span>` : "";
    const snip = isObj && s.snippet ? escapeHtml(s.snippet) : "";
    const title = escapeHtml(sourceName(s));
    const pct = relPct(s);
    if (kind === "top") {
      return `<a class="src-card src-top"${anchor} ${link}>`
        + `<span class="src-rule" aria-hidden="true"></span>`
        + `<div class="src-top-head">`
        +   `<div class="src-ids"><span class="src-num">${n}</span>`
        +     `<span class="src-badge">TOP SOURCE</span>${game}</div>`
        +   (pct == null ? "" : `<span class="src-match">${pct}% match</span>`)
        + `</div>`
        + `<div class="src-title">${title}</div>`
        + (snip ? `<div class="src-snip">${snip}</div>` : "")
        + (pct == null ? "" : `<div class="src-bar"><span style="width:${pct}%"></span></div>`)
        + `</a>`;
    }
    if (kind === "low") {
      return `<a class="src-card src-low"${anchor} ${link}>`
        + `<span class="src-num">${n}</span>${game}`
        + `<span class="src-title">${title}</span>`
        + `<span class="src-snip">${snip}</span>`
        + (pct == null ? "" : `<span class="src-match">${pct}%</span>`)
        + `</a>`;
    }
    return `<a class="src-card src-mid"${anchor} ${link}>`
      + `<div class="src-mid-head"><div class="src-ids"><span class="src-num">${n}</span>${game}</div>`
      +   (pct == null ? "" : `<span class="src-match">${pct}%</span>`)
      + `</div>`
      + `<div class="src-title">${title}</div>`
      + (snip ? `<div class="src-snip">${snip}</div>` : "")
      + `</a>`;
  }

  // Build the Sources block. Accepts the rich payload [{url,title,game,snippet,relevance,tier}] or
  // legacy [url]. Rendered as a native <details> collapsed by default (a "GROUNDED IN N WIKI PAGES"
  // toggle) so a multi-turn thread isn't cluttered with citation cards. The single most-relevant
  // source (list index 0 — rag.py orders sources best-first) is the TOP SOURCE hero card; the rest
  // are mid cards, except low-tier ones which drop to compact rows. With a turnId, each card carries
  // the anchor id (src-<turnId>-<n>, 1-based, list order = citation number order) so the answer's
  // inline [n] markers, click-to-scroll, and the hover tooltip all stay in sync.
  function sourcesHtml(sources, turnId) {
    if (!sources || !sources.length) return "";
    const tid = turnId == null ? null : escapeAttr(turnId);
    const isLow = (s) => s && typeof s === "object" && s.tier === "low";
    const top = sourceCard(sources[0], 1, "top", tid);
    let mids = "", lows = "";
    for (let i = 1; i < sources.length; i++) {
      const s = sources[i];
      if (isLow(s)) lows += sourceCard(s, i + 1, "low", tid);
      else mids += sourceCard(s, i + 1, "mid", tid);
    }
    const shield = `<svg class="src-shield" width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">`
      + `<path d="M12 3l7 3v5c0 4.4-3 8-7 10-4-2-7-5.6-7-10V6l7-3z" stroke="currentColor" stroke-width="1.8"`
      + ` fill="color-mix(in srgb, var(--accent) 16%, transparent)"/>`
      + `<path d="M9 12l2 2 4-4.5" stroke="currentColor" stroke-width="1.9"/></svg>`;
    const chevron = `<span class="src-chev" aria-hidden="true">`
      + `<svg width="13" height="13" viewBox="0 0 12 12"><path d="M2 4l4 4 4-4" stroke="currentColor" stroke-width="1.7" fill="none"/></svg></span>`;
    return `<details class="sources"><summary class="src-summary">`
      + `<span class="src-grounded">${shield}<span>GROUNDED IN ${sources.length} WIKI PAGES</span></span>`
      + chevron
      + `</summary>`
      + `<div class="sources-body">`
      +   top
      +   (mids ? `<div class="src-mids">${mids}</div>` : "")
      +   (lows ? `<div class="src-lows">${lows}</div>` : "")
      + `</div></details>`;
  }

  // One conversation turn: the "YOUR QUESTION" block (accent left-bar + text), then the answer card
  // (a phase indicator / caret stream into the `.answer` element), a copy button, and that turn's
  // sources. Returns an HTML string (testable without a DOM); the page streams tokens into `.answer`
  // afterwards. aria-live makes the streamed answer announce to screen readers. The id / class hooks
  // (`id="turn-N"`, `.answer`, `.copy-btn[data-copy]`, `.turn-sources`) are what runTurn + the
  // copy/citation delegation query, so they must stay stable.
  function answerBlockHtml(turn) {
    const id = turn.id == null ? "" : String(turn.id);
    const tid = "turn-" + escapeAttr(id);
    const q = escapeHtml(turn.question || "");
    const answerHtml = turn.answerHtml || "";
    const src = turn.sources ? sourcesHtml(turn.sources, id) : "";
    return `<div class="turn" id="${tid}">`
      + `<div class="q">`
      +   `<span class="q-bar" aria-hidden="true"></span>`
      +   `<div class="q-body"><div class="q-label">YOUR QUESTION</div><div class="q-text">${q}</div></div>`
      + `</div>`
      + `<div class="a-card">`
      +   `<div class="answer" aria-live="polite">${answerHtml}</div>`
      +   `<div class="a-foot"><button type="button" class="copy-btn" data-copy="${tid}">Copy</button><span class="a-tier"></span></div>`
      +   `<div class="turn-sources">${src}</div>`
      + `</div>`
      + `</div>`;
  }

  // Which answer tier the router picked for a turn, shown as a quiet caption in the answer footer.
  const TIER_LABELS = { fast: "Fast mode", thinking: "Thinking mode", scholar: "Scholar mode" };
  function tierCaptionHtml(tier) {
    const label = Object.prototype.hasOwnProperty.call(TIER_LABELS, tier) ? TIER_LABELS[tier] : "";
    return label ? `<span class="tier-tag">${label}</span>` : "";
  }

  // Set (never append) a turn's `.a-tier` caption from an SSE `tier` event payload ({tier, source}).
  // A question can carry TWO tier events in one stream: the initial routing decision, then (SP3) a
  // second one when the answerability check escalates to scholar depth
  // ({tier: "scholar", source: "escalated"}). Assignment, not concatenation, is what makes the second
  // event REPLACE the caption instead of the two stacking up side by side. `root` is any element the
  // caption slot can be found under (index.html passes the turn's container).
  function applyTierCaption(root, payload) {
    const slot = root && typeof root.querySelector === "function" ? root.querySelector(".a-tier") : null;
    if (slot && payload) slot.innerHTML = tierCaptionHtml(payload.tier);
  }

  // Empty-state starter questions; clicking one fills the box and asks (wired via data-example on
  // `.example`, kept for the existing click delegation). Each renders as an arrow row.
  function examplesHtml(list) {
    if (!list || !list.length) return "";
    const items = list.map((q) =>
      `<button type="button" class="example" data-example="${escapeAttr(q)}">`
      + `<span class="ex-text">${escapeHtml(q)}</span>`
      + `<span class="ex-arrow" aria-hidden="true">&rarr;</span>`
      + `</button>`
    ).join("");
    return `<div class="examples-label">TRY ASKING</div><div class="examples-grid">${items}</div>`;
  }

  // Parse one SSE block (the text between blank lines) into {event, data}. `event` is the declared
  // `event:` name, or "message" for a bare `data:` block (the answer text deltas). `data` is the
  // JSON-decoded payload. Returns null for a block with no data line or unparseable JSON, so a
  // garbled frame is skipped instead of throwing out of the read loop and wrecking the turn.
  function parseSseEvent(block) {
    let event = "message";
    const data = [];
    for (const line of String(block).split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    if (!data.length) return null;
    try { return { event, data: JSON.parse(data.join("\n")) }; } catch (e) { return null; }
  }

  return {
    parseSseEvent, safeHref, escapeHtml, escapeAttr, inline, renderMarkdown, sourceName, sourcesHtml,
    answerBlockHtml, examplesHtml, tierCaptionHtml, applyTierCaption,
  };
});
