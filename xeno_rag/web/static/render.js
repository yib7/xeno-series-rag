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

  function inline(s) {
    // Stash code spans and links FIRST as opaque tokens, so emphasis (*, _) can't corrupt
    // URLs that contain those characters (e.g. .../wiki/Infinity_Blade_(XC3)). The placeholder is
    // wrapped in NUL bytes (\x00<idx>\x00) so it can never collide with a literal number in the
    // prose — a bare-digit token would swallow real numbers like "100 HP" on restore.
    const stash = [];
    const keep = (html) => { stash.push(html); return `\x00${stash.length - 1}\x00`; };
    s = s.replace(/`([^`]+)`/g, (_, c) => keep(`<code>${c}</code>`));
    s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      (_, t, u) => keep(`<a href="${u}" target="_blank" rel="noopener">${t}</a>`));
    s = s.replace(/(https?:\/\/[^\s<]+)/g, (m, url) => {
      let trail = "", mm;
      // Peel trailing sentence punctuation, but keep a ")" that balances a "(" in the URL
      // (Xeno wiki pages legitimately end in "(XC3)", "(XC1)", etc.).
      while ((mm = url.match(/[.,;:!?)]$/))) {
        if (mm[0] === ")" && (url.match(/\)/g) || []).length <= (url.match(/\(/g) || []).length) break;
        trail = mm[0] + trail; url = url.slice(0, -1);
      }
      return keep(`<a href="${url}" target="_blank" rel="noopener">${url}</a>`) + trail;
    });
    // Emphasis on the remaining plain text only.
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
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
  function buildTable(header, rows) {
    const th = header.map((c) => `<th>${inline(c)}</th>`).join("");
    const trs = rows
      .map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`)
      .join("");
    return `<table><thead><tr>${th}</tr></thead><tbody>${trs}</tbody></table>`;
  }

  function renderMarkdown(md) {
    const lines = escapeHtml(md).split("\n");
    let html = "", list = null, para = [];
    const flushPara = () => { if (para.length) { html += `<p>${inline(para.join("<br>"))}</p>`; para = []; } };
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
        html += buildTable(header, rows);
        continue;
      }
      const h = line.match(/^(#{1,6})\s+(.*)$/);
      const ul = line.match(/^\s*[-*]\s+(.*)$/);
      const ol = line.match(/^\s*\d+\.\s+(.*)$/);
      if (line.trim() === "") { flushPara(); closeList(); i++; continue; }
      if (h) { flushPara(); closeList(); const lvl = Math.min(h[1].length + 2, 4); html += `<h${lvl}>${inline(h[2])}</h${lvl}>`; i++; continue; }
      if (ul) { flushPara(); if (list !== "ul") { closeList(); list = "ul"; html += "<ul>"; } html += `<li>${inline(ul[1])}</li>`; i++; continue; }
      if (ol) { flushPara(); if (list !== "ol") { closeList(); list = "ol"; html += "<ol>"; } html += `<li>${inline(ol[1])}</li>`; i++; continue; }
      closeList(); para.push(line); i++;
    }
    flushPara(); closeList();
    return html;
  }

  function escapeAttr(s) {
    return escapeHtml(String(s == null ? "" : s)).replace(/"/g, "&quot;");
  }

  // The display name for a source: a dict's title, else derive a readable page name from a wiki URL.
  function sourceName(s) {
    if (s && typeof s === "object") return s.title || s.url || "";
    let name = String(s);
    try { name = decodeURIComponent(name.split("/wiki/")[1] || name).replace(/_/g, " "); } catch (e) {}
    return name;
  }

  // Build the Sources block. Accepts the rich payload [{url,title,game,snippet}] or legacy [url].
  // Rendered as a native <details> collapsed by default — sources stay out of the way until the
  // reader toggles "Sources (N)" open, so a multi-turn thread isn't cluttered with citation cards.
  function sourcesHtml(sources) {
    if (!sources || !sources.length) return "";
    const chips = sources.map((s) => {
      const isObj = s && typeof s === "object";
      const url = isObj ? s.url : s;
      const name = sourceName(s);
      // Size the bubble by how correlated the source is (high/med/low); legacy sources stay plain.
      const tier = isObj && s.tier ? ` tier-${s.tier}` : "";
      const game = isObj && s.game ? `<span class="src-game">${escapeHtml(s.game)}</span>` : "";
      const snippet = isObj && s.snippet
        ? `<span class="src-snippet">${escapeHtml(s.snippet)}</span>` : "";
      return `<a class="chip${tier}" href="${escapeAttr(url)}" target="_blank" rel="noopener" title="${escapeAttr(url)}">`
        + `<span class="src-head"><span class="dot"></span>${game}<span class="src-name">${escapeHtml(name)}</span></span>`
        + snippet
        + `</a>`;
    }).join("");
    return `<details class="sources"><summary>Sources (${sources.length})</summary>`
      + `<div class="chips">${chips}</div></details>`;
  }

  // One conversation turn: the user's question, the (streamed) answer with a copy button, and that
  // turn's sources. Returns an HTML string (testable without a DOM); the page streams tokens into
  // the `.answer` element afterwards. aria-live makes the streamed answer announce to screen readers.
  function answerBlockHtml(turn) {
    const id = turn.id == null ? "" : String(turn.id);
    const tid = "turn-" + escapeAttr(id);
    const q = escapeHtml(turn.question || "");
    const answerHtml = turn.answerHtml || "";
    const src = turn.sources ? sourcesHtml(turn.sources) : "";
    return `<div class="turn" id="${tid}">`
      + `<div class="q"><span class="q-label">You</span><span class="q-text">${q}</span></div>`
      + `<div class="a-card">`
      +   `<div class="answer" aria-live="polite">${answerHtml}</div>`
      +   `<div class="a-foot"><button type="button" class="copy-btn" data-copy="${tid}">Copy</button></div>`
      +   `<div class="turn-sources">${src}</div>`
      + `</div>`
      + `</div>`;
  }

  // Empty-state starter questions; clicking one fills the box and asks (wired via data-example).
  function examplesHtml(list) {
    if (!list || !list.length) return "";
    const items = list.map((q) =>
      `<button type="button" class="example" data-example="${escapeAttr(q)}">${escapeHtml(q)}</button>`
    ).join("");
    return `<p class="examples-label">Try asking</p><div class="examples-grid">${items}</div>`;
  }

  return {
    escapeHtml, escapeAttr, inline, renderMarkdown, sourceName, sourcesHtml,
    answerBlockHtml, examplesHtml,
  };
});
