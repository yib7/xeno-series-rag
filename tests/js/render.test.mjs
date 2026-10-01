// Unit tests for the answer-pane Markdown renderer (xeno_rag/web/static/render.js).
// Run: `node --test tests/js/`. Node imports the CommonJS module as the default export.
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const {
  renderMarkdown, inline, escapeHtml, sourcesHtml, answerBlockHtml, examplesHtml, tierCaptionHtml,
  applyTierCaption, parseSseEvent, safeHref,
} = require("../../xeno_rag/web/static/render.js");

// A minimal DOM stand-in for applyTierCaption's tests: just enough of Element for
// `root.querySelector(".a-tier")` to find a child by class and for `.innerHTML` assignment/reads to
// work, with no real DOM (jsdom) dependency in this project.
function fakeTurnEl() {
  const tierSlot = { innerHTML: "" };
  return { tierSlot, querySelector: (sel) => (sel === ".a-tier" ? tierSlot : null) };
}

test("numbers in prose survive (regression: the '100 -> undefined' bug)", () => {
  const out = renderMarkdown("Metal Face appears at level 10 with 124 HP.");
  assert.match(out, /level 10 with 124 HP/);
  assert.doesNotMatch(out, /undefined/);
});

test("plain numbers with no code/links are untouched", () => {
  const out = inline("just 100 and 250 numbers");
  assert.equal(out, "just 100 and 250 numbers");
});

test("a code span AND a number both render correctly", () => {
  const out = inline("The `Monado` deals 100 damage.");
  assert.match(out, /<code>Monado<\/code>/);
  assert.match(out, /deals 100 damage/);
  assert.doesNotMatch(out, /undefined/);
});

test("multiple code spans and numbers stay aligned", () => {
  const out = inline("Use `Stream Edge` for 250 power and `Sword Strike` at level 17.");
  assert.match(out, /<code>Stream Edge<\/code> for 250 power/);
  assert.match(out, /<code>Sword Strike<\/code> at level 17/);
  assert.doesNotMatch(out, /undefined/);
});

test("wiki links with balanced (XCn) parens stay intact", () => {
  const out = inline("See https://www.xenoserieswiki.org/wiki/Monado_(XC3).");
  assert.match(out, /href="https:\/\/www\.xenoserieswiki\.org\/wiki\/Monado_\(XC3\)"/);
  // The trailing sentence period is peeled off the link.
  assert.match(out, /<\/a>\.$/);
});

test("bold and italic still render", () => {
  assert.match(inline("**bold** and *italic*"), /<strong>bold<\/strong> and <em>italic<\/em>/);
});

test("bold wrapping italic renders both, no literal asterisks", () => {
  // LLM answers often emit "**In *Torna*:**" (bold that contains an italic); the bold must still
  // render instead of leaking raw ** markers to the reader.
  const out = inline("**In *Torna* land:**");
  assert.match(out, /<strong>In <em>Torna<\/em> land:<\/strong>/);
  assert.doesNotMatch(out, /\*\*/);
});

test("two bolds on a line don't merge across the middle", () => {
  // Guards the non-greedy match: "**a** x **b**" must be two spans, not one swallowing " x ".
  assert.match(inline("**a** x **b**"), /<strong>a<\/strong> x <strong>b<\/strong>/);
});

test("a Markdown table renders as <table>", () => {
  const md = [
    "| Stat | Value |",
    "| --- | --- |",
    "| HP | 124 |",
    "| Level | 10 |",
  ].join("\n");
  const out = renderMarkdown(md);
  assert.match(out, /<table>/);
  assert.match(out, /<th>Stat<\/th>/);
  assert.match(out, /<th>Value<\/th>/);
  assert.match(out, /<td>HP<\/td>/);
  assert.match(out, /<td>124<\/td>/);
  assert.doesNotMatch(out, /undefined/);
});

test("table cells render inline markdown (numbers + code survive)", () => {
  const md = ["| Move | Power |", "| --- | --- |", "| `Slash` | 100 |"].join("\n");
  const out = renderMarkdown(md);
  assert.match(out, /<td><code>Slash<\/code><\/td>/);
  assert.match(out, /<td>100<\/td>/);
});

test("HTML in input is escaped (no injection)", () => {
  const out = renderMarkdown("a <script>alert(1)</script> b");
  assert.doesNotMatch(out, /<script>/);
  assert.match(out, /&lt;script&gt;/);
});

test("markdown-link URL with an embedded quote cannot inject an attribute (XSS)", () => {
  // A model-emitted link like [x](https://e/"onmouseover="x) must not break out of the quoted
  // href value: HTML5 tokenizers accept an attribute starting right after a closing quote, so an
  // unescaped `"` in the URL would make onmouseover a live event-handler attribute.
  const out = inline('[x](https://e/"onmouseover="x)');
  assert.doesNotMatch(out, /"onmouseover/, "quote in URL escaped the href attribute");
  assert.match(out, /href="https:\/\/e\/&quot;onmouseover=&quot;x"/);
});

test("autolinked URL with an embedded quote cannot inject an attribute (XSS)", () => {
  // The autolink branch repeats the URL as the link *text* (safe: quotes are inert in text
  // content), so the assertion targets the opening <a ...> tag, where a raw quote would
  // terminate the href value and promote the rest into live attributes.
  const out = inline('see https://e/"onmouseover="x now');
  const tag = out.match(/<a\s[^>]*>/)[0];
  assert.doesNotMatch(tag, /"onmouseover/, "quote in URL escaped for the href attribute");
  assert.match(tag, /href="https:\/\/e\/&quot;onmouseover=&quot;x"/);
});

// ---- sources (SP5: richer payload + snippet previews) ----

test("sourcesHtml renders title, game and snippet for dict sources", () => {
  const html = sourcesHtml([
    { url: "https://w/Rex", title: "Rex (XC2)", game: "XC2", snippet: "Rex is the salvager protagonist." },
  ]);
  assert.match(html, /Rex \(XC2\)/);                 // title shown
  assert.match(html, /salvager protagonist/);        // snippet preview present
  assert.match(html, /href="https:\/\/w\/Rex"/);     // links out
  assert.match(html, /XC2/);                          // game badge
});

test("sourcesHtml tolerates legacy string urls", () => {
  const html = sourcesHtml(["https://www.xenoserieswiki.org/wiki/Rex_(XC2)"]);
  assert.match(html, /Rex \(XC2\)/);                  // name derived from the URL
  assert.match(html, /href="https:\/\/www\.xenoserieswiki\.org\/wiki\/Rex_\(XC2\)"/);
});

test("sourcesHtml of nothing is empty", () => {
  assert.equal(sourcesHtml([]), "");
  assert.equal(sourcesHtml(null), "");
});

test("sourcesHtml is collapsed by default behind a 'GROUNDED IN N' toggle", () => {
  const html = sourcesHtml([{ url: "https://w/a", title: "A" }, { url: "https://w/b", title: "B" }]);
  assert.match(html, /<details[^>]*class="sources"/);          // a native collapsible
  assert.doesNotMatch(html, /<details[^>]*\sopen/);            // hidden until toggled on
  assert.match(html, /GROUNDED IN 2 WIKI PAGES/);              // count in the toggle label
});

test("sourcesHtml makes the first source the top card and buckets the rest by tier", () => {
  const html = sourcesHtml([
    { url: "https://w/A", title: "A", tier: "high", relevance: 0.98 },
    { url: "https://w/B", title: "B", tier: "med", relevance: 0.80 },
    { url: "https://w/C", title: "C", tier: "low", relevance: 0.50 },
  ]);
  assert.match(html, /class="src-card src-top"/);    // the single most-relevant is the hero card
  assert.match(html, /TOP SOURCE/);
  assert.match(html, /class="src-card src-mid"/);    // a mid-tier card
  assert.match(html, /class="src-card src-low"/);    // a low-tier compact row
  // rendered in the given (best-first) order
  assert.ok(html.indexOf(">A<") < html.indexOf(">B<"));
  assert.ok(html.indexOf(">B<") < html.indexOf(">C<"));
});

test("sourcesHtml shows a % match + relevance bar on the top card", () => {
  const html = sourcesHtml([{ url: "https://w/A", title: "A", relevance: 0.97 }]);
  assert.match(html, /97% match/);
  assert.match(html, /class="src-bar"/);
});

test("sourcesHtml omits % match for legacy sources without a relevance", () => {
  const html = sourcesHtml([{ url: "https://w/A", title: "A" }]);
  assert.doesNotMatch(html, /% match/);
  assert.doesNotMatch(html, /class="src-bar"/);
});

test("sourcesHtml of a single source renders just the top card", () => {
  const html = sourcesHtml([{ url: "https://w/x", title: "X" }]);
  assert.match(html, /class="src-card src-top"/);
  assert.doesNotMatch(html, /src-mid/);
  assert.doesNotMatch(html, /src-low/);
});

test("sourcesHtml escapes injection in title/snippet", () => {
  const html = sourcesHtml([{ url: "https://w/x", title: "<b>x</b>", snippet: "<script>bad</script>" }]);
  assert.doesNotMatch(html, /<script>bad/);
  assert.doesNotMatch(html, /<b>x<\/b>/);
});

// ---- inline citations: [n] markers link to that turn's source cards ----

test("renderMarkdown links [n] markers to source anchors when n is in range", () => {
  const out = renderMarkdown("Rex wields the Aegis [2] and lives on Gramps [1].",
    { citations: { count: 2, turnId: 1 } });
  assert.match(out, /<sup class="cite"><a class="cite-link" href="#src-1-2">\[2\]<\/a><\/sup>/);
  assert.match(out, /<sup class="cite"><a class="cite-link" href="#src-1-1">\[1\]<\/a><\/sup>/);
});

test("renderMarkdown leaves out-of-range markers as plain text", () => {
  const out = renderMarkdown("A claim [9] and another [0].", { citations: { count: 2, turnId: 1 } });
  assert.doesNotMatch(out, /cite-link/);
  assert.match(out, /\[9\]/);            // stays literal prose
  assert.match(out, /\[0\]/);
});

test("renderMarkdown without citation opts leaves markers as plain text (graceful)", () => {
  const out = renderMarkdown("A fact [1].");
  assert.doesNotMatch(out, /cite-link/);
  assert.match(out, /\[1\]/);
});

test("citation markers work inside table cells and lists", () => {
  const md = ["| Stat | Value |", "| --- | --- |", "| HP | 124 [1] |", "", "- Drops a gem [2]"].join("\n");
  const out = renderMarkdown(md, { citations: { count: 2, turnId: 3 } });
  assert.match(out, /<td>124 <sup class="cite"><a class="cite-link" href="#src-3-1">\[1\]<\/a><\/sup><\/td>/);
  assert.match(out, /<li>Drops a gem <sup class="cite"><a class="cite-link" href="#src-3-2">\[2\]<\/a><\/sup><\/li>/);
});

test("citation markers do not corrupt markdown links whose text is a number", () => {
  const out = renderMarkdown("See [1](https://w/x) and a real marker [1].",
    { citations: { count: 1, turnId: 1 } });
  assert.match(out, /<a href="https:\/\/w\/x"[^>]*>1<\/a>/);       // the link renders as a link
  assert.match(out, /href="#src-1-1"/);                            // the bare marker still links
});

test("adjacent markers [1][3] each link separately", () => {
  const out = renderMarkdown("Fact [1][3].", { citations: { count: 3, turnId: 2 } });
  assert.match(out, /href="#src-2-1"/);
  assert.match(out, /href="#src-2-3"/);
});

test("sourcesHtml with a turnId anchors and numbers each card in order", () => {
  const html = sourcesHtml([
    { url: "https://w/A", title: "A" },
    { url: "https://w/B", title: "B" },
  ], 7);
  assert.match(html, /id="src-7-1"/);
  assert.match(html, /id="src-7-2"/);
  assert.match(html, /<span class="src-num">1<\/span>/);
  assert.match(html, /<span class="src-num">2<\/span>/);
  assert.ok(html.indexOf("src-7-1") < html.indexOf("src-7-2"));    // list order = citation order
});

test("sourcesHtml without a turnId still numbers cards but omits anchor ids", () => {
  const html = sourcesHtml([{ url: "https://w/A", title: "A" }]);
  assert.doesNotMatch(html, /id="src-/);                     // no anchor without a turn to link to
  assert.match(html, /<span class="src-num">1<\/span>/);     // the citation number is always shown
});

// ---- SP6: conversation thread blocks + example questions ----

test("answerBlockHtml builds a turn with question, answer, copy button, sources", () => {
  const html = answerBlockHtml({
    id: 1, question: "Who is Rex?", answerHtml: "<p>Rex is the protagonist.</p>",
    sources: [{ url: "https://w/Rex", title: "Rex (XC2)", game: "XC2", snippet: "salvager" }],
  });
  assert.match(html, /id="turn-1"/);
  assert.match(html, /Who is Rex\?/);                  // the question
  assert.match(html, /Rex is the protagonist\./);      // the answer
  assert.match(html, /class="copy-btn"/);              // copy button
  assert.match(html, /aria-live="polite"/);            // streamed answer is announced
  assert.match(html, /Rex \(XC2\)/);                   // sources rendered into the block
});

test("answerBlockHtml escapes the question (no injection)", () => {
  const html = answerBlockHtml({ id: 2, question: "<img src=x onerror=alert(1)>", answerHtml: "", sources: null });
  assert.doesNotMatch(html, /<img src=x/);
  assert.match(html, /&lt;img/);
});

test("examplesHtml renders clickable example questions", () => {
  const html = examplesHtml(["What is the Monado?", "Who is Pyra?"]);
  assert.match(html, /What is the Monado\?/);
  assert.match(html, /Who is Pyra\?/);
  assert.match(html, /data-example="What is the Monado\?"/);   // wired for click-to-fill
});

test("examplesHtml of nothing is empty", () => {
  assert.equal(examplesHtml([]), "");
});

// ---- SP4: tier caption (replaces the removed answer-style selector) ----

test("tierCaptionHtml labels known tiers and ignores unknown ones", () => {
  assert.match(tierCaptionHtml("fast"), /Fast mode/);
  assert.match(tierCaptionHtml("thinking"), /Thinking mode/);
  assert.match(tierCaptionHtml("scholar"), /Scholar mode/);
  assert.equal(tierCaptionHtml("ultra"), "");
  assert.equal(tierCaptionHtml(undefined), "");
});

test("answerBlockHtml has an empty tier slot in the answer footer", () => {
  const out = answerBlockHtml({ id: 1, question: "q", answerHtml: "" });
  assert.match(out, /<span class="a-tier"><\/span>/);
});

test("answerBlockHtml keeps the copy button before the tier caption (copy button stays left)", () => {
  const out = answerBlockHtml({ id: 1, question: "q", answerHtml: "" });
  assert.ok(out.indexOf('class="copy-btn"') < out.indexOf('class="a-tier"'));
});

// ---- applyTierCaption (SP3: a repeated tier SSE event -- the answerability escalation -- must
// REPLACE the caption, not append to it) ----

test("applyTierCaption sets the caption from a tier payload", () => {
  const turn = fakeTurnEl();
  applyTierCaption(turn, { tier: "fast", source: "jev" });
  assert.match(turn.tierSlot.innerHTML, /Fast mode/);
});

test("a second tier event REPLACES the caption rather than appending to it", () => {
  // This is the exact SP3 scenario: the initial routing tier, then a second event when the
  // answerability check escalates to scholar depth.
  const turn = fakeTurnEl();
  applyTierCaption(turn, { tier: "fast", source: "jev" });
  applyTierCaption(turn, { tier: "scholar", source: "escalated" });
  assert.match(turn.tierSlot.innerHTML, /Scholar mode/);
  assert.doesNotMatch(turn.tierSlot.innerHTML, /Fast mode/);
  // exactly one caption span, not two stacked up
  assert.equal((turn.tierSlot.innerHTML.match(/tier-tag/g) || []).length, 1);
});

test("applyTierCaption does nothing when the turn has no .a-tier slot", () => {
  const empty = { querySelector: () => null };
  assert.doesNotThrow(() => applyTierCaption(empty, { tier: "fast" }));
});

test("applyTierCaption ignores a falsy payload (never blanks an existing caption)", () => {
  const turn = fakeTurnEl();
  applyTierCaption(turn, { tier: "thinking", source: "jev" });
  applyTierCaption(turn, null);
  assert.match(turn.tierSlot.innerHTML, /Thinking mode/);
});

// ---- parseSseEvent / safeHref ----

test("parseSseEvent reads a named tier event and a bare data block", () => {
  const tier = parseSseEvent('event: tier\ndata: {"tier": "scholar", "source": "escalated"}');
  assert.deepEqual(tier, { event: "tier", data: { tier: "scholar", source: "escalated" } });
  const text = parseSseEvent('data: "hello \\nworld"');
  assert.deepEqual(text, { event: "message", data: "hello \nworld" });
});

test("parseSseEvent: a stream of tier, text, tier, sources keeps its order and kinds", () => {
  const wire = [
    'event: tier\ndata: {"tier":"fast","source":"router"}',
    'data: "part one "',
    'event: tier\ndata: {"tier":"scholar","source":"escalated"}',
    'event: sources\ndata: []',
  ].join("\n\n");
  const kinds = wire.split("\n\n").map((b) => parseSseEvent(b).event);
  assert.deepEqual(kinds, ["tier", "message", "tier", "sources"]);
});

test("parseSseEvent returns null for garbled, empty or data-less frames instead of throwing", () => {
  assert.equal(parseSseEvent("event: tier\ndata: {not json"), null);
  assert.equal(parseSseEvent(""), null);
  assert.equal(parseSseEvent("event: sources"), null);
  assert.equal(parseSseEvent(": keepalive comment"), null);
});

test("safeHref allows http(s) only; other schemes become an inert #", () => {
  assert.equal(safeHref("https://xenoserieswiki.org/wiki/Shulk"), "https://xenoserieswiki.org/wiki/Shulk");
  assert.equal(safeHref("javascript:alert(1)"), "#");
  assert.equal(safeHref("  JavaScript:alert(1)"), "#");
  assert.equal(safeHref("data:text/html,x"), "#");
  assert.equal(safeHref(undefined), "#");
});

test("a source card never emits a javascript: href", () => {
  const html = sourcesHtml([{ title: "Evil", url: "javascript:alert(1)", relevance: 0.9 }], 1);
  assert.doesNotMatch(html, /href="javascript:/i);
  assert.match(html, /href="#"/);
});

// --- Phase 4 security probes: hostile model/server text through every renderer entry point ---

test("a NUL-forged stash token in the text cannot duplicate or smuggle stashed HTML", () => {
  const out = renderMarkdown("\x000\x00 \x001\x00 [x](https://a.com) `code`");
  assert.equal((out.match(/<a /g) || []).length, 1, "only the real link is an anchor");
  assert.equal((out.match(/<code>/g) || []).length, 1, "only the real code span is code");
  assert.ok(!out.includes("\x00"), "no NUL survives into the HTML");
});

test("a URL directly followed by a code span keeps both and leaves no stash token in the href", () => {
  const out = renderMarkdown("`a`https://a.com/`b`");
  assert.ok(!out.includes("\x00"));
  assert.match(out, /<code>a<\/code>/);
  assert.match(out, /<code>b<\/code>/);
  assert.match(out, /href="https:\/\/a\.com\/"/);
});

test("javascript: and data: links stay inert text, never an href", () => {
  for (const md of ["[x](javascript:alert(1))", "[x](data:text/html;base64,AAAA)", "javascript:alert(1)"]) {
    assert.doesNotMatch(renderMarkdown(md), /href=/, md);
  }
});

test("decline and off-topic canned messages render as plain escaped text", () => {
  const out = renderMarkdown("The wiki pages I found don't seem to cover that. The closest matches are listed below — try rephrasing.");
  assert.match(out, /^<p>The wiki pages/);
  assert.doesNotMatch(out, /<(?!\/?p>)/);
});

test("raw HTML in headings, tables, bold and citations is escaped everywhere", () => {
  const md = "# <img src=x onerror=1>\n\n| <b>h</b> | b |\n|---|---|\n| <i>c</i> | **<script>x</script>** [1] |";
  const out = renderMarkdown(md, { citations: { count: 1, turnId: '1"><x' } });
  assert.doesNotMatch(out, /<(img|script|b|i)[ >]/);
  assert.doesNotMatch(out, /href="#src-1"><x/, "turn id is attribute-escaped in the citation href");
  assert.match(out, /href="#src-1&quot;&gt;&lt;x-1"/);
});

test("tierCaptionHtml only ever emits a fixed label, even for prototype-chain and markup names", () => {
  for (const t of ["<img onerror=1>", "__proto__", "constructor", "toString", "hasOwnProperty", null, undefined, 5, {}]) {
    assert.equal(tierCaptionHtml(t), "", String(t));
  }
});
