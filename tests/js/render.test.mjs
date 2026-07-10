// Unit tests for the answer-pane Markdown renderer (xeno_rag/web/static/render.js).
// Run: `node --test tests/js/`. Node imports the CommonJS module as the default export.
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const {
  renderMarkdown, inline, escapeHtml, sourcesHtml, answerBlockHtml, examplesHtml,
} = require("../../xeno_rag/web/static/render.js");

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

test("sourcesHtml is collapsed by default with a labelled count", () => {
  const html = sourcesHtml([{ url: "https://w/a", title: "A" }, { url: "https://w/b", title: "B" }]);
  assert.match(html, /<details[^>]*class="sources"/);          // a native collapsible
  assert.doesNotMatch(html, /<details[^>]*\sopen/);            // hidden until toggled on
  assert.match(html, /<summary[^>]*>\s*Sources \(2\)\s*<\/summary>/);  // count in the toggle label
});

test("sourcesHtml sizes bubbles by tier and keeps correlation order", () => {
  const html = sourcesHtml([
    { url: "https://w/A", title: "A", tier: "high" },
    { url: "https://w/B", title: "B", tier: "med" },
    { url: "https://w/C", title: "C", tier: "low" },
  ]);
  assert.match(html, /class="chip tier-high"/);
  assert.match(html, /class="chip tier-med"/);
  assert.match(html, /class="chip tier-low"/);
  // rendered in the given (best-first) order
  assert.ok(html.indexOf(">A<") < html.indexOf(">B<"));
  assert.ok(html.indexOf(">B<") < html.indexOf(">C<"));
});

test("sourcesHtml without a tier stays a plain chip (legacy)", () => {
  const html = sourcesHtml([{ url: "https://w/x", title: "X" }]);
  assert.match(html, /class="chip"/);            // no tier class when none supplied
});

test("sourcesHtml escapes injection in title/snippet", () => {
  const html = sourcesHtml([{ url: "https://w/x", title: "<b>x</b>", snippet: "<script>bad</script>" }]);
  assert.doesNotMatch(html, /<script>bad/);
  assert.doesNotMatch(html, /<b>x<\/b>/);
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
