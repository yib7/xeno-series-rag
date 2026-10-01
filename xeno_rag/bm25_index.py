"""A lexical BM25 retriever over the chunk corpus, backed by SQLite FTS5.

Dense (Qwen3-Embedding) retrieval misses exact proper-noun / concept queries when many near-duplicate ancillary
pages (weapon SKUs, music tracks, boss-instances) crowd the canonical page out of the candidate
window, e.g. "What are mimeosomes?" returned only Skell weapon part-numbers. BM25 scores exact term
overlap, so the page that literally says "mimeosome" ranks first.

FTS5 is built into Python's bundled sqlite3: persistent, scales to the full ~169k chunks, low memory,
no heavy new dependency, and supports the per-game filter via a side metadata table. The index lives
in its own sqlite file (separate from ChromaDB's store) and is built from the live collection so its
``game`` column matches the re-tagged metadata exactly.
"""

import os
import re
import sqlite3
import threading

from .parse_wikitext import _BASE_GAMES, filter_membership, membership_from_game

# Word tokens for the MATCH query. Wrapping each token in double quotes neutralizes FTS5 operators
# (-, *, :, parentheses, NEAR), so an arbitrary user question can never be a malformed FTS expression.
# Unicode-aware (digits and letters of any script, no underscore): FTS5's unicode61 tokenizer folds
# diacritics on both the indexed text and the quoted query token, so "Rhéa" must reach it whole
# and match "Rhea". An ASCII-only pattern split it into the junk token "rh".
_WORD = re.compile(r"[^\W_]+")

DEFAULT_PATH = os.path.join("data", "vectorstore", "bm25.sqlite3")

# The most common English function words: OR-joining these matches most of the 169k rows and forces
# FTS5 to score a near-full index before LIMIT. They are dropped from the MATCH expression whenever
# at least one content token remains (an all-stopword query keeps them, so it still returns
# something). Deliberately small: no NLP dependency, and rare-but-real names ("Who is N?": N is an
# XC3 character) must never be swallowed.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "does", "for", "from", "had",
    "has", "have", "he", "her", "his", "how", "i", "in", "is", "it", "its", "of", "on", "or",
    "she", "that", "the", "their", "there", "they", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "you",
})


def _games_str(chunk: dict) -> str:
    """Comma-joined membership for a chunk's ``games`` filter column: an explicit ``games`` list if
    present, else derived from the single ``game`` label. Wrapped with leading/trailing commas at
    query time so a ``LIKE '%,G,%'`` test matches whole codes only."""
    games = chunk.get("games")
    member = set(games) if games else membership_from_game(chunk.get("game"))
    return ",".join(sorted(member))


def _match_query(text: str) -> str:
    """Turn a free-text question into a safe FTS5 MATCH string: quoted tokens joined with OR (recall-
    friendly; bm25 still rewards documents matching more / rarer terms). Single-character tokens are
    kept: quoting makes them safe FTS5 syntax and some are real names ("N" in XC3). Stopwords are
    dropped when at least one content token remains; an all-stopword query falls back to using them
    all. Empty if no usable tokens."""
    toks = _WORD.findall(text.lower())
    content = [t for t in toks if t not in _STOPWORDS]
    return " OR ".join(f'"{t}"' for t in (content or toks))


class Bm25Index:
    def __init__(self, path: str | None = None, cfg: dict | None = None):
        if path is None:
            path = (cfg or {}).get("paths", {}).get("bm25", DEFAULT_PATH)
        self.path = path
        # read-only-ish connection reused for searches; check_same_thread off so the web server's
        # worker threads can share it. Sharing one connection is only safe when the sqlite library
        # is compiled fully serialized (sqlite3.threadsafety == 3, true for python.org builds, not
        # guaranteed everywhere), so `search` serializes access with a lock regardless: an FTS read
        # is sub-millisecond next to model latency, making contention irrelevant and the code
        # correct on every build (audit suspicion S2).
        self._con = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()

    def close(self) -> None:
        """Close the sqlite connection (releases the file lock, required on Windows before the
        index file can be replaced by a rebuild)."""
        self._con.close()

    @classmethod
    def build(cls, chunks, path: str | None = None, cfg: dict | None = None, batch: int = 5000,
              expected_count: int | None = None) -> "Bm25Index":
        """(Re)build the FTS index from an iterable of chunk dicts ({chunk_id, game, title, text}).

        With ``expected_count`` the temp index is verified BEFORE it replaces the live one: its row
        count must match and no chunk_id may repeat, else the temp file is deleted and a RuntimeError
        names the problem, leaving the working index untouched.

        Builds into a temp file in the same directory, then ``os.replace``s it into place (atomic
        on POSIX *and* Windows) so a reader never sees a missing or half-written index. If a
        running server holds the destination open, Windows blocks the replace: that surfaces as a
        RuntimeError telling the operator to stop the server, not a raw PermissionError."""
        if path is None:
            path = (cfg or {}).get("paths", {}).get("bm25", DEFAULT_PATH)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp_path = path + ".tmp"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)             # stale leftover from an interrupted build
        con = sqlite3.connect(tmp_path)
        con.execute("PRAGMA journal_mode=WAL")
        # contentless-ish: text in FTS5, identity/filter columns in a parallel table keyed by rowid.
        con.execute("CREATE VIRTUAL TABLE docs USING fts5(text, tokenize='porter unicode61')")
        # `game` = single display label; `games` = comma-joined multi-tag membership for filtering.
        con.execute("CREATE TABLE meta (rowid INTEGER PRIMARY KEY, chunk_id TEXT, game TEXT, "
                    "games TEXT, title TEXT)")
        ins_meta = "INSERT INTO meta(rowid, chunk_id, game, games, title) VALUES (?, ?, ?, ?, ?)"
        rows_fts = []
        rows_meta = []
        n = 0
        for rowid, c in enumerate(chunks, start=1):
            rows_fts.append((rowid, c.get("text") or ""))
            rows_meta.append((rowid, c["chunk_id"], c.get("game"), _games_str(c), c.get("title")))
            if len(rows_fts) >= batch:
                con.executemany("INSERT INTO docs(rowid, text) VALUES (?, ?)", rows_fts)
                con.executemany(ins_meta, rows_meta)
                rows_fts.clear()
                rows_meta.clear()
            n = rowid
        if rows_fts:
            con.executemany("INSERT INTO docs(rowid, text) VALUES (?, ?)", rows_fts)
            con.executemany(ins_meta, rows_meta)
        con.execute("CREATE INDEX idx_meta_game ON meta(game)")
        con.commit()
        if expected_count is not None:
            problem = None
            if n != expected_count:
                problem = (f"BM25 index built {n} rows but the collection has {expected_count} "
                           "(offset-paginated get() may have skipped or duplicated chunks)")
            else:
                dup = con.execute("SELECT chunk_id, COUNT(*) c FROM meta GROUP BY chunk_id "
                                  "HAVING c > 1 LIMIT 1").fetchone()
                if dup is not None:
                    problem = (f"BM25 index has a duplicate chunk_id {dup[0]!r} "
                               "(offset-paginated get() returned a chunk twice)")
            if problem:
                con.close()
                os.remove(tmp_path)
                raise RuntimeError(problem)
        con.close()                          # the builder's own handle must not block the replace
        try:
            os.replace(tmp_path, path)
        except PermissionError as exc:
            raise RuntimeError(
                f"Cannot replace BM25 index at {path}: the file is open in another process "
                "(on Windows an open handle blocks replacement, a running server holds the "
                "index). Stop the server, then rerun the rebuild; the new index was built to "
                f"{tmp_path} and is not lost."
            ) from exc
        idx = cls(path=path)
        idx.count = n
        return idx

    @classmethod
    def from_collection(cls, cfg: dict, page: int = 10000) -> "Bm25Index":
        """Build the index from the LIVE ChromaDB collection, so the BM25 ``game`` column matches the
        current (re-tagged) metadata exactly rather than a possibly-stale chunks.jsonl."""
        import chromadb
        from chromadb.config import Settings

        from .errors import SetupError

        path = cfg["paths"]["vectorstore"]
        name = cfg.get("collection_name", "xeno_wiki")
        # get_collection, never get_or_create: building the index must not conjure an empty store.
        if not os.path.isfile(os.path.join(path, "chroma.sqlite3")):
            raise SetupError(f"Vector store not found at {path}. Run `python -m scripts.setup` to "
                             "download the prebuilt store before building the BM25 index.")
        client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
        try:
            col = client.get_collection(name)
        except Exception as exc:
            raise SetupError(f"The vector store at {path} is unreadable or has no collection named {name!r} "
                             f"({type(exc).__name__}); check `collection_name` in config.yaml.") from exc
        return cls._from_collection_obj(col, cfg, page=page)

    @classmethod
    def _from_collection_obj(cls, col, cfg: dict, page: int = 10000) -> "Bm25Index":
        """Build from an already-opened collection object (split out from ``from_collection`` so it
        is testable with a fake collection, no real ChromaDB store needed)."""
        total = col.count()
        if total == 0:
            # An empty collection would replace a good index with a 0-row one and report success.
            raise RuntimeError("the collection is empty; refusing to build a 0-row BM25 index")

        def it():
            off = 0
            while off < total:
                got = col.get(include=["documents", "metadatas"], limit=page, offset=off)
                for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
                    m = meta or {}
                    # Reconstruct multi-tag membership from the per-game `g_<game>` flags so the BM25
                    # filter matches the collection's metadata exactly (falls back to the label).
                    member = [g for g in _BASE_GAMES if m.get(f"g_{g}")]
                    yield {"chunk_id": cid, "game": m.get("game"), "games": member,
                           "title": m.get("title"), "text": doc}
                off += page

        # P2-5: chromadb's get() ordering across offset-paginated pages is not contractually
        # guaranteed to be stable (pinned chromadb>=1.5.9,<1.6 happens to be stable in practice,
        # but nothing enforces it) - the collection is static during the build, so a stable get()
        # must yield exactly `total` rows once each. `expected_count` makes build() verify that on the
        # temp file before it replaces the live index; a lossy or duplicated index is never shipped.
        return cls.build(it(), cfg=cfg, expected_count=total)

    def search(self, query: str, n: int = 60, game_filter: str | None = None):
        """Return up to ``n`` chunk_ids ranked best-first (lowest bm25 score) for the query, honoring
        the multi-tag membership game filter. Empty list if the query has no usable terms."""
        match = _match_query(query)
        if not match:
            return []
        sql = ("SELECT m.chunk_id FROM docs d JOIN meta m ON d.rowid = m.rowid "
               "WHERE docs MATCH ?")
        params = [match]
        game = filter_membership(game_filter)   # base game whose membership flag must be present
        if game is not None:
            sql += " AND (',' || m.games || ',') LIKE ?"
            params.append(f"%,{game},%")
        sql += " ORDER BY bm25(docs) LIMIT ?"
        params.append(n)
        with self._lock:
            return [r[0] for r in self._con.execute(sql, params).fetchall()]


def run(cfg: dict) -> int:
    """Pipeline entry: (re)build the BM25 index from the collection. Returns the chunk count."""
    idx = Bm25Index.from_collection(cfg)
    count = idx.count
    idx.close()                  # release the file handle (Windows blocks replacing an open file)
    return count
