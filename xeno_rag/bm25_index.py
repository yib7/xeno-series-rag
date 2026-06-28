"""A lexical BM25 retriever over the chunk corpus, backed by SQLite FTS5.

Dense (Qwen3-Embedding) retrieval misses exact proper-noun / concept queries when many near-duplicate ancillary
pages (weapon SKUs, music tracks, boss-instances) crowd the canonical page out of the candidate
window — e.g. "What are mimeosomes?" returned only Skell weapon part-numbers. BM25 scores exact term
overlap, so the page that literally says "mimeosome" ranks first.

FTS5 is built into Python's bundled sqlite3: persistent, scales to the full ~169k chunks, low memory,
no heavy new dependency, and supports the per-game filter via a side metadata table. The index lives
in its own sqlite file (separate from ChromaDB's store) and is built from the live collection so its
``game`` column matches the re-tagged metadata exactly.
"""

import os
import re
import sqlite3

from .parse_wikitext import _BASE_GAMES, filter_membership, membership_from_game

# Word tokens for the MATCH query. Wrapping each token in double quotes neutralizes FTS5 operators
# (-, *, :, parentheses, NEAR), so an arbitrary user question can never be a malformed FTS expression.
_WORD = re.compile(r"[0-9A-Za-z]+")

DEFAULT_PATH = os.path.join("data", "vectorstore", "bm25.sqlite3")


def _games_str(chunk: dict) -> str:
    """Comma-joined membership for a chunk's ``games`` filter column: an explicit ``games`` list if
    present, else derived from the single ``game`` label. Wrapped with leading/trailing commas at
    query time so a ``LIKE '%,G,%'`` test matches whole codes only."""
    games = chunk.get("games")
    member = set(games) if games else membership_from_game(chunk.get("game"))
    return ",".join(sorted(member))


def _match_query(text: str) -> str:
    """Turn a free-text question into a safe FTS5 MATCH string: quoted tokens joined with OR (recall-
    friendly; bm25 still rewards documents matching more / rarer terms). Empty if no usable tokens."""
    toks = _WORD.findall(text.lower())
    toks = [t for t in toks if len(t) > 1]
    return " OR ".join(f'"{t}"' for t in toks)


class Bm25Index:
    def __init__(self, path: str = None, cfg: dict = None):
        if path is None:
            path = (cfg or {}).get("paths", {}).get("bm25", DEFAULT_PATH)
        self.path = path
        # read-only-ish connection reused for searches; check_same_thread off so the web server's
        # worker threads can share it (FTS5 reads are safe to share).
        self._con = sqlite3.connect(path, check_same_thread=False)

    @classmethod
    def build(cls, chunks, path: str = None, cfg: dict = None, batch: int = 5000) -> "Bm25Index":
        """(Re)build the FTS index from an iterable of chunk dicts ({chunk_id, game, title, text}).
        Overwrites any existing file so a rebuild is clean."""
        if path is None:
            path = (cfg or {}).get("paths", {}).get("bm25", DEFAULT_PATH)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if os.path.exists(path):
            os.remove(path)
        con = sqlite3.connect(path)
        con.execute("PRAGMA journal_mode=WAL")
        # contentless-ish: text in FTS5, identity/filter columns in a parallel table keyed by rowid.
        con.execute("CREATE VIRTUAL TABLE docs USING fts5(text, tokenize='porter unicode61')")
        # `game` = single display label; `games` = comma-joined multi-tag membership for filtering.
        con.execute("CREATE TABLE meta (rowid INTEGER PRIMARY KEY, chunk_id TEXT, game TEXT, "
                    "games TEXT, title TEXT)")
        ins_meta = "INSERT INTO meta(rowid, chunk_id, game, games, title) VALUES (?, ?, ?, ?, ?)"
        rows_fts = []
        rows_meta = []
        rowid = 0
        n = 0
        for c in chunks:
            rowid += 1
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
        con.close()
        idx = cls(path=path)
        idx.count = n
        return idx

    @classmethod
    def from_collection(cls, cfg: dict, page: int = 10000) -> "Bm25Index":
        """Build the index from the LIVE ChromaDB collection, so the BM25 ``game`` column matches the
        current (re-tagged) metadata exactly rather than a possibly-stale chunks.jsonl."""
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(path=cfg["paths"]["vectorstore"],
                                           settings=Settings(anonymized_telemetry=False))
        col = client.get_or_create_collection(name=cfg.get("collection_name", "xeno_wiki"),
                                              metadata={"hnsw:space": "cosine"})
        total = col.count()

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

        return cls.build(it(), cfg=cfg)

    def search(self, query: str, n: int = 60, game_filter: str = None):
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
        return [r[0] for r in self._con.execute(sql, params).fetchall()]


def run(cfg: dict) -> int:
    """Pipeline entry: (re)build the BM25 index from the collection. Returns the chunk count."""
    idx = Bm25Index.from_collection(cfg)
    return idx.count
