import os
import time

from smartsearch.chunker import chunk_text
from smartsearch.search import SearchFilters, parse_since


def test_chunker_respects_size():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 80 for i in range(20))
    chunks = chunk_text(text, size=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 600 for c in chunks)


def test_chunker_short_text():
    assert chunk_text("hello world") == ["hello world"]
    assert chunk_text("   ") == []


def test_index_skips_excluded_and_hidden(engine):
    paths = engine.store.all_paths()
    assert len(paths) == 4
    assert not any("node_modules" in p or ".secret" in p for p in paths)


def test_keyword_search(engine):
    res = engine.search("charitable donations", mode="keyword")
    assert res and res[0].name == "tax_return_2025.txt"


def test_hybrid_search_ranks_relevant_first(engine):
    res = engine.search("pecorino eggs spaghetti")
    assert res[0].name == "pasta.md"


def test_filename_match(engine):
    res = engine.search("budget")
    assert res[0].name == "budget.csv"
    assert "filename" in res[0].matched_by


def test_extension_filter(engine):
    res = engine.search("tax", filters=SearchFilters(extensions=["md"]))
    assert all(r.ext == ".md" for r in res)


def test_incremental_update_and_delete(engine, corpus):
    f = corpus / "recipes" / "pasta.md"
    f.write_text("Completely different: quantum chromodynamics lecture notes")
    os.utime(f, (time.time() + 5, time.time() + 5))
    rep = engine.indexer.run()
    assert rep.updated == 1 and rep.unchanged == 3
    assert engine.search("chromodynamics")[0].name == "pasta.md"

    f.unlink()
    rep = engine.indexer.run()
    assert rep.removed == 1
    assert not engine.search("chromodynamics", mode="keyword")


def test_parse_since():
    assert abs(parse_since("1d") - (time.time() - 86400)) < 5
    assert parse_since(None) is None
