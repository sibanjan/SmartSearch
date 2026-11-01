import pytest

from smartsearch.config import Config
from smartsearch.engine import Engine


@pytest.fixture
def corpus(tmp_path):
    docs = tmp_path / "docs"
    (docs / "finance").mkdir(parents=True)
    (docs / "recipes").mkdir()
    (docs / "node_modules").mkdir()
    (docs / ".secret").mkdir()

    (docs / "finance" / "tax_return_2025.txt").write_text(
        "Annual income tax return. Deductions for home office and charitable donations."
    )
    (docs / "finance" / "budget.csv").write_text("category,amount\nrent,1800\ngroceries,450\n")
    (docs / "recipes" / "pasta.md").write_text(
        "# Carbonara\n\nCook spaghetti. Whisk eggs with pecorino cheese and black pepper. "
        "Fry guanciale until crisp."
    )
    (docs / "notes.py").write_text(
        "def train_model(data):\n    # gradient boosting classifier\n    pass\n"
    )
    (docs / "node_modules" / "ignored.txt").write_text("tax return should not be indexed")
    (docs / ".secret" / "hidden.txt").write_text("tax return hidden")
    return docs


@pytest.fixture
def engine(tmp_path, corpus):
    cfg = Config(roots=[str(corpus)], index_dir=str(tmp_path / "index"), embedder="hash")
    eng = Engine(cfg)
    eng.indexer.run()
    return eng
