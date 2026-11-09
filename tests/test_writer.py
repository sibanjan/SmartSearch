"""While the SmartSearch daemon runs on an index, the CLI's write commands refuse."""

import subprocess
import sys

import pytest
from typer.testing import CliRunner

from smartsearch.cli import app
from smartsearch.writer import IndexBusy, daemon_pid, ensure_can_write


@pytest.fixture
def other_process():
    """A live process that is not this one, standing in for the daemon."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield p.pid
    p.kill()
    p.wait()


@pytest.fixture
def config(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("hello world")
    index = tmp_path / "index"
    index.mkdir()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"roots:\n- {docs.as_posix()}\nindex_dir: {index.as_posix()}\nembedder: hash\n")
    return cfg, index


def test_live_daemon_blocks_writes(tmp_path, other_process):
    (tmp_path / "daemon.pid").write_text(str(other_process))
    assert daemon_pid(tmp_path) == other_process
    with pytest.raises(IndexBusy, match="only writer"):
        ensure_can_write(tmp_path)


def test_stale_or_missing_pid_file_allows_writes(tmp_path):
    ensure_can_write(tmp_path)  # no pid file
    (tmp_path / "daemon.pid").write_text("999999")  # no such process
    assert daemon_pid(tmp_path) is None
    ensure_can_write(tmp_path)


def test_index_command_refuses_while_daemon_runs(config, other_process):
    cfg, index = config
    (index / "daemon.pid").write_text(str(other_process))
    r = CliRunner().invoke(app, ["index", "--config", str(cfg)])
    assert r.exit_code == 1 and "only writer" in " ".join(r.output.split())
    stats = CliRunner().invoke(app, ["stats", "--config", str(cfg)])  # reading still works
    assert stats.exit_code == 0, stats.output


def test_index_command_runs_without_daemon(config):
    cfg, _ = config
    r = CliRunner().invoke(app, ["index", "--config", str(cfg)])
    assert r.exit_code == 0, r.output
