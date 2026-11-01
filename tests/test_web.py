from fastapi.testclient import TestClient

from smartsearch.web.app import create_app


def test_search_api(engine):
    client = TestClient(create_app(engine, watch=False, index_on_start=False))
    r = client.get("/api/search", params={"q": "pecorino"})
    assert r.status_code == 200
    assert r.json()["results"][0]["name"] == "pasta.md"


def test_open_rejects_unindexed(engine):
    client = TestClient(create_app(engine, watch=False, index_on_start=False))
    r = client.post("/api/open", json={"path": "/etc/passwd"})
    assert r.status_code == 404


def test_status_and_page(engine):
    client = TestClient(create_app(engine, watch=False, index_on_start=False))
    assert client.get("/api/status").json()["files"] == 4
    assert "SmartSearch" in client.get("/").text
