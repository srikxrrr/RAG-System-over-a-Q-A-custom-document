from fastapi.testclient import TestClient


def test_health_and_ask(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from civilrag import api
    c = TestClient(api.app)
    assert c.get("/api/health").status_code == 200
    r = c.post("/api/ask", json={"question": "What is the minimum curing period for blended cements?"})
    assert r.status_code == 200 and r.json()["sources"]
    assert c.post("/api/ask", json={"question": "x"}).status_code == 422


def test_upload_rejects_bad_type():
    from civilrag import api
    c = TestClient(api.app)
    r = c.post("/api/upload", files={"files": ("evil.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400
