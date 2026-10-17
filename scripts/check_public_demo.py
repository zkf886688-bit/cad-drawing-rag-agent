"""Exercise public demo retrieval, evidence and multi-turn API behavior offline."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["CAD_GENERATE"] = "extractive"
os.environ["CAD_RAG_BACKEND"] = "bm25"

from fastapi.testclient import TestClient
from app.main import app

with TestClient(app) as client:
    assert client.get("/health").status_code == 200
    assert client.get("/ui").status_code == 200
    for question, expected in [
        ("工程示意平面图的图号是什么", "DEMO-ENG-001"),
        ("它有哪些图层", "TITLE"),
        ("工程示意平面图梁长尺寸是多少", "6000"),
    ]:
        response = client.post("/chat", json={
            "message": question, "session_id": "public-demo-check",
            "generate_mode": "extractive"
        })
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["drawing"] == "demo_engineering_sheet", result
        assert result["sources"], result
        assert expected in result["answer"], result
        assert result["generate_mode"] == "extractive", result
        print("PASS:", question)
print("Public demo API checks passed.")
