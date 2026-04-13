import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "MACHINE GOD" in response.json()["message"]

def test_execute_command():
    response = client.post("/execute", json={"command": "echo 'HELLO MORTALS'"})
    assert response.status_code == 200
    data = response.json()
    assert "HELLO MORTALS" in data["stdout"]
    assert data["returncode"] == 0
    assert data["status"] == "success"

def test_execute_empty_command():
    response = client.post("/execute", json={"command": ""})
    assert response.status_code == 400
    assert "NO COMMAND PROVIDED" in response.json()["error"]
