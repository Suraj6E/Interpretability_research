import pytest
from fastapi.testclient import TestClient

from interp.serve import create_app


@pytest.fixture(scope="module")
def client(model, spec):
    return TestClient(create_app(model, spec))


def test_info(client):
    info = client.get("/api/info").json()
    assert info["model"] == "gpt2-small"
    assert (info["n_layers"], info["n_heads"]) == (12, 12)


def test_analyze_returns_viewer_format(client):
    response = client.post("/api/analyze", json={"prompt": "The Eiffel Tower is in the city of"})
    assert response.status_code == 200
    result = response.json()
    assert result["tokens"][0] == "<|endoftext|>"
    assert len(result["activation"]["heads"]) == 12
    assert len(result["top_predictions"]) == 5


@pytest.mark.parametrize("prompt", ["", "   ", "x" * 5000])
def test_bad_prompts_are_refused(client, prompt):
    assert client.post("/api/analyze", json={"prompt": prompt}).status_code == 400
