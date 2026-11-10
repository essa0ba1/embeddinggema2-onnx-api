import pytest
from fastapi.testclient import TestClient

from main import app, get_embedding_model


class FakeEmbedding:
    def tolist(self):
        return [[0.1, 0.2, 0.3]]


class FakeModel:
    def __init__(self):
        self.text_request = None
        self.image = None

    def process_text(self, **request):
        self.text_request = request
        return FakeEmbedding()

    def process_vision(self, image):
        self.image = image
        return FakeEmbedding(), FakeEmbedding()


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def client(fake_model):
    app.dependency_overrides[get_embedding_model] = lambda: fake_model
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_text_embedding_returns_embedding_and_passes_request(client, fake_model):
    response = client.post(
        "/v1/embeddings/text",
        json={"input_text": "hello", "task": "query_task"},
    )

    assert response.status_code == 200
    assert response.json() == {"text_embedding": [[0.1, 0.2, 0.3]]}
    assert fake_model.text_request["input_text"] == "hello"
    assert fake_model.text_request["task"] == "query_task"


@pytest.mark.parametrize(
    "payload",
    [
        {"input_text": "   ", "task": "query_task"},
        {"input_text": "hello", "task": "unknown_task"},
        {
            "input_text": ["one", "two"],
            "title": ["only one", "two", "three"],
            "task": "document_task",
        },
    ],
)
def test_text_embedding_rejects_invalid_request(client, payload):
    response = client.post("/v1/embeddings/text", json=payload)

    assert response.status_code == 422


def test_vision_embedding_accepts_uploaded_file(client, fake_model):
    image_bytes = b"test image contents"
    response = client.post(
        "/v1/embeddings/vision",
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "vision_embedding": [[0.1, 0.2, 0.3]],
        "text_embedding": [[0.1, 0.2, 0.3]],
    }
    assert fake_model.image == image_bytes
