# EmbeddingGemma-2 ONNX API

FastAPI service that embeds **text** and **images** with int8 ONNX EmbeddingGemma-2 models via ONNX Runtime (CUDA when available, otherwise CPU).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` uses `onnxruntime-gpu`. For CPU-only hosts, replace it with `onnxruntime`.

### Download models

```bash
export MODEL_DIR=models/embeddinggemma-2-int8-onnx
python download_model.py
```

Expected files under `MODEL_DIR`:

- `model_q8.onnx`
- `vision_encoder_dynamic_q8.onnx`
- `audio_encoder_q8.onnx`
- `tokenizer.json`

Repo: [`aissabakhil/embeddinggemma-2-int8-onnx`](https://huggingface.co/aissabakhil/embeddinggemma-2-int8-onnx)

## Run

```bash
export MODEL_DIR=models/embeddinggemma-2-int8-onnx
uvicorn main:app --host 0.0.0.0 --port 8000
# or: fastapi dev
```

OpenAPI docs: `http://localhost:8000/docs`

## API

### `POST /v1/embeddings/text`

JSON body:

| Field | Required | Notes |
|-------|----------|--------|
| `input_text` | yes | string or list (1–100), each 1–2000 chars |
| `task` | yes | `query_task` or `document_task` |
| `prefix` | no | query only; default `task: search result \| query: ` |
| `title` | no | document only; string or list aligned with `input_text` |

Query example:

```bash
curl -s http://localhost:8000/v1/embeddings/text \
  -H 'Content-Type: application/json' \
  -d '{"input_text":"red sneakers","task":"query_task"}'
```

Document example:

```bash
curl -s http://localhost:8000/v1/embeddings/text \
  -H 'Content-Type: application/json' \
  -d '{"input_text":"Comfortable running shoes","title":"Product","task":"document_task"}'
```

Response: `{ "text_embedding": [[...]] }`

### `POST /v1/embeddings/vision`

Multipart upload (`JPEG` / `PNG` / `WEBP`, max 5 MB, max 20M pixels):

```bash
curl -s http://localhost:8000/v1/embeddings/vision \
  -F 'file=@image.jpg'
```

Response: `{ "vision_embedding": [...], "text_embedding": [...] }`  
Use `text_embedding` for retrieval against text embeddings.

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_DIR` | `models/embeddinggemma-2-int8-onnx` | ONNX + tokenizer directory |
| `CUDA_DEVICE_ID` | `0` | GPU index when CUDA is available |
| `ORT_INTRA_THREADS` | CPU count | ORT intra-op threads |

Provider selection prefers `CUDAExecutionProvider`, then falls back to CPU. Check logs for `Session providers in use`.

## Docker

```bash
docker build -t embeddinggemma2 .
docker run --rm -p 8000:8000 \
  -e MODEL_DIR=/models/embeddinggemma-2-int8-onnx \
  -v "$PWD/models/embeddinggemma-2-int8-onnx:/models/embeddinggemma-2-int8-onnx:ro" \
  embeddinggemma2
```

For NVIDIA GPU: install the NVIDIA Container Toolkit and run with `--gpus all` (image must have a working `onnxruntime-gpu` + CUDA stack).

## Tests

```bash
pytest
```

## Layout

- `main.py` — FastAPI routes + request validation
- `embedder.py` — ONNX sessions, text/vision encoding
- `download_model.py` — Hugging Face snapshot download
