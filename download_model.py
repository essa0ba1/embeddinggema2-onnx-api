import os
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_DIR = Path(
    os.getenv("MODEL_DIR", "/models/embeddinggemma-2-int8-onnx")
)

snapshot_download(
    repo_id="aissabakhil/embeddinggemma-2-int8-onnx",
    local_dir=MODEL_DIR,
)

