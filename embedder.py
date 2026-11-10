import io

from typing import List
import numpy as np
import os
import time
import onnxruntime as ort
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import urlparse
from PIL import Image
import requests
import math
from tokenizers import Tokenizer
from huggingface_hub import snapshot_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#models/embeddinggemma-2-int8-onnx/model_q8.onnx
ROOT_MODEL_DIR = Path(
    os.getenv(
        "MODEL_DIR",
        "models/embeddinggemma-2-int8-onnx"
    )
)


PATH  = Path(ROOT_MODEL_DIR).expanduser()

text_encoder_path_q8 = str(PATH / "model_q8.onnx")
vision_encoder_path_q8 = str(PATH / "vision_encoder_dynamic_q8.onnx")
audio_encoder_path_q8 = str(PATH / "audio_encoder_q8.onnx")
tokenizer_path = str(PATH / "tokenizer.json")





PATCH, POOL, MAX_PATCHES = 16, 3, 280 * 3**2
VIDEO8TOK, IMAGE_TOK, BOS, BOI, EOI = 258884 , 258880, 2, 255999, 258882  # vision params from processor_config.json
EMPTY = np.zeros((0, 512), np.float32) 


def is_valid_syntax(url: str) -> bool:
    try:
        result = urlparse(url)
        # Ensure it has a scheme (http/https) and a domain name (netloc)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except ValueError:
        return False

@dataclass
class ImageInput:
    image :str | Path | Image.Image | np.ndarray | bytes

    

    def load(self):
        if isinstance(self.image, str):
            if is_valid_syntax(self.image):
                self.image = Image.open(requests.get(self.image, stream=True).raw).convert("RGB")
                self.image = np.array(self.image)
            else:
                self.image = np.array(Image.open(self.image).convert("RGB"))
                
        elif isinstance(self.image, Path):
            self.image = np.array(Image.open(str(self.image)).convert("RGB"))
        elif isinstance(self.image, bytes):
            self.image = np.array(Image.open(io.BytesIO(self.image)).convert("RGB"))
        elif isinstance(self.image, Image.Image):
            self.image = np.array(self.image.convert("RGB"))
        elif isinstance(self.image, np.ndarray):
            self.image = np.array(Image.fromarray(self.image).convert("RGB"))
        return self.image

class EmbeddingModel:
    def __init__(self, text_encoder_path, vision_encoder_path, audio_encoder_path, tokenizer_path):
        self.text_encoder = text_encoder_path
        self.vision_encoder = vision_encoder_path
        self.audio_encoder = audio_encoder_path
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.text_encoder_session = self.create_session(text_encoder_path)
        self.vision_encoder_session = self.create_session(vision_encoder_path)
        self.audio_encoder_session = self.create_session(audio_encoder_path)
    
    def create_session(self, model_path):
        session_options = ort.SessionOptions()
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.intra_op_num_threads = int(os.getenv("ORT_INTRA_THREADS", os.cpu_count() or 1))
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available = ort.get_available_providers()
        logger.info(f"Available providers: {available}")

        # Prefer NVIDIA when truly available; always keep CPU fallback
        preferred = []
        if "CUDAExecutionProvider" in available:
            preferred.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": int(os.getenv("CUDA_DEVICE_ID", "0")),
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": "1",
                    },
                )
            )
        preferred.append("CPUExecutionProvider")

        session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=preferred,
        )
        logger.info(f"Session providers in use: {session.get_providers()}")
        return session
        
        

    def process_vision(self,  image : str | Path | Image.Image | np.ndarray):

        img = ImageInput(image=image).load()
        h, w = img.shape[:2]
        img = Image.fromarray(img)

        
        # Resize keeping the aspect ratio; both sides must be multiples of POOL * PATCH.
        side, scale = POOL * PATCH, math.sqrt(MAX_PATCHES * PATCH**2 / (w * h))
        th, tw = int(scale * h // side) * side, int(scale * w // side) * side
        px = np.asarray(img.resize((tw, th), Image.BICUBIC), dtype=np.float32) 
        px = px.transpose(2, 0, 1) / 255.0
        ph, pw = th // PATCH, tw // PATCH
        patches = px.reshape(3, ph, PATCH, pw, PATCH).transpose(1, 3, 2, 4, 0).reshape(ph * pw, -1)
        grid = np.stack(np.meshgrid(np.arange(pw), np.arange(ph), indexing="xy"), -1).reshape(-1, 2)
       
     
        start_time = time.time()
        vision_features = np.asarray(self.vision_encoder_session.run(["image_features"], {
        "pixel_values": patches[None],
        "pixel_position_ids": grid[None].astype(np.int64)})[0], np.float32)
        end_time = time.time()
        logger.info(f"Vision encoding time: {end_time - start_time:.4f} seconds")
        n = len(vision_features)
        input_ids = np.array([[BOS, BOI] + [IMAGE_TOK] * n + [EOI, 1]], np.int64)
        attention_mask = np.ones((1, n + 4), np.int64)
      
        start_time = time.time()
        out = self.text_encoder_session.run(["sentence_embedding"], {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_features": vision_features, "video_features": EMPTY, "audio_features": EMPTY})[0]
        end_time = time.time()
        logger.info(f"Text encoding with vision features time: {end_time - start_time:.4f} seconds")
   
        return vision_features, out[0]

    


    def process_text(self, input_text, prefix:str|List[str], task:str, title :str | None | List[str] ):
        if task == "query_task":
            if isinstance(input_text, str):

                input_text = [prefix + input_text] if prefix else [input_text]
            else :
                input_text = [prefix + t for t in input_text]

            logger.info(f"Processing text with task {task}, input_text: {input_text}")
        
        if task == "document_task":
            if isinstance(input_text, str):
                template = f"title: {title[0] if isinstance(title, list) else title} | text: {input_text}" if title else f"text: {input_text}"
                input_text = [template]
            else:
                if title is None:
                    input_text = [f"text: {t}" for t in input_text]
                else:
                    len_title = len(title) if isinstance(title, list) else 1
                    if len_title == len(input_text):
                        input_text = [f"title: {ti} | text: {t}" if ti else f"text: {t}" for t, ti in zip(input_text, title)]
                    elif len_title == 1:
                        input_text = [f"title: {title[0] if isinstance(title, list) else title} | text: {t}" for t in input_text]
                    else:
                        raise ValueError("Length of title list must be 1 or equal to the length of input_text list.")
            logger.info(f"Processing text with task {task}  -> input_text: {input_text}")



        encs = self.tokenizer.encode_batch(input_text)
        n = max(len(e.ids) for e in encs)
        ids  = np.array([e.ids + [0] * (n - len(e.ids)) for e in encs], np.int64)
        mask = np.array([[1] * len(e.ids) + [0] * (n - len(e.ids)) for e in encs], np.int64)

        return self.text_encoder_session.run(["sentence_embedding"], {
            "input_ids": ids, "attention_mask": mask,
            "image_features": EMPTY, "video_features": EMPTY, "audio_features": EMPTY})[0]


    
if __name__ == "__main__":
    import time
        
       
    model = EmbeddingModel(text_encoder_path, vision_encoder_path, audio_encoder_path, tokenizer_path)
    img = "Giraffes.webp"
    start_time = time.time()
    _ , vision = model.process_vision(image=img)
    end_time = time.time()
    logger.info(f"Total time for vision embedding: {end_time - start_time:.4f} seconds")
  

        
      