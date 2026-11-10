from functools import lru_cache
from typing import Any
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field, StringConstraints , model_validator 
from typing import Annotated, Literal
from typing_extensions import Self
import logging 
import warnings
from PIL import Image, UnidentifiedImageError
from io import BytesIO


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    from embedder import (
        EmbeddingModel,
        audio_encoder_path_q8,
        text_encoder_path_q8,
        tokenizer_path,
        vision_encoder_path_q8,
    )

    return EmbeddingModel(
        text_encoder_path_q8, vision_encoder_path_q8, audio_encoder_path_q8, tokenizer_path
    )





NonEmptyStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
]

NonEmptyStrList = Annotated[
    list[NonEmptyStr],
    Field(min_length=1, max_length=100)
]


ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 10 MB


def validate_image(image_bytes: bytes):

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning
            )

            with Image.open(BytesIO(image_bytes)) as image:

                if image.format not in ALLOWED_FORMATS:
                    raise HTTPException(
                        status_code=415,
                        detail="Unsupported image format."
                    )

                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(
                        status_code=413,
                        detail="Image dimensions are too large."
                    )

                image.verify()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsafe image."
        )

    return True


class TextEmbeddingRequest(BaseModel):
    input_text : NonEmptyStr | NonEmptyStrList
    title : NonEmptyStr | NonEmptyStrList | None = None
    task : Literal["document_task", "query_task"]
    prefix : Literal["task: search result | query: ", "task: captioning | query: ", "task: classification | query: "] | None = "task: search result | query: " 
    @model_validator(mode="after")
    def validate_title(self) -> Self:
        length_title = len(self.title) if isinstance(self.title, list) else 1
        length_input_text = len(self.input_text) if isinstance(self.input_text, list) else 1
        if length_title > 1 and length_title != length_input_text:
            raise ValueError("Length of title list must be 1 or equal to the length of input_text list.")
        return self

    @model_validator(mode="after")
    def validate_prefix(self) -> Self:
        if self.task == "document_task" and self.prefix is not None:
            self.prefix = None
        
        return self
        



@app.post("/v1/embeddings/text")
def process_text(
    user_request: TextEmbeddingRequest,
    model: Any = Depends(get_embedding_model),
):

    dict_request = user_request.model_dump()
    logger.info(f"Prefix: {dict_request['prefix']}, Task: {dict_request['task']}, Title: {dict_request['title']}, Input Text: {dict_request['input_text']}")
    text_embedding = model.process_text(**dict_request)
    return {"text_embedding": text_embedding.tolist()}


# for vision embedding we need to pass the image file or link as a form-data parameter
@app.post("/v1/embeddings/vision")
async def process_vision(
    file: UploadFile = File(...),
    model: Any = Depends(get_embedding_model),
):
    
       
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty image file."
        )
    elif len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image file size exceeds the maximum limit."
        )
    validate_image(image_bytes)
    vision_embedding, text_embedding = model.process_vision(image=image_bytes)
    return {
        "vision_embedding": vision_embedding.tolist(),
        "text_embedding": text_embedding.tolist()
    }
