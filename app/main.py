"""Add a trained MNIST GAN to the existing course API."""

from contextlib import asynccontextmanager
import math
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse

from app.classifier import ImageClassifier, InvalidImageError, ImageLimitError, MAX_UPLOAD_BYTES
from app.gan_service import GANService
from app.advanced_service import AdvancedService
from pydantic import BaseModel, ConfigDict, Field

from app.bigram_model import BigramModel
from app.embeddings import (
    MODEL_NAME,
    InvalidWordError,
    UnknownWordError,
    WordEmbeddingService,
)
from app.schemas import (
    ClassificationResponse,
    EmbeddingResponse,
    HealthResponse,
    TextGenerationRequest,
    TextGenerationResponse,
    GANRequest,
    GANResponse,
)

CORPUS = [
    "The cat reads books",
    "The dog reads notes",
    "The cat writes notes",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.embeddings = WordEmbeddingService()
    app.state.bigram = BigramModel(CORPUS)
    app.state.classifier = ImageClassifier()
    app.state.gan = GANService()
    app.state.ebm = AdvancedService('ebm')
    app.state.diffusion = AdvancedService('diffusion')
    yield
    del app.state.embeddings
    del app.state.bigram
    del app.state.classifier
    del app.state.gan


app = FastAPI(
    title="Assignment 4: CIFAR-10 Energy and Diffusion Models",
    description="Trained CIFAR-10 EBM and U-Net diffusion generators, extending the existing course API.",
    version="4.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    # The JSON decoder accepts NaN/Infinity, and even valid JSON 1e309 becomes
    # infinity. Preserve validation details without echoing invalid JSON numbers.
    content = jsonable_encoder(
        {"detail": exc.errors()},
        custom_encoder={float: lambda value: value if math.isfinite(value) else str(value)},
    )
    return JSONResponse(status_code=422, content=content)


@app.get("/", tags=["Status"])
def read_root():
    return {"message": "Assignment 4 API is running", "documentation": "/docs",
            "demo": "/advanced/demo", "energy": "/ebm/generate", "diffusion": "/diffusion/generate"}


@app.get("/health", response_model=HealthResponse, tags=["Status"])
def health(request: Request):
    service = request.app.state.embeddings
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_version": service.model_version,
        "dimensions": service.dimensions,
    }


@app.get(
    "/embedding",
    response_model=EmbeddingResponse,
    tags=["Word embeddings"],
    responses={
        404: {"description": "No pretrained vector exists for the word."},
        422: {"description": "A single alphabetic word is required."},
    },
)
def get_embedding(
    request: Request,
    word: Annotated[
        str,
        Query(min_length=1, max_length=100, description="One English word, for example apple"),
    ],
):
    """Return all 300 components of the lowercase word's static embedding."""
    try:
        return request.app.state.embeddings.embed(word)
    except InvalidWordError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except UnknownWordError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/generate", response_model=TextGenerationResponse, tags=["Text generation"])
def generate_text(request: Request, body: TextGenerationRequest):
    """Sample up to length words from the small course bigram model."""
    try:
        generated = request.app.state.bigram.generate_text(body.start_word, body.length)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"generated_text": generated}


@app.get("/model", tags=["Image classification"])
def model_info(request: Request):
    """Inspect the bundled CNN checkpoint, class order and preprocessing."""
    return request.app.state.classifier.metadata


@app.post(
    "/classify", response_model=ClassificationResponse, tags=["Image classification"],
    responses={413: {"description": "Image byte or pixel limit exceeded."},
               422: {"description": "Missing, corrupt or unsupported image."}},
)
def classify_image(request: Request, file: Annotated[UploadFile, File(description="PNG/JPEG; maximum 5 MiB and 20 million pixels")]):
    """Return one of the ten CIFAR-10 classes for an uploaded image."""
    try:
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        return request.app.state.classifier.classify(content)
    except ImageLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        file.file.close()


@app.get("/gan/model", tags=["GAN digit generation"])
def gan_model(request: Request):
    """Inspect the bundled generator, training scope and model checksum."""
    return request.app.state.gan.metadata


@app.post("/gan/generate", response_model=GANResponse, tags=["GAN digit generation"])
def generate_digits(request: Request, body: GANRequest):
    """Return every generated 28x28 grayscale PNG plus a contact sheet as Base64."""
    return request.app.state.gan.generate(body.count, body.seed)


@app.get("/gan/demo", response_class=HTMLResponse, tags=["GAN digit generation"])
def gan_demo():
    """Small offline browser client for the same documented API."""
    return (Path(__file__).parent / "static" / "gan_demo.html").read_text(encoding="utf-8")


class EBMRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    count: int = Field(default=4, ge=1, le=16, strict=True)
    seed: int | None = Field(default=42, ge=0, le=2**31-1, strict=True)
    steps: int = Field(default=60, ge=1, le=120, strict=True)


class DiffusionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    count: int = Field(default=4, ge=1, le=16, strict=True)
    seed: int | None = Field(default=42, ge=0, le=2**31-1, strict=True)
    steps: int = Field(default=50, ge=2, le=100, strict=True)


class AdvancedResponse(GANResponse):
    steps: int


@app.get('/ebm/model', tags=['Energy generation'])
def ebm_info(request: Request):
    return request.app.state.ebm.metadata


@app.post('/ebm/generate', response_model=AdvancedResponse, tags=['Energy generation'])
def ebm_generate(request: Request, body: EBMRequest):
    return request.app.state.ebm.generate(body.count, body.seed, body.steps)


@app.get('/diffusion/model', tags=['Diffusion generation'])
def diffusion_info(request: Request):
    return request.app.state.diffusion.metadata


@app.post('/diffusion/generate', response_model=AdvancedResponse, tags=['Diffusion generation'])
def diffusion_generate(request: Request, body: DiffusionRequest):
    return request.app.state.diffusion.generate(body.count, body.seed, body.steps)


@app.get('/advanced/demo', response_class=HTMLResponse, tags=['Image generation demo'])
def advanced_demo():
    return (Path(__file__).parent/'static'/'advanced_demo.html').read_text(encoding='utf-8')
