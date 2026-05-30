"""
main.py
=======
FastAPI REST API for the waste image classifier.

Endpoints:
  GET  /health          liveness check (used by Cloud Run / load balancer)
  GET  /classes         list all supported waste classes
  POST /predict         classify a waste image

Usage (local):
  pip install fastapi uvicorn python-multipart tensorflow Pillow numpy
  uvicorn main:app --host 0.0.0.0 --port 8080 --reload

  Then open: http://localhost:8080/docs  (automatic Swagger UI)

Environment variables:
  MODEL_PATH          path to model_best.keras  (default: ./model/model_best.keras)
  CLASS_INDICES_PATH  path to class_indices.json (default: ./model/class_indices.json)
  MAX_FILE_SIZE_MB    max upload size in MB      (default: 10)
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from predictor import WastePredictor, preprocess_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config from environment variables (easy to override in Docker / Cloud Run)
# ---------------------------------------------------------------------------

MODEL_PATH         = os.getenv("MODEL_PATH",         "./model/model_best.keras")
CLASS_INDICES_PATH = os.getenv("CLASS_INDICES_PATH",  "./model/class_indices.json")
MAX_FILE_SIZE_MB   = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_B    = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_ORIGINS = ["*"]

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/bmp", "image/webp",
}


# ---------------------------------------------------------------------------
# Startup / shutdown: load model once, reuse for every request
# ---------------------------------------------------------------------------

MODEL_LOAD_TIMEOUT_S = int(os.getenv("MODEL_LOAD_TIMEOUT_S", "120"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    logger.info("Starting up - loading model...")
    try:
        loop = asyncio.get_event_loop()
        predictor = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: WastePredictor(MODEL_PATH, CLASS_INDICES_PATH),
            ),
            timeout=MODEL_LOAD_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.error(
            "Model failed to load within %ds - aborting startup",
            MODEL_LOAD_TIMEOUT_S,
        )
        raise RuntimeError("Model load timeout")

    app.state.predictor = predictor
    logger.info("API ready")
    yield
    logger.info("Shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Waste Classifier API",
    description="Classifies household waste images into 10 categories using neural network model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_content_type(file: UploadFile):
    """Validate the upload's MIME type. Raises HTTPException on failure."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported file type: " + str(file.content_type) +
                ". Accepted: jpeg, png, bmp, webp."
            ),
        )


def validate_size(num_bytes: int):
    """Validate the upload's size. Raises HTTPException on failure."""
    if num_bytes == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    if num_bytes > MAX_FILE_SIZE_B:
        raise HTTPException(
            status_code=413,
            detail=(
                "File too large. Max size: " + str(MAX_FILE_SIZE_MB) + " MB. "
                "Got: " + str(round(num_bytes / 1024 / 1024, 1)) + " MB."
            ),
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
def health(request: Request):
    """
    Liveness check for Cloud Run and load balancers.
    Returns 200 when the model is loaded and ready.
    Returns 503 if the model has not loaded yet.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "ready": False},
        )
    return {"status": "ok", "ready": True}


@app.get("/classes", tags=["model"])
def get_classes(request: Request):
    """
    Return all waste classes the model can recognize.
    Useful for building UI dropdowns or documentation.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    classes = list(predictor.idx_to_class.values())
    return {
        "num_classes": len(classes),
        "classes":     sorted(classes),
    }


@app.post("/predict", tags=["inference"])
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Waste image (jpeg/png/bmp/webp)"),
):
    """
    Classify a single waste image.

    Returns the predicted class, confidence score, and scores for all classes.

    Example response:
    ```json
    {
      "class": "plastic",
      "confidence": 0.9423,
      "all_scores": {
        "plastic":    0.9423,
        "metal":      0.0312,
        "cardboard":  0.0098,
        ...
      },
      "latency_ms": 47.3,
      "filename": "bottle.jpg"
    }
    ```
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # 1. Validate content type before reading anything.
    validate_content_type(file)

    # 2. Pre-check Content-Length header so we don't pull a multi-GB payload
    #    into memory just to reject it. The header can be spoofed, so we
    #    re-validate after reading.
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_FILE_SIZE_B:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "File too large. Max size: " + str(MAX_FILE_SIZE_MB) + " MB."
                    ),
                )
        except ValueError:
            pass  # malformed header, fall through to post-read check

    # 3. Read and re-validate actual size.
    content = await file.read()
    validate_size(len(content))

    # 4. Preprocessing errors (bad/corrupt image) -> 422.
    try:
        arr = preprocess_image(content)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail="Could not decode image: " + str(e),
        )

    # 5. Inference errors (OOM, model failure) -> 500.
    try:
        result = predictor.predict_from_array(arr)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Inference failed: " + str(e),
        )

    result["filename"] = file.filename
    return result
