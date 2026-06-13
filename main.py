# ml-service/main.py
#
# -> entry point FastAPI ML service
#    -> load InsightFacePipeline sebagai singleton saat startup (sekali load, reuse per request)
#    -> expose 2 endpoint: POST /face/register, POST /face/inference
# -> jalankan: uvicorn main:app --host 0.0.0.0 --port 8001 --reload

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.face_pipeline import InsightFacePipeline
from app.routes.face import router as face_router

load_dotenv()

# path ke folder model ONNX - ambil dari env, default ke ../ml-models
MODEL_DIR = os.getenv("MODEL_DIR", str(Path(__file__).parent.parent.parent / "ml-models"))
PORT      = int(os.getenv("PORT", "8001"))


# ========================================================================================
# LIFESPAN - load model sekali saat startup, cleanup saat shutdown
# ========================================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # validasi semua model tersedia sebelum startup
    model_dir = Path(MODEL_DIR)
    required  = ["det_10g.onnx", "w600k_r50.onnx", "genderage.onnx"]
    missing   = [m for m in required if not (model_dir / m).exists()]

    if missing:
        raise RuntimeError(
            f"Model ONNX tidak ditemukan: {missing}\n"
            f"Path: {model_dir}\n"
            f"Jalankan ml-models/download_models.sh terlebih dahulu."
        )

    print(f"[startup] loading InsightFace models dari: {model_dir}")
    # load pipeline sekali - simpan di app.state supaya bisa diakses semua route
    app.state.pipeline = InsightFacePipeline(str(model_dir))
    print("[startup] ML service siap")

    yield  # service berjalan

    # cleanup saat shutdown (ONNX session akan di-GC secara otomatis)
    print("[shutdown] ML service berhenti")


# ========================================================================================
# APP SETUP
# ========================================================================================

app = FastAPI(
    title="Smart Door - ML Face Recognition Service",
    description=(
        "Microservice Python untuk face registration dan inference\n"
        "menggunakan InsightFace pipeline (det_10g + w600k_r50 + genderage ONNX)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# register routes
app.include_router(face_router)


# ========================================================================================
# HEALTH CHECK
# ========================================================================================

# endpoint health check
# output : { status: "ok", model_dir: str }
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_dir": MODEL_DIR}


# ========================================================================================
# ENTRYPOINT (dev)
# ========================================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
