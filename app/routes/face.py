# ml-service/app/routes/face.py
#
# -> FastAPI router untuk face recognition endpoints
#    -> POST /register  : daftarkan wajah baru dengan 3 foto
#    -> POST /inference : kenali wajah dari 1 foto
# -> request pakai multipart/form-data (foto sebagai file upload)
# -> tidak perlu auth (public endpoint, seperti /access di Node.js backend)

from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse
from app.services.face_service import register_face, infer_face
import os

router = APIRouter(prefix="/face", tags=["Face Recognition"])

# ambil threshold dari env
MATCH_THRESHOLD     = float(os.getenv("MATCH_THRESHOLD", "0.40"))
DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", "0.5"))


# ========================================================================================
# POST /face/register
# ========================================================================================

# endpoint register wajah baru dengan 3 foto
# input  : multipart/form-data
#   - person_name : str (required) -> nama orang
#   - user_id     : int (optional) -> link ke tabel users
#   - photo_1     : file (required) -> JPEG atau PNG
#   - photo_2     : file (required) -> JPEG atau PNG
#   - photo_3     : file (required) -> JPEG atau PNG
# output : JSON {
#   success : bool
#   message : str
#   data    : { person_name, embeddings_saved, similarity_scores }
# }
@router.post("/register")
async def register_face_endpoint(
    request:     Request,
    person_name: str        = Form(..., description="Nama orang yang didaftarkan"),
    user_id:     int | None = Form(None, description="ID user di tabel users (opsional)"),
    photo_1:     UploadFile = File(..., description="Foto pertama (front-facing)"),
    photo_2:     UploadFile = File(..., description="Foto kedua"),
    photo_3:     UploadFile = File(..., description="Foto ketiga"),
):
    pipeline = request.app.state.pipeline

    # baca bytes dari semua file upload
    try:
        photos = [
            await photo_1.read(),
            await photo_2.read(),
            await photo_3.read(),
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal baca file: {str(e)}")

    # validasi semua file tidak kosong
    for i, p in enumerate(photos):
        if len(p) == 0:
            raise HTTPException(status_code=400, detail=f"Foto ke-{i + 1} kosong")

    try:
        result = register_face(
            pipeline=pipeline,
            photos=photos,
            person_name=person_name,
            user_id=user_id,
            det_threshold=DETECTION_THRESHOLD,
            sim_threshold=MATCH_THRESHOLD,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    return JSONResponse(
        status_code=201,
        content={
            "success": True,
            "message": "Face registered successfully",
            "data":    result,
        },
    )


# ========================================================================================
# POST /face/inference
# ========================================================================================

# endpoint kenali wajah dari 1 foto
# input  : multipart/form-data
#   - photo : file (required) -> JPEG atau PNG
# output : JSON {
#   success : bool
#   message : str
#   data    : { matched, person_name, user_id, similarity, gender, age, bbox, score }
# }
@router.post("/inference")
async def infer_face_endpoint(
    request: Request,
    photo:   UploadFile = File(..., description="Foto untuk dikenali"),
):
    pipeline = request.app.state.pipeline

    try:
        photo_bytes = await photo.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal baca file: {str(e)}")

    if len(photo_bytes) == 0:
        raise HTTPException(status_code=400, detail="Foto kosong")

    try:
        result = infer_face(
            pipeline=pipeline,
            photo_bytes=photo_bytes,
            match_threshold=MATCH_THRESHOLD,
            det_threshold=DETECTION_THRESHOLD,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    matched = result["matched"]
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": "Match found" if matched else "No match found",
            "data":    result,
        },
    )
