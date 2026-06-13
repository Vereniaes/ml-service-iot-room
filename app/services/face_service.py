# ml-service/app/services/face_service.py
#
# -> handling business logic face recognition (Stateless)
#    -> register_face  : proses 3 foto, validasi same person, kembalikan 3 embedding array
#    -> infer_face     : proses 1 foto, kembalikan 1 embedding array
# -> tidak ada koneksi database di sini (pure compute node)
# -> semua error dilempar sebagai ValueError yang ditangkap di router

import numpy as np
import cv2
from app.models.face_pipeline import InsightFacePipeline, cosine_similarity

# helper ---------------------------------------------------------------------------------

# fungsi convert bytes foto (dari upload) ke np.ndarray BGR
# input param : photo_bytes -> bytes raw image (JPEG/PNG)
# output : np.ndarray BGR shape (H, W, 3)
# error  : ValueError jika gagal decode (bukan format gambar valid)
def _bytes_to_bgr(photo_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(photo_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gagal decode gambar - pastikan format JPEG atau PNG")
    return img

# end of helper --------------------------------------------------------------------------


# ========================================================================================
# REGISTER FACE
# ========================================================================================

# fungsi utama registrasi wajah dari 3 foto
# input param : pipeline    -> InsightFacePipeline instance (sudah loaded)
#               photos      -> list of bytes [photo1, photo2, photo3]
#               det_threshold  -> float confidence detection (default 0.5)
#               sim_threshold  -> float min similarity antar foto (default 0.40)
# output : dict {
#   embeddings        : list[list[float]] -> 3 array embedding (tiap array isi 512 float)
#   similarity_scores : list[float]       -> sim foto1 vs foto2, foto1 vs foto3
# }
# error  : ValueError jika wajah tidak terdeteksi atau foto tidak sama orang
def register_face(
    pipeline: InsightFacePipeline,
    photos: list[bytes],
    det_threshold: float = 0.5,
    sim_threshold: float = 0.40,
) -> dict:
    if len(photos) != 3:
        raise ValueError(f"Butuh tepat 3 foto, diterima {len(photos)}")

    embeddings        = []
    similarity_scores = []
    main_embedding    = None

    for i, photo_bytes in enumerate(photos):
        img    = _bytes_to_bgr(photo_bytes)
        result = pipeline.detect_and_embed(img, threshold=det_threshold)

        if result is None:
            raise ValueError(f"Wajah tidak terdeteksi di foto ke-{i + 1}")

        emb = result["embedding"]

        if i == 0:
            # foto pertama jadi referensi
            main_embedding = emb
        else:
            # foto 2 & 3 harus mirip dengan foto pertama
            sim = cosine_similarity(main_embedding, emb)
            similarity_scores.append(round(float(sim), 4))

            if sim < sim_threshold:
                raise ValueError(
                    f"Foto ke-{i + 1} tidak cocok dengan foto pertama "
                    f"(similarity={sim:.3f} < threshold={sim_threshold}). "
                    f"Pastikan ketiga foto adalah orang yang sama."
                )

        embeddings.append(emb.tolist())

    return {
        "embeddings": embeddings,
        "similarity_scores": similarity_scores,
    }


# ========================================================================================
# INFERENCE FACE
# ========================================================================================

# fungsi utama inference wajah dari 1 foto
# input param : pipeline       -> InsightFacePipeline instance
#               photo_bytes    -> bytes raw foto
#               det_threshold  -> float min detection confidence (default 0.5)
# output : dict {
#   embedding   : list[float] -> 512 dimensi
#   gender      : str
#   age         : int
#   bbox        : [x1, y1, x2, y2]
#   score       : float  - detection confidence
# }
# error  : ValueError jika wajah tidak terdeteksi di foto input
def infer_face(
    pipeline: InsightFacePipeline,
    photo_bytes: bytes,
    det_threshold: float = 0.5,
) -> dict:
    img    = _bytes_to_bgr(photo_bytes)
    result = pipeline.detect_and_embed(img, threshold=det_threshold)

    if result is None:
        raise ValueError("Wajah tidak terdeteksi di foto yang diberikan")

    return {
        "embedding":   result["embedding"].tolist(),
        "gender":      result["gender"],
        "age":         result["age"],
        "bbox":        [round(v, 1) for v in result["bbox"]],
        "score":       round(float(result["score"]), 4),
    }
