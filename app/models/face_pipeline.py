# ml-service/app/models/face_pipeline.py
#
# -> wrapper InsightFace pipeline menggunakan ONNX Runtime
#    -> load 3 model: det_10g (detector), w600k_r50 (recognizer), genderage (opsional)
#    -> expose InsightFacePipeline sebagai singleton yang di-load sekali saat startup
# -> refactor dari ml-models/inefence-example.py

import numpy as np
import onnxruntime as ort
import cv2
from pathlib import Path


# ========================================================================================
# MATH HELPERS
# ========================================================================================

# helper ---------------------------------------------------------------------------------

# fungsi buat L2 normalize vektor embedding
# input  : vec -> np.ndarray float
# output : np.ndarray float dengan magnitude = 1.0
# math   : v_norm = v / ||v||_2
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


# fungsi buat hitung cosine similarity antara 2 embedding
# input  : a, b -> np.ndarray float (512-dim)
# output : float di range [0.0, 1.0]
# math   : raw_cosine = dot(a, b) / (||a|| * ||b||)
#          scale ke [0, 1] : sim = (raw_cosine + 1) / 2
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = l2_normalize(a)
    b_norm = l2_normalize(b)
    raw = float(np.dot(a_norm, b_norm))
    return (raw + 1.0) / 2.0


# dimensi embedding output w600k_r50.onnx - harus 512
EMBEDDING_DIM = 512

# end of helper --------------------------------------------------------------------------


# ========================================================================================
# FACE DETECTOR - det_10g.onnx (InsightFace RetinaFace multi-scale)
# ========================================================================================

STRIDES     = [8, 16, 32]
NUM_ANCHORS = 2
NMS_THRESHOLD = 0.4


# helper ---------------------------------------------------------------------------------

# fungsi generate anchor centers berdasarkan feature map size dan stride
# input  : height, width -> feature map dimensions
#          stride        -> downsampling factor (8, 16, atau 32)
# output : array shape (H*W*num_anchors, 2) berisi [cx, cy]
def _generate_anchors(height: int, width: int, stride: int) -> np.ndarray:
    cx = np.arange(width) * stride + stride // 2
    cy = np.arange(height) * stride + stride // 2
    grid_x, grid_y = np.meshgrid(cx, cy)
    centers = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    centers = np.repeat(centers, NUM_ANCHORS, axis=0)
    return centers.astype(np.float32)


# fungsi decode raw bbox ke absolute pixel coords
# input  : pred_bbox -> raw output (N, 4)
#          anchors   -> anchor centers (N, 2)
#          stride    -> scaling factor
# output : array (N, 4) berisi [x1, y1, x2, y2]
def _decode_boxes(pred_bbox: np.ndarray, anchors: np.ndarray, stride: int) -> np.ndarray:
    x1 = anchors[:, 0] - pred_bbox[:, 0] * stride
    y1 = anchors[:, 1] - pred_bbox[:, 1] * stride
    x2 = anchors[:, 0] + pred_bbox[:, 2] * stride
    y2 = anchors[:, 1] + pred_bbox[:, 3] * stride
    return np.stack([x1, y1, x2, y2], axis=1)


# fungsi decode 5 keypoints dari raw output
# input  : pred_kps -> raw output (N, 10)
#          anchors  -> anchor centers (N, 2)
#          stride   -> scaling factor
# output : array (N, 5, 2) berisi koordinat pixel [x, y]
def _decode_keypoints(pred_kps: np.ndarray, anchors: np.ndarray, stride: int) -> np.ndarray:
    kps = pred_kps.reshape(-1, 5, 2)
    kps[:, :, 0] = anchors[:, 0:1] + kps[:, :, 0] * stride
    kps[:, :, 1] = anchors[:, 1:2] + kps[:, :, 1] * stride
    return kps


# fungsi non-maximum suppression
# input  : boxes         -> (N, 4) [x1, y1, x2, y2]
#          scores        -> (N,)
#          iou_threshold -> float
# output : array index yang survive NMS
def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.4) -> np.ndarray:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_threshold]

    return np.array(keep, dtype=np.int32)

# end of helper --------------------------------------------------------------------------


class FaceDetector:
    """
    wrapper det_10g.onnx
    -> multi-scale RetinaFace (stride 8, 16, 32)
    -> input  : image BGR float32, shape (1, 3, H, W)
    -> output : list of dict { bbox, score, kps }
    """

    # buat inisialisasi model dari path file .onnx
    # input param : model_path -> str path ke det_10g.onnx
    def __init__(self, model_path: str):
        self.session    = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = (640, 640)

    # fungsi deteksi wajah dari image BGR
    # input  : img       -> np.ndarray BGR shape (H, W, 3)
    #          threshold -> min confidence (default 0.5)
    # output : list of { bbox: [x1,y1,x2,y2], score: float, kps: (5,2) } sorted desc
    def detect(self, img: np.ndarray, threshold: float = 0.5) -> list[dict]:
        img_h, img_w = img.shape[:2]
        det_w, det_h = self.input_size

        scale  = min(det_w / img_w, det_h / img_h)
        new_w  = int(img_w * scale)
        new_h  = int(img_h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        canvas = np.zeros((det_h, det_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        # normalize: (pixel - 127.5) / 128.0
        blob = (canvas.astype(np.float32) - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, :]

        outputs  = self.session.run(None, {self.input_name: blob})
        out_names = [o.name for o in self.session.get_outputs()]
        out_map  = {name: val for name, val in zip(out_names, outputs)}

        all_boxes, all_scores, all_kps = [], [], []

        for stride in STRIDES:
            scores_raw = out_map[f"score_{stride}"].reshape(-1)
            scores     = 1.0 / (1.0 + np.exp(-scores_raw))  # sigmoid
            keep_idx   = np.where(scores >= threshold)[0]

            if len(keep_idx) == 0:
                continue

            scores  = scores[keep_idx]
            feat_h  = det_h // stride
            feat_w  = det_w // stride
            anchors = _generate_anchors(feat_h, feat_w, stride).reshape(-1, 2)[keep_idx]

            bbox_raw = out_map[f"bbox_{stride}"].reshape(-1, 4)[keep_idx]
            boxes    = _decode_boxes(bbox_raw, anchors, stride)

            kps_raw = out_map[f"kps_{stride}"].reshape(-1, 10)[keep_idx]
            kps     = _decode_keypoints(kps_raw, anchors, stride)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_kps.append(kps)

        if not all_boxes:
            return []

        all_boxes  = np.concatenate(all_boxes,  axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_kps    = np.concatenate(all_kps,    axis=0)

        keep       = _nms(all_boxes, all_scores, NMS_THRESHOLD)
        all_boxes  = all_boxes[keep]
        all_scores = all_scores[keep]
        all_kps    = all_kps[keep]

        inv_scale  = 1.0 / scale
        all_boxes *= inv_scale
        all_kps   *= inv_scale

        results = [
            {"bbox": all_boxes[i].tolist(), "score": float(all_scores[i]), "kps": all_kps[i]}
            for i in range(len(all_boxes))
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


# ========================================================================================
# FACE ALIGNER - affine transform ke 112x112 (ArcFace standard)
# ========================================================================================

# 5-point reference landmarks ArcFace (InsightFace standard)
# urutan: left_eye, right_eye, nose, mouth_left, mouth_right
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


# fungsi hitung affine matrix dari 5 keypoints ke reference landmarks
# input  : src_pts -> array (5, 2) keypoints dari detector
# output : 2x3 affine matrix untuk cv2.warpAffine
def _estimate_affine(src_pts: np.ndarray) -> np.ndarray:
    tform, _ = cv2.estimateAffinePartial2D(
        src_pts.astype(np.float32),
        ARCFACE_DST,
        method=cv2.LMEDS,
    )
    return tform


# fungsi align dan crop wajah ke 112x112
# input  : img -> np.ndarray BGR original
#          kps -> array (5, 2) keypoints dari detector
# output : np.ndarray BGR shape (112, 112, 3)
def align_face(img: np.ndarray, kps: np.ndarray) -> np.ndarray:
    M       = _estimate_affine(kps)
    aligned = cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR)
    return aligned


# ========================================================================================
# FACE RECOGNIZER - w600k_r50.onnx (ArcFace ResNet50)
# ========================================================================================

class FaceRecognizer:
    """
    wrapper w600k_r50.onnx
    -> ArcFace ResNet50 trained on WebFace600K
    -> input  : aligned face BGR float32, shape (1, 3, 112, 112)
    -> output : 512-dim embedding float32 (L2 normalized)
    """

    # buat inisialisasi model dari path file .onnx
    # input param : model_path -> str path ke w600k_r50.onnx
    def __init__(self, model_path: str):
        self.session    = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    # fungsi extract 512-dim embedding dari aligned face
    # input  : face_bgr -> np.ndarray BGR shape (112, 112, 3)
    # output : np.ndarray float32 shape (512,) sudah L2 normalized
    def get_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        # normalize: (pixel - 127.5) / 127.5 (berbeda dengan detector yang pakai / 128.0)
        blob   = (face_bgr.astype(np.float32) - 127.5) / 127.5
        blob   = blob.transpose(2, 0, 1)[np.newaxis, :]
        output = self.session.run(None, {self.input_name: blob})
        return l2_normalize(output[0].flatten())


# ========================================================================================
# GENDER & AGE ESTIMATOR - genderage.onnx (opsional)
# ========================================================================================

class GenderAgeEstimator:
    """
    wrapper genderage.onnx
    -> input  : aligned face BGR, shape (1, 3, 96, 96)
    -> output : gender (male/female) + age (int)
    """

    # buat inisialisasi model dari path file .onnx
    # input param : model_path -> str path ke genderage.onnx
    def __init__(self, model_path: str):
        self.session    = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    # fungsi estimate gender dan age dari aligned face
    # input  : face_bgr -> np.ndarray BGR shape (112, 112, 3)
    # output : dict { gender: str, age: int }
    def predict(self, face_bgr: np.ndarray) -> dict:
        resized = cv2.resize(face_bgr, (96, 96))
        blob    = (resized.astype(np.float32) - 127.5) / 127.5
        blob    = blob.transpose(2, 0, 1)[np.newaxis, :]
        output  = self.session.run(None, {self.input_name: blob})[0].flatten()
        gender  = "male" if float(output[0]) > 0.5 else "female"
        age     = int(np.round(output[2] * 100))
        return {"gender": gender, "age": age}


# ========================================================================================
# PIPELINE LENGKAP: detect + align + embed + gender/age
# ========================================================================================

class InsightFacePipeline:
    """
    high-level pipeline: detector + recognizer + gender-age
    -> singleton - load sekali saat startup FastAPI
    -> replicates cermin-new pipeline behavior
    """

    # buat inisialisasi semua model dari folder
    # input param : model_dir -> str path ke folder .onnx files
    def __init__(self, model_dir: str):
        model_dir       = Path(model_dir)
        self.detector   = FaceDetector(str(model_dir / "det_10g.onnx"))
        self.recognizer = FaceRecognizer(str(model_dir / "w600k_r50.onnx"))
        self.gender_age = GenderAgeEstimator(str(model_dir / "genderage.onnx"))
        print(f"[InsightFace] loaded dari {model_dir}")

    # fungsi detect wajah + extract embedding dari 1 image
    # input  : img_bgr   -> np.ndarray BGR dari cv2.imread / cv2.imdecode
    #          threshold -> min detection confidence (default 0.5)
    # output : dict atau None jika tidak ada wajah
    #   {
    #     embedding : np.ndarray(512,)   - L2-normalized
    #     bbox      : [x1, y1, x2, y2]  - pixel coords
    #     score     : float              - detection confidence
    #     gender    : str                - male / female
    #     age       : int                - estimated age
    #   }
    def detect_and_embed(self, img_bgr: np.ndarray, threshold: float = 0.5) -> dict | None:
        detections = self.detector.detect(img_bgr, threshold)
        if not detections:
            return None

        best    = detections[0]
        aligned = align_face(img_bgr, best["kps"])
        embedding = self.recognizer.get_embedding(aligned)
        ga      = self.gender_age.predict(aligned)

        return {
            "embedding": embedding,
            "bbox":      best["bbox"],
            "score":     best["score"],
            "gender":    ga["gender"],
            "age":       ga["age"],
        }

    # fungsi cari best match dari list embedding database
    # input  : query_emb    -> np.ndarray(512,) embedding query
    #          db_entries   -> list of { person_name, user_id, embedding: list[float] }
    #          threshold    -> min cosine similarity (default 0.40)
    # output : dict { person_name, user_id, similarity, matched }
    def match_face(
        self,
        query_emb: np.ndarray,
        db_entries: list[dict],
        threshold: float = 0.40,
    ) -> dict:
        if not db_entries:
            return {"person_name": "unknown", "user_id": None, "similarity": -1.0, "matched": False}

        best_sim  = -1.0
        best_name = "unknown"
        best_uid  = None
        skipped   = 0

        for entry in db_entries:
            emb = np.array(entry["embedding"], dtype=np.float32)
            if emb.size == 0 or emb.shape[0] != EMBEDDING_DIM:
                skipped += 1
                continue

            sim = cosine_similarity(query_emb, emb)
            if sim > best_sim:
                best_sim  = sim
                best_name = entry["person_name"]
                best_uid  = entry.get("user_id")

        if skipped > 0:
            print(f"  [match_face] skip {skipped} entry (bukan {EMBEDDING_DIM}-dim)")

        matched = best_sim >= threshold
        return {
            "person_name": best_name if matched else "unknown",
            "user_id":     best_uid  if matched else None,
            "similarity":  best_sim,
            "matched":     matched,
        }
