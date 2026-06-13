import cv2
from app.models.face_pipeline import InsightFacePipeline

def test():
    # Model ada di public URL, download manual untuk testing
    import urllib.request
    import os
    os.makedirs('models', exist_ok=True)
    models = ['det_10g.onnx', 'w600k_r50.onnx', 'genderage.onnx']
    for m in models:
        if not os.path.exists(f'models/{m}'):
            print(f"Downloading {m}...")
            urllib.request.urlretrieve(f"https://storage.googleapis.com/ml-models-iot/{m}", f'models/{m}')

    print("Loading pipeline...")
    pipeline = InsightFacePipeline('models')
    
    print("Reading image...")
    img = cv2.imread('../smart-room-access-backend/face1.jpg')
    if img is None:
        print("Image not found!")
        return

    print("Testing pipeline...")
    res = pipeline.detect_and_embed(img)
    if res:
        print(f"Success! Detected face. Bbox: {res['bbox']}, Score: {res['score']}, Gender: {res['gender']}, Age: {res['age']}")
        print(f"Embedding size: {len(res['embedding'])}")
    else:
        print("No face detected.")

if __name__ == "__main__":
    test()
