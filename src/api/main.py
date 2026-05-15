from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import io
import numpy as np
from PIL import Image

from src.core.config import CLASS_INFO, CLASSES, PATHS
from src.inference.classifier import WasteClassifier

classifier = None

@asynccontextmanager
async def lifespan(app):
    """Khởi tạo và dọn dẹp tài nguyên khi server start/stop"""
    global classifier
    print("🚀 Khởi động API và load model AI...")
    try:
        classifier = WasteClassifier(PATHS['model_save'])
        print("✅ Model đã sẵn sàng!")
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
    yield
    # Cleanup khi shutdown (nếu cần)
    print("🛑 API đang tắt...")

app = FastAPI(
    title="Waste Classification API",
    description="API cho hệ thống phân loại rác thải bằng AI",
    version="1.0.0",
    lifespan=lifespan
)

# Cấu hình CORS để Web App (Frontend) có thể gọi API mà không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Kiểm tra trạng thái server"""
    return {
        "status": "online",
        "model_loaded": classifier is not None
    }

@app.get("/info")
async def get_info():
    """Lấy danh sách các loại rác và hướng dẫn xử lý"""
    return {
        "classes": CLASSES,
        "details": CLASS_INFO
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Phân loại rác từ hình ảnh tải lên"""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file ảnh (.png, .jpg, .jpeg, .webp)")
    
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng. Vui lòng thử lại sau.")

    try:
        # Đọc dữ liệu ảnh từ request
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Chuyển đổi sang numpy array RGB cho model
        image_np = np.array(image)
        
        # Dự đoán – classifier.preprocess_image() nhận RGB numpy array
        result = classifier.predict(image_np)
        
        return {
            "success": True,
            "filename": file.filename,
            "prediction": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")
