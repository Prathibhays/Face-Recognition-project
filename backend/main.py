from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import create_router
from model_loader import load_model
from services.recognition_service import RecognitionService


app = FastAPI(
    title="PCA Face Recognition API",
    description="Face recognition using PCA, eigenfaces and projections.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine, model_info = load_model()
recognition_service = RecognitionService(engine)

app.include_router(create_router(recognition_service, model_info))


@app.get("/")
def root():
    return {
        "message": "PCA Face Recognition API is running",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
