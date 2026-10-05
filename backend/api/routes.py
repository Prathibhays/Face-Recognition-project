from fastapi import APIRouter, File, HTTPException, UploadFile


def create_router(recognition_service, model_info):
    router = APIRouter(prefix="/api")

    @router.get("/health")
    def health():
        return {"status": "ok", "service": "PCA Face Recognition API"}

    @router.get("/model-info")
    def get_model_info():
        return model_info()

    @router.post("/recognize")
    async def recognize(file: UploadFile = File(...)):
        allowed = {"image/jpeg", "image/png", "image/jpg"}

        if file.content_type not in allowed:
            raise HTTPException(
                status_code=400,
                detail="Please upload a JPG or PNG image.",
            )

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")

        try:
            return {
                "success": True,
                "filename": file.filename,
                "result": recognition_service.recognize(content),
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Recognition failed: {exc}",
            )

    return router
