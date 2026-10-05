from .preprocessing import preprocess_uploaded_image


class RecognitionService:
    def __init__(self, engine, threshold=None):
        self.engine = engine
        self.threshold = threshold

    def recognize(self, image_bytes):
        face_vector, detected = preprocess_uploaded_image(image_bytes)
        result = self.engine.predict(face_vector)

        distance = result["distance"]
        similarity = 100.0 / (1.0 + distance)
        person = result["person"]

        if self.threshold is not None and distance > self.threshold:
            person = "Unknown"

        return {
            "person": person,
            "distance": round(float(distance), 4),
            "similarity": round(float(similarity), 2),
            "face_detected": detected,
            "neighbors": result["neighbors"],
            "components": len(result["projection"]),
        }
