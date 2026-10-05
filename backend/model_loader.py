from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

from algorithms.pca import PCAEngine
from services.recognition import FaceRecognitionEngine


def load_model():
    print("Loading Olivetti Faces dataset...")
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)

    X, y = faces.data, faces.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pca = PCAEngine(n_components=50)
    pca.fit(X_train)

    engine = FaceRecognitionEngine(pca, n_neighbors=3)
    engine.train(X_train, y_train)

    accuracy, _ = engine.evaluate(X_test, y_test)

    print(f"Validation accuracy: {accuracy * 100:.2f}%")

    def info():
        return {
            "dataset": "Olivetti Faces",
            "image_size": "64x64",
            "original_features": int(X.shape[1]),
            "training_samples": int(len(X_train)),
            "testing_samples": int(len(X_test)),
            "pca_components": int(len(pca.components_)),
            "validation_accuracy": round(accuracy * 100, 2),
            "explained_variance": round(
                float(pca.cumulative_variance_[-1] * 100), 2
            ),
        }

    return engine, info
