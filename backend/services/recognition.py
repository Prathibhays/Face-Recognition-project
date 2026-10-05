import numpy as np


class FaceRecognitionEngine:
    def __init__(self, pca, n_neighbors=3):
        self.pca = pca
        self.n_neighbors = n_neighbors
        self.X_train_pca = None
        self.y_train = None

    def train(self, X, y):
        self.X_train_pca = self.pca.transform(X)
        self.y_train = np.asarray(y)

    def project(self, face):
        return self.pca.transform(face)

    def nearest_neighbors(self, projected, k=None):
        k = k or self.n_neighbors
        distances = np.linalg.norm(
            self.X_train_pca - projected[0],
            axis=1,
        )
        indices = np.argsort(distances)[:k]
        return [
            {
                "person": int(self.y_train[i]),
                "distance": float(distances[i]),
            }
            for i in indices
        ]

    def predict(self, face):
        projected = self.project(face)
        neighbors = self.nearest_neighbors(projected)
        labels = [item["person"] for item in neighbors]
        unique, counts = np.unique(labels, return_counts=True)
        prediction = int(unique[np.argmax(counts)])

        return {
            "person": prediction,
            "distance": neighbors[0]["distance"],
            "neighbors": neighbors,
            "projection": projected[0],
        }

    def evaluate(self, X, y):
        predictions = [self.predict(face)["person"] for face in X]
        predictions = np.asarray(predictions)
        y = np.asarray(y)
        return float(np.mean(predictions == y)), predictions
