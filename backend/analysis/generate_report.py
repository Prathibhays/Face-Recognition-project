from pathlib import Path
import csv
import numpy as np

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from algorithms.pca import PCAEngine
from analysis.experiments import component_experiment
from analysis.visualization import (
    save_mean_face,
    save_eigenfaces,
    save_explained_variance,
    save_projection_2d,
    save_confusion_matrix,
    save_component_accuracy,
)
from services.recognition import FaceRecognitionEngine


class SimpleKNN:
    def __init__(self, k=3):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        return self

    def predict(self, X):
        predictions = []
        for row in np.asarray(X):
            distances = np.linalg.norm(self.X - row, axis=1)
            indices = np.argsort(distances)[:self.k]
            labels = self.y[indices]
            unique, counts = np.unique(labels, return_counts=True)
            predictions.append(unique[np.argmax(counts)])
        return np.asarray(predictions)


def main():
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

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

    save_mean_face(
        pca.mean_,
        faces.images.shape[1:],
        results_dir / "mean_face.png",
    )
    save_eigenfaces(
        pca,
        faces.images.shape[1:],
        results_dir / "eigenfaces.png",
    )
    save_explained_variance(
        pca,
        results_dir / "explained_variance.png",
    )

    pca2 = PCAEngine(n_components=2)
    X2 = pca2.fit_transform(X)
    save_projection_2d(
        X2,
        y,
        results_dir / "pca_2d_projection.png",
    )

    engine = FaceRecognitionEngine(pca, n_neighbors=3)
    engine.train(X_train, y_train)
    accuracy, predictions = engine.evaluate(X_test, y_test)

    save_confusion_matrix(
        confusion_matrix(y_test, predictions),
        results_dir / "confusion_matrix.png",
    )

    experiment = component_experiment(
        X_train,
        X_test,
        y_train,
        y_test,
        pca_factory=lambda k: PCAEngine(k),
        classifier_factory=lambda: SimpleKNN(k=3),
    )

    save_component_accuracy(
        experiment,
        results_dir / "accuracy_vs_components.png",
    )

    with open(
        results_dir / "component_experiment.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["components", "accuracy", "time_ms"],
        )
        writer.writeheader()
        writer.writerows(experiment)

    print(f"50-component accuracy: {accuracy * 100:.2f}%")
    print(f"Results: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
