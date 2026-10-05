import numpy as np
from algorithms.pca import PCAEngine
from services.recognition import FaceRecognitionEngine


def test_recognition():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 20))
    y = np.repeat(np.arange(3), 10)

    pca = PCAEngine(5)
    pca.fit(X)

    engine = FaceRecognitionEngine(pca, 3)
    engine.train(X, y)
    result = engine.predict(X[0])

    assert "person" in result
    assert "distance" in result
    assert len(result["neighbors"]) == 3
