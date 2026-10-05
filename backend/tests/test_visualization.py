import numpy as np
from algorithms.pca import PCAEngine
from analysis.visualization import save_mean_face, save_eigenfaces


def test_visualizations(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.random((20, 64))

    pca = PCAEngine(5)
    pca.fit(X)

    mean_path = tmp_path / "mean.png"
    eigen_path = tmp_path / "eigen.png"

    save_mean_face(pca.mean_, (8, 8), mean_path)
    save_eigenfaces(pca, (8, 8), eigen_path, count=5)

    assert mean_path.exists()
    assert eigen_path.exists()
