import numpy as np
from algorithms.pca import PCAEngine


def test_pca_shapes():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 16))

    pca = PCAEngine(5)
    Z = pca.fit_transform(X)

    assert Z.shape == (20, 5)
    assert pca.components_.shape == (5, 16)
    assert len(pca.eigenvalues_) == 5
