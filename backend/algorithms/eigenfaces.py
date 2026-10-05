import numpy as np


def get_eigenfaces(pca, image_shape=(64, 64), count=10):
    count = min(count, len(pca.components_))
    return [
        pca.components_[i].reshape(image_shape)
        for i in range(count)
    ]


def normalize_for_display(image):
    image = np.asarray(image, dtype=np.float64)
    minimum, maximum = image.min(), image.max()
    if maximum - minimum < 1e-12:
        return np.zeros_like(image)
    return (image - minimum) / (maximum - minimum)
