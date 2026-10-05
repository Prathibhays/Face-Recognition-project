import numpy as np


def project_face(face_vector, pca):
    vector = np.asarray(face_vector, dtype=np.float64)
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    return pca.transform(vector)


def reconstruct_face(projected_vector, pca):
    return pca.inverse_transform(projected_vector)
