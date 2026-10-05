import numpy as np


def projectData(Xcentered, principalComponents):
    # Project centered data into PCA space

    Xcentered = np.asarray(Xcentered)
    principalComponents = np.asarray(principalComponents)

    if Xcentered.shape[1] != principalComponents.shape[0]:
        raise ValueError(
            "Number of pixels in the data must match "
            "the number of rows in the principal components."
        )

    return Xcentered @ principalComponents


def projectSingleImage(image, meanFace, principalComponents):
    # Project one new face image into PCA space

    image = np.asarray(image).reshape(1, -1)

    centeredImage = image - meanFace

    return centeredImage @ principalComponents
