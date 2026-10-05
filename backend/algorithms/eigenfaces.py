import numpy as np


def createEigenfaces(principalComponents, imageHeight, imageWidth):
    # Convert principal components into image-shaped eigenfaces

    numberOfPixels = imageHeight * imageWidth

    if principalComponents.shape[0] != numberOfPixels:
        raise ValueError("Image dimensions do not match.")

    numberOfComponents = principalComponents.shape[1]

    eigenfaces = principalComponents.T.reshape(
        numberOfComponents,
        imageHeight,
        imageWidth
    )

    return eigenfaces


def normalizeEigenfaces(eigenfaces):
    # Normalize eigenfaces to values between 0 and 1

    normalizedFaces = []

    for face in eigenfaces:
        minimum = np.min(face)
        maximum = np.max(face)

        if maximum == minimum:
            normalizedFace = np.zeros_like(face)
        else:
            normalizedFace = (
                (face - minimum) /
                (maximum - minimum)
            )

        normalizedFaces.append(normalizedFace)

    return np.array(normalizedFaces)
