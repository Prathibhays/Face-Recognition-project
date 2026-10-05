import numpy as np


def flattenImages(images):
    # Convert images into rows of pixel values

    images = np.asarray(images)

    return images.reshape(images.shape[0], -1)


def computeMeanFace(X):
    # Calculate the mean value of each pixel

    return np.mean(X, axis=0)


def centerData(X, meanFace):
    # Subtract the mean face from each image

    return X - meanFace


def computeCovariance(Xcentered):
    # Compute the compact covariance matrix

    numberOfImages = Xcentered.shape[0]

    if numberOfImages < 2:
        raise ValueError("At least two images are required.")

    return (Xcentered @ Xcentered.T) / (numberOfImages - 1)


def computeEigen(covarianceMatrix):
    # Calculate eigenvalues and eigenvectors

    eigenvalues, eigenvectors = np.linalg.eigh(covarianceMatrix)

    # Sort from largest eigenvalue to smallest
    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return eigenvalues, eigenvectors


def computePrincipalComponents(
    Xcentered,
    eigenvalues,
    eigenvectors,
    numberOfComponents
):
    # Convert compact eigenvectors into principal components

    numberOfImages = Xcentered.shape[0]

    numberOfComponents = min(
        numberOfComponents,
        numberOfImages
    )

    selectedEigenvalues = eigenvalues[:numberOfComponents]
    selectedEigenvectors = eigenvectors[:, :numberOfComponents]

    components = []

    for i in range(numberOfComponents):

        if selectedEigenvalues[i] > 1e-12:

            component = (
                Xcentered.T @ selectedEigenvectors[:, i]
            ) / np.sqrt(
                selectedEigenvalues[i]
                * (numberOfImages - 1)
            )

            components.append(component)

    if len(components) == 0:
        raise ValueError("No valid principal components found.")

    return np.column_stack(components)


def computeExplainedVariance(eigenvalues):
    # Calculate the fraction of total variance explained

    totalVariance = np.sum(eigenvalues)

    if totalVariance <= 0:
        raise ValueError("Total variance must be positive.")

    return eigenvalues / totalVariance


def fitPCA(X, numberOfComponents):
    # Run the complete PCA pipeline

    X = flattenImages(X)

    meanFace = computeMeanFace(X)

    Xcentered = centerData(
        X,
        meanFace
    )

    covarianceMatrix = computeCovariance(
        Xcentered
    )

    eigenvalues, eigenvectors = computeEigen(
        covarianceMatrix
    )

    principalComponents = computePrincipalComponents(
        Xcentered,
        eigenvalues,
        eigenvectors,
        numberOfComponents
    )

    explainedVariance = computeExplainedVariance(
        eigenvalues
    )

    return {
        "meanFace": meanFace,
        "centeredData": Xcentered,
        "covarianceMatrix": covarianceMatrix,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "principalComponents": principalComponents,
        "explainedVariance": explainedVariance
    }
