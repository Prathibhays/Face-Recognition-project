import numpy as np
import matplotlib.pyplot as plt

from pca import fitPCA
from eigenfaces import createEigenfaces, normalizeEigenfaces
from projection import projectData


def showMeanFace(meanFace, imageHeight, imageWidth):
    """Display the mean face."""

    meanFaceImage = meanFace.reshape(
        imageHeight,
        imageWidth
    )

    plt.figure()
    plt.imshow(meanFaceImage, cmap="gray")
    plt.title("Mean Face")
    plt.axis("off")
    plt.show()


def showEigenfaces(eigenfaces):
    """Display the first 9 eigenfaces."""

    numberOfFaces = min(9, len(eigenfaces))

    plt.figure(figsize=(10, 10))

    for i in range(numberOfFaces):

        plt.subplot(3, 3, i + 1)

        plt.imshow(
            eigenfaces[i],
            cmap="gray"
        )

        plt.title(f"Eigenface {i + 1}")
        plt.axis("off")

    plt.suptitle("Eigenfaces")
    plt.tight_layout()
    plt.show()


def showExplainedVariance(explainedVariance):
    """Display explained variance."""

    components = np.arange(
        1,
        len(explainedVariance) + 1
    )

    plt.figure()

    plt.plot(
        components,
        explainedVariance
    )

    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance")
    plt.title("Explained Variance")
    plt.grid()

    plt.show()


def showCumulativeVariance(explainedVariance):
    """Display cumulative explained variance."""

    cumulativeVariance = np.cumsum(
        explainedVariance
    )

    components = np.arange(
        1,
        len(cumulativeVariance) + 1
    )

    plt.figure()

    plt.plot(
        components,
        cumulativeVariance
    )

    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance")
    plt.grid()

    plt.show()


def showPCAProjection(projectedData):
    """Display PCA projection using PC1 and PC2."""

    if projectedData.shape[1] < 2:
        raise ValueError(
            "At least 2 principal components are required."
        )

    plt.figure()

    plt.scatter(
        projectedData[:, 0],
        projectedData[:, 1]
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Projection")

    plt.grid()
    plt.show()


def runVisualization(
    images,
    imageHeight,
    imageWidth,
    numberOfComponents=50
):
    """
    Run Person 2's PCA code and create
    Person 4's visualizations.
    """

    # Run PCA
    result = fitPCA(
        images,
        numberOfComponents
    )

    # Get PCA results
    meanFace = result["meanFace"]
    principalComponents = result["principalComponents"]
    explainedVariance = result["explainedVariance"]
    centeredData = result["centeredData"]

    # Convert principal components into eigenfaces
    eigenfaces = createEigenfaces(
        principalComponents,
        imageHeight,
        imageWidth
    )

    # Normalize eigenfaces for display
    eigenfaces = normalizeEigenfaces(
        eigenfaces
    )

    # Project images into PCA space
    projectedData = projectData(
        centeredData,
        principalComponents
    )

    # Display results
    showMeanFace(
        meanFace,
        imageHeight,
        imageWidth
    )

    showEigenfaces(
        eigenfaces
    )

    showExplainedVariance(
        explainedVariance
    )

    showCumulativeVariance(
        explainedVariance
    )

    showPCAProjection(
        projectedData
    )
