import time
import csv
import numpy as np
import matplotlib.pyplot as plt

from pca import fitPCA
from recognition import FaceRecognitionEngine


# PCA component values for the experiment
COMPONENTS = [10, 20, 30, 40, 50, 75, 100]


class PCAAdapter:
    """
    Adapter that connects Person 2's PCA code
    with Person 3's FaceRecognitionEngine.

    Person 2:
        fitPCA()

    Person 3 expects:
        pca.transform()
    """

    def __init__(self, X_train, numberOfComponents):

        self.result = fitPCA(
            X_train,
            numberOfComponents
        )

        self.meanFace = self.result["meanFace"]

        self.principalComponents = (
            self.result["principalComponents"]
        )

    def transform(self, X):

        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Center the data using Person 2's mean face
        Xcentered = X - self.meanFace

        # Project into PCA space
        projected = (
            Xcentered @ self.principalComponents
        )

        return projected


def runExperiments(
    X_train,
    y_train,
    X_test,
    y_test
):

    results = []

    for numberOfComponents in COMPONENTS:

        print("\n" + "=" * 50)

        print(
            f"Running experiment with "
            f"{numberOfComponents} PCA components"
        )

        print("=" * 50)

        startTime = time.time()

        # -----------------------------------------
        # 1. Create PCA model
        #    Uses Person 2's fitPCA()
        # -----------------------------------------

        pcaModel = PCAAdapter(
            X_train,
            numberOfComponents
        )

        # -----------------------------------------
        # 2. Create recognition engine
        #    Uses Person 3's recognition.py
        # -----------------------------------------

        recognitionEngine = FaceRecognitionEngine(
            pcaModel,
            n_neighbors=3
        )

        # -----------------------------------------
        # 3. Train recognition model
        # -----------------------------------------

        recognitionEngine.train(
            X_train,
            y_train
        )

        # -----------------------------------------
        # 4. Evaluate on test data
        # -----------------------------------------

        evaluation = recognitionEngine.evaluate(
            X_test,
            y_test
        )

        # -----------------------------------------
        # 5. Get accuracy
        # -----------------------------------------

        accuracy = evaluation[
            "accuracy_percentage"
        ]

        # -----------------------------------------
        # 6. Calculate execution time
        # -----------------------------------------

        executionTime = (
            time.time() - startTime
        )

        # -----------------------------------------
        # 7. Store experiment results
        # -----------------------------------------

        results.append({
            "components": numberOfComponents,
            "accuracy": accuracy,
            "execution_time": executionTime,
            "dimensionality": numberOfComponents
        })

        print(
            f"Components       : {numberOfComponents}"
        )

        print(
            f"Accuracy         : {accuracy:.2f}%"
        )

        print(
            f"Execution time   : {executionTime:.4f} seconds"
        )

    return results


def saveResults(
    results,
    filename="accuracy_results.csv"
):

    with open(
        filename,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "components",
                "accuracy",
                "execution_time",
                "dimensionality"
            ]
        )

        writer.writeheader()

        writer.writerows(results)

    print(
        f"\nResults saved to {filename}"
    )


def plotAccuracy(results):

    components = [
        result["components"]
        for result in results
    ]

    accuracy = [
        result["accuracy"]
        for result in results
    ]

    plt.figure()

    plt.plot(
        components,
        accuracy,
        marker="o"
    )

    plt.xlabel(
        "Number of PCA Components"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        "Accuracy vs Number of PCA Components"
    )

    plt.grid()

    plt.show()


def plotExecutionTime(results):

    components = [
        result["components"]
        for result in results
    ]

    executionTime = [
        result["execution_time"]
        for result in results
    ]

    plt.figure()

    plt.plot(
        components,
        executionTime,
        marker="o"
    )

    plt.xlabel(
        "Number of PCA Components"
    )

    plt.ylabel(
        "Execution Time (seconds)"
    )

    plt.title(
        "Execution Time vs Number of PCA Components"
    )

    plt.grid()

    plt.show()
