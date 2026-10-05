import matplotlib.pyplot as plt
import numpy as np


def save_mean_face(mean_face, image_shape, output):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mean_face.reshape(image_shape), cmap="gray")
    ax.set_title("Mean Face")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_eigenfaces(pca, image_shape, output, count=10):
    count = min(count, len(pca.components_))
    cols = 5
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5))
    axes = np.asarray(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i < count:
            ax.imshow(
                pca.components_[i].reshape(image_shape),
                cmap="gray",
            )
            ax.set_title(f"Eigenface {i + 1}")
        ax.axis("off")

    fig.suptitle("Eigenfaces")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_explained_variance(pca, output):
    x = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    cumulative = pca.cumulative_variance_ * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, cumulative, marker="o", markersize=3)
    ax.set_xlabel("Principal Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("Explained Variance")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_projection_2d(X2, y, output):
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        X2[:, 0],
        X2[:, 1],
        c=y,
        cmap="tab20",
        s=20,
        alpha=0.8,
    )
    ax.set_title("2D PCA Projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="Person")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_confusion_matrix(cm, output):
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("Face Recognition Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_component_accuracy(results, output):
    x = [r["components"] for r in results]
    y = [r["accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y, marker="o")
    ax.set_title("Accuracy vs PCA Components")
    ax.set_xlabel("PCA Components")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
