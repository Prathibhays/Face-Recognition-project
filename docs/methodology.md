# Methodology

1. Load labeled face images.
2. Convert images to grayscale.
3. Resize to 64x64.
4. Flatten images into vectors.
5. Calculate the mean face.
6. Center the data.
7. Calculate covariance.
8. Find eigenvalues and eigenvectors.
9. Sort eigenvectors by descending eigenvalue.
10. Select principal components.
11. Project faces into PCA space.
12. Compare projected faces using Euclidean distance.
13. Use KNN to recognize identity.
14. Evaluate on a held-out test set.
15. Visualize eigenfaces, variance, projection and confusion matrix.
