# Architecture

```text
React Frontend
      |
      | REST
      v
FastAPI
      |
      +--> preprocessing
      |
      +--> PCA engine
      |       +--> mean face
      |       +--> covariance
      |       +--> eigenvectors
      |       +--> eigenfaces
      |       +--> projection
      |
      +--> KNN recognition
      |
      v
Prediction
```
