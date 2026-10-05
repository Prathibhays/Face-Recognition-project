import time
import numpy as np


def component_experiment(
    X_train,
    X_test,
    y_train,
    y_test,
    pca_factory,
    classifier_factory,
    components=(5, 10, 20, 30, 40, 50, 75, 100),
):
    results = []
    max_k = min(X_train.shape[0], X_train.shape[1])

    for k in components:
        if k > max_k:
            continue

        start = time.perf_counter()
        pca = pca_factory(k)
        Xtr = pca.fit_transform(X_train)
        Xte = pca.transform(X_test)

        clf = classifier_factory()
        clf.fit(Xtr, y_train)
        pred = clf.predict(Xte)

        elapsed = (time.perf_counter() - start) * 1000
        accuracy = float(np.mean(pred == y_test) * 100)

        results.append({
            "components": int(k),
            "accuracy": round(accuracy, 4),
            "time_ms": round(elapsed, 4),
        })

    return results
