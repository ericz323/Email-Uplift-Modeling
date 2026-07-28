import numpy as np
from causalml.inference.tree import UpliftRandomForestClassifier


def train_uplift_rf(
        X_train, t_train, y_train, X_test,
        n_estimators=100,
        max_depth=5,
        evaluationFunction="KL",
        control_name="control",
        random_state=42,
        n_jobs=1,
        **kwargs
) -> np.ndarray:
    t_train_str = np.where(np.asarray(t_train) == 1, "treatment", control_name)
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train)

    uplift_rf = UpliftRandomForestClassifier(
        control_name=control_name,
        n_estimators=n_estimators,
        max_depth=max_depth,
        evaluationFunction=evaluationFunction,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )

    uplift_rf.fit(X_train, t_train_str, y_train)

    preds = uplift_rf.predict(X_test)

    treatment_groups = list(uplift_rf.classes_)[1:]
    uplift_scores = preds[:, treatment_groups.index("treatment")]

    return uplift_scores