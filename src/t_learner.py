from sklearn.ensemble import RandomForestClassifier


def fit_t_learner(
        X_train, t_train, y_train,
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=50,
        random_state=42,
        **rf_kwargs
):
    rf_treatment = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        **rf_kwargs)
    rf_control = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        **rf_kwargs)

    X_train_treatment = X_train[t_train == 1]
    X_train_control = X_train[t_train == 0]
    y_train_treatment = y_train[t_train == 1]
    y_train_control = y_train[t_train == 0]

    rf_treatment.fit(X_train_treatment, y_train_treatment)
    rf_control.fit(X_train_control, y_train_control)

    return rf_treatment, rf_control


def predict_uplift(model_treatment, model_control, X):
    return model_treatment.predict_proba(X)[:, 1] - model_control.predict_proba(X)[:, 1]


def run_two_model(X_train, t_train, y_train, X_test, **rf_kwargs):
    model_treatment, model_control = fit_t_learner(X_train, t_train, y_train, **rf_kwargs)

    return predict_uplift(model_treatment, model_control, X_test)