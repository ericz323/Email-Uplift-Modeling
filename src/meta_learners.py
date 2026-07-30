import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _default_outcome_estimator(random_state, n_jobs):
    # Stage 1 fits one model per arm, so each sees ~half the rows. Depth is
    # capped and leaves kept large because these predictions become the
    # baseline that stage 2 subtracts -- overfitting here leaks straight into D.
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=50,
        max_features="sqrt",
        n_jobs=n_jobs,
        random_state=random_state,
    )


def _default_effect_estimator(random_state, n_jobs):
    # D is a noisy pseudo-outcome (roughly +/-1 residuals around a small true
    # effect), so the effect models are regularized harder than the outcome
    # models -- shallower, with leaves big enough to average the noise out.
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=100,
        max_features="sqrt",
        n_jobs=n_jobs,
        random_state=random_state,
    )


def _default_propensity_estimator(random_state):
    # Features are on wildly different scales (history in dollars vs 0/1
    # dummies), so logistic regression needs scaling to converge sensibly.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=random_state),
    )


def fit_x_learner(
        X_train, treatment_train, y_train,
        outcome_estimator=None, effect_estimator=None, propensity_estimator=None,
        random_state=42, n_jobs=1
):
    """Fit the four X-learner stages.

    Returns the fitted sub-models rather than just the final scorer, so the
    pieces stay available for diagnostics -- in particular the propensity model,
    whose predicted distribution is the overlap check.
    """
    if outcome_estimator is None:
        outcome_estimator = _default_outcome_estimator(random_state, n_jobs)
    if effect_estimator is None:
        effect_estimator = _default_effect_estimator(random_state, n_jobs)
    if propensity_estimator is None:
        propensity_estimator = _default_propensity_estimator(random_state)

    treatment_train = np.asarray(treatment_train)
    y_train = np.asarray(y_train)
    is_treated = treatment_train == 1

    X_train_t = X_train[is_treated]
    X_train_c = X_train[~is_treated]
    y_train_t = y_train[is_treated]
    y_train_c = y_train[~is_treated]

    # Stage 1 -- outcome model per arm
    mu1 = clone(outcome_estimator).fit(X_train_t, y_train_t)
    mu0 = clone(outcome_estimator).fit(X_train_c, y_train_c)

    # Stage 2 -- impute individual effects, each arm scored by the OTHER arm's
    # model. Both differences are oriented treated-minus-control.
    d1 = y_train_t - mu0.predict_proba(X_train_t)[:, 1]
    d0 = mu1.predict_proba(X_train_c)[:, 1] - y_train_c

    # Stage 3 -- effect models on the imputed D
    tau1 = clone(effect_estimator).fit(X_train_t, d1)
    tau0 = clone(effect_estimator).fit(X_train_c, d0)

    # Stage 4 -- propensity, i.e. P(treated | x), NOT the outcome
    g = clone(propensity_estimator).fit(X_train, treatment_train)

    return {
        "mu1": mu1,
        "mu0": mu0,
        "tau1": tau1,
        "tau0": tau0,
        "propensity_model": g,
        "d1": d1,
        "d0": d0,
    }


def predict_x_learner_uplift(fitted_x_learner, X):
    tau1 = fitted_x_learner["tau1"]
    tau0 = fitted_x_learner["tau0"]
    g = fitted_x_learner["propensity_model"]

    # tau1/tau0 are regressors on D -- they predict an effect directly, so
    # there is no probability to extract
    tau1_pred = tau1.predict(X)
    tau0_pred = tau0.predict(X)
    propensity = g.predict_proba(X)[:, 1]

    uplift = propensity * tau0_pred + (1 - propensity) * tau1_pred

    return uplift


def run_x_learner(X_train, treatment_train, y_train, X_test, **kwargs):
    fitted_x_learner = fit_x_learner(X_train, treatment_train, y_train, **kwargs)

    return predict_x_learner_uplift(fitted_x_learner, X_test)
