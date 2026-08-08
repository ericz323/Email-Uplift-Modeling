import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

from src.data_prep import load_and_prepare
from src.t_learner import run_two_model
from src.uplift_trees import train_uplift_rf
from src.meta_learners import run_x_learner
from src.evaluation import evaluate_model, plot_qini_curve, plot_uplift_by_decile

def _run_one(split_seed, model_seed, model_type, arm, outcome, normalize=None):
    try:
        X_train, X_test, y_train, y_test, treatment_train, treatment_test = load_and_prepare(treatment_arm=arm, random_state=split_seed)

        if model_type == "T-Learner":
            uplift_scores = run_two_model(
                X_train, treatment_train, y_train[outcome], X_test, random_state=model_seed, n_jobs=1
            )
        elif model_type == "Uplift Trees":
            uplift_scores = train_uplift_rf(
                X_train, treatment_train, y_train[outcome], X_test,
                random_state=model_seed, max_depth=8, n_estimators=300, evaluationFunction="KL", n_jobs=1
            )
        elif model_type == "X-Learner":
            uplift_scores = run_x_learner(
                X_train, treatment_train, y_train[outcome], X_test, random_state=model_seed, n_jobs=1
            )
        else:
            raise ValueError(f"unknown model_type {model_type!r}")

        results = evaluate_model(model_type, uplift_scores, treatment_test, y_test[outcome], normalize=normalize)

        return {
            "split_seed": split_seed,
            "model_seed": model_seed,
            "model": model_type,
            "arm": arm,
            "outcome": outcome,
            "error": None,
            **results}
    except Exception as e:
        return {
            "split_seed": split_seed,
            "model_seed": model_seed,
            "model": model_type,
            "arm": arm,
            "outcome": outcome,
            "error": f"{type(e).__name__}: {e}",
        }


def run_stability_check(
        n_splits=30,
        n_model_seeds=2,
        models=("T-Learner", "Uplift Trees", "X-Learner"),
        arm="Womens E-Mail",
        outcome="visit",
        max_workers=12,
        normalize=None
):
    jobs = [
        (split_seed, model_seed, model_type)
        for split_seed in range(n_splits)
        for model_seed in range(n_model_seeds)
        for model_type in models
    ]
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one, split_seed, model_seed, model_type, arm, outcome, normalize):
                       (split_seed, model_seed, model_type) for split_seed, model_seed, model_type in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except Exception:
                print(f"Job {job} failed at the executor level:")
                traceback.print_exc()

    df = pd.DataFrame(results)
    n_errors = df["error"].notna().sum() if "error" in df else 0
    if n_errors:
        print(f"Warning: {n_errors} of {len(df)} jobs errored.")

    return df


# # 1. Get real train/test splits, already encoded
# X_train, X_test, y_train, y_test, treatment_train, treatment_test = load_and_prepare(treatment_arm="Womens E-Mail")
#
# # 2. Fit + score
# uplift_scores_t_learner = run_two_model(X_train, treatment_train, y_train["visit"], X_test, random_state=42, n_jobs=1)
# uplift_scores_uplift_rf = train_uplift_rf(
#     X_train, treatment_train, y_train["visit"], X_test,
#     random_state=42, max_depth=8, n_estimators=300, evaluationFunction="KL", n_jobs=1)
#
# # 3. Evaluate on test set
# t_learner_results = evaluate_model("T-Learner", uplift_scores_t_learner, treatment_test, y_test["visit"])
# uplift_rf_results = evaluate_model("Uplift Trees", uplift_scores_uplift_rf, treatment_test, y_test["visit"])
#
# print(t_learner_results)
# print(uplift_rf_results)
#
# # 4. Visual check
# ax1 = plot_qini_curve(uplift_scores_t_learner, treatment_test, y_test["visit"], outcome_name="visit", label="T-Learner")
# plot_qini_curve(uplift_scores_uplift_rf, treatment_test, y_test["visit"], outcome_name="visit", label="Uplift Trees", ax=ax1)
# plot_uplift_by_decile(uplift_scores_t_learner, treatment_test, y_test["visit"], outcome_name="visit", label="T-Learner")
# plot_uplift_by_decile(uplift_scores_uplift_rf, treatment_test, y_test["visit"], outcome_name="visit", label="Uplift Trees")
#
# plt.show()


if __name__ == "__main__":
    df = run_stability_check(outcome="conversion")
    summary = df.groupby("model")["qini_coefficient"].agg(["mean", "std", "count"])

    pivoted = df.pivot_table(
        index=["split_seed", "model_seed"], columns="model", values="qini_coefficient"
    )
    win_rate = (pivoted["X-Learner"] > pivoted["Uplift Trees"]).mean()

    print(summary)
    print(f"X-Learner wins {win_rate}")

