import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _rank_by_score(uplift_scores, treatment, outcome):
    sorted_indices = np.argsort(uplift_scores)[::-1]

    return uplift_scores[sorted_indices], treatment[sorted_indices], outcome[sorted_indices]


def _cumulative_incremental_gain(treatment_sorted, outcome_sorted):
    is_treated = treatment_sorted == 1
    is_control = ~is_treated
    conv_treated_cum = np.cumsum(outcome_sorted * is_treated)
    conv_control_cum = np.cumsum(outcome_sorted * is_control)
    n_treated_cum = np.cumsum(is_treated)
    n_control_cum = np.cumsum(is_control)

    ratio = np.divide(n_treated_cum, n_control_cum,
                      out=np.zeros(len(treatment_sorted)), where=n_control_cum != 0)

    return conv_treated_cum - conv_control_cum * ratio


def qini_curve(uplift_scores, treatment, outcome):
    n = len(treatment)
    pct_targeted = np.concatenate([[0.0], np.arange(1, n+1) / n])

    uplift_sorted, treatment_sorted, outcome_sorted = _rank_by_score(uplift_scores, treatment, outcome)
    cumulative_gain = np.concatenate([[0.0], _cumulative_incremental_gain(treatment_sorted, outcome_sorted)])

    total_gain = cumulative_gain[-1]
    random_diagonal = pct_targeted * total_gain

    return pct_targeted, cumulative_gain, random_diagonal


def plot_qini_curve(uplift_scores, treatment, outcome, label=None, ax=None):
    pct_targeted, cumulative_gain, random_diagonal = qini_curve(uplift_scores, treatment, outcome)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(pct_targeted, cumulative_gain, label=label)
    ax.plot(pct_targeted, random_diagonal)

    if label is not None:
        ax.legend()

    return ax


def qini_coefficient(uplift_scores, treatment, outcome):
    pct_targeted, cumulative_gain, random_diagonal = qini_curve(uplift_scores, treatment, outcome)
    qini_coefficient = np.trapezoid(y=cumulative_gain, x=pct_targeted) - np.trapezoid(y=random_diagonal, x=pct_targeted)

    return qini_coefficient


def uplift_by_decile(uplift_scores, treatment, outcome, n_deciles=10):
    order = np.argsort(uplift_scores)[::-1]
    ranks = np.empty(len(uplift_scores), dtype=int)
    ranks[order] = np.arange(len(uplift_scores))

    decile = pd.qcut(ranks, n_deciles, labels=False) + 1
    df = pd.DataFrame({"decile": decile, "treatment": treatment, "outcome": outcome})

    n_treated, n_control, treated_rate, control_rate, observed_uplift = [], [], [], [], []
    for d, g in df.groupby("decile"):
        treated = g[g.treatment == 1]
        control = g[g.treatment == 0]
        d_treated_rate = treated["outcome"].mean() if len(treated) else np.nan
        d_control_rate = control["outcome"].mean() if len(control) else np.nan
        d_observed_uplift = d_treated_rate - d_control_rate

        n_treated.append(len(treated))
        n_control.append(len(control))
        treated_rate.append(d_treated_rate)
        control_rate.append(d_control_rate)
        observed_uplift.append(d_observed_uplift)

    output = pd.DataFrame({
        "decile": np.arange(1, n_deciles + 1),
        "n_treated": n_treated,
        "n_control": n_control,
        "treated_rate": treated_rate,
        "control_rate": control_rate,
        "observed_uplift": observed_uplift
    })

    return output


def plot_uplift_by_decile(uplift_scores, treatment, outcome, ax=None):
    uplift_df = uplift_by_decile(uplift_scores, treatment, outcome)

    if ax is None:
        fig, ax = plt.subplots()

    colors = np.where(uplift_df["observed_uplift"] < 0, "tab:red", "tab:blue")
    ax.bar(uplift_df["decile"], uplift_df["observed_uplift"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    for row in uplift_df.itertuples():
        if row.observed_uplift < 0:
            print(f"Decile {row.decile} has negative uplift: {row.observed_uplift}")

    return ax


def evaluate_model(model_name, uplift_scores, treatment, outcome):
    qini_coef = qini_coefficient(uplift_scores, treatment, outcome)
    decile_table = uplift_by_decile(uplift_scores, treatment, outcome)

    return {
        "model": model_name,
        "qini_coefficient": qini_coef,
        "top_decile_uplift": decile_table.loc[decile_table.decile == 1, "observed_uplift"].iloc[0],
        "bottom_decile_uplift": decile_table.loc[decile_table.decile == 10, "observed_uplift"].iloc[0],
        "monotonic": decile_table["observed_uplift"].is_monotonic_decreasing,
    }