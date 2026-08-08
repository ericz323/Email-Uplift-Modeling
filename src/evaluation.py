import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _rank_by_score(uplift_scores, treatment, outcome):
    # np.asarray so pandas Series index labels can't turn positional sorting
    # into label-based lookups
    uplift_scores = np.asarray(uplift_scores)
    treatment = np.asarray(treatment)
    outcome = np.asarray(outcome)
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


def plot_qini_curve(
        uplift_scores, treatment, outcome,
        curve_color="b", diagonal_color="r",
        outcome_name=None, label=None, ax=None, show_diagonal=True
):
    pct_targeted, cumulative_gain, random_diagonal = qini_curve(uplift_scores, treatment, outcome)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(pct_targeted, cumulative_gain, color=curve_color, label=label)
    if show_diagonal:
        ax.plot(pct_targeted, random_diagonal, color=diagonal_color)

    ax.set_title(f"Qini Plot: {outcome_name}")

    if label is not None:
        ax.legend()

    return ax


def sample_qini_curves(df, run_lookup, model, curve_colors, diagonal_color, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    model_df = df[df["model"] == model]

    best = model_df.loc[model_df["qini_coefficient"].idxmax()]
    worst = model_df.loc[model_df["qini_coefficient"].idxmin()]
    median_index = (model_df["qini_coefficient"] - model_df["qini_coefficient"].median()).abs().idxmin()
    median = model_df.loc[median_index]

    for index, (row_name, row) in enumerate({"max": best, "median": median, "min": worst}.items()):
        uplift_scores, treatment, outcome = run_lookup(row["split_seed"], row["model_seed"], model_type=model)

        show_diagonal = True if row_name == "median" else False
        plot_qini_curve(
            uplift_scores, treatment, outcome,
            outcome_name=row["outcome"], label=f"{model} - {row_name}",
            curve_color=curve_colors[index], diagonal_color=diagonal_color, ax=ax, show_diagonal=show_diagonal
        )


    return ax


def _perfect_ranking(treatment, outcome):
    """Scores for the ranking that maximizes the Qini curve on observed data.

    We never see both potential outcomes, so the oracle is defined over the four
    observable cells rather than over true individual uplift. Ordering them:

      (t=1, y=1)  treated responder      -- each one lifts the curve by 1
      (t=0, y=0)  control non-responder  -- consistent with uplift, lifts by 0
      (t=1, y=0)  treated non-responder  -- flat, but raises the treated/control
                                            ratio, so it must come before...
      (t=0, y=1)  control responder      -- each one pulls the curve back down

    All four cells must appear, so every ranking ends at the same total gain;
    the oracle is the one that rises fastest and falls last. Note this requires
    knowing y at scoring time, so it is an upper bound no X-based model can hit.
    """
    treatment = np.asarray(treatment)
    outcome = np.asarray(outcome)

    return np.select(
        [
            (treatment == 1) & (outcome == 1),
            (treatment == 0) & (outcome == 0),
            (treatment == 1) & (outcome == 0),
        ],
        [3.0, 2.0, 1.0],
        default=0.0,
    )


def qini_coefficient(uplift_scores, treatment, outcome, normalize=None):
    """Area between the Qini curve and the random-targeting diagonal.

    normalize:
      None      -- raw area, in incremental-event units. Scales with sample size
                   and base rate, so it is NOT comparable across outcomes.
      "gain"    -- divide by the campaign's total incremental gain. Unitless,
                   roughly [-0.5, 0.5]; the uplift analogue of a Gini. Reads as
                   "how front-loaded the gain is under this ranking".
      "perfect" -- divide by the best achievable curve on this data. Unitless,
                   <= 1; reads as "fraction of attainable uplift captured".
    """
    pct_targeted, cumulative_gain, random_diagonal = qini_curve(uplift_scores, treatment, outcome)
    raw = np.trapezoid(y=cumulative_gain, x=pct_targeted) - np.trapezoid(y=random_diagonal, x=pct_targeted)

    if normalize is None:
        return raw

    if normalize == "gain":
        total_gain = abs(cumulative_gain[-1])

        return raw / total_gain if total_gain > 0 else np.nan

    if normalize == "perfect":
        perfect_pct, perfect_gain, perfect_diagonal = qini_curve(
            _perfect_ranking(treatment, outcome), treatment, outcome
        )
        best = np.trapezoid(y=perfect_gain, x=perfect_pct) - np.trapezoid(y=perfect_diagonal, x=perfect_pct)

        return raw / best if best > 0 else np.nan

    raise ValueError(f"unknown normalize {normalize!r}")


def plot_qini_distribution(df, metric="qini_coefficient", ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    n_error = df["error"].notna().sum() if "error" in df else 0
    if n_error:
        print(f"Dropping {n_error} errored runs")

    clean_df = df[df["error"].isna()].copy()

    sns.violinplot(data=clean_df, x=metric, y="model", ax=ax)
    sns.stripplot(data=clean_df, x=metric, y="model", ax=ax, alpha=0.4, jitter=True)

    return ax


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


def plot_uplift_by_decile(uplift_scores, treatment, outcome, outcome_name, ax=None, label=None):
    uplift_df = uplift_by_decile(uplift_scores, treatment, outcome)

    if ax is None:
        fig, ax = plt.subplots()

    colors = np.where(uplift_df["observed_uplift"] < 0, "tab:red", "tab:blue")
    ax.bar(uplift_df["decile"], uplift_df["observed_uplift"], color=colors, label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    for row in uplift_df.itertuples():
        if row.observed_uplift < 0:
            print(f"Decile {row.decile} has negative uplift: {row.observed_uplift}")

    ax.set_title(outcome_name)

    if label is not None:
        ax.legend()

    return ax


def evaluate_model(model_name, uplift_scores, treatment, outcome, normalize=None):
    qini_coef = qini_coefficient(uplift_scores, treatment, outcome, normalize=normalize)
    decile_table = uplift_by_decile(uplift_scores, treatment, outcome)

    return {
        "model": model_name,
        "qini_coefficient": qini_coef,
        "qini_gain_normalized": qini_coefficient(uplift_scores, treatment, outcome, normalize="gain"),
        "qini_perfect_normalized": qini_coefficient(uplift_scores, treatment, outcome, normalize="perfect"),
        "top_decile_uplift": decile_table.loc[decile_table.decile == 1, "observed_uplift"].iloc[0],
        "bottom_decile_uplift": decile_table.loc[decile_table.decile == 10, "observed_uplift"].iloc[0],
        "monotonic": decile_table["observed_uplift"].is_monotonic_decreasing,
    }