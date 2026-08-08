# Hillstrom Uplift Modeling

Uplift (heterogeneous treatment effect) modeling on the MineThatData e-mail challenge: which
customers should receive a marketing e-mail, given that some would have purchased anyway and some
are put off by being contacted?

The dataset is 64,000 customers randomly assigned to one of three arms—a women's merchandise
e-mail, a men's merchandise e-mail, or no e-mail—with `visit`, `conversion` and `spend`
recorded over the following two weeks. Because assignment was randomized, treated and control
customers are exchangeable, so the incremental effect of the e-mail
identifiable.

Three learners are compared on repeated train/test splits, scored by
Qini.

## Structure

```
src/
  data_prep.py                  # load, filter to one arm vs control, split, encode
  t_learner.py                  # T-Learner (two independent random forests)
  uplift_trees.py               # causalml UpliftRandomForestClassifier
  meta_learners.py              # X-Learner (four-stage)
  evaluation.py                 # Qini curve/coefficient, decile tables, plots
  stability.py                  # repeated-split sweep across models, in parallel
notebooks/
  eda.ipynb                     # randomization checks, base rates, naive uplift by recency
  results.ipynb                 # the writeup: methodology, results, business policy
```



## Running

Everything imports as `src.<module>`, so run from the repository root.

Prepare the data and print split diagnostics:

```bash
python -m src.data_prep
```

Run the repeated-split model comparison:

```bash
python -m src.stability
```

Or, from a notebook / REPL at the repo root:

```python
from src.data_prep import load_and_prepare
from src.meta_learners import run_x_learner
from src.evaluation import evaluate_model

X_train, X_test, y_train, y_test, t_train, t_test = load_and_prepare(treatment_arm="Womens E-Mail")
scores = run_x_learner(X_train, t_train, y_train["conversion"], X_test)
print(evaluate_model("X-Learner", scores, t_test, y_test["conversion"], normalize="perfect"))
```

`notebooks/results.ipynb` is the main writeup with the full methodology, the plots, and the
interpretation.

## Approach

### Data preparation

`load_and_prepare` filters to a single treatment arm versus `No E-Mail`, builds a binary
`treatment` flag, and makes a stratified 25% test split on the treatment × outcome cross so both
arms and both outcome classes stay proportional (important at a ~0.7% conversion rate).
`history_segment` is ordinally encoded in its natural order; `zip_code` and `channel` are one-hot
encoded. The encoder is fit on train only.

Both outcomes are returned together as dicts (`y_train["visit"]`, `y_train["conversion"]`) so a
single split can be scored on either.

### Models

| Model | Idea                                                                                                                                                                                                                                                                      |
|---|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **T-Learner** (`t_learner.py`) | Fit one random forest per arm, take the difference of predicted probabilities. The difference of two independently-fit models is noisy.                                                                                                                                   |
| **Uplift Trees** (`uplift_trees.py`) | `causalml`'s uplift random forest, splitting directly on a treatment-effect criterion (KL divergence) rather than on outcome purity.                                                                                                                                      |
| **X-Learner** (`meta_learners.py`) | Four stages: outcome model per arm → impute each unit's individual effect using the *other* arm's model → fit effect models on the imputed pseudo-outcome → weight the two effect models by propensity.|


### Evaluation

Scoring is entirely rank-based, because the business policy—mail the top N%, suppress the rest—only uses the ordering of the scores, not their magnitude.

- **Qini curve** (`qini_curve`): cumulative incremental events as a function of targeting depth,
  with the control arm rescaled by the running treated/control ratio, against the
  random-targeting diagonal.
- **Qini coefficient** (`qini_coefficient`): area between curve and diagonal. Three
  normalizations:
  - `None` — raw, in incremental-event units; not comparable across outcomes.
  - `"gain"` — divided by total incremental gain; the uplift analogue of a Gini.
  - `"perfect"` — divided by the best curve achievable on the observed data, so it reads as
    "fraction of attainable uplift captured". What the notebook reports.
- **Decile tables** (`uplift_by_decile`): observed treated-minus-control rate per score decile, a
  direct monotonicity check.

### Stability

Uplift is a difference of two noisy rates, so on a rare outcome a single split can rank models by luck.
`run_stability_check` runs over many different splits (which
customers land in test) and model seeds (the learner's own randomness), fitting every model on
every combination, and returns each run in its own row.

## Results

`Womens E-Mail` vs `No E-Mail`, 15 splits × 2 model seeds = 30 runs per model, Qini normalized
against the perfect ranking.

**`visit`** (~13% event rate):

| model | mean | std |
|---|---|---|
| T-Learner | 0.057 | 0.016 |
| Uplift Trees | 0.057 | 0.014 |
| X-Learner | 0.059 | 0.017 |

All three capture ~6% of attainable uplift and all sit above zero, so every ranking beats random
targeting. However, the spread between model means is about an eighth of any one model's standard
deviation. On `visit`, no learner is meaningfully better than the others.

**`conversion`** (~0.57% control vs ~0.88% treated):

| model | mean | std |
|---|---|---|
| T-Learner | 0.039 | 0.048 |
| Uplift Trees | 0.052 | 0.041 |
| **X-Learner** | **0.084** | 0.050 |

The X-Learner more than doubles the T-Learner mean and beats it on 93% of runs (83% versus Uplift
Trees). However, the run-to-run standard deviation is roughly the size of the gap itself, so this result should be taken with a grain of salt.

### Business translation

The policy the analysis supports: score the mailable base, sort descending, mail the top N%,
suppress the rest. On the median X-Learner conversion run, contacting the top 50% captures ~95% of the incremental
conversions available from mailing everyone, at roughly half the sends (~1.9× random targeting at
that depth). The curve is non-monotonic at other depths, which at ~8 events per decile is noise. 
