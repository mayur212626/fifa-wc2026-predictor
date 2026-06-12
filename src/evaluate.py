"""
evaluate.py — scoring rules, calibration, and World Cup backtests.

This module answers the question that separates a real forecasting system
from a toy: *are the probabilities any good?* It does that three ways:

1. Proper scoring rules — multiclass Brier score and log-loss. Both reward
   honest, well-calibrated probabilities; you cannot game them.
2. Baselines — every score is meaningless without a reference. We compare
   against a uniform (1/3, 1/3, 1/3) forecast and a climatology baseline
   (the historical win/draw/loss frequencies in neutral matches).
3. Out-of-sample backtests — refit the model using ONLY matches played
   before the 2018 and 2022 World Cups, then score its predictions on the
   actual tournament matches. No future data leaks into training; this is
   the chronological split from src.data.

Reference points (3-outcome problem):
  * Uniform Brier = 0.667, uniform log-loss = ln(3) = 1.099.
  * Good international-football models typically land around Brier
    0.55-0.62 and log-loss 0.95-1.05 on World Cup matches. The betting
    market is usually near the bottom of those ranges.

Usage:
    python -m src.evaluate        # runs both backtests + calibration plot
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# World Cup opening dates (train strictly before these)
BACKTESTS = {
    "2018 World Cup": ("2018-06-14", "2018-07-16"),
    "2022 World Cup": ("2022-11-20", "2022-12-19"),
}


# ----------------------------------------------------------------------------
# Outcomes and scoring rules
# ----------------------------------------------------------------------------

def outcome_index(home_goals: int, away_goals: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Multiclass Brier score, averaged over matches. Lower is better.

    probs: (n_matches, 3) array of [p_home, p_draw, p_away].
    outcomes: (n_matches,) array of outcome indices (0/1/2).
    """
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcomes)), outcomes] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean negative log-likelihood of the realized outcome. Lower is better."""
    p_actual = probs[np.arange(len(outcomes)), outcomes]
    return float(-np.log(np.clip(p_actual, 1e-12, 1.0)).mean())


def calibration_table(probs: np.ndarray, outcomes: np.ndarray,
                      n_bins: int = 8) -> pd.DataFrame:
    """Bin all predicted probabilities and compare to realized frequency.

    Pools the three outcome columns (each prediction-outcome pair is one
    observation), which is standard for small samples. Perfect calibration
    means predicted ~= observed in every bin.
    """
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcomes)), outcomes] = 1.0
    p = probs.ravel()
    y = onehot.ravel()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = which == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": f"{bins[b]:.2f}-{bins[b + 1]:.2f}",
            "n": int(mask.sum()),
            "predicted": float(p[mask].mean()),
            "observed": float(y[mask].mean()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Generating predictions for a set of matches
# ----------------------------------------------------------------------------

def predict_matches(model, matches: pd.DataFrame) -> tuple[np.ndarray,
                                                           np.ndarray,
                                                           pd.DataFrame]:
    """Run the model over a matches DataFrame. Returns (probs, outcomes,
    skipped) where skipped lists matches with teams unseen in training."""
    probs, outcomes, skipped = [], [], []
    for row in matches.itertuples(index=False):
        try:
            pred = model.predict(row.home_team, row.away_team,
                                 neutral=bool(row.neutral))
        except KeyError:
            skipped.append((row.home_team, row.away_team))
            continue
        probs.append([pred["p_home"], pred["p_draw"], pred["p_away"]])
        outcomes.append(outcome_index(int(row.home_score),
                                      int(row.away_score)))
    return (np.array(probs), np.array(outcomes),
            pd.DataFrame(skipped, columns=["home", "away"]))


def climatology_baseline(train: pd.DataFrame) -> np.ndarray:
    """Historical W/D/L frequencies in neutral-venue matches — the simplest
    forecast that uses any data at all."""
    neutral = train[train.neutral]
    counts = np.zeros(3)
    for row in neutral.itertuples(index=False):
        counts[outcome_index(int(row.home_score), int(row.away_score))] += 1
    return counts / counts.sum()


# ----------------------------------------------------------------------------
# Backtest harness
# ----------------------------------------------------------------------------

def backtest_world_cup(df: pd.DataFrame, start: str, end: str,
                       train_since: str = "2010-01-01",
                       label: str = "") -> dict:
    """Train strictly before `start`, evaluate on the actual World Cup
    matches between start and end. Returns a dict of scores."""
    from src.dixon_coles import DixonColes

    train = df[(df.date >= pd.Timestamp(train_since))
               & (df.date < pd.Timestamp(start))].reset_index(drop=True)
    test = df[(df.tournament == "FIFA World Cup")
              & (df.date >= pd.Timestamp(start))
              & (df.date <= pd.Timestamp(end))].reset_index(drop=True)

    print(f"\n=== {label} ===")
    print(f"Train: {len(train):,} matches up to {start} "
          f"| Test: {len(test)} World Cup matches")

    model = DixonColes().fit(train)
    probs, outcomes, skipped = predict_matches(model, test)
    if len(skipped):
        print(f"Skipped {len(skipped)} matches with unseen teams:")
        print(skipped.to_string(index=False))

    clim = climatology_baseline(train)
    clim_probs = np.tile(clim, (len(outcomes), 1))
    uniform = np.full((len(outcomes), 3), 1 / 3)

    results = {
        "label": label,
        "n_matches": len(outcomes),
        "model_brier": brier_score(probs, outcomes),
        "model_logloss": log_loss(probs, outcomes),
        "climatology_brier": brier_score(clim_probs, outcomes),
        "climatology_logloss": log_loss(clim_probs, outcomes),
        "uniform_brier": brier_score(uniform, outcomes),
        "uniform_logloss": log_loss(uniform, outcomes),
        "probs": probs,
        "outcomes": outcomes,
    }

    print(f"\n{'':>14}  {'Brier':>7}  {'LogLoss':>8}   (lower = better)")
    print(f"{'Model':>14}  {results['model_brier']:>7.4f}  "
          f"{results['model_logloss']:>8.4f}")
    print(f"{'Climatology':>14}  {results['climatology_brier']:>7.4f}  "
          f"{results['climatology_logloss']:>8.4f}")
    print(f"{'Uniform':>14}  {results['uniform_brier']:>7.4f}  "
          f"{results['uniform_logloss']:>8.4f}")
    return results


def calibration_plot(all_probs: np.ndarray, all_outcomes: np.ndarray,
                     path: str = "reports/calibration.png") -> pd.DataFrame:
    """Save a reliability diagram pooling all backtest predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = calibration_table(all_probs, all_outcomes)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(table.predicted, table.observed, "o-", label="model")
    for _, r in table.iterrows():
        ax.annotate(f"n={r.n}", (r.predicted, r.observed),
                    textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration — 2018 + 2022 World Cup backtests")
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"\nCalibration plot saved to {path}")
    return table


if __name__ == "__main__":
    from src.data import load_results

    df = load_results()

    all_probs, all_outcomes = [], []
    summary_rows = []
    for label, (start, end) in BACKTESTS.items():
        res = backtest_world_cup(df, start, end, label=label)
        all_probs.append(res["probs"])
        all_outcomes.append(res["outcomes"])
        summary_rows.append({k: v for k, v in res.items()
                             if k not in ("probs", "outcomes")})

    probs = np.vstack(all_probs)
    outcomes = np.concatenate(all_outcomes)

    print("\n=== Pooled calibration (both tournaments, "
          f"{len(outcomes)} matches) ===")
    table = calibration_plot(probs, outcomes)
    print(table.round(3).to_string(index=False))

    pd.DataFrame(summary_rows).to_csv("reports/backtest_summary.csv",
                                      index=False)
    print("\nSummary saved to reports/backtest_summary.csv")
