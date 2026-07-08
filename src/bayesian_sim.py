"""
bayesian_sim.py — tournament simulation with full uncertainty propagation.

The Dixon-Coles simulation treats the fitted team strengths as exact truth.
This module does the Bayesian version: every simulated tournament samples a
DIFFERENT plausible set of team strengths from the posterior, so parameter
uncertainty flows all the way into the title odds.

Why this matters: the live calibration data showed the point-estimate model
underestimates upsets in the tails (predicted ~8%, observed ~20%). Posterior
sampling is the principled fix — teams the model is unsure about get
genuinely uncertain simulated outcomes, which should fatten the tails.
Running this side-by-side with the Dixon-Coles odds tests that hypothesis.

Method note: one posterior draw per tournament REPLICATION (not per match),
so that a "strong Argentina" draw stays strong for that whole tournament —
this preserves the correlated uncertainty that per-match resampling would
wash out. Scorelines use independent Poisson (the Bayesian model has no rho
term); the low-score correction matters little at tournament level.

Usage (takes several minutes: MCMC fit + 5,000 simulations):
    python -m src.bayesian_sim
"""

from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.data import load_results, filter_matches
from src.bayesian import BayesianGoalModel
from src.simulate import (GROUPS, R32, R16, QF, SF,
                          dataset_name, validate_teams, assign_thirds)
from src.update import (played_2026, build_lookups, cond_group,
                        cond_knockout, real_third_assignment)

REPORTS = Path(__file__).resolve().parents[1] / "reports"
N_SIMS = 5000
MAX_GOALS = 10


class BayesianAdapter:
    """Wraps posterior draws behind the same interface the simulator uses
    (`score_matrix`, `_expected_goals`, `_idx`). Set `.k` to choose which
    posterior draw is active; the tournament loop re-samples k per
    replication."""

    def __init__(self, draws: list[dict], max_goals: int = MAX_GOALS):
        self.draws = draws
        self.max_goals = max_goals
        self._idx = draws[0]["teams"]          # {team_name: index}
        self.k = 0

    def _expected_goals(self, home: str, away: str, neutral: bool):
        d = self.draws[self.k]
        hi, ai = self._idx[home], self._idx[away]
        adv = 0.0 if neutral else 1.0
        lam_h = float(np.exp(d["mu"] + d["home_adv"] * adv
                             + d["att"][hi] - d["def_"][ai]))
        lam_a = float(np.exp(d["mu"] + d["att"][ai] - d["def_"][hi]))
        return lam_h, lam_a

    def score_matrix(self, home: str, away: str,
                     neutral: bool = True) -> np.ndarray:
        lam_h, lam_a = self._expected_goals(home, away, neutral)
        goals = np.arange(self.max_goals + 1)
        m = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))
        return m / m.sum()


def simulate_bayesian(adapter: BayesianAdapter, results: dict,
                      shootouts: dict, n_sims: int = N_SIMS,
                      seed: int = 42,
                      fixed_thirds: dict | None = None) -> pd.DataFrame:
    """Same conditioned tournament loop as src.update, but re-sampling a
    posterior parameter draw at the start of every replication. If
    fixed_thirds is given (the real bracket), it replaces random slot
    assignment so simulations follow the true pairings."""
    rng = np.random.default_rng(seed)
    stages = ["r32", "r16", "qf", "sf", "final", "champion"]
    counts = {t: defaultdict(int) for ts in GROUPS.values() for t in ts}
    n_draws = len(adapter.draws)

    for _ in range(n_sims):
        adapter.k = int(rng.integers(n_draws))      # new world each rep

        firsts, seconds, thirds = {}, {}, []
        for g, teams in GROUPS.items():
            table = cond_group(adapter, teams, results, rng)
            firsts[g] = table[0][0]
            seconds[g] = table[1][0]
            thirds.append((g, table[2][0], table[2][1], table[2][2],
                           table[2][3]))
        third_assign = (fixed_thirds if fixed_thirds is not None
                        else assign_thirds(thirds, rng))

        slot_i, r32_pairs = 0, []
        for sa, sb in R32:
            pair = []
            for spec in (sa, sb):
                if spec[0] == "1":
                    pair.append(firsts[spec[1]])
                elif spec[0] == "2":
                    pair.append(seconds[spec[1]])
                else:
                    pair.append(third_assign[f"{spec}#{slot_i}"])
                    slot_i += 1
            r32_pairs.append(tuple(pair))

        for a, b in r32_pairs:
            counts[a]["r32"] += 1
            counts[b]["r32"] += 1
        w32 = [cond_knockout(adapter, a, b, results, shootouts, rng)
               for a, b in r32_pairs]

        r16_pairs = [(w32[i], w32[j]) for i, j in R16]
        for a, b in r16_pairs:
            counts[a]["r16"] += 1
            counts[b]["r16"] += 1
        w16 = [cond_knockout(adapter, a, b, results, shootouts, rng)
               for a, b in r16_pairs]

        qf_pairs = [(w16[i], w16[j]) for i, j in QF]
        for a, b in qf_pairs:
            counts[a]["qf"] += 1
            counts[b]["qf"] += 1
        w8 = [cond_knockout(adapter, a, b, results, shootouts, rng)
              for a, b in qf_pairs]

        sf_pairs = [(w8[i], w8[j]) for i, j in SF]
        for a, b in sf_pairs:
            counts[a]["sf"] += 1
            counts[b]["sf"] += 1
        w4 = [cond_knockout(adapter, a, b, results, shootouts, rng)
              for a, b in sf_pairs]

        counts[w4[0]]["final"] += 1
        counts[w4[1]]["final"] += 1
        champ = cond_knockout(adapter, w4[0], w4[1], results, shootouts, rng)
        counts[champ]["champion"] += 1

    rows = [[t] + [counts[t][s] / n_sims for s in stages] for t in counts]
    return (pd.DataFrame(rows, columns=["team"] + stages)
            .sort_values("champion", ascending=False)
            .reset_index(drop=True))


if __name__ == "__main__":
    # 1. Fit the Bayesian model (MCMC — a few minutes; keep the laptop awake)
    df_all = load_results()
    train = filter_matches(df_all, since="2015-01-01")
    print(f"Fitting hierarchical Bayesian model on {len(train):,} matches "
          f"(through {train.date.max().date()}) — MCMC, please wait ...")
    bayes = BayesianGoalModel().fit(train, draws=1000, tune=1000, chains=2)

    print("\nSampler diagnostics (want r_hat < 1.01):")
    print(bayes.diagnostics().to_string())

    # 2. Wrap posterior draws for the simulator
    adapter = BayesianAdapter(bayes.posterior_draws(n=1000))
    missing = validate_teams(adapter)
    if missing:
        raise SystemExit(f"Team name mapping broken for: {missing}")

    # 3. Condition on everything actually played, then simulate
    played = played_2026(df_all)
    results, shootouts = build_lookups(played)
    fixed = real_third_assignment(played, results)
    if fixed is not None:
        print("Round of 32 fully known — pinning simulations to the real "
              "bracket (eliminated teams will show 0).")
    print(f"\nConditioning on {len(results)} real results. Simulating "
          f"{N_SIMS:,} tournaments, one posterior draw per replication ...")
    bayes_table = simulate_bayesian(adapter, results, shootouts,
                                    fixed_thirds=fixed)

    pct = (bayes_table.set_index("team") * 100).round(1)
    print("\nBayesian title odds (top 12):\n")
    print(pct.head(12).to_string())

    REPORTS.mkdir(exist_ok=True)
    pct.to_csv(REPORTS / "simulation_2026_bayesian.csv")

    # 4. Compare against the latest Dixon-Coles snapshot
    hist_path = REPORTS / "odds_history.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        hist["ts"] = pd.to_datetime(
            hist["run_ts"].str.replace(" UTC", "", regex=False),
            errors="coerce")
        dc = (hist[hist["ts"] == hist["ts"].max()]
              .set_index("team")["champion"] * 100)
        comp = pd.DataFrame({
            "dixon_coles_%": dc.round(1),
            "bayesian_%": pct["champion"],
        }).dropna().sort_values("bayesian_%", ascending=False)
        comp["diff"] = (comp["bayesian_%"] - comp["dixon_coles_%"]).round(1)

        print("\nModel comparison — champion odds (top 12):\n")
        print(comp.head(12).to_string())
        comp.to_csv(REPORTS / "model_comparison.csv")

        top1 = comp["bayesian_%"].iloc[0]
        print(f"\nRead: if the Bayesian favourite ({comp.index[0]}, "
              f"{top1:.1f}%) sits BELOW the Dixon-Coles number, uncertainty "
              f"is spreading probability toward the field — the predicted "
              f"tail-fattening effect.")
    print("\nSaved: simulation_2026_bayesian.csv, model_comparison.csv")
