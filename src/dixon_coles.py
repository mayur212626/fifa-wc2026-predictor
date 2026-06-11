"""
dixon_coles.py — Dixon-Coles bivariate Poisson scoreline model.

Where Elo gives one number per team, this models *goals*. Each team gets an
attack strength and a defense strength; combined with a home-advantage term
these give the expected goals for each side, and Poisson distributions turn
those into the probability of every possible scoreline. The Dixon-Coles
correction adjusts the four low-scoring results (0-0, 1-0, 0-1, 1-1), which
a plain independent-Poisson model gets slightly wrong.

A time-decay weight makes recent matches count more than old ones, so the
ratings reflect current form rather than treating a 2005 result as equal to
a 2025 one.

References: Maher (1982); Dixon & Coles (1997).

Usage:
    from src.data import load_results, filter_matches
    from src.dixon_coles import DixonColes

    df = filter_matches(load_results(), since="2010-01-01")
    model = DixonColes().fit(df)
    model.predict("Brazil", "Argentina", neutral=True)
    model.score_matrix("Brazil", "Argentina")   # full grid of scoreline probs
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


class DixonColes:
    """Maher/Dixon-Coles goal model fitted by maximum likelihood.

    Parameters
    ----------
    xi : time-decay rate. Larger -> recent matches dominate more.
         0.0018/day roughly halves a match's weight every ~1 year.
    max_goals : grid size for scoreline probability matrices.
    """

    def __init__(self, xi: float = 0.0018, max_goals: int = 10):
        self.xi = xi
        self.max_goals = max_goals
        self.teams: list[str] = []
        self.params: np.ndarray | None = None
        self._idx: dict[str, int] = {}

    # -- the low-score correction -----------------------------------------
    @staticmethod
    def _tau(hg, ag, lam_h, lam_a, rho):
        """Dixon-Coles dependence correction for the four lowest scores.
        Returns a multiplier applied to the independent-Poisson probability."""
        hg, ag = np.asarray(hg), np.asarray(ag)
        out = np.ones(np.broadcast(hg, ag, lam_h, lam_a).shape, dtype=float)
        out = np.where((hg == 0) & (ag == 0), 1.0 - lam_h * lam_a * rho, out)
        out = np.where((hg == 0) & (ag == 1), 1.0 + lam_h * rho, out)
        out = np.where((hg == 1) & (ag == 0), 1.0 + lam_a * rho, out)
        out = np.where((hg == 1) & (ag == 1), 1.0 - rho, out)
        return out

    # -- fitting -----------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "DixonColes":
        """Estimate attack/defense/home/rho by maximizing the (weighted)
        log-likelihood of all observed scorelines."""
        self.teams = sorted(set(df.home_team) | set(df.away_team))
        self._idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        hg = df.home_score.to_numpy()
        ag = df.away_score.to_numpy()
        hi = df.home_team.map(self._idx).to_numpy()
        ai = df.away_team.map(self._idx).to_numpy()
        neutral = df.neutral.to_numpy().astype(float)

        # time-decay weights: recent matches matter more
        age_days = (df.date.max() - df.date).dt.days.to_numpy()
        weights = np.exp(-self.xi * age_days)

        # parameter vector: [attack(n), defense(n), home_adv, rho]
        init = np.concatenate([np.zeros(n), np.zeros(n), [0.25, -0.1]])

        def neg_log_likelihood(p):
            attack = p[:n]
            defense = p[n:2 * n]
            home_adv = p[2 * n]
            rho = p[2 * n + 1]

            # expected goals (home advantage only on non-neutral grounds)
            lam_h = np.exp(attack[hi] - defense[ai] + home_adv * (1 - neutral))
            lam_a = np.exp(attack[ai] - defense[hi])

            tau = self._tau(hg, ag, lam_h, lam_a, rho)
            log_lik = (np.log(np.maximum(tau, 1e-10))
                       + poisson.logpmf(hg, lam_h)
                       + poisson.logpmf(ag, lam_a))
            return -np.sum(weights * log_lik)

        # identifiability: pin team 0's attack to zero by centering inside
        # the likelihood — lets us use the faster unconstrained L-BFGS-B
        def neg_log_likelihood_centered(p):
            p = p.copy()
            p[:n] = p[:n] - p[:n].mean()   # center attacks each evaluation
            return neg_log_likelihood(p)

        result = minimize(
            neg_log_likelihood_centered, init, method="L-BFGS-B",
            options={"maxiter": 5000, "maxfun": 2_000_000,
                     "ftol": 1e-7, "gtol": 1e-4},
        )
        result.x[:len(self.teams)] -= result.x[:len(self.teams)].mean()
        if not result.success:
            print(f"[warning] optimizer did not fully converge: {result.message}")
        self.params = result.x
        return self

    # -- prediction --------------------------------------------------------
    def _expected_goals(self, home: str, away: str, neutral: bool):
        if self.params is None:
            raise RuntimeError("Model not fitted. Call .fit(df) first.")
        for t in (home, away):
            if t not in self._idx:
                raise KeyError(f"Unknown team: {t!r} (not in training data)")
        n = len(self.teams)
        attack, defense = self.params[:n], self.params[n:2 * n]
        home_adv = self.params[2 * n]
        hi, ai = self._idx[home], self._idx[away]
        lam_h = np.exp(attack[hi] - defense[ai] + home_adv * (0 if neutral else 1))
        lam_a = np.exp(attack[ai] - defense[hi])
        return float(lam_h), float(lam_a)

    def score_matrix(self, home: str, away: str,
                     neutral: bool = True) -> np.ndarray:
        """Full (max_goals+1) x (max_goals+1) matrix of scoreline probabilities,
        entry [i, j] = P(home scores i, away scores j)."""
        lam_h, lam_a = self._expected_goals(home, away, neutral)
        rho = self.params[-1]
        goals = np.arange(self.max_goals + 1)

        ph = poisson.pmf(goals, lam_h)
        pa = poisson.pmf(goals, lam_a)
        matrix = np.outer(ph, pa)

        # apply the low-score correction to the 2x2 corner
        for i in (0, 1):
            for j in (0, 1):
                matrix[i, j] *= float(self._tau(i, j, lam_h, lam_a, rho))

        return matrix / matrix.sum()

    def predict(self, home: str, away: str, neutral: bool = True) -> dict:
        """Win/draw/loss probabilities and expected goals for a fixture."""
        m = self.score_matrix(home, away, neutral)
        lam_h, lam_a = self._expected_goals(home, away, neutral)
        return {
            "p_home": float(np.tril(m, -1).sum()),   # home goals > away goals
            "p_draw": float(np.trace(m)),            # diagonal
            "p_away": float(np.triu(m, 1).sum()),    # away goals > home goals
            "xg_home": lam_h,
            "xg_away": lam_a,
        }

    def team_ratings(self) -> pd.DataFrame:
        """Fitted attack and defense strengths per team (for inspection)."""
        n = len(self.teams)
        return (pd.DataFrame({
            "team": self.teams,
            "attack": self.params[:n],
            "defense": self.params[n:2 * n],
        }).sort_values("attack", ascending=False).reset_index(drop=True))


if __name__ == "__main__":
    from src.data import load_results, filter_matches

    # a ~15-year window balances enough data against current relevance
    df = filter_matches(load_results(), since="2010-01-01")
    print(f"Fitting Dixon-Coles on {len(df):,} matches "
          f"({df.date.min().date()} -> {df.date.max().date()}) ...")

    model = DixonColes().fit(df)
    print(f"Home advantage (log scale): {model.params[-2]:.3f}")
    print(f"Rho (low-score correction): {model.params[-1]:.3f}\n")

    print("Top 10 attacking sides (fitted):")
    print(model.team_ratings().head(10).to_string(index=False))

    print("\nExample — Brazil vs Argentina (neutral):")
    pred = model.predict("Brazil", "Argentina", neutral=True)
    for k, v in pred.items():
        print(f"  {k}: {v:.3f}")
