"""
bayesian.py — hierarchical Bayesian goal model (PyMC).

The same attack/defense/home structure as Dixon-Coles, but estimated as a
Bayesian hierarchical model (Baio & Blangiardo, 2010). Three things this
buys us over the maximum-likelihood version:

1. Partial pooling — attack and defense strengths share a population prior,
   so teams with few matches get shrunk toward the average instead of
   getting wild, unstable estimates.
2. Full posteriors — every parameter is a distribution, not a point. We can
   see *how sure* the model is about each team.
3. Uncertainty propagation — the tournament simulation can draw a different
   plausible set of team strengths each run, so the final title odds include
   parameter uncertainty, not just match randomness.

Model (per match m, home h(m), away a(m)):
    log lam_home = mu + home_adv * (1 - neutral) + att[h] - def[a]
    log lam_away = mu + att[a] - def[h]
    goals ~ Poisson(lam)
    att[t] ~ Normal(0, sigma_att),  def[t] ~ Normal(0, sigma_def)   [pooled]

Time-decay enters as per-match weights on the log-likelihood (recent
matches count more), via pm.Potential.

NOTE on runtime: MCMC is much slower than the MLE fit — expect several
minutes up to ~20+ depending on your machine. Keep the laptop awake.

Usage:
    from src.data import load_results, filter_matches
    from src.bayesian import BayesianGoalModel

    df = filter_matches(load_results(), since="2015-01-01")
    model = BayesianGoalModel().fit(df)        # runs MCMC
    model.predict("Brazil", "Argentina")       # posterior-averaged probs
    draws = model.posterior_draws(n=500)       # for the simulator
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import poisson

import pymc as pm
import arviz as az


class BayesianGoalModel:
    """Hierarchical Bayesian bivariate Poisson model.

    Parameters
    ----------
    xi : time-decay rate per day (same role as in Dixon-Coles).
    max_goals : grid size for scoreline matrices.
    """

    def __init__(self, xi: float = 0.0018, max_goals: int = 10):
        self.xi = xi
        self.max_goals = max_goals
        self.teams: list[str] = []
        self._idx: dict[str, int] = {}
        self.trace = None          # ArviZ InferenceData after .fit()
        self._post = None          # flattened posterior draws (dict of arrays)

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, draws: int = 1000, tune: int = 1000,
            chains: int = 2, target_accept: float = 0.9,
            seed: int = 42) -> "BayesianGoalModel":
        """Build the PyMC model and sample the posterior with NUTS."""
        self.teams = sorted(set(df.home_team) | set(df.away_team))
        self._idx = {t: i for i, t in enumerate(self.teams)}
        n_teams = len(self.teams)

        hg = df.home_score.to_numpy()
        ag = df.away_score.to_numpy()
        hi = df.home_team.map(self._idx).to_numpy()
        ai = df.away_team.map(self._idx).to_numpy()
        neutral = df.neutral.to_numpy().astype(float)

        age_days = (df.date.max() - df.date).dt.days.to_numpy()
        weights = np.exp(-self.xi * age_days)

        with pm.Model() as model:
            # --- population-level (hyper) priors ---------------------------
            mu = pm.Normal("mu", 0.0, 1.0)                  # baseline log-rate
            home_adv = pm.Normal("home_adv", 0.2, 0.2)      # home edge
            sigma_att = pm.HalfNormal("sigma_att", 0.5)     # spread of attacks
            sigma_def = pm.HalfNormal("sigma_def", 0.5)     # spread of defenses

            # --- team-level effects (non-centered for sampler stability) ---
            att_raw = pm.Normal("att_raw", 0.0, 1.0, shape=n_teams)
            def_raw = pm.Normal("def_raw", 0.0, 1.0, shape=n_teams)
            # center so attack/defense are identified (sum-to-zero)
            att = pm.Deterministic(
                "att", (att_raw - att_raw.mean()) * sigma_att)
            dfn = pm.Deterministic(
                "def_", (def_raw - def_raw.mean()) * sigma_def)

            # --- match-level rates -----------------------------------------
            log_lam_h = mu + home_adv * (1 - neutral) + att[hi] - dfn[ai]
            log_lam_a = mu + att[ai] - dfn[hi]

            # --- weighted Poisson likelihood via Potential ------------------
            # (Potential lets us apply the time-decay weights per match)
            pm.Potential(
                "lik_home",
                (weights * pm.logp(pm.Poisson.dist(pm.math.exp(log_lam_h)),
                                   hg)).sum(),
            )
            pm.Potential(
                "lik_away",
                (weights * pm.logp(pm.Poisson.dist(pm.math.exp(log_lam_a)),
                                   ag)).sum(),
            )

            self.trace = pm.sample(
                draws=draws, tune=tune, chains=chains,
                target_accept=target_accept, random_seed=seed,
                progressbar=True,
            )

        # flatten posterior into plain numpy for fast prediction
        post = self.trace.posterior
        self._post = {
            "mu": post["mu"].values.reshape(-1),
            "home_adv": post["home_adv"].values.reshape(-1),
            "att": post["att"].values.reshape(-1, n_teams),
            "def_": post["def_"].values.reshape(-1, n_teams),
        }
        return self

    # ----------------------------------------------------------- diagnostics
    def diagnostics(self) -> pd.DataFrame:
        """R-hat and effective sample size for the key parameters.
        Rule of thumb: r_hat should be < 1.01 for a trustworthy fit."""
        return az.summary(self.trace,
                          var_names=["mu", "home_adv",
                                     "sigma_att", "sigma_def"])

    def team_table(self, top: int = 15) -> pd.DataFrame:
        """Posterior mean and uncertainty for each team's attack/defense."""
        att = self._post["att"]
        dfn = self._post["def_"]
        out = pd.DataFrame({
            "team": self.teams,
            "attack_mean": att.mean(axis=0),
            "attack_sd": att.std(axis=0),
            "defense_mean": dfn.mean(axis=0),
            "defense_sd": dfn.std(axis=0),
        }).sort_values("attack_mean", ascending=False).reset_index(drop=True)
        return out.head(top)

    # ------------------------------------------------------------ prediction
    def _rates_per_draw(self, home: str, away: str, neutral: bool,
                        thin: int = 1):
        """Per-posterior-draw expected goals for a fixture."""
        for t in (home, away):
            if t not in self._idx:
                raise KeyError(f"Unknown team: {t!r} (not in training data)")
        hi, ai = self._idx[home], self._idx[away]
        p = self._post
        sl = slice(None, None, thin)
        adv = 0.0 if neutral else 1.0
        lam_h = np.exp(p["mu"][sl] + p["home_adv"][sl] * adv
                       + p["att"][sl, hi] - p["def_"][sl, ai])
        lam_a = np.exp(p["mu"][sl] + p["att"][sl, ai] - p["def_"][sl, hi])
        return lam_h, lam_a

    def predict(self, home: str, away: str, neutral: bool = True,
                thin: int = 4) -> dict:
        """Posterior-averaged win/draw/loss probabilities.

        Computes scoreline probabilities for every posterior draw and
        averages — so the answer integrates over parameter uncertainty
        rather than using one point estimate.
        """
        lam_h, lam_a = self._rates_per_draw(home, away, neutral, thin)
        goals = np.arange(self.max_goals + 1)

        # (draws, goals) pmfs for each side, then batched outer products
        ph = poisson.pmf(goals[None, :], lam_h[:, None])
        pa = poisson.pmf(goals[None, :], lam_a[:, None])
        mats = ph[:, :, None] * pa[:, None, :]          # (draws, g, g)
        mats /= mats.sum(axis=(1, 2), keepdims=True)

        tril = np.tril(np.ones((len(goals), len(goals))), -1)
        triu = np.triu(np.ones((len(goals), len(goals))), 1)
        eye = np.eye(len(goals))

        return {
            "p_home": float((mats * tril).sum(axis=(1, 2)).mean()),
            "p_draw": float((mats * eye).sum(axis=(1, 2)).mean()),
            "p_away": float((mats * triu).sum(axis=(1, 2)).mean()),
            "xg_home": float(lam_h.mean()),
            "xg_away": float(lam_a.mean()),
        }

    def posterior_draws(self, n: int = 500, seed: int = 0) -> list[dict]:
        """Sample n posterior parameter sets for the tournament simulator.
        Each entry has mu, home_adv, and full att/def vectors — the simulator
        uses one per tournament replication to propagate uncertainty."""
        rng = np.random.default_rng(seed)
        total = len(self._post["mu"])
        pick = rng.choice(total, size=min(n, total), replace=False)
        return [
            {
                "mu": self._post["mu"][k],
                "home_adv": self._post["home_adv"][k],
                "att": self._post["att"][k],
                "def_": self._post["def_"][k],
                "teams": self._idx,
            }
            for k in pick
        ]


if __name__ == "__main__":
    from src.data import load_results, filter_matches

    # shorter window than the MLE model: MCMC cost scales with data size,
    # and the time-decay means pre-2015 matches carry little weight anyway
    df = filter_matches(load_results(), since="2015-01-01")
    print(f"Sampling hierarchical Bayesian model on {len(df):,} matches "
          f"({df.date.min().date()} -> {df.date.max().date()})")
    print("This runs MCMC — several minutes is normal. Keep the laptop awake.\n")

    model = BayesianGoalModel().fit(df)

    print("\nSampler diagnostics (want r_hat < 1.01):")
    print(model.diagnostics().to_string())

    print("\nTop 15 attacks (posterior mean ± sd):")
    print(model.team_table(15).round(3).to_string(index=False))

    print("\nExample — Brazil vs Argentina (neutral), posterior-averaged:")
    pred = model.predict("Brazil", "Argentina", neutral=True)
    for k, v in pred.items():
        print(f"  {k}: {v:.3f}")
