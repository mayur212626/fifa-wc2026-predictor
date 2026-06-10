"""
elo.py — Elo ratings for national teams.

A simple, interpretable measure of team strength built sequentially from
match history. Each team starts at a base rating; after every match the
winner takes points from the loser, scaled by the result, the margin of
victory, and how surprising the result was. This gives us a single number
per team to sanity-check (the usual contenders should rise to the top)
before moving on to the Dixon-Coles and Bayesian models.

Reference: the World Football Elo Ratings method, with a FiveThirtyEight-
style margin-of-victory multiplier.

Usage:
    from src.data import load_results, filter_matches
    from src.elo import EloModel

    df = filter_matches(load_results(), since="2002-01-01")
    elo = EloModel().fit(df)
    elo.top(20)
    elo.match_probs("Brazil", "Argentina", neutral=True)
"""

from __future__ import annotations
import math
import pandas as pd


class EloModel:
    """Sequential Elo rating model.

    Parameters
    ----------
    k : base update size (how much ratings move per match).
    home_adv : rating points added to the home side on non-neutral grounds.
    base : starting rating for a team never seen before.
    """

    def __init__(self, k: float = 30.0, home_adv: float = 65.0,
                 base: float = 1500.0):
        self.k = k
        self.home_adv = home_adv
        self.base = base
        self.ratings: dict[str, float] = {}

    # -- internals ---------------------------------------------------------
    def _get(self, team: str) -> float:
        return self.ratings.get(team, self.base)

    @staticmethod
    def _expected(rating_a: float, rating_b: float) -> float:
        """Expected score for A vs B (logistic, 400-point scale)."""
        return 1.0 / (1.0 + 10 ** (-(rating_a - rating_b) / 400.0))

    @staticmethod
    def _mov_multiplier(goal_diff: int, rating_diff: float) -> float:
        """Margin-of-victory multiplier. Bigger wins move ratings more, but
        the effect is dampened when a strong team beats a weak one (so blowouts
        against minnows don't over-inflate ratings)."""
        return math.log1p(abs(goal_diff)) * (2.2 / (abs(rating_diff) * 0.001 + 2.2))

    # -- fitting -----------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "EloModel":
        """Process all matches in chronological order to build final ratings.

        Expects columns: date, home_team, away_team, home_score, away_score,
        neutral. (Use src.data.load_results to get this shape.)
        """
        self.ratings = {}
        df = df.sort_values("date")

        for row in df.itertuples(index=False):
            home, away = row.home_team, row.away_team
            rh, ra = self._get(home), self._get(away)

            # apply home advantage only on non-neutral grounds
            adv = 0.0 if row.neutral else self.home_adv
            exp_home = self._expected(rh + adv, ra)

            gd = int(row.home_score) - int(row.away_score)
            if gd > 0:
                result = 1.0
            elif gd == 0:
                result = 0.5
            else:
                result = 0.0

            mult = self._mov_multiplier(gd, (rh + adv) - ra)
            change = self.k * mult * (result - exp_home)

            self.ratings[home] = rh + change
            self.ratings[away] = ra - change

        return self

    # -- inspection & prediction ------------------------------------------
    def rating(self, team: str) -> float:
        return self._get(team)

    def top(self, n: int = 20) -> pd.DataFrame:
        """Return the top-n teams by current rating."""
        s = (pd.Series(self.ratings, name="elo")
             .sort_values(ascending=False)
             .head(n)
             .round(0)
             .astype(int))
        out = s.reset_index()
        out.columns = ["team", "elo"]
        out.index = range(1, len(out) + 1)
        return out

    def match_probs(self, home: str, away: str,
                    neutral: bool = True) -> dict[str, float]:
        """Win/draw/loss probabilities for a single fixture.

        Elo gives an expected *score*, not a draw rate directly, so we map
        the rating gap to win/draw/loss using an empirical draw model. This
        is a rough approximation — the Dixon-Coles model handles scorelines
        properly — but it's a useful baseline.
        """
        adv = 0.0 if neutral else self.home_adv
        diff = (self._get(home) + adv) - self._get(away)

        # expected score for the home side in [0, 1]
        exp_home = self._expected(self._get(home) + adv, self._get(away))

        # empirical draw probability: highest when teams are even, decaying
        # as the rating gap grows. ~0.28 peak is typical for internationals.
        p_draw = 0.28 * math.exp(-abs(diff) / 200.0)

        # split the remaining probability around the expected score
        p_home = (1 - p_draw) * exp_home
        p_away = (1 - p_draw) * (1 - exp_home)
        return {"p_home": p_home, "p_draw": p_draw, "p_away": p_away}


if __name__ == "__main__":
    from src.data import load_results, filter_matches

    df = filter_matches(load_results(), since="2002-01-01")
    elo = EloModel().fit(df)

    print("Top 20 national teams by Elo:\n")
    print(elo.top(20).to_string())

    print("\nExample fixture — Brazil vs Argentina (neutral):")
    probs = elo.match_probs("Brazil", "Argentina", neutral=True)
    for k, v in probs.items():
        print(f"  {k}: {v:.3f}")
