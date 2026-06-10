"""
data.py — loading, cleaning, and splitting the international results.

This is the single place the rest of the project gets match data from, so
every model (Elo, Dixon-Coles, Bayesian) trains on exactly the same clean
input. Keeping it here avoids subtle differences between scripts.

Typical use:
    from src.data import load_results, filter_matches, train_test_split_by_date

    df = load_results()                       # clean full history
    train = filter_matches(df, since="2018-01-01")
    train, test = train_test_split_by_date(df, cutoff="2022-11-20")
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

# default location written by data/download_data.py
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "results.csv"

# tournaments we treat as "competitive" (everything else is friendly-ish).
# Used optionally — competitive matches are better signal than friendlies.
COMPETITIVE_KEYWORDS = (
    "FIFA World Cup", "UEFA Euro", "Copa América", "Copa America",
    "African Cup of Nations", "AFC Asian Cup", "qualification",
    "Nations League", "Confederations Cup", "Gold Cup",
)


def load_results(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Load results.csv and apply baseline cleaning.

    Expected columns: date, home_team, away_team, home_score, away_score,
    tournament, city, country, neutral.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/download_data.py` first."
        )

    df = pd.read_csv(path, parse_dates=["date"])

    # drop rows without a recorded score (can't learn from those)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # strip whitespace on team names to avoid silent duplicates
    df["home_team"] = df["home_team"].str.strip()
    df["away_team"] = df["away_team"].str.strip()

    # normalize the neutral flag to a clean boolean
    df["neutral"] = df["neutral"].astype(bool)

    df = df.sort_values("date").reset_index(drop=True)
    return df


def _is_competitive(tournament: pd.Series) -> pd.Series:
    pattern = "|".join(COMPETITIVE_KEYWORDS)
    return tournament.str.contains(pattern, case=False, regex=True, na=False)


def filter_matches(
    df: pd.DataFrame,
    since: str | None = None,
    until: str | None = None,
    drop_friendlies: bool = False,
    competitive_only: bool = False,
) -> pd.DataFrame:
    """Filter the match set by date window and/or match importance."""
    out = df
    if since is not None:
        out = out[out["date"] >= pd.Timestamp(since)]
    if until is not None:
        out = out[out["date"] <= pd.Timestamp(until)]
    if drop_friendlies:
        out = out[out["tournament"] != "Friendly"]
    if competitive_only:
        out = out[_is_competitive(out["tournament"])]
    return out.reset_index(drop=True)


def train_test_split_by_date(
    df: pd.DataFrame, cutoff: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: train on everything strictly before `cutoff`,
    test on everything on/after it. This is the correct split for
    forecasting — never let future matches leak into training.

    For backtesting a past World Cup, set cutoff to that tournament's
    opening date (e.g. 2022-11-20 for Qatar, 2018-06-14 for Russia).
    """
    cut = pd.Timestamp(cutoff)
    train = df[df["date"] < cut].reset_index(drop=True)
    test = df[df["date"] >= cut].reset_index(drop=True)
    return train, test


def get_teams(df: pd.DataFrame) -> list[str]:
    """Sorted list of every team appearing as home or away."""
    return sorted(set(df["home_team"]) | set(df["away_team"]))


if __name__ == "__main__":
    # quick smoke test
    df = load_results()
    print(f"Loaded {len(df):,} matches "
          f"({df.date.min().date()} -> {df.date.max().date()})")
    print(f"Teams: {len(get_teams(df))}")
    recent = filter_matches(df, since="2018-01-01")
    print(f"Since 2018: {len(recent):,} matches")
    tr, te = train_test_split_by_date(df, "2022-11-20")
    print(f"Backtest split @2022-11-20 -> train {len(tr):,}, test {len(te):,}")
