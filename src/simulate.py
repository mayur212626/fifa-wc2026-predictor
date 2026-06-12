"""
simulate.py — Monte Carlo simulation of the 2026 FIFA World Cup.

Plays the whole tournament thousands of times using a fitted goal model
(Dixon-Coles), sampling a scoreline for every match, applying the official
rules, and counting how often each team reaches each stage.

2026 format implemented here:
  * 12 groups (A-L) of 4, single round robin.
  * Tiebreakers: points, goal difference, goals scored. (FIFA's further
    tiebreakers — conduct points, FIFA ranking — are replaced by a random
    tiebreak; they decide a negligible share of simulations.)
  * Top 2 per group + 8 best third-placed teams -> Round of 32.
  * Single elimination from R32: extra time, then penalties if level.
  * Host nations (USA, Mexico, Canada) get home advantage in their matches.

Groups below are the CONFIRMED final groups (post March-2026 playoffs).
The R32 slot table follows FIFA's published bracket; the third-place slot
assignment uses a feasibility-respecting greedy match to each slot's
allowed-groups list (equivalent in spirit to FIFA's allocation table).

Usage:
    python -m src.simulate            # fits Dixon-Coles, runs 5,000 sims
"""

from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# The confirmed 2026 field
# ----------------------------------------------------------------------------

GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Map official names -> names used in the martj42 dataset.
# validate_teams() will tell you if any mapping is still wrong.
RENAMES: dict[str, str] = {
    "USA": "United States",
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
}

HOSTS = {"United States", "Mexico", "Canada"}

# ----------------------------------------------------------------------------
# Round-of-32 bracket (FIFA published schedule, matches 73-88).
# "1A" = winner of group A, "2A" = runner-up, "3:ABCDF" = a third-placed
# team drawn from one of those groups.
# ----------------------------------------------------------------------------

R32: list[tuple[str, str]] = [
    ("2A", "2B"),          # M73
    ("1E", "3:ABCDF"),     # M74
    ("1F", "2C"),          # M75
    ("1C", "2F"),          # M76
    ("1I", "3:CDFGH"),     # M77
    ("2E", "2I"),          # M78
    ("1A", "3:CEFHI"),     # M79
    ("1L", "3:EHIJK"),     # M80
    ("1D", "3:BEFIJ"),     # M81
    ("1G", "3:AEHIJ"),     # M82
    ("2K", "2L"),          # M83
    ("1H", "2J"),          # M84
    ("1B", "3:EFGIJ"),     # M85
    ("1J", "2H"),          # M86
    ("1K", "3:DEIJL"),     # M87
    ("2D", "2G"),          # M88
]

# Round of 16: pairs of R32 match indices (0-based into the winners list).
R16 = [(1, 4), (0, 2), (3, 5), (6, 7), (10, 11), (8, 9), (13, 15), (12, 14)]
QF = [(0, 1), (4, 5), (2, 3), (6, 7)]
SF = [(0, 1), (2, 3)]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def dataset_name(team: str) -> str:
    return RENAMES.get(team, team)


def validate_teams(model) -> list[str]:
    """Check every 2026 team exists in the fitted model; return missing."""
    missing = []
    for teams in GROUPS.values():
        for t in teams:
            if dataset_name(t) not in model._idx:
                missing.append(t)
    return missing


def sample_score(model, home: str, away: str, neutral: bool,
                 rng: np.random.Generator) -> tuple[int, int]:
    """Draw one scoreline from the model's scoreline distribution."""
    m = model.score_matrix(dataset_name(home), dataset_name(away), neutral)
    flat = m.ravel()
    k = rng.choice(flat.size, p=flat / flat.sum())
    return divmod(int(k), m.shape[1])


def play_match(model, a: str, b: str, rng) -> tuple[int, int]:
    """Play one match between a and b (in that order), giving a host nation
    home advantage when one is involved. Returns (goals_a, goals_b)."""
    if dataset_name(a) in HOSTS:
        ga, gb = sample_score(model, a, b, neutral=False, rng=rng)
        return ga, gb
    if dataset_name(b) in HOSTS:
        gb, ga = sample_score(model, b, a, neutral=False, rng=rng)
        return ga, gb
    ga, gb = sample_score(model, a, b, neutral=True, rng=rng)
    return ga, gb


def knockout_winner(model, a: str, b: str, rng) -> str:
    """Single-elimination tie: 90 minutes, then extra time (~1/3 rates),
    then a penalty shootout (modeled as a coin flip)."""
    ga, gb = play_match(model, a, b, rng)
    if ga != gb:
        return a if ga > gb else b

    # extra time: approximate 30 minutes as independent Poisson at 1/3 rate
    neutral = (dataset_name(a) not in HOSTS) and (dataset_name(b) not in HOSTS)
    lam_a, lam_b = model._expected_goals(dataset_name(a), dataset_name(b),
                                         neutral)
    ea = rng.poisson(lam_a / 3.0)
    eb = rng.poisson(lam_b / 3.0)
    if ea != eb:
        return a if ea > eb else b

    # penalties: empirically close to a coin flip
    return a if rng.random() < 0.5 else b


# ----------------------------------------------------------------------------
# Group stage
# ----------------------------------------------------------------------------

def simulate_group(model, teams: list[str], rng) -> list[tuple]:
    """Round-robin one group. Returns standings as a sorted list of
    (team, points, goal_diff, goals_for)."""
    stats = {t: [0, 0, 0] for t in teams}  # [pts, gd, gf]
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = teams[i], teams[j]
            ga, gb = play_match(model, a, b, rng)
            stats[a][1] += ga - gb
            stats[b][1] += gb - ga
            stats[a][2] += ga
            stats[b][2] += gb
            if ga > gb:
                stats[a][0] += 3
            elif gb > ga:
                stats[b][0] += 3
            else:
                stats[a][0] += 1
                stats[b][0] += 1

    order = sorted(teams,
                   key=lambda t: (stats[t][0], stats[t][1], stats[t][2],
                                  rng.random()),
                   reverse=True)
    return [(t, *stats[t]) for t in order]


def assign_thirds(third_rows: list[tuple], rng) -> dict[str, str]:
    """Rank the 12 third-placed teams, keep the best 8, and assign them to
    the eight 3:xxx bracket slots respecting each slot's allowed groups.

    third_rows: list of (group_letter, team, pts, gd, gf).
    Returns {slot_key -> team} where slot_key is '3:xxx#<position>'.
    Greedy + backtracking; FIFA's published allocation table accomplishes
    the same feasibility goal deterministically.
    """
    ranked = sorted(third_rows,
                    key=lambda r: (r[2], r[3], r[4], rng.random()),
                    reverse=True)
    best8 = ranked[:8]
    group_of = {r[0]: r[1] for r in best8}

    slot_specs = []
    for slot_a, slot_b in R32:
        for spec in (slot_a, slot_b):
            if spec.startswith("3:"):
                slot_specs.append(spec)

    def backtrack(i: int, remaining: set, assign: dict):
        if i == len(slot_specs):
            return assign
        spec = slot_specs[i]
        allowed = [g for g in spec[2:] if g in remaining]
        rng.shuffle(allowed)
        for g in allowed:
            res = backtrack(i + 1, remaining - {g},
                            {**assign, f"{spec}#{i}": group_of[g]})
            if res is not None:
                return res
        return None

    result = backtrack(0, set(group_of), {})
    if result is None:
        # rare infeasible combination under this slot table: fall back to
        # rank-order assignment ignoring group constraints
        result = {f"{spec}#{i}": best8[i][1]
                  for i, spec in enumerate(slot_specs)}
    return result


# ----------------------------------------------------------------------------
# Full tournament
# ----------------------------------------------------------------------------

def simulate_tournament(model, n_sims: int = 5000,
                        seed: int = 42) -> pd.DataFrame:
    """Run the tournament n_sims times. Returns per-team probabilities of
    reaching each stage (r32, r16, qf, sf, final, champion)."""
    rng = np.random.default_rng(seed)
    stages = ["r32", "r16", "qf", "sf", "final", "champion"]
    counts = {t: defaultdict(int)
              for teams in GROUPS.values() for t in teams}

    for _ in range(n_sims):
        # ---- group stage --------------------------------------------------
        firsts, seconds, thirds = {}, {}, []
        for g, teams in GROUPS.items():
            table = simulate_group(model, teams, rng)
            firsts[g] = table[0][0]
            seconds[g] = table[1][0]
            thirds.append((g, table[2][0], table[2][1], table[2][2],
                           table[2][3]))

        third_assign = assign_thirds(thirds, rng)

        # ---- resolve R32 pairings ------------------------------------------
        slot_counter = 0
        r32_pairs = []
        for slot_a, slot_b in R32:
            pair = []
            for spec in (slot_a, slot_b):
                if spec[0] == "1":
                    pair.append(firsts[spec[1]])
                elif spec[0] == "2":
                    pair.append(seconds[spec[1]])
                else:
                    pair.append(third_assign[f"{spec}#{slot_counter}"])
                    slot_counter += 1
            r32_pairs.append(tuple(pair))

        for a, b in r32_pairs:
            counts[a]["r32"] += 1
            counts[b]["r32"] += 1

        # ---- knockouts ------------------------------------------------------
        w32 = [knockout_winner(model, a, b, rng) for a, b in r32_pairs]

        r16_pairs = [(w32[i], w32[j]) for i, j in R16]
        for a, b in r16_pairs:
            counts[a]["r16"] += 1
            counts[b]["r16"] += 1
        w16 = [knockout_winner(model, a, b, rng) for a, b in r16_pairs]

        qf_pairs = [(w16[i], w16[j]) for i, j in QF]
        for a, b in qf_pairs:
            counts[a]["qf"] += 1
            counts[b]["qf"] += 1
        w8 = [knockout_winner(model, a, b, rng) for a, b in qf_pairs]

        sf_pairs = [(w8[i], w8[j]) for i, j in SF]
        for a, b in sf_pairs:
            counts[a]["sf"] += 1
            counts[b]["sf"] += 1
        w4 = [knockout_winner(model, a, b, rng) for a, b in sf_pairs]

        counts[w4[0]]["final"] += 1
        counts[w4[1]]["final"] += 1
        champion = knockout_winner(model, w4[0], w4[1], rng)
        counts[champion]["champion"] += 1

    rows = [[t] + [counts[t][s] / n_sims for s in stages] for t in counts]
    out = pd.DataFrame(rows, columns=["team"] + stages)
    return out.sort_values("champion", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from pathlib import Path
    from src.data import load_results, filter_matches
    from src.dixon_coles import DixonColes

    df = filter_matches(load_results(), since="2010-01-01")
    print(f"Fitting Dixon-Coles on {len(df):,} matches ...")
    model = DixonColes().fit(df)

    missing = validate_teams(model)
    if missing:
        print("\n!! These 2026 teams were NOT found in the dataset — fix the")
        print("   RENAMES mapping in src/simulate.py before trusting results:")
        for t in missing:
            print(f"   - {t}")
        raise SystemExit(1)

    print("All 48 team names validated against the dataset.\n")
    print("Simulating the 2026 World Cup 5,000 times (this takes a while)...")
    table = simulate_tournament(model, n_sims=5000)

    pct = (table.set_index("team") * 100).round(1)
    print("\nTop 15 — probability (%) of reaching each stage:\n")
    print(pct.head(15).to_string())

    Path("reports").mkdir(exist_ok=True)
    pct.to_csv("reports/simulation_2026.csv")
    print("\nFull table saved to reports/simulation_2026.csv")
