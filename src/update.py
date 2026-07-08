"""
update.py — daily update loop while the 2026 World Cup is running.

Each run does five things:
  1. Re-downloads the latest results (the martj42 repo updates during
     tournaments) plus shootouts.csv for penalty winners.
  2. Refits the Dixon-Coles model on everything through today.
  3. Re-simulates the tournament, CONDITIONED on what has actually
     happened: played matches use their real scores (and real shootout
     winners); only the remaining matches are sampled from the model.
  4. Appends a snapshot of every team's stage probabilities to
     reports/odds_history.csv — so you can chart how title odds moved
     match by match across the tournament.
  5. Logs the model's win/draw/loss prediction for every not-yet-played
     group fixture to reports/predictions_log.csv, and scores earlier
     logged predictions once their matches have been played (running
     Brier / log-loss against reality).

Run it once a day (or after each matchday):
    python -m src.update
"""

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from src.data import load_results, filter_matches
from src.dixon_coles import DixonColes
from src.evaluate import brier_score, log_loss, outcome_index
from src.simulate import (GROUPS, R32, R16, QF, SF, HOSTS,
                          dataset_name, validate_teams,
                          play_match, knockout_winner, assign_thirds)

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
REPORTS = Path(__file__).resolve().parents[1] / "reports"
BASE = "https://raw.githubusercontent.com/martj42/international_results/master"

WC_START = pd.Timestamp("2026-06-11")
N_SIMS = 5000


# ----------------------------------------------------------------------------
# 1. Refresh data
# ----------------------------------------------------------------------------

def refresh_data() -> None:
    for fname in ("results.csv", "shootouts.csv"):
        try:
            r = requests.get(f"{BASE}/{fname}", timeout=60)
            r.raise_for_status()
            (RAW / fname).write_bytes(r.content)
            print(f"Refreshed {fname}")
        except requests.RequestException as e:
            print(f"[warning] could not refresh {fname}: {e} — "
                  f"using the existing local copy")


# ----------------------------------------------------------------------------
# 2. What has actually happened so far
# ----------------------------------------------------------------------------

def played_2026(df: pd.DataFrame) -> pd.DataFrame:
    """All 2026 World Cup matches already in the dataset."""
    return df[(df.tournament == "FIFA World Cup")
              & (df.date >= WC_START)].reset_index(drop=True)


def build_lookups(played: pd.DataFrame):
    """Played-fixture lookup keyed by the unordered team pair (dataset
    names), plus penalty-shootout winners for knockout draws."""
    results = {}
    for row in played.itertuples(index=False):
        key = frozenset((row.home_team, row.away_team))
        results[key] = (row.home_team, int(row.home_score),
                        int(row.away_score))

    shootout_winner = {}
    shoot_path = RAW / "shootouts.csv"
    if shoot_path.exists():
        sh = pd.read_csv(shoot_path, parse_dates=["date"])
        sh = sh[sh.date >= WC_START]
        for row in sh.itertuples(index=False):
            shootout_winner[frozenset((row.home_team, row.away_team))] = \
                row.winner
    return results, shootout_winner


def real_third_assignment(played: pd.DataFrame, results: dict):
    """Once every group match AND the Round of 32 have been played, read the
    ACTUAL R32 pairings out of the results and pin the third-place slot
    assignment to reality.

    Why: pre-tournament we approximated FIFA's third-place allocation table
    with a random feasible assignment. That was fine when everything was
    simulated, but once the real bracket exists, random assignment lets
    simulations diverge into brackets that never happened — which gives
    already-eliminated teams phantom odds. Pinning to the real pairings
    fixes that: eliminated teams drop to zero because every simulation now
    replays the true bracket.

    Returns {slot_key: official_team_name} or None if the R32 isn't fully
    known yet (in which case the caller falls back to random assignment).
    """
    # every intra-group fixture, keyed by unordered dataset-name pair
    group_keys = set()
    for teams in GROUPS.values():
        for i in range(4):
            for j in range(i + 1, 4):
                group_keys.add(frozenset((dataset_name(teams[i]),
                                          dataset_name(teams[j]))))

    is_group = played.apply(
        lambda r: frozenset((r.home_team, r.away_team)) in group_keys, axis=1)
    group_played = played[is_group]
    knockout = played[~is_group].sort_values("date")

    if len(group_played) < 72 or len(knockout) < 16:
        return None                       # real bracket not fully known yet

    r32_matches = knockout.head(16)

    # real group winners (all group fixtures are played, so cond_group never
    # touches the model — passing None is safe) and the set of thirds
    rng = np.random.default_rng(0)
    firsts, third_teams = {}, set()
    for g, teams in GROUPS.items():
        table = cond_group(None, teams, results, rng)
        firsts[g] = table[0][0]
        third_teams.add(dataset_name(table[2][0]))

    ds_to_official = {dataset_name(t): t
                      for ts in GROUPS.values() for t in ts}

    assign, slot_i = {}, 0
    for sa, sb in R32:
        for spec in (sa, sb):
            if not spec.startswith("3:"):
                continue
            # every third slot is paired with a group winner ("1X")
            partner = sb if spec == sa else sa
            anchor_ds = dataset_name(firsts[partner[1]])
            opp_ds = None
            for row in r32_matches.itertuples(index=False):
                if anchor_ds in (row.home_team, row.away_team):
                    opp_ds = (row.away_team if row.home_team == anchor_ds
                              else row.home_team)
                    break
            if opp_ds is None or opp_ds not in third_teams:
                return None               # bracket looks inconsistent; bail
            assign[f"{spec}#{slot_i}"] = ds_to_official[opp_ds]
            slot_i += 1
    return assign


# ----------------------------------------------------------------------------
# 3. Conditioned simulation
# ----------------------------------------------------------------------------

def cond_match(model, a: str, b: str, results: dict, rng):
    """Real score if the fixture has been played, else sample the model.
    Returns (goals_a, goals_b, was_real)."""
    key = frozenset((dataset_name(a), dataset_name(b)))
    if key in results:
        home, hg, ag = results[key]
        if home == dataset_name(a):
            return hg, ag, True
        return ag, hg, True
    ga, gb = play_match(model, a, b, rng)
    return ga, gb, False


def cond_group(model, teams, results, rng):
    stats = {t: [0, 0, 0] for t in teams}  # pts, gd, gf
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = teams[i], teams[j]
            ga, gb, _ = cond_match(model, a, b, results, rng)
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


def cond_knockout(model, a, b, results, shootouts, rng):
    key = frozenset((dataset_name(a), dataset_name(b)))
    if key in results:
        home, hg, ag = results[key]
        if hg != ag:
            winner_ds = home if hg > ag else \
                (dataset_name(b) if home == dataset_name(a)
                 else dataset_name(a))
            return a if dataset_name(a) == winner_ds else b
        # draw on the night -> decided on penalties
        if key in shootouts:
            return a if dataset_name(a) == shootouts[key] else b
        return a if rng.random() < 0.5 else b
    return knockout_winner(model, a, b, rng)


def simulate_remaining(model, results, shootouts,
                       n_sims=N_SIMS, seed=42,
                       fixed_thirds: dict | None = None) -> pd.DataFrame:
    """Same tournament loop as src.simulate, but every fixture checks the
    real-results lookup first. If fixed_thirds is given (the real bracket,
    from real_third_assignment), it replaces the random slot assignment."""
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    stages = ["r32", "r16", "qf", "sf", "final", "champion"]
    counts = {t: defaultdict(int) for ts in GROUPS.values() for t in ts}

    for _ in range(n_sims):
        firsts, seconds, thirds = {}, {}, []
        for g, teams in GROUPS.items():
            table = cond_group(model, teams, results, rng)
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
        w32 = [cond_knockout(model, a, b, results, shootouts, rng)
               for a, b in r32_pairs]

        r16_pairs = [(w32[i], w32[j]) for i, j in R16]
        for a, b in r16_pairs:
            counts[a]["r16"] += 1
            counts[b]["r16"] += 1
        w16 = [cond_knockout(model, a, b, results, shootouts, rng)
               for a, b in r16_pairs]

        qf_pairs = [(w16[i], w16[j]) for i, j in QF]
        for a, b in qf_pairs:
            counts[a]["qf"] += 1
            counts[b]["qf"] += 1
        w8 = [cond_knockout(model, a, b, results, shootouts, rng)
              for a, b in qf_pairs]

        sf_pairs = [(w8[i], w8[j]) for i, j in SF]
        for a, b in sf_pairs:
            counts[a]["sf"] += 1
            counts[b]["sf"] += 1
        w4 = [cond_knockout(model, a, b, results, shootouts, rng)
              for a, b in sf_pairs]

        counts[w4[0]]["final"] += 1
        counts[w4[1]]["final"] += 1
        champ = cond_knockout(model, w4[0], w4[1], results, shootouts, rng)
        counts[champ]["champion"] += 1

    rows = [[t] + [counts[t][s] / n_sims for s in stages] for t in counts]
    return (pd.DataFrame(rows, columns=["team"] + stages)
            .sort_values("champion", ascending=False)
            .reset_index(drop=True))


# ----------------------------------------------------------------------------
# 4/5. Logging: odds history + match-level predictions and scoring
# ----------------------------------------------------------------------------

def append_odds_history(table: pd.DataFrame, run_ts: str) -> None:
    path = REPORTS / "odds_history.csv"
    snap = table.copy()
    snap.insert(0, "run_ts", run_ts)
    snap.to_csv(path, mode="a", header=not path.exists(), index=False)
    print(f"Appended snapshot to {path.name}")


def log_group_predictions(model, results: dict, run_ts: str) -> None:
    """Log model probabilities for every group fixture not yet played."""
    path = REPORTS / "predictions_log.csv"
    rows = []
    for teams in GROUPS.values():
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = teams[i], teams[j]
                if frozenset((dataset_name(a), dataset_name(b))) in results:
                    continue
                # host plays home, mirroring how the match will be recorded
                if dataset_name(b) in HOSTS and dataset_name(a) not in HOSTS:
                    a, b = b, a
                neutral = dataset_name(a) not in HOSTS
                p = model.predict(dataset_name(a), dataset_name(b),
                                  neutral=neutral)
                rows.append([run_ts, dataset_name(a), dataset_name(b),
                             round(p["p_home"], 4), round(p["p_draw"], 4),
                             round(p["p_away"], 4)])
    if rows:
        out = pd.DataFrame(rows, columns=["run_ts", "home", "away",
                                          "p_home", "p_draw", "p_away"])
        out.to_csv(path, mode="a", header=not path.exists(), index=False)
        print(f"Logged {len(rows)} upcoming-fixture predictions")


def score_logged_predictions(played: pd.DataFrame, run_ts: str) -> None:
    """Score the most recent pre-match prediction for each played fixture,
    print the running Brier/log-loss, and append them to score_history.csv so
    the dashboard can chart how the model's live accuracy evolves."""
    path = REPORTS / "predictions_log.csv"
    if not path.exists() or played.empty:
        return
    log = pd.read_csv(path, parse_dates=["run_ts"])

    probs, outs = [], []
    for row in played.itertuples(index=False):
        cutoff = pd.Timestamp(row.date + pd.Timedelta(days=1), tz="UTC")
        m = log[(log.home == row.home_team) & (log.away == row.away_team)
                & (log.run_ts < cutoff)]
        if m.empty:
            continue
        last = m.sort_values("run_ts").iloc[-1]
        probs.append([last.p_home, last.p_draw, last.p_away])
        outs.append(outcome_index(int(row.home_score), int(row.away_score)))

    if probs:
        probs, outs = np.array(probs), np.array(outs)
        brier = brier_score(probs, outs)
        ll = log_loss(probs, outs)
        print(f"\nRunning score on {len(outs)} played matches the model "
              f"predicted in advance:")
        print(f"  Brier:    {brier:.4f}  (uniform = 0.6667)")
        print(f"  Log-loss: {ll:.4f}  (uniform = 1.0986)")

        # append to a history file so the dashboard can chart accuracy over time
        score_path = REPORTS / "score_history.csv"
        snap = pd.DataFrame([{
            "run_ts": run_ts,
            "n_matches": len(outs),
            "brier": round(float(brier), 4),
            "logloss": round(float(ll), 4),
            "uniform_brier": 0.6667,
            "uniform_logloss": 1.0986,
        }])
        snap.to_csv(score_path, mode="a", header=not score_path.exists(),
                    index=False)
        print(f"Logged score to {score_path.name}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"=== Update run: {run_ts} ===\n")

    refresh_data()
    df = load_results()
    played = played_2026(df)
    print(f"\n2026 World Cup matches played so far: {len(played)}")

    train = filter_matches(df, since="2010-01-01")
    print(f"Refitting Dixon-Coles on {len(train):,} matches "
          f"(through {train.date.max().date()}) ...")
    model = DixonColes().fit(train)

    missing = validate_teams(model)
    if missing:
        raise SystemExit(f"Team name mapping broken for: {missing}")

    results, shootouts = build_lookups(played)
    fixed = real_third_assignment(played, results)
    if fixed is not None:
        print("Round of 32 fully known — pinning simulations to the real "
              "bracket (eliminated teams will show 0).")
    print(f"Conditioning on {len(results)} real results. "
          f"Simulating the rest {N_SIMS:,} times ...")
    table = simulate_remaining(model, results, shootouts, fixed_thirds=fixed)

    pct = (table.set_index("team") * 100).round(1)
    print("\nCurrent title odds (top 10):\n")
    print(pct.head(10).to_string())

    REPORTS.mkdir(exist_ok=True)
    append_odds_history(table, run_ts)
    score_logged_predictions(played, run_ts)
    log_group_predictions(model, results, run_ts)

    print("\nDone. Run this again after the next matchday.")
