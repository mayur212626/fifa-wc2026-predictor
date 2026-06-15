"""
test_models.py — correctness tests for the forecasting pipeline.

These check the properties that must always hold regardless of the data:
probabilities are valid, the models reject nonsense input, scoreline matrices
are proper distributions, and the simulator returns a coherent tournament.

A small Dixon-Coles model is fit ONCE (session-scoped fixture) and shared
across tests, because fitting is the slow part. Tests use a short training
window to keep that fit fast.

Run from the project root:
    pytest -q
"""

import numpy as np
import pytest

from src.data import (load_results, filter_matches,
                      train_test_split_by_date, get_teams)
from src.elo import EloModel
from src.dixon_coles import DixonColes


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def matches():
    """Recent-window match data, loaded once for the whole test session."""
    return filter_matches(load_results(), since="2018-01-01")


@pytest.fixture(scope="session")
def dc_model(matches):
    """A fitted Dixon-Coles model, shared across tests (fitting is slow)."""
    return DixonColes().fit(matches)


@pytest.fixture(scope="session")
def elo_model(matches):
    return EloModel().fit(matches)


# a pair of strong teams that are certain to exist in any recent window
PAIR = ("Brazil", "France")


# ----------------------------------------------------------------------------
# Data layer
# ----------------------------------------------------------------------------

def test_data_loads_and_is_clean(matches):
    assert len(matches) > 1000
    # required columns present
    for col in ["date", "home_team", "away_team", "home_score",
                "away_score", "neutral"]:
        assert col in matches.columns
    # no missing scores survived cleaning
    assert matches["home_score"].notna().all()
    assert matches["away_score"].notna().all()
    # scores are non-negative integers
    assert (matches["home_score"] >= 0).all()
    assert (matches["away_score"] >= 0).all()


def test_chronological_split_has_no_leakage(matches):
    cutoff = "2022-11-20"
    train, test = train_test_split_by_date(matches, cutoff)
    assert train["date"].max() < test["date"].min()
    assert len(train) + len(test) == len(matches)


def test_get_teams_is_sorted_and_unique(matches):
    teams = get_teams(matches)
    assert teams == sorted(teams)
    assert len(teams) == len(set(teams))


# ----------------------------------------------------------------------------
# Probability axioms (the core property of any forecaster)
# ----------------------------------------------------------------------------

def test_dixon_coles_probs_sum_to_one(dc_model):
    p = dc_model.predict(*PAIR, neutral=True)
    total = p["p_home"] + p["p_draw"] + p["p_away"]
    assert total == pytest.approx(1.0, abs=1e-6)


def test_dixon_coles_probs_are_valid(dc_model):
    p = dc_model.predict(*PAIR, neutral=True)
    for key in ("p_home", "p_draw", "p_away"):
        assert 0.0 <= p[key] <= 1.0


def test_elo_probs_sum_to_one(elo_model):
    p = elo_model.match_probs(*PAIR, neutral=True)
    total = p["p_home"] + p["p_draw"] + p["p_away"]
    assert total == pytest.approx(1.0, abs=1e-6)


def test_score_matrix_is_a_distribution(dc_model):
    m = dc_model.score_matrix(*PAIR, neutral=True)
    assert m.shape == (dc_model.max_goals + 1, dc_model.max_goals + 1)
    assert (m >= 0).all()                       # no negative probabilities
    assert m.sum() == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------------------
# Sanity of the modelled quantities
# ----------------------------------------------------------------------------

def test_expected_goals_are_reasonable(dc_model):
    p = dc_model.predict(*PAIR, neutral=True)
    # international matches: expected goals should sit in a believable band
    assert 0.2 < p["xg_home"] < 5.0
    assert 0.2 < p["xg_away"] < 5.0


def test_home_advantage_helps(dc_model):
    """Playing at home should not lower a team's win probability."""
    home = dc_model.predict(*PAIR, neutral=False)["p_home"]
    neutral = dc_model.predict(*PAIR, neutral=True)["p_home"]
    assert home >= neutral - 1e-9


def test_stronger_team_is_favoured(dc_model):
    """A top side vs a weak side: the favourite's win prob should dominate."""
    teams = set(dc_model.teams)
    weak = next((t for t in ("San Marino", "Gibraltar", "Liechtenstein",
                             "Andorra") if t in teams), None)
    if weak is None:
        pytest.skip("no clearly-weak team available in this dataset window")
    p = dc_model.predict("Brazil", weak, neutral=True)
    assert p["p_home"] > p["p_away"]


# ----------------------------------------------------------------------------
# Error handling
# ----------------------------------------------------------------------------

def test_unknown_team_raises(dc_model):
    with pytest.raises(KeyError):
        dc_model.predict("Brazil", "Wakanda", neutral=True)


def test_unfitted_model_raises():
    fresh = DixonColes()
    with pytest.raises(RuntimeError):
        fresh.predict(*PAIR, neutral=True)


# ----------------------------------------------------------------------------
# Simulator
# ----------------------------------------------------------------------------

def test_all_2026_teams_known_to_model(dc_model):
    """Every team in the 2026 field must resolve to a name the model knows —
    otherwise the simulation would crash mid-run."""
    from src.simulate import validate_teams
    # validate against a full-window model to match production fitting
    full = DixonColes().fit(filter_matches(load_results(), since="2010-01-01"))
    missing = validate_teams(full)
    assert missing == [], f"unmapped 2026 teams: {missing}"


def test_simulation_probabilities_are_coherent():
    """A short simulation should yield valid per-team probabilities with
    champion <= final <= ... <= r32 for every team."""
    from src.simulate import simulate_tournament
    model = DixonColes().fit(filter_matches(load_results(), since="2010-01-01"))
    table = simulate_tournament(model, n_sims=200, seed=1)

    stages = ["r32", "r16", "qf", "sf", "final", "champion"]
    # all probabilities in [0, 1]
    assert ((table[stages] >= 0) & (table[stages] <= 1)).all().all()
    # exactly one champion per simulation -> champion column sums to ~1
    assert table["champion"].sum() == pytest.approx(1.0, abs=1e-6)
    # monotonic: reaching a later stage is never more likely than an earlier one
    for i in range(len(stages) - 1):
        later, earlier = stages[i + 1], stages[i]
        assert (table[later] <= table[earlier] + 1e-9).all()
