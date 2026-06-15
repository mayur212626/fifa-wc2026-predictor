"""
main.py — FastAPI service for the 2026 World Cup forecaster.

Exposes the model over a small REST API so predictions can be consumed
programmatically (by the dashboard, a notebook, or any client):

  GET  /                      -> service info
  GET  /health                -> liveness check (for Render/uptime monitors)
  GET  /teams                 -> every team the model knows, with ratings
  GET  /predict?home=..&away=..&neutral=true
                              -> win/draw/loss probs + expected goals
  GET  /odds?top=15           -> current title odds from the latest sim run

The Dixon-Coles model is fit ONCE at startup (it takes a minute) and reused
for every request — fitting per-request would be far too slow.

Run locally:
    uvicorn api.main:app --reload
Then open the interactive docs at http://127.0.0.1:8000/docs
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from src.data import load_results, filter_matches
from src.dixon_coles import DixonColes

REPORTS = Path(__file__).resolve().parents[1] / "reports"
TRAIN_SINCE = "2010-01-01"

# module-level holder for the fitted model and metadata
STATE: dict = {"model": None, "trained_through": None, "n_matches": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fit the model once when the service boots."""
    print("Loading data and fitting Dixon-Coles model (one-time startup)...")
    df = filter_matches(load_results(), since=TRAIN_SINCE)
    model = DixonColes().fit(df)
    STATE["model"] = model
    STATE["trained_through"] = str(df.date.max().date())
    STATE["n_matches"] = int(len(df))
    print(f"Model ready: {STATE['n_matches']:,} matches "
          f"through {STATE['trained_through']}")
    yield
    STATE.clear()


app = FastAPI(
    title="World Cup 2026 Forecaster",
    description="Dixon-Coles bivariate-Poisson match & tournament forecasts.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_model() -> DixonColes:
    model = STATE.get("model")
    if model is None:
        raise HTTPException(status_code=503,
                            detail="Model still warming up. Try again shortly.")
    return model


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "World Cup 2026 Forecaster",
        "model": "Dixon-Coles bivariate Poisson",
        "trained_through": STATE.get("trained_through"),
        "n_training_matches": STATE.get("n_matches"),
        "endpoints": ["/health", "/teams", "/predict", "/odds", "/docs"],
    }


@app.get("/health")
def health():
    """Liveness probe — returns ok once the model has finished loading."""
    ready = STATE.get("model") is not None
    return {"status": "ok" if ready else "warming_up", "model_ready": ready}


@app.get("/teams")
def teams():
    """List every team the model can predict, with attack/defense ratings."""
    model = get_model()
    ratings = model.team_ratings()
    return {
        "count": len(ratings),
        "teams": ratings.to_dict(orient="records"),
    }


@app.get("/predict")
def predict(
    home: str = Query(..., description="Home (or first) team name"),
    away: str = Query(..., description="Away (or second) team name"),
    neutral: bool = Query(True, description="True for a neutral venue"),
):
    """Predict a single match. Team names must match the dataset spelling
    (see /teams). Returns win/draw/loss probabilities and expected goals."""
    model = get_model()
    try:
        pred = model.predict(home, away, neutral=neutral)
    except KeyError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown team {e}. Check spelling against /teams.",
        )
    return {
        "home": home,
        "away": away,
        "neutral": neutral,
        "probabilities": {
            "home_win": round(pred["p_home"], 4),
            "draw": round(pred["p_draw"], 4),
            "away_win": round(pred["p_away"], 4),
        },
        "expected_goals": {
            "home": round(pred["xg_home"], 3),
            "away": round(pred["xg_away"], 3),
        },
    }


@app.get("/odds")
def odds(top: int = Query(15, ge=1, le=48, description="How many teams")):
    """Current tournament title odds, read from the latest simulation output.
    Run `python -m src.update` (or src.simulate) to refresh the underlying CSV."""
    # prefer the live history snapshot, fall back to the frozen forecast
    hist_file = REPORTS / "odds_history.csv"
    sim_file = REPORTS / "simulation_2026.csv"

    if hist_file.exists():
        df = pd.read_csv(hist_file)
        df["ts"] = pd.to_datetime(
            df["run_ts"].str.replace(" UTC", "", regex=False), errors="coerce")
        df = df[df["ts"] == df["ts"].max()].drop(columns=["run_ts", "ts"])
        # history is stored as 0-1 proportions -> percent
        for c in ["r32", "r16", "qf", "sf", "final", "champion"]:
            if c in df:
                df[c] = (df[c] * 100).round(1)
        source = "live snapshot"
    elif sim_file.exists():
        df = pd.read_csv(sim_file).rename(columns={"Unnamed: 0": "team"})
        source = "frozen forecast"
    else:
        raise HTTPException(
            status_code=404,
            detail="No odds available yet. Run `python -m src.simulate` first.",
        )

    df = df.sort_values("champion", ascending=False).head(top)
    return {
        "source": source,
        "title_odds": df.to_dict(orient="records"),
    }
