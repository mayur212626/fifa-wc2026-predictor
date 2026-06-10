# FIFA World Cup 2026 - Forecasting & Simulation

> [One or two sentences in your own words: what this project does, and that
>  it is a pre-tournament forecast made before kickoff.]

## Motivation
[Why you picked this and what question you are answering.]

## Approach
[The pipeline in your words: Elo team strength, Dixon-Coles bivariate
 Poisson scoreline model, the Bayesian (PyMC) upgrade with partial pooling
 and posterior uncertainty, and the Monte Carlo simulation of the 48-team
 bracket.]

## Key Result
[Fill in AFTER you have numbers. Lead with calibration, not accuracy, e.g.
 "Backtested on 2018 & 2022, Brier score X vs Y for the bookmaker baseline."
 Add the 2026 title-odds table here.]

## Data
- Match history: International football results (1872-present), Mart Jurisoo.
  Pulled via data/download_data.py.
- FIFA rankings: [source]
- Historical odds (2018/2022): [source]

## How to Run
## Validation
[Out-of-sample backtest on past World Cups, proper scoring rules (Brier,
 log-loss), calibration curves, and the betting-market benchmark.]

## Limitations & Honest Notes
[Football is high-variance; the model cannot capture injuries, form, or
 tactics; parameter uncertainty is real. Being candid here builds credibility.]

## License
[Your choice - MIT is common for portfolio repos.]
