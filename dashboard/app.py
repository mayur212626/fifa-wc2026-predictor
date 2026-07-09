"""
app.py — live dashboard for the 2026 World Cup forecast.

Three tabs, each answering one question:
  * Title Race        — who's winning, right now?
  * Model Lab         — is the model any good, and what did the Bayesian
                        upgrade change?
  * The Forecast Story— what did I predict before kickoff, and how did
                        reality treat it?

Reads only the CSVs the pipeline commits (odds_history, score_history,
simulation_2026, model_comparison, plus the calibration plot), so the
deployed app needs no model stack.

Run from the project root:  streamlit run dashboard/app.py
"""

from pathlib import Path
import pandas as pd
import altair as alt
import streamlit as st

# ----------------------------------------------------------------------------
# Setup & palette
# ----------------------------------------------------------------------------

REPORTS = Path(__file__).resolve().parents[1] / "reports"
SIM_FILE = REPORTS / "simulation_2026.csv"
HIST_FILE = REPORTS / "odds_history.csv"
SCORE_FILE = REPORTS / "score_history.csv"
COMP_FILE = REPORTS / "model_comparison.csv"
SCORECARD_FILE = REPORTS / "match_scorecard.csv"
CALIB_IMG = REPORTS / "calibration.png"

STAGES = ["r32", "r16", "qf", "sf", "final", "champion"]
STAGE_LABELS = {"r32": "R32", "r16": "R16", "qf": "QF",
                "sf": "SF", "final": "Final", "champion": "Champion"}

BG = "#0B0E14"
GRID = "#222A38"
TEXT = "#E6E9EF"
MUTED = "#8A93A6"
GOLD = "#F2C14E"
BLUE = "#4EA8DE"
RED = "#E4572E"

st.set_page_config(page_title="World Cup 2026 — Title Race",
                   page_icon="🏆", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&display=swap');
      [class*="css"], .stMarkdown, p, span, div { font-feature-settings: "tnum"; }
      .hero { padding: .4rem 0 1.1rem 0; margin-bottom: .6rem;
              border-bottom: 1px solid #222A38; }
      .hero .rule { height: 3px; width: 64px; background: #F2C14E;
              border-radius: 3px; margin-bottom: .9rem; }
      .hero .eyebrow { font-family:'Space Grotesk',sans-serif; letter-spacing:.24em;
              font-size:.7rem; font-weight:700; color:#F2C14E; text-transform:uppercase; }
      .hero h1 { font-family:'Space Grotesk',sans-serif; font-weight:700;
              font-size:2.7rem; line-height:1.04; margin:.35rem 0 .5rem 0; color:#F4F6FA; }
      .hero .sub { color:#8A93A6; font-size:1rem; margin:0; max-width:64ch; }
      h2, h3 { font-family:'Space Grotesk',sans-serif !important; color:#E6E9EF; }
      [data-testid="stMetric"] { background:#141925; border:1px solid #222A38;
              border-radius:14px; padding:16px 18px; }
      [data-testid="stMetricValue"] { font-family:'Space Grotesk',sans-serif; color:#F4F6FA; }
      [data-testid="stMetricLabel"] { color:#8A93A6; }
      .stCaption, .stCaption p { color:#6E778A !important; }
      button[data-baseweb="tab"] { font-family:'Space Grotesk',sans-serif;
              font-weight:600; font-size:1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_chart(chart):
    return (chart
            .configure(background="transparent")
            .configure_view(strokeWidth=0, fill="transparent")
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID,
                            domainColor=GRID, tickColor=GRID,
                            labelFontSize=12, titleFontSize=12)
            .configure_legend(labelColor=TEXT, titleColor=MUTED))


# ----------------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------------

def _to_percent(df):
    df = df.copy()
    cols = [c for c in STAGES if c in df.columns]
    if cols and df[cols].to_numpy().max() <= 1.5:
        df[cols] = df[cols] * 100.0
    return df


@st.cache_data(ttl=600)
def load_simulation():
    if not SIM_FILE.exists():
        return None
    df = pd.read_csv(SIM_FILE)
    return _to_percent(df.rename(columns={df.columns[0]: "team"}))


@st.cache_data(ttl=600)
def load_history():
    if not HIST_FILE.exists():
        return None
    df = _to_percent(pd.read_csv(HIST_FILE))
    df["ts"] = pd.to_datetime(df["run_ts"].str.replace(" UTC", "", regex=False),
                              errors="coerce")
    return df.dropna(subset=["ts"])


@st.cache_data(ttl=600)
def load_scores():
    if not SCORE_FILE.exists():
        return None
    df = pd.read_csv(SCORE_FILE)
    df["ts"] = pd.to_datetime(df["run_ts"].str.replace(" UTC", "", regex=False),
                              errors="coerce")
    return df.dropna(subset=["ts"]).sort_values("ts")


@st.cache_data(ttl=600)
def load_comparison():
    if not COMP_FILE.exists():
        return None
    df = pd.read_csv(COMP_FILE)
    return df.rename(columns={df.columns[0]: "team"})


@st.cache_data(ttl=600)
def load_scorecard():
    if not SCORECARD_FILE.exists():
        return None
    return pd.read_csv(SCORECARD_FILE)


sim = load_simulation()
hist = load_history()
scores = load_scores()
comp = load_comparison()
card = load_scorecard()

if hist is not None and not hist.empty:
    latest_ts = hist["ts"].max()
    current = hist[hist["ts"] == latest_ts].copy()
    source_note = f"Live snapshot · {latest_ts:%b %d, %Y · %H:%M} UTC"
elif sim is not None:
    current = sim.copy()
    source_note = "Frozen pre-tournament forecast"
else:
    current, source_note = None, ""

if current is not None:
    current = (current.sort_values(STAGES[::-1], ascending=False)
               .reset_index(drop=True))


# ----------------------------------------------------------------------------
# Header + top metrics
# ----------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
      <div class="rule"></div>
      <div class="eyebrow">Live forecast · model-driven · auto-updating</div>
      <h1>World Cup 2026 — Title Race</h1>
      <p class="sub">A Dixon-Coles bivariate-Poisson goal model, re-fit and
      re-simulated 5,000 times after every matchday. Played matches lock in
      their real results; only the remaining bracket is simulated.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if current is None:
    st.error("No forecast data found. Run `python -m src.simulate`, then "
             "`python -m src.update`.")
    st.stop()

fav = current.iloc[0]
alive = int((current["champion"] > 0).sum())
n_snapshots = hist["ts"].nunique() if hist is not None else 0
c1, c2, c3, c4 = st.columns(4)
c1.metric("🏆 Favorite", fav["team"], f"{fav['champion']:.1f}% to win")
c2.metric("Chasing", current.iloc[1]["team"],
          f"{current.iloc[1]['champion']:.1f}%")
c3.metric("Still alive", f"{alive}", "teams with a path to the title")
c4.metric("Snapshots logged", f"{n_snapshots}", "matchday updates")
st.caption(source_note)

tab_race, tab_lab, tab_story = st.tabs(
    ["🏆 Title Race", "🔬 Model Lab", "📜 The Forecast Story"])


# ============================================================================
# TAB 1 — TITLE RACE
# ============================================================================

with tab_race:
    st.subheader("Title odds right now")
    top_n = st.slider("Teams to show", 6, 24, 10)
    bars_df = current.head(top_n)

    bar = (
        alt.Chart(bars_df)
        .mark_bar(color=GOLD, cornerRadiusEnd=3, height=18)
        .encode(
            x=alt.X("champion:Q", title="Probability of winning the cup (%)"),
            y=alt.Y("team:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("team:N", title="Team"),
                     alt.Tooltip("champion:Q", title="Champion %", format=".1f"),
                     alt.Tooltip("final:Q", title="Reach final %", format=".1f"),
                     alt.Tooltip("sf:Q", title="Reach semi %", format=".1f")],
        )
        .properties(height=30 * len(bars_df))
    )
    st.altair_chart(style_chart(bar), width='stretch')

    st.subheader("How far each team is likely to go")
    st.caption("Probability (%) of reaching each stage — brighter gold = "
               "more likely. Hard 0s and 100s are real results, locked in.")

    heat_src = (current.head(16)[["team"] + STAGES]
                .melt(id_vars="team", var_name="stage", value_name="prob"))
    heat_src["stage"] = heat_src["stage"].map(STAGE_LABELS)
    team_order = current.head(16)["team"].tolist()
    stage_order = [STAGE_LABELS[s] for s in STAGES]

    base = alt.Chart(heat_src).encode(
        x=alt.X("stage:N", sort=stage_order, title=None,
                axis=alt.Axis(orient="top", labelAngle=0,
                              labelFontWeight="bold")),
        y=alt.Y("team:N", sort=team_order, title=None),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("prob:Q", scale=alt.Scale(range=["#0E1320", GOLD]),
                        legend=alt.Legend(title="%")),
        tooltip=[alt.Tooltip("team:N", title="Team"),
                 alt.Tooltip("stage:N", title="Stage"),
                 alt.Tooltip("prob:Q", title="Probability %", format=".1f")],
    )
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text("prob:Q", format=".0f"),
        color=alt.condition("datum.prob > 45", alt.value(BG),
                            alt.value("#AEB6C4")),
    )
    st.altair_chart(
        style_chart((heat + text).properties(height=34 * len(team_order))),
        width='stretch')

    st.subheader("How the odds have moved")
    if hist is None or hist["ts"].nunique() < 2:
        st.info("Run `python -m src.update` after each matchday — trends "
                "appear once two or more snapshots exist.")
    else:
        default_teams = current.head(6)["team"].tolist()
        chosen = st.multiselect("Teams to chart",
                                sorted(hist["team"].unique()),
                                default=default_teams)
        if chosen:
            sub = hist[hist["team"].isin(chosen)]
            line = (
                alt.Chart(sub)
                .mark_line(point=True, strokeWidth=2.5)
                .encode(
                    x=alt.X("ts:T", title=None),
                    y=alt.Y("champion:Q", title="Title probability (%)"),
                    color=alt.Color("team:N", title="Team",
                                    scale=alt.Scale(scheme="tableau10")),
                    tooltip=[alt.Tooltip("team:N", title="Team"),
                             alt.Tooltip("ts:T", title="When",
                                         format="%b %d %H:%M"),
                             alt.Tooltip("champion:Q", title="Champion %",
                                         format=".1f")],
                )
                .properties(height=420)
            )
            st.altair_chart(style_chart(line), width='stretch')

    st.subheader("Explore a team")
    pick = st.selectbox("Pick a team", current["team"].tolist())
    row = current[current["team"] == pick].iloc[0]

    e1, e2 = st.columns(2)
    with e1:
        st.markdown(f"**{pick} — chance of reaching each stage**")
        team_stages = pd.DataFrame({
            "stage": [STAGE_LABELS[s] for s in STAGES],
            "prob": [row[s] for s in STAGES],
        })
        chart = (
            alt.Chart(team_stages)
            .mark_bar(color=BLUE, height=18)
            .encode(
                x=alt.X("prob:Q", title="Probability (%)"),
                y=alt.Y("stage:N", sort=[STAGE_LABELS[s] for s in STAGES],
                        title=None),
                tooltip=[alt.Tooltip("prob:Q", format=".1f")],
            )
            .properties(height=220)
        )
        st.altair_chart(style_chart(chart), width='stretch')
    with e2:
        if hist is not None and hist["team"].eq(pick).any():
            st.markdown(f"**{pick} — title odds over time**")
            t = hist[hist["team"] == pick]
            trend = (
                alt.Chart(t)
                .mark_line(point=True, color=GOLD, strokeWidth=2.5)
                .encode(
                    x=alt.X("ts:T", title=None),
                    y=alt.Y("champion:Q", title="Champion %"),
                    tooltip=[alt.Tooltip("ts:T", format="%b %d %H:%M"),
                             alt.Tooltip("champion:Q", format=".1f")],
                )
                .properties(height=220)
            )
            st.altair_chart(style_chart(trend), width='stretch')
        else:
            st.info("Title-odds history appears after a few update runs.")


# ============================================================================
# TAB 2 — MODEL LAB
# ============================================================================

with tab_lab:
    st.subheader("Is the model any good? (live accuracy)")
    st.caption("Brier score of pre-match predictions vs real results, "
               "against a coin-flip baseline. Lower is better — below the "
               "dashed line means the model is winning.")

    if scores is None or scores.empty:
        st.info("Accuracy history appears once matches the model predicted "
                "have been played and scored.")
    else:
        latest = scores.iloc[-1]
        beating = latest["brier"] < latest["uniform_brier"]
        s1, s2, s3 = st.columns(3)
        s1.metric("Current Brier", f"{latest['brier']:.3f}",
                  f"{latest['brier'] - latest['uniform_brier']:+.3f} vs baseline",
                  delta_color="inverse")
        s2.metric("Matches scored", f"{int(latest['n_matches'])}")
        s3.metric("Verdict",
                  "Beating baseline" if beating else "Below baseline")

        long = scores.melt(id_vars="ts",
                           value_vars=["brier", "uniform_brier"],
                           var_name="series", value_name="score")
        long["series"] = long["series"].map(
            {"brier": "Model", "uniform_brier": "Coin-flip baseline"})
        acc = (
            alt.Chart(long)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X("ts:T", title=None),
                y=alt.Y("score:Q", title="Brier score (lower = better)",
                        scale=alt.Scale(zero=False)),
                color=alt.Color("series:N", title=None,
                                scale=alt.Scale(
                                    domain=["Model", "Coin-flip baseline"],
                                    range=[GOLD, MUTED])),
                strokeDash=alt.condition(
                    "datum.series == 'Coin-flip baseline'",
                    alt.value([6, 4]), alt.value([0])),
                tooltip=[alt.Tooltip("series:N", title=""),
                         alt.Tooltip("ts:T", title="When",
                                     format="%b %d %H:%M"),
                         alt.Tooltip("score:Q", title="Brier",
                                     format=".3f")],
            )
            .properties(height=340)
        )
        st.altair_chart(style_chart(acc), width='stretch')

    st.subheader("Best calls and biggest surprises")
    st.caption("Every scored match: the probability the model gave to what "
               "actually happened. High = a called shot; low = an upset it "
               "didn't see coming.")

    if card is None or card.empty:
        st.info("The scorecard appears after the next update run "
                "(`python -m src.update`).")
    else:
        best = card.sort_values("p_outcome", ascending=False).head(4)
        worst = card.sort_values("p_outcome").head(4)
        cb, cw = st.columns(2)
        with cb:
            st.markdown("**🎯 Called it**")
            for r in best.itertuples(index=False):
                st.markdown(
                    f"{r.home} **{r.score}** {r.away} — gave "
                    f"**{r.p_outcome * 100:.0f}%** to the {r.outcome}")
        with cw:
            st.markdown("**💥 Didn't see it coming**")
            for r in worst.itertuples(index=False):
                st.markdown(
                    f"{r.home} **{r.score}** {r.away} — gave only "
                    f"**{r.p_outcome * 100:.0f}%** to the {r.outcome}")
        with st.expander("Full match-by-match record"):
            show = card.sort_values("date", ascending=False)[
                ["date", "home", "score", "away", "outcome", "p_outcome"]]
            show = show.rename(columns={"p_outcome": "prob given to outcome"})
            st.dataframe(show, width='stretch', height=380, hide_index=True)

    st.subheader("Point estimates vs full uncertainty (Dixon-Coles vs Bayesian)")
    st.caption("The Dixon-Coles model treats fitted team strengths as exact. "
               "The hierarchical Bayesian version samples a different "
               "plausible strength set per simulated tournament — parameter "
               "uncertainty flows into the odds. Positive bars: the Bayesian "
               "model gives that team MORE title probability.")

    if comp is None or comp.empty:
        st.info("Run `python -m src.bayesian_sim` to generate the model "
                "comparison.")
    else:
        cshow = comp[(comp["dixon_coles_%"] > 0) | (comp["bayesian_%"] > 0)]
        cshow = cshow.sort_values("diff")
        div = (
            alt.Chart(cshow)
            .mark_bar(cornerRadiusEnd=2, height=16)
            .encode(
                x=alt.X("diff:Q",
                        title="Bayesian − Dixon-Coles (title odds, pp)"),
                y=alt.Y("team:N", sort=alt.EncodingSortField("diff"),
                        title=None),
                color=alt.condition("datum.diff >= 0", alt.value(GOLD),
                                    alt.value(RED)),
                tooltip=[alt.Tooltip("team:N", title="Team"),
                         alt.Tooltip("dixon_coles_%:Q", title="Dixon-Coles %",
                                     format=".1f"),
                         alt.Tooltip("bayesian_%:Q", title="Bayesian %",
                                     format=".1f"),
                         alt.Tooltip("diff:Q", title="Difference",
                                     format="+.1f")],
            )
            .properties(height=26 * len(cshow))
        )
        st.altair_chart(style_chart(div), width='stretch')
        st.caption("Pattern to notice: confident favourites shed probability, "
                   "uncertain outsiders gain it — uncertainty propagation "
                   "fattens the tails, the predicted fix for the model's "
                   "documented upset-underestimation.")

    st.subheader("Out-of-sample backtests (2018 & 2022)")
    b1, b2 = st.columns(2)
    b1.metric("2018 World Cup — Brier", "0.566", "-0.083 vs climatology",
              delta_color="inverse")
    b2.metric("2022 World Cup — Brier", "0.614", "-0.033 vs climatology",
              delta_color="inverse")
    if CALIB_IMG.exists():
        st.image(str(CALIB_IMG),
                 caption="Reliability diagram, pooled 2018+2022 backtests "
                         "(128 matches). Mid-range is well calibrated; the "
                         "lowest bin shows the tail-upset underestimation.")


# ============================================================================
# TAB 3 — THE FORECAST STORY
# ============================================================================

with tab_story:
    st.subheader("What I said before a ball was kicked — vs now")
    st.caption("The forecast was frozen on June 12 with the training data "
               "cut at June 8 (see reports/forecast_2026.md — the commit "
               "timestamp is the proof). Gold dots: frozen forecast. Blue "
               "dots: today. The line is each team's journey.")

    if sim is None:
        st.info("Frozen forecast file not found.")
    else:
        frozen = sim.set_index("team")["champion"]
        now = current.set_index("team")["champion"]
        story = pd.DataFrame({"frozen": frozen, "now": now}).dropna()
        story = story[(story["frozen"] > 0.5) | (story["now"] > 0.5)]
        story = (story.sort_values("now", ascending=False)
                 .reset_index().rename(columns={"index": "team"}))
        order = story["team"].tolist()

        rule = (
            alt.Chart(story)
            .mark_rule(color=GRID, strokeWidth=2)
            .encode(y=alt.Y("team:N", sort=order, title=None),
                    x=alt.X("frozen:Q",
                            title="Title probability (%)"),
                    x2="now:Q")
        )
        long = story.melt(id_vars="team", value_vars=["frozen", "now"],
                          var_name="when", value_name="prob")
        long["when"] = long["when"].map({"frozen": "Frozen (Jun 12)",
                                         "now": "Now"})
        dots = (
            alt.Chart(long)
            .mark_point(filled=True, size=110)
            .encode(
                y=alt.Y("team:N", sort=order, title=None),
                x=alt.X("prob:Q"),
                color=alt.Color("when:N", title=None,
                                scale=alt.Scale(
                                    domain=["Frozen (Jun 12)", "Now"],
                                    range=[GOLD, BLUE])),
                tooltip=[alt.Tooltip("team:N", title="Team"),
                         alt.Tooltip("when:N", title=""),
                         alt.Tooltip("prob:Q", title="Champion %",
                                     format=".1f")],
            )
        )
        st.altair_chart(
            style_chart((rule + dots).properties(height=30 * len(story))),
            width='stretch')

        gain = (story["now"] - story["frozen"])
        biggest_up = story.loc[gain.idxmax()]
        biggest_dn = story.loc[gain.idxmin()]
        g1, g2 = st.columns(2)
        g1.metric("Biggest riser since the freeze", biggest_up["team"],
                  f"{biggest_up['now'] - biggest_up['frozen']:+.1f} pp")
        g2.metric("Biggest faller since the freeze", biggest_dn["team"],
                  f"{biggest_dn['now'] - biggest_dn['frozen']:+.1f} pp")

    st.markdown(
        "The full frozen forecast, method notes, and honest limitations live "
        "in the repo: `reports/forecast_2026.md`. After the final on "
        "July 19, this tab becomes the post-mortem — scoring the June "
        "probabilities against the completed tournament."
    )


st.divider()
st.caption(
    "Model: Dixon-Coles bivariate Poisson with time-decay weighting, fit on "
    "international results 2010-present (data: Mart Jurisoo), plus a "
    "hierarchical Bayesian comparison model (PyMC). Probabilities come from "
    "5,000 Monte Carlo simulations of the remaining bracket, pinned to the "
    "real knockout pairings. Model estimates, not certainties — a 17% "
    "favourite still loses the title 83% of the time."
)
