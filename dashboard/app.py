"""
app.py — live dashboard for the 2026 World Cup forecast.

Reads the CSVs the pipeline produces and turns them into a polished, live
picture of the title race:
  * reports/simulation_2026.csv  — the frozen pre-tournament forecast
  * reports/odds_history.csv      — a snapshot appended on each
                                    `python -m src.update` run (powers trends)

Run it from the project root:
    streamlit run dashboard/app.py

Design: a dark broadcast theme built around gold — the colour of the trophy
this whole dashboard is about. Everything else stays quiet so the data and
the gold accent carry the page.
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

STAGES = ["r32", "r16", "qf", "sf", "final", "champion"]
STAGE_LABELS = {
    "r32": "R32", "r16": "R16", "qf": "QF",
    "sf": "SF", "final": "Final", "champion": "Champion",
}

BG = "#0B0E14"
SURFACE = "#141925"
GRID = "#222A38"
TEXT = "#E6E9EF"
MUTED = "#8A93A6"
GOLD = "#F2C14E"
BLUE = "#4EA8DE"

st.set_page_config(page_title="World Cup 2026 — Title Race",
                   page_icon="🏆", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&display=swap');
      [class*="css"], .stMarkdown, p, span, div { font-feature-settings: "tnum"; }

      .hero { padding: 0.4rem 0 1.1rem 0; margin-bottom: 1.3rem;
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
      [data-testid="stMetricValue"] { font-family:'Space Grotesk',sans-serif;
              color:#F4F6FA; }
      [data-testid="stMetricLabel"] { color:#8A93A6; }

      .stCaption, .stCaption p { color:#6E778A !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_chart(chart):
    """Apply the dark theme consistently to any Altair chart."""
    return (
        chart
        .configure(background="transparent")
        .configure_view(strokeWidth=0, fill="transparent")
        .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID,
                        domainColor=GRID, tickColor=GRID,
                        labelFontSize=12, titleFontSize=12)
        .configure_legend(labelColor=TEXT, titleColor=MUTED)
    )


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def _to_percent(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = [c for c in STAGES if c in df.columns]
    if cols and df[cols].to_numpy().max() <= 1.5:
        df[cols] = df[cols] * 100.0
    return df


def load_simulation():
    if not SIM_FILE.exists():
        return None
    df = pd.read_csv(SIM_FILE)
    return _to_percent(df.rename(columns={df.columns[0]: "team"}))


def load_history():
    if not HIST_FILE.exists():
        return None
    df = _to_percent(pd.read_csv(HIST_FILE))
    df["ts"] = pd.to_datetime(df["run_ts"].str.replace(" UTC", "", regex=False),
                              errors="coerce")
    return df


sim = load_simulation()
hist = load_history()

if hist is not None and not hist.empty:
    latest_ts = hist["ts"].max()
    current = hist[hist["ts"] == latest_ts].copy()
    source_note = f"Live snapshot · {latest_ts:%b %d, %Y · %H:%M} UTC"
elif sim is not None:
    current = sim.copy()
    source_note = "Frozen pre-tournament forecast"
else:
    current = None
    source_note = ""

if current is not None:
    current = current.sort_values("champion", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
      <div class="rule"></div>
      <div class="eyebrow">Live forecast · model-driven</div>
      <h1>World Cup 2026 — Title Race</h1>
      <p class="sub">A Dixon-Coles bivariate-Poisson goal model, re-fit and
      re-simulated 5,000 times after every matchday. Played matches are locked
      in; only the rest of the bracket is simulated.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if current is None:
    st.error("No forecast data found. Run `python -m src.simulate` to create "
             "reports/simulation_2026.csv, then `python -m src.update` to begin "
             "the live history.")
    st.stop()

fav = current.iloc[0]
n_snapshots = hist["ts"].nunique() if hist is not None else 0
c1, c2, c3, c4 = st.columns(4)
c1.metric("🏆 Favorite", fav["team"], f"{fav['champion']:.1f}% to win")
c2.metric("Chasing", current.iloc[1]["team"], f"{current.iloc[1]['champion']:.1f}%")
c3.metric("In contention", f"{int((current['champion'] >= 1).sum())}",
          "≥ 1% title chance")
c4.metric("Snapshots logged", f"{n_snapshots}", "matchday updates")
st.caption(source_note)
st.write("")


# ----------------------------------------------------------------------------
# Current title odds (gold bars)
# ----------------------------------------------------------------------------

st.subheader("Title odds right now")
top_n = st.slider("Teams to show", 8, 24, 14)
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


# ----------------------------------------------------------------------------
# Stage heatmap (gold scale on dark — replaces the plain table)
# ----------------------------------------------------------------------------

st.subheader("How far each team is likely to go")
st.caption("Probability (%) of reaching each stage — brighter gold = more likely.")

heat_src = (current.head(16)[["team"] + STAGES]
            .melt(id_vars="team", var_name="stage", value_name="prob"))
heat_src["stage"] = heat_src["stage"].map(STAGE_LABELS)
team_order = current.head(16)["team"].tolist()
stage_order = [STAGE_LABELS[s] for s in STAGES]

base = alt.Chart(heat_src).encode(
    x=alt.X("stage:N", sort=stage_order, title=None,
            axis=alt.Axis(orient="top", labelAngle=0, labelFontWeight="bold")),
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
    color=alt.condition("datum.prob > 45", alt.value(BG), alt.value("#AEB6C4")),
)
st.altair_chart(style_chart((heat + text).properties(height=34 * len(team_order))),
                width='stretch')


# ----------------------------------------------------------------------------
# Odds over time
# ----------------------------------------------------------------------------

st.subheader("How the odds have moved")

if hist is None or hist["ts"].nunique() < 2:
    st.info("This chart fills in as the tournament progresses. Run "
            "`python -m src.update` after each matchday — with two or more "
            "snapshots you'll see each contender's title odds trend here.")
else:
    default_teams = current.head(6)["team"].tolist()
    chosen = st.multiselect("Teams to chart", sorted(hist["team"].unique()),
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
                         alt.Tooltip("ts:T", title="When", format="%b %d %H:%M"),
                         alt.Tooltip("champion:Q", title="Champion %", format=".1f")],
            )
            .properties(height=420)
        )
        st.altair_chart(style_chart(line), width='stretch')

    first = hist[hist["ts"] == hist["ts"].min()].set_index("team")["champion"]
    now = current.set_index("team")["champion"]
    delta = (now - first).dropna().sort_values()
    if len(delta):
        st.markdown("**Biggest movers** since the forecast was frozen")
        m1, m2 = st.columns(2)
        with m1:
            st.caption("Rising")
            for team, d in delta.tail(3)[::-1].items():
                st.metric(team, f"{now[team]:.1f}%", f"{d:+.1f}")
        with m2:
            st.caption("Falling")
            for team, d in delta.head(3).items():
                st.metric(team, f"{now[team]:.1f}%", f"{d:+.1f}")


# ----------------------------------------------------------------------------
# Team explorer
# ----------------------------------------------------------------------------

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
            y=alt.Y("stage:N", sort=[STAGE_LABELS[s] for s in STAGES], title=None),
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
        st.info("Title-odds history appears here after a few update runs.")


st.divider()
st.caption(
    "Model: Dixon-Coles bivariate Poisson with time-decay weighting, fit on "
    "international results 2010-present (data: Mart Jurisoo). Probabilities come "
    "from 5,000 Monte Carlo simulations of the remaining bracket. These are model "
    "estimates, not certainties — a 17% favorite still loses the title 83% of the time."
)
