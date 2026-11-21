"""
Plotly Visualizations for Matchup & Opponent Intelligence.

Generates interactive, dark-themed charts across all 3 confrontation modes:
1. Mode 1 (Batter vs Bowler): Format split, Phase confrontation, Shot distribution donut.
2. Mode 2 (Batter vs Team): Format mastery, Top bowlers faced, Phase attack profile.
3. Mode 3 (Bowler vs Team): Format mastery, Top batters dismissed, Dismissal anatomy donut.
"""

from typing import Dict, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

THEME_BG = "#ffffff"
CARD_BG = "#ffffff"
TEXT_COLOR = "#1a1a1a"
ACCENT_BLUE = "#38bdf8"
ACCENT_GREEN = "#22c55e"
ACCENT_ORANGE = "#f97316"
ACCENT_RED = "#ef4444"
ACCENT_PURPLE = "#a855f7"
GRID_COLOR = "#e0e0e0"


def _apply_dark_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    """Applies clean full-width dark theme styling."""
    fig.update_layout(
        autosize=True,
        title=None,
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font={"color": TEXT_COLOR, "family": "Inter, system-ui, sans-serif", "size": 12},
        height=height,
        margin={"l": 60, "r": 40, "t": 20, "b": 50},
        legend={
            "orientation": "h",
            "yanchor": "bottom", "y": 1.01,
            "xanchor": "right",  "x": 1,
            "font": {"size": 11},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
        },
        hovermode="x unified",
        hoverlabel={"bgcolor": "#ffffff", "bordercolor": "#e0e0e0", "font": {"color": TEXT_COLOR, "size": 12}},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e0e0e0", gridwidth=1, zeroline=False, tickfont={"size": 11}, linecolor="#e0e0e0")
    fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0", gridwidth=1, zeroline=False, tickfont={"size": 11}, linecolor="#e0e0e0")
    return fig


def _render_fig(fig: go.Figure) -> str:
    """Renders responsive 100% wide Plotly HTML with zero binary encoding bug."""
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displayModeBar": False},
        default_width="100%",
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MODE 1: BATTER VS BOWLER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def create_h2h_format_plot(df_format: pd.DataFrame) -> Optional[str]:
    """Dual-axis chart of H2H runs and strike rate per format."""
    if df_format.empty:
        return None

    formats = df_format["format"].tolist()
    runs = [int(v) for v in df_format["total_runs"].tolist()]
    balls = [int(v) for v in df_format["balls_faced"].tolist()]
    dismissals = [int(v) for v in df_format["dismissals"].tolist()]
    strike_rate = [round(float(v), 1) for v in df_format["strike_rate"].tolist()]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=formats,
            y=runs,
            name="Runs Scored",
            marker_color=ACCENT_BLUE,
            text=runs,
            textposition="auto",
            hovertemplate="Format: %{x}<br>Runs: %{y}<br>Balls: %{customdata[0]}<br>Dismissals: %{customdata[1]}<extra></extra>",
            customdata=[[b, d] for b, d in zip(balls, dismissals)],
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=formats,
            y=strike_rate,
            name="Strike Rate",
            mode="lines+markers+text",
            text=[f"{sr:.1f}" for sr in strike_rate],
            textposition="top center",
            line={"color": ACCENT_ORANGE, "width": 3},
            marker={"size": 9, "color": ACCENT_ORANGE},
            hovertemplate="Format: %{x}<br>Strike Rate: %{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Runs Scored", secondary_y=False, gridcolor="#e0e0e0")
    fig.update_yaxes(title_text="Strike Rate (Runs / 100 Balls)", secondary_y=True, showgrid=False)

    _apply_dark_layout(fig)
    return _render_fig(fig)


def create_h2h_phase_plot(df_phase: pd.DataFrame) -> Optional[str]:
    """Dual-axis chart of H2H runs and strike rate across match phases."""
    if df_phase.empty:
        return None

    phases = df_phase["match_phase"].tolist()
    runs = [int(v) for v in df_phase["total_runs"].tolist()]
    balls = [int(v) for v in df_phase["balls_faced"].tolist()]
    dismissals = [int(v) for v in df_phase["dismissals"].tolist()]
    strike_rate = [round(float(v), 1) for v in df_phase["strike_rate"].tolist()]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=phases,
            y=runs,
            name="Runs Scored",
            marker_color=ACCENT_BLUE,
            text=runs,
            textposition="auto",
            hovertemplate="Phase: %{x}<br>Runs: %{y}<br>Balls: %{customdata[0]}<br>Dismissals: %{customdata[1]}<extra></extra>",
            customdata=[[b, d] for b, d in zip(balls, dismissals)],
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=phases,
            y=strike_rate,
            name="Strike Rate",
            mode="lines+markers+text",
            text=[f"{sr:.1f}" for sr in strike_rate],
            textposition="top center",
            line={"color": ACCENT_ORANGE, "width": 3},
            marker={"size": 9, "color": ACCENT_ORANGE},
            hovertemplate="Phase: %{x}<br>Strike Rate: %{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Runs Scored", secondary_y=False, gridcolor="#e0e0e0")
    fig.update_yaxes(title_text="Strike Rate", secondary_y=True, showgrid=False)

    _apply_dark_layout(fig)
    return _render_fig(fig)


def create_h2h_shots_plot(shots: Dict[str, int]) -> Optional[str]:
    """Interactive Donut chart of scoring shot distribution."""
    total = sum(shots.values())
    if total == 0:
        return None

    labels = list(shots.keys())
    values = [int(v) for v in shots.values()]
    colors = [GRID_COLOR, ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE, ACCENT_ORANGE, ACCENT_RED]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker={"colors": colors[: len(labels)]},
                textinfo="label+percent",
                hovertemplate="Delivery Outcome: %{label}<br>Count: %{value}<br>Share: %{percent}<extra></extra>",
            )
        ]
    )
    _apply_dark_layout(fig)
    return _render_fig(fig)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MODE 2: BATTER VS OPPOSITION TEAM
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def create_bat_team_format_plot(df_format: pd.DataFrame) -> Optional[str]:
    """Dual-axis chart of Batter performance against team by format."""
    if df_format.empty:
        return None

    formats = df_format["format"].tolist()
    runs = [int(v) for v in df_format["total_runs"].tolist()]
    dismissals = [int(v) for v in df_format["dismissals"].tolist()]
    strike_rate = [round(float(v), 1) for v in df_format["strike_rate"].tolist()]
    avg = [round(float(v), 1) for v in df_format["batting_average"].tolist()]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=formats,
            y=runs,
            name="Runs Scored",
            marker_color=ACCENT_BLUE,
            text=runs,
            textposition="auto",
            hovertemplate="Format: %{x}<br>Runs: %{y}<br>Dismissals: %{customdata[0]}<br>Average: %{customdata[1]}<extra></extra>",
            customdata=[[d, a] for d, a in zip(dismissals, avg)],
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=formats,
            y=strike_rate,
            name="Strike Rate",
            mode="lines+markers+text",
            text=[f"{sr:.1f}" for sr in strike_rate],
            textposition="top center",
            line={"color": ACCENT_ORANGE, "width": 3},
            marker={"size": 9, "color": ACCENT_ORANGE},
            hovertemplate="Format: %{x}<br>Strike Rate: %{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Runs Scored", secondary_y=False, gridcolor="#e0e0e0")
    fig.update_yaxes(title_text="Strike Rate", secondary_y=True, showgrid=False)

    _apply_dark_layout(fig)
    return _render_fig(fig)


def create_bat_team_bowlers_plot(df_bowlers: pd.DataFrame) -> Optional[str]:
    """Horizontal bar chart showing bowlers from opposition team faced most often."""
    if df_bowlers.empty:
        return None

    df_sorted = df_bowlers.sort_values(by="runs_scored", ascending=True)
    bowlers = df_sorted["bowler"].tolist()
    runs = [int(v) for v in df_sorted["runs_scored"].tolist()]
    dismissals = [int(v) for v in df_sorted["dismissals"].tolist()]
    balls = [int(v) for v in df_sorted["balls_faced"].tolist()]
    sr = [round(float(v), 1) for v in df_sorted["strike_rate"].tolist()]

    labels = [f"{r} runs (Out: {d}x, SR {s})" for r, d, s in zip(runs, dismissals, sr)]

    fig = go.Figure(
        go.Bar(
            x=runs,
            y=bowlers,
            orientation="h",
            marker={"color": ACCENT_BLUE},
            text=labels,
            textposition="auto",
            hovertemplate="Bowler: %{y}<br>Runs: %{x}<br>Balls: %{customdata[0]}<br>Dismissals: %{customdata[1]}<br>SR: %{customdata[2]}<extra></extra>",
            customdata=[[b, d, s] for b, d, s in zip(balls, dismissals, sr)],
        )
    )
    _apply_dark_layout(fig, height=450)
    return _render_fig(fig)


def create_bat_team_phase_plot(df_phase: pd.DataFrame) -> Optional[str]:
    """Dual-axis chart of Batter scoring against team per phase."""
    if df_phase.empty:
        return None

    phases = df_phase["match_phase"].tolist()
    runs = [int(v) for v in df_phase["total_runs"].tolist()]
    balls = [int(v) for v in df_phase["balls_faced"].tolist()]
    strike_rate = [round(float(v), 1) for v in df_phase["strike_rate"].tolist()]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=phases,
            y=runs,
            name="Runs Scored",
            marker_color=ACCENT_BLUE,
            text=runs,
            textposition="auto",
            hovertemplate="Phase: %{x}<br>Runs: %{y}<br>Balls: %{customdata}<extra></extra>",
            customdata=balls,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=phases,
            y=strike_rate,
            name="Strike Rate",
            mode="lines+markers+text",
            text=[f"{sr:.1f}" for sr in strike_rate],
            textposition="top center",
            line={"color": ACCENT_ORANGE, "width": 3},
            marker={"size": 9, "color": ACCENT_ORANGE},
            hovertemplate="Phase: %{x}<br>Strike Rate: %{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Runs Scored", secondary_y=False, gridcolor="#e0e0e0")
    fig.update_yaxes(title_text="Strike Rate", secondary_y=True, showgrid=False)

    _apply_dark_layout(fig)
    return _render_fig(fig)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MODE 3: BOWLER VS OPPOSITION TEAM
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def create_bowl_team_format_plot(df_format: pd.DataFrame) -> Optional[str]:
    """Dual-axis chart of Bowler wickets and economy rate against team by format."""
    if df_format.empty:
        return None

    formats = df_format["format"].tolist()
    wickets = [int(v) for v in df_format["wickets"].tolist()]
    overs = df_format["overs"].tolist()
    economy = [round(float(v), 2) for v in df_format["economy_rate"].tolist()]
    avg = [round(float(v), 1) for v in df_format["bowling_average"].tolist()]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=formats,
            y=wickets,
            name="Wickets Taken",
            marker_color=ACCENT_BLUE,
            text=wickets,
            textposition="auto",
            hovertemplate="Format: %{x}<br>Wickets: %{y}<br>Overs: %{customdata[0]}<br>Average: %{customdata[1]}<extra></extra>",
            customdata=[[o, a] for o, a in zip(overs, avg)],
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=formats,
            y=economy,
            name="Economy Rate",
            mode="lines+markers+text",
            text=[f"{e:.2f}" for e in economy],
            textposition="top center",
            line={"color": ACCENT_ORANGE, "width": 3},
            marker={"size": 9, "color": ACCENT_ORANGE},
            hovertemplate="Format: %{x}<br>Economy: %{y:.2f} rpo<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Wickets Taken", secondary_y=False, gridcolor="#e0e0e0")
    fig.update_yaxes(title_text="Economy Rate (Runs / Over)", secondary_y=True, showgrid=False)

    _apply_dark_layout(fig)
    return _render_fig(fig)


def create_bowl_team_batters_plot(df_batters: pd.DataFrame) -> Optional[str]:
    """Horizontal bar chart showing top victims from opposition team dismissed most often."""
    if df_batters.empty:
        return None

    df_sorted = df_batters.sort_values(by="wickets", ascending=True)
    batters = df_sorted["batter"].tolist()
    wickets = [int(v) for v in df_sorted["wickets"].tolist()]
    runs = [int(v) for v in df_sorted["runs_conceded"].tolist()]
    balls = [int(v) for v in df_sorted["balls_bowled"].tolist()]

    labels = [f"{w} wickets ({r} runs in {b}b)" for w, r, b in zip(wickets, runs, balls)]

    fig = go.Figure(
        go.Bar(
            x=wickets,
            y=batters,
            orientation="h",
            marker={"color": ACCENT_BLUE},
            text=labels,
            textposition="auto",
            hovertemplate="Batsman: %{y}<br>Wickets: %{x}<br>Runs Conceded: %{customdata[0]}<br>Balls Bowled: %{customdata[1]}<extra></extra>",
            customdata=[[r, b] for r, b in zip(runs, balls)],
        )
    )
    _apply_dark_layout(fig, height=450)
    return _render_fig(fig)


def create_bowl_team_dismissal_plot(df_dismissals: pd.DataFrame) -> Optional[str]:
    """Donut chart of dismissal modes against this opposition team."""
    if df_dismissals.empty:
        return None

    labels = [str(k).title() for k in df_dismissals["dismissal_kind"].tolist()]
    values = [int(v) for v in df_dismissals["dismissal_count"].tolist()]
    colors = [ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_PURPLE, ACCENT_RED]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker={"colors": colors[: len(labels)]},
                textinfo="label+percent",
                hovertemplate="Dismissal Mode: %{label}<br>Count: %{value}<br>Share: %{percent}<extra></extra>",
            )
        ]
    )
    _apply_dark_layout(fig)
    return _render_fig(fig)


