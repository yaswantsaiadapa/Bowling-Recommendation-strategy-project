"""
Plotly Visualizations for Team & Venue Intelligence.

Generates interactive, dark-themed charts:
1. Venue:
   - Test Cricket: 4 Innings pitch decay progression, match duration breakdown (Day 3 / 4 / 5 / Draw).
   - Limited Overs (T20I / ODI): Match outcomes donut (Bat 1st vs Chase), Phase scoring run rate.
   - All Formats: Format-by-Format match outcome and benchmark comparison.
2. Team: Opponents Win/Loss comparison, Phase scoring DNA, Fortress grounds win %.
"""

from typing import Any, Dict, List, Optional
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
ACCENT_TEAL = "#14b8a6"
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
# VENUE VISUALIZATIONS (TEST CRICKET)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def create_test_innings_decay_plot(df_innings: pd.DataFrame) -> Optional[str]:
    """Bar + line chart showing 4 innings score decay progression in Test matches."""
    if df_innings.empty:
        return None

    inn_labels = {
        1: "Innings 1 (Day 1-2)",
        2: "Innings 2 (Day 2-3)",
        3: "Innings 3 (Day 3-4)",
        4: "Innings 4 (Day 4-5 Target)",
    }
    
    df_sorted = df_innings.sort_values(by="inning").copy()
    labels = [inn_labels.get(int(i), f"Innings {i}") for i in df_sorted["inning"].tolist()]
    avg_scores = [round(float(v), 1) for v in df_sorted["avg_score"].tolist()]
    samples = [int(v) for v in df_sorted["sample_size"].tolist()]
    max_scores = [int(v) for v in df_sorted["max_score"].tolist()] if "max_score" in df_sorted.columns else []
    min_scores = [int(v) for v in df_sorted["min_score"].tolist()] if "min_score" in df_sorted.columns else []

    bar_colors = [ACCENT_BLUE, ACCENT_TEAL, ACCENT_ORANGE, ACCENT_RED]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=avg_scores,
            name="Average Score",
            marker_color=bar_colors[:len(labels)],
            text=[f"{int(s)} runs" for s in avg_scores],
            textposition="auto",
            hovertemplate="Stage: %{x}<br>Avg Total: %{y:.1f} runs<br>Sample Innings: %{customdata[0]}<extra></extra>",
            customdata=[[s] for s in samples],
        )
    )

    fig.add_trace(
        go.Scatter(
            x=labels,
            y=avg_scores,
            name="Pitch Decay Trajectory",
            mode="lines+markers",
            line={"color": "#fbbf24", "width": 3, "dash": "dot"},
            marker={"size": 10, "color": "#fbbf24"},
            hovertemplate="Decay Curve: %{y:.1f} runs<extra></extra>",
        )
    )

    fig.update_yaxes(title_text="Average Runs Scored")
    _apply_dark_layout(fig, height=400)
    return _render_fig(fig)


def create_test_duration_plot(durations: Dict[str, int]) -> Optional[str]:
    """Donut chart illustrating match completion days in Test cricket."""
    total = sum(durations.values())
    if total == 0:
        return None

    labels = list(durations.keys())
    values = [int(v) for v in durations.values()]
    
    color_map = {
        "Day 3 or earlier": ACCENT_RED,
        "Day 4": ACCENT_BLUE,
        "Day 5": ACCENT_GREEN,
        "Draw": ACCENT_PURPLE,
    }
    colors = [color_map.get(k, ACCENT_ORANGE) for k in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker={"colors": colors},
                textinfo="label+percent",
                hovertemplate="Completion: %{label}<br>Tests: %{value}<br>Share: %{percent}<extra></extra>",
            )
        ]
    )
    _apply_dark_layout(fig, height=400)
    return _render_fig(fig)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# VENUE VISUALIZATIONS (LIMITED OVERS & ALL FORMATS)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def create_venue_outcomes_plot(outcomes: Dict[str, int]) -> Optional[str]:
    """Donut chart showing Batting 1st Wins vs Chasing Wins vs Ties/Draws."""
    total = sum(outcomes.values())
    if total == 0:
        return None

    labels = list(outcomes.keys())
    values = [int(v) for v in outcomes.values()]
    colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker={"colors": colors[: len(labels)]},
                textinfo="label+percent",
                hovertemplate="Outcome: %{label}<br>Matches: %{value}<br>Share: %{percent}<extra></extra>",
            )
        ]
    )
    _apply_dark_layout(fig)
    return _render_fig(fig)


def create_venue_format_comparison_plot(format_cards: List[Dict[str, Any]]) -> Optional[str]:
    """Comparative grouped bar chart for All Formats view."""
    if not format_cards:
        return None

    formats = [c["format"] for c in format_cards]
    matches = [int(c["matches"]) for c in format_cards]
    avg_1st = [int(c.get("avg_1st", 0)) for c in format_cards]
    avg_2nd = [int(c.get("avg_2nd", 0)) for c in format_cards]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=formats,
            y=avg_1st,
            name="Avg 1st Innings Runs",
            marker_color=ACCENT_BLUE,
            text=avg_1st,
            textposition="auto",
            hovertemplate="Format: %{x}<br>Avg 1st Innings: %{y} runs<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=formats,
            y=avg_2nd,
            name="Avg 2nd Innings Runs",
            marker_color=ACCENT_GREEN,
            text=avg_2nd,
            textposition="auto",
            hovertemplate="Format: %{x}<br>Avg 2nd Innings: %{y} runs<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=formats,
            y=matches,
            name="Matches Hosted",
            mode="lines+markers+text",
            text=[f"{m} matches" for m in matches],
            textposition="top center",
            line={"color": ACCENT_ORANGE, "width": 3},
            marker={"size": 10, "color": ACCENT_ORANGE},
            hovertemplate="Format: %{x}<br>Matches Hosted: %{y}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Average Innings Total (Runs)", secondary_y=False)
    fig.update_yaxes(title_text="Total Matches Hosted", secondary_y=True, showgrid=False)

    _apply_dark_layout(fig, height=420)
    return _render_fig(fig)


def create_venue_phase_plot(df_phase: pd.DataFrame) -> Optional[str]:
    """Dual-axis chart of scoring run rate and boundary % across phases."""
    if df_phase.empty:
        return None

    phases = df_phase["match_phase"].tolist()
    run_rate = [round(float(v), 2) for v in df_phase["run_rate"].tolist()]
    boundary_pct = [round(float(v), 1) for v in df_phase["boundary_pct"].tolist()]
    runs = [int(v) for v in df_phase["runs"].tolist()]
    balls = [int(v) for v in df_phase["balls"].tolist()]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=phases,
            y=run_rate,
            name="Run Rate (RPO)",
            marker_color=ACCENT_BLUE,
            text=[f"{rr:.2f}" for rr in run_rate],
            textposition="auto",
            hovertemplate="Phase: %{x}<br>Run Rate: %{y:.2f} rpo<br>Runs: %{customdata[0]}<br>Balls: %{customdata[1]}<extra></extra>",
            customdata=[[r, b] for r, b in zip(runs, balls)],
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=phases,
            y=boundary_pct,
            name="Boundary %",
            mode="lines+markers+text",
            text=[f"{bp:.1f}%" for bp in boundary_pct],
            textposition="top center",
            line={"color": ACCENT_ORANGE, "width": 3},
            marker={"size": 9, "color": ACCENT_ORANGE},
            hovertemplate="Phase: %{x}<br>Boundary: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Scoring Run Rate (Runs / Over)", secondary_y=False, gridcolor="#e0e0e0")
    fig.update_yaxes(title_text="Boundary Percentage (%)", secondary_y=True, showgrid=False)

    _apply_dark_layout(fig)
    return _render_fig(fig)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TEAM VISUALIZATIONS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def create_team_opponents_plot(df_opp: pd.DataFrame) -> Optional[str]:
    """Grouped bar chart showing Wins vs Losses against major opponents."""
    if df_opp.empty:
        return None

    df_sorted = df_opp.sort_values(by="matches", ascending=True)
    opponents = df_sorted["opponent"].tolist()
    wins = [int(v) for v in df_sorted["wins"].tolist()]
    losses = [int(v) for v in df_sorted["losses"].tolist()]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=opponents,
            x=wins,
            name="Wins",
            orientation="h",
            marker_color=ACCENT_GREEN,
            text=wins,
            textposition="auto",
            hovertemplate="Opponent: %{y}<br>Wins: %{x}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            y=opponents,
            x=losses,
            name="Losses",
            orientation="h",
            marker_color=ACCENT_RED,
            text=losses,
            textposition="auto",
            hovertemplate="Opponent: %{y}<br>Losses: %{x}<extra></extra>",
        )
    )

    fig.update_layout(barmode="group")
    _apply_dark_layout(fig, height=460)
    return _render_fig(fig)


def create_team_phase_plot(df_phase: pd.DataFrame) -> Optional[str]:
    """Dual-axis chart of team batting scoring run rate and boundary %."""
    if df_phase.empty:
        return None

    phases = df_phase["match_phase"].tolist()
    run_rate = [round(float(v), 2) for v in df_phase["run_rate"].tolist()]
    boundary_pct = [round(float(v), 1) for v in df_phase["boundary_pct"].tolist()]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=phases,
            y=run_rate,
            name="Run Rate (RPO)",
            marker_color=ACCENT_BLUE,
            text=[f"{rr:.2f}" for rr in run_rate],
            textposition="auto",
            hovertemplate="Phase: %{x}<br>Run Rate: %{y:.2f} rpo<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=phases,
            y=boundary_pct,
            name="Boundary %",
            mode="lines+markers+text",
            text=[f"{bp:.1f}%" for bp in boundary_pct],
            textposition="top center",
            line={"color": ACCENT_ORANGE, "width": 3},
            marker={"size": 9, "color": ACCENT_ORANGE},
            hovertemplate="Phase: %{x}<br>Boundary: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Run Rate (Runs / Over)", secondary_y=False, gridcolor="#e0e0e0")
    fig.update_yaxes(title_text="Boundary %", secondary_y=True, showgrid=False)

    _apply_dark_layout(fig)
    return _render_fig(fig)


def create_team_fortress_plot(df_fortress: pd.DataFrame) -> Optional[str]:
    """Horizontal bar chart of top grounds by win percentage."""
    if df_fortress.empty:
        return None

    df_sorted = df_fortress.sort_values(by="win_pct", ascending=True)
    grounds = df_sorted["ground_name"].tolist()
    win_pct = [round(float(v), 1) for v in df_sorted["win_pct"].tolist()]
    matches = [int(v) for v in df_sorted["matches"].tolist()]
    wins = [int(v) for v in df_sorted["wins"].tolist()]

    labels = [f"{wp}% ({w}/{m} wins)" for wp, w, m in zip(win_pct, wins, matches)]

    fig = go.Figure(
        go.Bar(
            x=win_pct,
            y=grounds,
            orientation="h",
            marker={"color": ACCENT_BLUE},
            text=labels,
            textposition="auto",
            hovertemplate="Ground: %{y}<br>Win Rate: %{x}%<br>Wins: %{customdata[0]}<br>Matches: %{customdata[1]}<extra></extra>",
            customdata=[[w, m] for w, m in zip(wins, matches)],
        )
    )
    fig.update_xaxes(range=[0, 105])
    _apply_dark_layout(fig, height=450)
    return _render_fig(fig)


