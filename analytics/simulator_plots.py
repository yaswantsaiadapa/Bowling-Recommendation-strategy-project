"""
Plotly Visualizations for Pre-Game Strategy & Match Simulator.

Generates interactive, dark-themed charts with zero binary encoding bug:
1. Win Probability & Toss Scenarios (Team 1 vs Team 2 under Bat 1st vs Chase).
2. Pitch Conditions & Wicket Distribution (Test: 4 Innings Decay; Limited Overs: Pace vs Spin %).
3. Projected Score Trajectories against Venue Par Score in this era.
"""

from typing import Any, Dict, Optional
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


def _apply_dark_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    """Applies clean full-width dark theme styling."""
    fig.update_layout(
        autosize=True,
        title=None,
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font={"color": TEXT_COLOR, "family": "Inter, system-ui, sans-serif", "size": 12},
        height=height,
        margin={"l": 50, "r": 30, "t": 20, "b": 50},
        legend={
            "orientation": "h",
            "yanchor": "bottom", "y": 1.02,
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


def create_win_probability_plot(sim_data: Dict[str, Any]) -> Optional[str]:
    """Horizontal stacked bar showing overall win probability and toss scenarios."""
    t1 = str(sim_data["team1"])
    t2 = str(sim_data["team2"])
    p1 = round(float(sim_data["t1_win_pct"]), 1)
    p2 = round(float(sim_data["t2_win_pct"]), 1)

    t1_bat = round(float(sim_data["t1_bat_first_win"]), 1)
    t1_chase = round(float(sim_data["t1_chase_win"]), 1)

    categories = ["Overall Match Win %", f"{t1} Bats 1st", f"{t1} Chases Target"]
    t1_probs = [p1, t1_bat, t1_chase]
    t2_probs = [p2, round(100.0 - t1_bat, 1), round(100.0 - t1_chase, 1)]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=categories,
            x=t1_probs,
            name=t1,
            orientation="h",
            marker_color=ACCENT_BLUE,
            text=[f"{v}%" for v in t1_probs],
            textposition="inside",
            hovertemplate=f"Scenario: %{{y}}<br>{t1} Win: %{{x}}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            y=categories,
            x=t2_probs,
            name=t2,
            orientation="h",
            marker_color=ACCENT_ORANGE,
            text=[f"{v}%" for v in t2_probs],
            textposition="inside",
            hovertemplate=f"Scenario: %{{y}}<br>{t2} Win: %{{x}}%<extra></extra>",
        )
    )

    fig.update_layout(barmode="stack")
    fig.update_xaxes(range=[0, 100], title_text="Win Probability (%)")
    _apply_dark_layout(fig, height=260)
    return _render_fig(fig)


def create_pitch_distribution_plot(pitch: Dict[str, Any]) -> Optional[str]:
    """
    Renders pitch character plot:
    - In Test: Bar + line plot of 4-Innings Average Scores Decay.
    - In Limited Overs: Donut chart of Pace vs Spin Wickets and averages.
    """
    is_test = pitch.get("is_test", False)

    if is_test:
        labels = ["Innings 1 (Day 1-2)", "Innings 2 (Day 2-3)", "Innings 3 (Day 3-4)", "Innings 4 (Day 4-5)"]
        scores = [
            int(pitch.get("avg_1st_innings", 320)),
            int(pitch.get("avg_2nd_innings", 290)),
            int(pitch.get("avg_3rd_innings", 240)),
            int(pitch.get("avg_4th_innings", 180)),
        ]
        bar_colors = [ACCENT_BLUE, ACCENT_TEAL, ACCENT_ORANGE, ACCENT_RED]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=labels,
                y=scores,
                name="Average Score",
                marker_color=bar_colors,
                text=[f"{s} runs" for s in scores],
                textposition="auto",
                hovertemplate="Stage: %{x}<br>Average Total: %{y} runs<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=scores,
                name="Deterioration Curve",
                mode="lines+markers",
                line={"color": "#fbbf24", "width": 3, "dash": "dot"},
                marker={"size": 10, "color": "#fbbf24"},
                hovertemplate="Pitch Decay: %{y} runs<extra></extra>",
            )
        )
        fig.update_yaxes(title_text="Average Runs Scored")
        _apply_dark_layout(fig, height=360)
        return _render_fig(fig)

    else:
        pace_pct = round(float(pitch.get("pace_wkt_pct", 65.0)), 1)
        spin_pct = round(float(pitch.get("spin_wkt_pct", 35.0)), 1)
        p_wkts = int(pitch.get("pace_wkts", 0))
        s_wkts = int(pitch.get("spin_wkts", 0))
        p_avg = float(pitch.get("pace_avg", 28.0))
        s_avg = float(pitch.get("spin_avg", 32.0))

        labels = ["Pace / Fast Bowling", "Spin Bowling"]
        values = [pace_pct, spin_pct]
        colors = [ACCENT_BLUE, ACCENT_ORANGE]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.55,
                    marker={"colors": colors, "line": {"color": "#0f172a", "width": 2}},
                    textinfo="percent+label",
                    textposition="inside",
                    insidetextorientation="horizontal",
                    showlegend=True,
                    hovertemplate="Bowling: %{label}<br>Wicket Share: %{value}%<br>Wickets: %{customdata[0]}<br>Average: %{customdata[1]}<extra></extra>",
                    customdata=[[p_wkts, p_avg], [s_wkts, s_avg]],
                )
            ]
        )
        fig.update_layout(
            autosize=True,
            title=None,
            paper_bgcolor=CARD_BG,
            plot_bgcolor=CARD_BG,
            font={"color": TEXT_COLOR, "family": "Inter, system-ui, sans-serif", "size": 12},
            height=360,
            margin={"l": 20, "r": 20, "t": 20, "b": 40},
            legend={
                "orientation": "h",
                "yanchor": "top", "y": -0.05,
                "xanchor": "center",  "x": 0.5,
                "font": {"size": 12},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
            },
            hoverlabel={"bgcolor": "#ffffff", "bordercolor": "#e0e0e0", "font": {"color": TEXT_COLOR, "size": 12}},
        )
        return _render_fig(fig)


def create_projected_scores_plot(sim_data: Dict[str, Any]) -> Optional[str]:
    """Grouped bar chart comparing Projected 1st Innings totals against Venue Par in this era."""
    t1 = str(sim_data["team1"])
    t2 = str(sim_data["team2"])
    t1_score = int(sim_data.get("t1_proj_runs", 165))
    t2_score = int(sim_data.get("t2_proj_runs", 165))
    par_score = int(sim_data.get("par_score", 165))

    labels = [f"{t1} Projected", "Venue Par Total", f"{t2} Projected"]
    scores = [t1_score, par_score, t2_score]
    colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=scores,
            marker_color=colors,
            text=[f"{s} runs" for s in scores],
            textposition="auto",
            hovertemplate="Entity: %{x}<br>Projected Total: %{y} runs<extra></extra>",
        )
    )

    fig.update_yaxes(title_text="1st Innings Projected Total (Runs)")
    _apply_dark_layout(fig, height=360)
    return _render_fig(fig)


