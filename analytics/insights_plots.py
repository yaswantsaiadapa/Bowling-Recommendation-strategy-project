"""
Plotly Visualizations for Global Cricket Insights & Historical Evolution.

Generates interactive, dark-themed charts with zero binary encoding bug:
1. Historical Scoring Evolution (Test: 4 distinct innings lines; ODI/T20I: average innings totals).
2. Run Rate & Scoring Tempo Trajectory.
3. Boundary Evolution (Fours, Sixes, and % of runs through boundaries).
4. Format Comparison Benchmark Bars (Test vs ODI vs T20I).
5. Test Cricket 4-Innings Decay & Draw Rate Decline.
6. Test 4th Innings Target Pursuits (Chase Win vs Draw Survived vs Defeat).
7. Batting Accumulation DNA (Stacked Area: Dots, Singles, Twos, Boundaries).
8. Target Chasing Brackets Success (Limited-Overs pure).
9. Bowling Wickets/100 Balls & Economy Evolution.
10. Dismissal Modes Donut Breakdown.
11. Top Batters (Leading run-scorers, batting average, strike rate).
12. Top Bowlers (Leading wicket-takers, bowling average, economy).
"""

from typing import Any, Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

THEME_BG = "#ffffff"
CARD_BG = "#ffffff"
TEXT_COLOR = "#1a1a1a"
GRID_COLOR = "#e0e0e0"

COLOR_INN1 = "#38bdf8"  # Cyan
COLOR_INN2 = "#14b8a6"  # Teal
COLOR_INN3 = "#f97316"  # Orange
COLOR_INN4 = "#ef4444"  # Red

COLOR_TEST = "#ef4444"  # Red / Cherry
COLOR_ODI = "#38bdf8"   # Blue
COLOR_T20I = "#a855f7"  # Purple
COLOR_GREEN = "#22c55e" # Green
COLOR_AMBER = "#fbbf24" # Amber


def _apply_dark_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    """Applies clean full-width dark theme styling."""
    fig.update_layout(
        autosize=True,
        title=None,
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font={"color": TEXT_COLOR, "family": "Inter, system-ui, sans-serif", "size": 12},
        height=height,
        margin={"l": 55, "r": 35, "t": 25, "b": 55},
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


def create_historical_scoring_plot(df_data: pd.DataFrame, format_filter: str) -> Optional[str]:
    """
    Renders Historical Scoring Evolution:
    - If format is 'Test': 4 distinct lines for Innings 1, 2, 3, and 4!
    - If format is 'ODI' or 'T20I': Single yearly average innings score line.
    - If format is 'All Formats': 4 lines for Test, plus ODI and T20I lines.
    """
    if df_data.empty:
        return None

    fig = go.Figure()

    if format_filter == "Test":
        colors = {1: COLOR_INN1, 2: COLOR_INN2, 3: COLOR_INN3, 4: COLOR_INN4}
        labels = {1: "1st Innings", 2: "2nd Innings", 3: "3rd Innings", 4: "4th Innings"}

        for inn in [1, 2, 3, 4]:
            sub = df_data[df_data["innings"] == inn]
            if not sub.empty:
                x_vals = [int(v) for v in sub["yr"]]
                y_vals = [float(v) for v in sub["avg_score"]]
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        name=labels[inn],
                        mode="lines+markers",
                        line={"color": colors[inn], "width": 2.5},
                        marker={"size": 6, "color": colors[inn]},
                        hovertemplate=f"{labels[inn]}: %{{y}} runs (%{{x}})<extra></extra>",
                    )
                )
        fig.update_yaxes(title_text="Average Innings Score (Runs)")

    elif format_filter in ("ODI", "T20I"):
        x_vals = [int(v) for v in df_data["yr"]]
        y_vals = [float(v) for v in df_data["avg_score"]]
        clr = COLOR_ODI if format_filter == "ODI" else COLOR_T20I
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                name=f"{format_filter} Average Score",
                mode="lines+markers",
                line={"color": clr, "width": 3},
                marker={"size": 7, "color": clr},
                hovertemplate=f"{format_filter} Avg Score: %{{y}} runs (%{{x}})<extra></extra>",
            )
        )
        fig.update_yaxes(title_text=f"{format_filter} Average Innings Score (Runs)")

    else:
        for fmt, clr in [("Test", COLOR_TEST), ("ODI", COLOR_ODI), ("T20I", COLOR_T20I)]:
            sub = df_data[df_data["match_type"] == fmt]
            if not sub.empty:
                x_vals = [int(v) for v in sub["yr"]]
                y_vals = [float(v) for v in sub["avg_score"]]
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        name=f"{fmt} Avg Score",
                        mode="lines+markers",
                        line={"color": clr, "width": 2.5},
                        marker={"size": 6, "color": clr},
                        hovertemplate=f"{fmt} Avg: %{{y}} runs (%{{x}})<extra></extra>",
                    )
                )
        fig.update_yaxes(title_text="Average Score by Format (Runs)")

    fig.update_xaxes(title_text="Year", dtick=2)
    _apply_dark_layout(fig, height=380)
    return _render_fig(fig)


def create_run_rate_evolution_plot(df_fmt: pd.DataFrame, format_filter: str) -> Optional[str]:
    """Line chart tracking scoring rate (RPO = runs / legal overs) across time."""
    if df_fmt.empty:
        return None

    fig = go.Figure()

    if format_filter in ("Test", "ODI", "T20I"):
        sub = df_fmt[df_fmt["match_type"] == format_filter] if "match_type" in df_fmt.columns else df_fmt
        if not sub.empty:
            x_vals = [int(v) for v in sub["yr"]]
            y_vals = [float(v) for v in sub["run_rate"]]
            clr = COLOR_TEST if format_filter == "Test" else (COLOR_ODI if format_filter == "ODI" else COLOR_T20I)
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    name=f"{format_filter} Run Rate",
                    mode="lines+markers",
                    line={"color": clr, "width": 3},
                    marker={"size": 7, "color": clr},
                    hovertemplate="Year: %{x}<br>Scoring Rate: %{y} RPO<extra></extra>",
                )
            )
    else:
        for fmt, clr in [("Test", COLOR_TEST), ("ODI", COLOR_ODI), ("T20I", COLOR_T20I)]:
            sub = df_fmt[df_fmt["match_type"] == fmt]
            if not sub.empty:
                x_vals = [int(v) for v in sub["yr"]]
                y_vals = [float(v) for v in sub["run_rate"]]
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        name=f"{fmt} RPO",
                        mode="lines+markers",
                        line={"color": clr, "width": 2.5},
                        marker={"size": 6, "color": clr},
                        hovertemplate=f"{fmt}: %{{y}} RPO (%{{x}})<extra></extra>",
                    )
                )

    fig.update_xaxes(title_text="Year", dtick=2)
    fig.update_yaxes(title_text="Run Rate (Runs per 6 Legal Balls)")
    _apply_dark_layout(fig, height=360)
    return _render_fig(fig)


def create_boundary_evolution_plot(df_data: pd.DataFrame) -> Optional[str]:
    """Grouped bar chart for Fours & Sixes per innings + line for Boundary Run %."""
    if df_data.empty:
        return None

    df_yr = df_data.groupby("yr").agg(
        avg_fours=("avg_fours", "mean"),
        avg_sixes=("avg_sixes", "mean"),
        bdry_run_pct=("bdry_run_pct", "mean")
    ).reset_index()

    x_vals = [int(v) for v in df_yr["yr"]]
    fours = [round(float(v), 1) for v in df_yr["avg_fours"]]
    sixes = [round(float(v), 1) for v in df_yr["avg_sixes"]]
    pcts = [round(float(v), 1) for v in df_yr["bdry_run_pct"]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=fours,
            name="Avg Fours / Inn",
            marker_color=COLOR_INN1,
            hovertemplate="Fours: %{y} per innings<extra></extra>",
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=sixes,
            name="Avg Sixes / Inn",
            marker_color=COLOR_INN3,
            hovertemplate="Sixes: %{y} per innings<extra></extra>",
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=pcts,
            name="Boundary Runs %",
            mode="lines+markers",
            line={"color": COLOR_AMBER, "width": 3},
            marker={"size": 7, "color": COLOR_AMBER},
            hovertemplate="Runs from Boundaries: %{y}%<extra></extra>",
        ),
        secondary_y=True
    )

    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Year", dtick=2)
    fig.update_yaxes(title_text="Boundaries per Innings", secondary_y=False)
    fig.update_yaxes(title_text="% Total Runs from Boundaries", secondary_y=True, range=[0, 80])
    _apply_dark_layout(fig, height=380)
    return _render_fig(fig)


def create_format_comparison_bars(matrix: Dict[str, Any]) -> Optional[str]:
    """Benchmark bar chart comparing Test vs ODI vs T20I across core metrics."""
    if not matrix:
        return None

    formats = [f for f in ["Test", "ODI", "T20I"] if f in matrix]
    if not formats:
        return None

    rpo = [float(matrix[f]["run_rate"]) for f in formats]
    dot_pct = [float(matrix[f]["dot_pct"]) for f in formats]
    bdry_pct = [float(matrix[f]["boundary_run_pct"]) for f in formats]
    wkts_100 = [float(matrix[f]["wkts_per_100_balls"]) for f in formats]

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=["Run Rate (RPO)", "Dot Ball Share %", "Boundary Run %", "Wickets / 100 Balls"]
    )

    colors = [COLOR_TEST, COLOR_ODI, COLOR_T20I]

    # RPO
    fig.add_trace(
        go.Bar(x=formats, y=rpo, marker_color=colors, text=[f"{v}" for v in rpo], textposition="auto", showlegend=False),
        row=1, col=1
    )
    # Dot %
    fig.add_trace(
        go.Bar(x=formats, y=dot_pct, marker_color=colors, text=[f"{v}%" for v in dot_pct], textposition="auto", showlegend=False),
        row=1, col=2
    )
    # Boundary %
    fig.add_trace(
        go.Bar(x=formats, y=bdry_pct, marker_color=colors, text=[f"{v}%" for v in bdry_pct], textposition="auto", showlegend=False),
        row=1, col=3
    )
    # Wickets / 100
    fig.add_trace(
        go.Bar(x=formats, y=wkts_100, marker_color=colors, text=[f"{v}" for v in wkts_100], textposition="auto", showlegend=False),
        row=1, col=4
    )

    _apply_dark_layout(fig, height=340)
    return _render_fig(fig)


def create_test_innings_decay_plot(df_summary: pd.DataFrame) -> Optional[str]:
    """Test 4-Innings decay curve: Average runs scored and all-out % by innings."""
    if df_summary.empty:
        return None

    inns = [f"Innings {int(r['innings'])}" for _, r in df_summary.iterrows()]
    scores = [float(r["avg_runs"]) for _, r in df_summary.iterrows()]
    all_out = [float(r["all_out_pct"]) for _, r in df_summary.iterrows()]
    colors = [COLOR_INN1, COLOR_INN2, COLOR_INN3, COLOR_INN4]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=inns,
            y=scores,
            name="Average Score",
            marker_color=colors,
            text=[f"{s:.0f} runs" for s in scores],
            textposition="auto",
            hovertemplate="Stage: %{x}<br>Average Total: %{y} runs<extra></extra>",
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=inns,
            y=all_out,
            name="All-Out %",
            mode="lines+markers",
            line={"color": COLOR_AMBER, "width": 3, "dash": "dot"},
            marker={"size": 9, "color": COLOR_AMBER},
            hovertemplate="All-Out Rate: %{y}%<extra></extra>",
        ),
        secondary_y=True
    )

    fig.update_yaxes(title_text="Average Score (Runs)", secondary_y=False)
    fig.update_yaxes(title_text="All-Out Percentage (%)", secondary_y=True, range=[0, 100])
    _apply_dark_layout(fig, height=360)
    return _render_fig(fig)


def create_test_draw_trajectory_plot(df_draws: pd.DataFrame) -> Optional[str]:
    """Test Match Outcomes Evolution: Dramatic drop of draw % over the modern era."""
    if df_draws.empty:
        return None

    x_vals = [int(v) for v in df_draws["yr"]]
    draw_pcts = [float(v) for v in df_draws["draw_pct"]]
    result_pcts = [float(v) for v in df_draws["result_pct"]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=result_pcts,
            name="Outright Result %",
            mode="lines+markers",
            line={"color": COLOR_GREEN, "width": 3},
            marker={"size": 6, "color": COLOR_GREEN},
            hovertemplate="Year: %{x}<br>Result %: %{y}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=draw_pcts,
            name="Draw %",
            mode="lines+markers",
            line={"color": COLOR_AMBER, "width": 3, "dash": "dash"},
            marker={"size": 6, "color": COLOR_AMBER},
            hovertemplate="Year: %{x}<br>Draw %: %{y}%<extra></extra>",
        )
    )

    fig.update_xaxes(title_text="Year", dtick=2)
    fig.update_yaxes(title_text="Match Outcome Percentage (%)", range=[0, 105])
    _apply_dark_layout(fig, height=360)
    return _render_fig(fig)


def create_test_chasing_brackets_plot(df_targets: pd.DataFrame) -> Optional[str]:
    """4th Innings Target Pursuits: Successful Chase vs Draw vs Defeat."""
    if df_targets.empty:
        return None

    brackets = [str(r["target_range"]) for _, r in df_targets.iterrows()]
    wins = [float(r["chase_win_pct"]) for _, r in df_targets.iterrows()]
    draws = [float(r["draw_pct"]) for _, r in df_targets.iterrows()]
    defeats = [float(r["defeat_pct"]) for _, r in df_targets.iterrows()]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=brackets,
            y=wins,
            name="Successful Chase Win %",
            marker_color=COLOR_GREEN,
            text=[f"{v}%" for v in wins],
            textposition="inside",
        )
    )

    fig.add_trace(
        go.Bar(
            x=brackets,
            y=draws,
            name="Draw (Batted Out) %",
            marker_color=COLOR_AMBER,
            text=[f"{v}%" for v in draws],
            textposition="inside",
        )
    )

    fig.add_trace(
        go.Bar(
            x=brackets,
            y=defeats,
            name="Defeat / All-Out %",
            marker_color=COLOR_INN4,
            text=[f"{v}%" for v in defeats],
            textposition="inside",
        )
    )

    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="4th Innings Target Size (Runs)")
    fig.update_yaxes(title_text="Outcome Share (%)", range=[0, 100])
    _apply_dark_layout(fig, height=360)
    return _render_fig(fig)


def create_batting_accumulation_stacked_plot(df_acc: pd.DataFrame) -> Optional[str]:
    """100% Stacked Area chart of batting accumulation DNA over the years."""
    if df_acc.empty:
        return None

    x_vals = [int(v) for v in df_acc["yr"]]
    dots = [float(v) for v in df_acc["dot_pct"]]
    singles = [float(v) for v in df_acc["single_pct"]]
    twos = [float(v) for v in df_acc["two_pct"]]
    fours = [float(v) for v in df_acc["four_pct"]]
    sixes = [float(v) for v in df_acc["six_pct"]]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals, y=dots, name="Dot Balls %", mode="lines",
        stackgroup="one", line={"width": 0.5, "color": "#64748b"}, fillcolor="rgba(100, 116, 139, 0.7)"
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=singles, name="Singles %", mode="lines",
        stackgroup="one", line={"width": 0.5, "color": "#38bdf8"}, fillcolor="rgba(56, 189, 248, 0.7)"
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=twos, name="Twos %", mode="lines",
        stackgroup="one", line={"width": 0.5, "color": "#14b8a6"}, fillcolor="rgba(20, 184, 166, 0.7)"
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=fours, name="Fours %", mode="lines",
        stackgroup="one", line={"width": 0.5, "color": "#f97316"}, fillcolor="rgba(249, 115, 22, 0.7)"
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=sixes, name="Sixes %", mode="lines",
        stackgroup="one", line={"width": 0.5, "color": "#ef4444"}, fillcolor="rgba(239, 68, 68, 0.8)"
    ))

    fig.update_xaxes(title_text="Year", dtick=2)
    fig.update_yaxes(title_text="Delivery Outcome Share (%)", range=[0, 100])
    _apply_dark_layout(fig, height=380)
    return _render_fig(fig)


def create_target_chasing_plot(df_brackets: pd.DataFrame, format_name: str) -> Optional[str]:
    """Limited-Overs Target Chasing Success by Target Bracket."""
    if df_brackets.empty:
        return None

    brackets = [str(r["target_bracket"]) for _, r in df_brackets.iterrows()]
    chase_win = [float(r["chase_success_pct"]) for _, r in df_brackets.iterrows()]
    defended = [float(r["bat_first_defended_pct"]) for _, r in df_brackets.iterrows()]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=brackets,
            y=chase_win,
            name="Chasing Team Win %",
            marker_color=COLOR_INN1,
            text=[f"{v}%" for v in chase_win],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            x=brackets,
            y=defended,
            name="Bat 1st Defended %",
            marker_color=COLOR_INN3,
            text=[f"{v}%" for v in defended],
            textposition="auto",
        )
    )

    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text=f"{format_name} Target Range (Runs)")
    fig.update_yaxes(title_text="Win Probability (%)", range=[0, 105])
    _apply_dark_layout(fig, height=360)
    return _render_fig(fig)


def create_bowling_evolution_plot(df_bowling: pd.DataFrame) -> Optional[str]:
    """Wickets per 100 balls and Economy Rate trajectory."""
    if df_bowling.empty:
        return None

    x_vals = [int(v) for v in df_bowling["yr"]]
    wkts_100 = [float(v) for v in df_bowling["wkts_per_100_balls"]]
    econ = [float(v) for v in df_bowling["economy_rate"]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=wkts_100,
            name="Wickets / 100 Balls",
            mode="lines+markers",
            line={"color": COLOR_INN4, "width": 3},
            marker={"size": 7, "color": COLOR_INN4},
            hovertemplate="Year: %{x}<br>Wickets/100 Balls: %{y}<extra></extra>",
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=econ,
            name="Economy Rate (RPO)",
            mode="lines+markers",
            line={"color": COLOR_INN1, "width": 3, "dash": "dot"},
            marker={"size": 7, "color": COLOR_INN1},
            hovertemplate="Year: %{x}<br>Economy: %{y} RPO<extra></extra>",
        ),
        secondary_y=True
    )

    fig.update_xaxes(title_text="Year", dtick=2)
    fig.update_yaxes(title_text="Wickets per 100 Legal Deliveries", secondary_y=False)
    fig.update_yaxes(title_text="Bowling Economy (Runs / Over)", secondary_y=True)
    _apply_dark_layout(fig, height=380)
    return _render_fig(fig)


def create_dismissal_types_donut(df_dismissals: pd.DataFrame) -> Optional[str]:
    """Centered donut chart of historical dismissal mode distribution."""
    if df_dismissals.empty:
        return None

    labels = [str(r["dismissal_group"]) for _, r in df_dismissals.iterrows()]
    values = [float(r["share_pct"]) for _, r in df_dismissals.iterrows()]
    counts = [int(r["wkt_count"]) for _, r in df_dismissals.iterrows()]

    palette = [COLOR_INN1, COLOR_INN3, COLOR_INN2, COLOR_INN4, COLOR_AMBER, "#818cf8", "#cbd5e1"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker={"colors": palette[:len(labels)]},
                textinfo="percent",
                textposition="inside",
                domain={"x": [0.15, 0.85], "y": [0.05, 0.95]},
                showlegend=True,
                hovertemplate="Mode: %{label}<br>Share: %{value}%<br>Dismissals: %{customdata:,}<extra></extra>",
                customdata=counts,
            )
        ]
    )

    fig.update_layout(
        legend={
            "orientation": "h",
            "yanchor": "bottom", "y": -0.15,
            "xanchor": "center",  "x": 0.5,
            "font": {"size": 12},
        }
    )
    _apply_dark_layout(fig, height=360)
    return _render_fig(fig)


def create_top_batters_plot(df_bat: pd.DataFrame, format_name: str) -> Optional[str]:
    """Horizontal bar chart showing top run-scoring batters in this format."""
    if df_bat.empty:
        return None

    df_plot = df_bat.iloc[::-1]

    names = [str(v) for v in df_plot["player_name"]]
    runs = [int(v) for v in df_plot["runs"]]
    avgs = [float(v) if pd.notnull(v) else 0.0 for v in df_plot["batting_avg"]]
    srs = [float(v) if pd.notnull(v) else 0.0 for v in df_plot["strike_rate"]]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=names,
            x=runs,
            orientation="h",
            marker_color=COLOR_INN1,
            text=[f"{r:,} runs (Avg: {a:.1f}, SR: {s:.1f})" for r, a, s in zip(runs, avgs, srs)],
            textposition="inside",
            hovertemplate="Batter: %{y}<br>Total Runs: %{x:,}<extra></extra>",
        )
    )

    fig.update_xaxes(title_text=f"Total {format_name} Runs")
    fig.update_layout(margin={"l": 110, "r": 30, "t": 20, "b": 40})
    _apply_dark_layout(fig, height=max(280, len(names) * 38))
    return _render_fig(fig)


def create_top_bowlers_plot(df_bowl: pd.DataFrame, format_name: str) -> Optional[str]:
    """Horizontal bar chart showing top wicket-taking bowlers in this format."""
    if df_bowl.empty:
        return None

    df_plot = df_bowl.iloc[::-1]

    names = [str(v) for v in df_plot["player_name"]]
    wkts = [int(v) for v in df_plot["wickets"]]
    avgs = [float(v) if pd.notnull(v) else 0.0 for v in df_plot["bowling_avg"]]
    econs = [float(v) if pd.notnull(v) else 0.0 for v in df_plot["economy_rate"]]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=names,
            x=wkts,
            orientation="h",
            marker_color=COLOR_INN3,
            text=[f"{w} wkts (Avg: {a:.1f}, Econ: {e:.2f})" for w, a, e in zip(wkts, avgs, econs)],
            textposition="inside",
            hovertemplate="Bowler: %{y}<br>Total Wickets: %{x}<extra></extra>",
        )
    )

    fig.update_xaxes(title_text=f"Total {format_name} Wickets")
    fig.update_layout(margin={"l": 110, "r": 30, "t": 20, "b": 40})
    _apply_dark_layout(fig, height=max(280, len(names) * 38))
    return _render_fig(fig)


