"""
Python Plotly Visualizations for Bowling Analytics.

Generates 6 interactive, dark-themed cricket visual charts:
1. create_bowler_phase_plot(): Dual-axis bar and line chart of wickets and economy rate per phase.
2. create_bowler_test_inning_plot(): Test innings 1-4 progression of wickets and bowling average.
3. create_bowler_defending_plot(): Grouped bar chart comparing 1st Innings (Restricting) vs 2nd Innings (Defending Target).
4. create_bowler_dismissal_plot(): Interactive Donut chart of bowler dismissal modes (bowled, caught, lbw, stumped).
5. create_bowler_opponent_plot(): Dual-axis chart of wickets and economy against opponent teams.
6. create_bowler_venue_plot(): Horizontal bar chart of top grounds by wickets, colored by economy rate.
"""

from typing import Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Dark theme palette for sports analytics
THEME_BG = "#ffffff"        # Dark slate background
CARD_BG = "#ffffff"         # Card background
TEXT_COLOR = "#1a1a1a"      # White / off-white text
ACCENT_BLUE = "#38bdf8"     # Sky blue
ACCENT_GREEN = "#22c55e"    # Vibrant emerald
ACCENT_ORANGE = "#f97316"   # Energetic orange
ACCENT_RED = "#ef4444"      # Crimson red
ACCENT_PURPLE = "#a855f7"   # Purple
GRID_COLOR = "#e0e0e0"      # Muted border


def _apply_dark_layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    """
    Applies clean full-width dark-theme styling.
    Title is intentionally blank â€” the HTML template renders a richer heading above each chart.
    """
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
        hoverlabel={
            "bgcolor": "#ffffff",
            "bordercolor": "#e0e0e0",
            "font": {"color": TEXT_COLOR, "size": 12},
        },
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="#e0e0e0", gridwidth=1,
        zeroline=False, tickfont={"size": 11},
        linecolor="#e0e0e0",
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#e0e0e0", gridwidth=1,
        zeroline=False, tickfont={"size": 11},
        linecolor="#e0e0e0",
    )
    return fig


def _render_fig(fig: go.Figure) -> str:
    """
    Renders Plotly figure with 100% responsive width and interactive resize.
    """
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displayModeBar": False},
        default_width="100%",
    )


def create_bowler_phase_plot(df_phase: pd.DataFrame) -> Optional[str]:
    """
    Creates a dual-axis chart:
    - Bar: Wickets taken in Powerplay, Middle, Death.
    - Line: Economy rate across phases.
    """
    if df_phase.empty:
        return None

    phases = df_phase["match_phase"].tolist()
    wickets = [int(v) for v in df_phase["wickets"].tolist()]
    economy = [round(float(v), 2) for v in df_phase["economy_rate"].tolist()]
    overs = [round(int(v) / 6.0, 1) for v in df_phase["legal_balls"].tolist()]
    runs = [int(v) for v in df_phase["runs_conceded"].tolist()]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar chart for Wickets
    fig.add_trace(
        go.Bar(
            x=phases,
            y=wickets,
            name="Wickets Taken",
            marker_color=ACCENT_BLUE,
            text=wickets,
            textposition="auto",
            hovertemplate="Phase: %{x}<br>Wickets: %{y}<br>Overs: %{customdata[0]}<br>Runs Conceded: %{customdata[1]}<extra></extra>",
            customdata=[[o, r] for o, r in zip(overs, runs)],
        ),
        secondary_y=False,
    )

    # Line chart for Economy Rate
    fig.add_trace(
        go.Scatter(
            x=phases,
            y=economy,
            name="Economy Rate",
            mode="lines+markers+text",
            text=[f"{e:.2f}" for e in economy],
            textposition="top center",
            line={"color": ACCENT_ORANGE, "width": 3},
            marker={"size": 9, "color": ACCENT_ORANGE},
            hovertemplate="Phase: %{x}<br>Economy: %{y:.2f} rpo<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Wickets Taken", secondary_y=False, gridcolor="#e0e0e0")
    fig.update_yaxes(title_text="Economy Rate (Runs / Over)", secondary_y=True, showgrid=False)

    _apply_dark_layout(fig)
    return _render_fig(fig)


def create_bowler_test_inning_plot(df_test: pd.DataFrame) -> Optional[str]:
    """
    Creates a Test Match Innings Progression Plot:
    - Bars: Wickets taken in Inning 1, Inning 2, Inning 3, Inning 4.
    - Line: Bowling Average across the 4 Test innings.
    """
    if df_test.empty:
        return None

    inning_labels = df_test["inning_label"].tolist()
    wickets = [int(v) for v in df_test["wickets"].tolist()]
    overs = df_test["overs"].tolist()
    runs = [int(v) for v in df_test["runs_conceded"].tolist()]
    avg = [round(float(v), 2) for v in df_test["bowling_average"].tolist()]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=inning_labels,
            y=wickets,
            name="Wickets Taken",
            marker_color=ACCENT_BLUE,
            text=wickets,
            textposition="auto",
            hovertemplate="Inning: %{x}<br>Wickets: %{y}<br>Overs: %{customdata[0]}<br>Runs: %{customdata[1]}<extra></extra>",
            customdata=[[o, r] for o, r in zip(overs, runs)],
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=inning_labels,
            y=avg,
            name="Bowling Average",
            mode="lines+markers+text",
            text=[f"{a:.1f}" if a > 0 else "N/A" for a in avg],
            textposition="top center",
            line={"color": ACCENT_GREEN, "width": 3},
            marker={"size": 9, "color": ACCENT_GREEN},
            hovertemplate="Inning: %{x}<br>Average: %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Wickets Taken", secondary_y=False, gridcolor="#e0e0e0")
    fig.update_yaxes(title_text="Bowling Average (Runs / Wicket)", secondary_y=True, showgrid=False)

    _apply_dark_layout(fig)
    return _render_fig(fig)


def create_bowler_defending_plot(df_defend: pd.DataFrame) -> Optional[str]:
    """
    Creates grouped comparison bars between Bowling 1st (Restricting) and Bowling 2nd (Defending Target).
    """
    if df_defend.empty:
        return None

    contexts = df_defend["innings_context"].tolist()
    economy = [round(float(v), 2) for v in df_defend["economy_rate"].tolist()]
    strike_rate = [round(float(v), 1) for v in df_defend["strike_rate"].tolist()]
    dot_pct = [round(float(v), 1) for v in df_defend["dot_pct"].tolist()]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=contexts,
            y=economy,
            name="Economy Rate",
            marker_color=ACCENT_BLUE,
            text=[f"{e:.2f}" for e in economy],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            x=contexts,
            y=strike_rate,
            name="Strike Rate (Balls/Wkt)",
            marker_color=ACCENT_GREEN,
            text=[f"{s:.1f}" for s in strike_rate],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            x=contexts,
            y=dot_pct,
            name="Dot Ball %",
            marker_color=ACCENT_ORANGE,
            text=[f"{d:.1f}%" for d in dot_pct],
            textposition="auto",
        )
    )

    fig.update_layout(barmode="group")
    _apply_dark_layout(fig)
    return _render_fig(fig)


def create_bowler_dismissal_plot(df_dismissal: pd.DataFrame) -> Optional[str]:
    """
    Creates an interactive Donut chart of bowler dismissal modes (caught, bowled, lbw, stumped, etc.).
    """
    if df_dismissal.empty:
        return None

    labels = [str(k).title() for k in df_dismissal["dismissal_kind"].tolist()]
    values = [int(v) for v in df_dismissal["dismissal_count"].tolist()]

    colors = [ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_PURPLE, ACCENT_RED, "#eab308"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker={"colors": colors[: len(df_dismissal)]},
                textinfo="label+percent",
                hovertemplate="Dismissal Mode: %{label}<br>Wickets: %{value}<br>Share: %{percent}<extra></extra>",
            )
        ]
    )

    _apply_dark_layout(fig)
    return _render_fig(fig)


def create_bowler_opponent_plot(df_opponent: pd.DataFrame) -> Optional[str]:
    """
    Plots performance against Opponent Teams:
    - Bars: Wickets taken against each team.
    - Line: Economy Rate against each team.
    """
    if df_opponent.empty:
        return None

    opponents = df_opponent["opponent_team"].tolist()
    wickets = [int(v) for v in df_opponent["wickets"].tolist()]
    matches = [int(v) for v in df_opponent["matches"].tolist()]
    runs = [int(v) for v in df_opponent["runs_conceded"].tolist()]
    economy = [round(float(v), 2) for v in df_opponent["economy_rate"].tolist()]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=opponents,
            y=wickets,
            name="Wickets Taken",
            marker_color=ACCENT_BLUE,
            text=wickets,
            textposition="auto",
            hovertemplate="Opponent: %{x}<br>Wickets: %{y}<br>Matches: %{customdata[0]}<br>Runs Conceded: %{customdata[1]}<extra></extra>",
            customdata=[[m, r] for m, r in zip(matches, runs)],
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=opponents,
            y=economy,
            name="Economy Rate",
            mode="lines+markers+text",
            text=[f"{e:.2f}" for e in economy],
            textposition="top center",
            line={"color": ACCENT_ORANGE, "width": 3},
            marker={"size": 9, "color": ACCENT_ORANGE},
            hovertemplate="Opponent: %{x}<br>Economy: %{y:.2f} rpo<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Wickets Taken", secondary_y=False, gridcolor="#e0e0e0")
    fig.update_yaxes(title_text="Economy Rate (Runs / Over)", secondary_y=True, showgrid=False)

    _apply_dark_layout(fig)
    return _render_fig(fig)


def create_bowler_venue_plot(df_venue: pd.DataFrame) -> Optional[str]:
    """
    Creates a horizontal bar chart showing top grounds by wickets taken, colored by Economy Rate.
    """
    if df_venue.empty:
        return None

    df_sorted = df_venue.sort_values(by="wickets", ascending=True)

    grounds = df_sorted["ground_name"].tolist()
    wickets = [int(v) for v in df_sorted["wickets"].tolist()]
    economy = [round(float(v), 2) for v in df_sorted["economy_rate"].tolist()]
    text_labels = [f"{w} wkts (Econ {e:.2f})" for w, e in zip(wickets, economy)]

    fig = go.Figure(
        go.Bar(
            x=wickets,
            y=grounds,
            orientation="h",
            marker={
                "color": economy,
                "colorscale": "Tealgrn_r",  # Lower economy is better, so reversed palette
                "showscale": True,
                "colorbar": {"title": "Economy"},
            },
            text=text_labels,
            textposition="auto",
            hovertemplate="Ground: %{y}<br>Wickets: %{x}<br>Economy: %{marker.color:.2f}<extra></extra>",
        )
    )

    _apply_dark_layout(fig, height=500)
    return _render_fig(fig)


