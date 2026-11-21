"""
Python Plotly Visualizations for Batting Analytics.

Generates 5 interactive, dark-themed cricket visual charts:
1. create_phase_plot(): Dual-axis bar and line chart of runs and strike rate per phase.
2. create_chase_plot(): Grouped bar chart comparing 1st Batting (Setting Target) vs 2nd Innings Chasing.
3. create_bowler_discipline_plot(): Grouped bar of scoring and dismissals against Bowler Disciplines.
4. create_dismissal_plot(): Interactive Donut chart of dismissal modes.
5. create_venue_plot(): Horizontal bar chart of top grounds by runs, colored by Strike Rate.
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
        autosize=True,                 # Stretch dynamically to container width
        title=None,                    # heading lives in HTML, not inside the chart
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font={"color": TEXT_COLOR, "family": "Inter, system-ui, sans-serif", "size": 12},
        height=height,
        # Tight margins so chart fills its container edge-to-edge
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


def create_phase_plot(df_phase: pd.DataFrame) -> Optional[str]:
    """
    Creates a dual-axis chart:
    - Bar: Runs scored in Powerplay, Middle, Death.
    - Line: Strike rate across phases.
    """
    if df_phase.empty:
        return None

    # Convert to native Python types to avoid Plotly 6.x binary encoding bug
    phases      = df_phase["match_phase"].tolist()
    runs        = [int(v) for v in df_phase["runs"].tolist()]
    balls       = [int(v) for v in df_phase["balls"].tolist()]
    strike_rate = [round(float(v), 2) for v in df_phase["strike_rate"].tolist()]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar chart for Runs
    fig.add_trace(
        go.Bar(
            x=phases,
            y=runs,
            name="Runs Scored",
            marker_color=ACCENT_BLUE,
            text=runs,
            textposition="auto",
            hovertemplate="Phase: %{x}<br>Runs: %{y}<br>Balls: %{customdata[0]}<extra></extra>",
            customdata=[[b] for b in balls],
        ),
        secondary_y=False,
    )

    # Line chart for Strike Rate
    fig.add_trace(
        go.Scatter(
            x=phases,
            y=strike_rate,
            name="Strike Rate",
            mode="lines+markers+text",
            text=[str(sr) for sr in strike_rate],
            textposition="top center",
            line={"color": ACCENT_ORANGE, "width": 3},
            marker={"size": 9, "color": ACCENT_ORANGE},
            hovertemplate="Phase: %{x}<br>Strike Rate: %{y}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Runs Scored", secondary_y=False, gridcolor="#e0e0e0")
    fig.update_yaxes(title_text="Strike Rate (Runs / 100 Balls)", secondary_y=True, showgrid=False)

    _apply_dark_layout(fig)
    return _render_fig(fig)


def create_test_inning_plot(df_test: pd.DataFrame) -> Optional[str]:
    """
    Creates a Test Match Innings Progression Plot:
    - Bars: Runs scored in Inning 1, Inning 2, Inning 3, Inning 4.
    - Line/Markers: Strike Rate across the 4 Test innings.
    """
    if df_test.empty:
        return None

    # Convert to native Python types to avoid Plotly 6.x binary encoding bug
    inning_labels = df_test["inning_label"].tolist()
    runs          = [int(v) for v in df_test["runs"].tolist()]
    balls         = [int(v) for v in df_test["balls"].tolist()]
    dismissals    = [int(v) for v in df_test["dismissals"].tolist()]
    strike_rate   = [round(float(v), 2) for v in df_test["strike_rate"].tolist()]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=inning_labels,
            y=runs,
            name="Runs Scored",
            marker_color=ACCENT_BLUE,
            text=runs,
            textposition="auto",
            hovertemplate="Inning: %{x}<br>Runs: %{y}<br>Balls: %{customdata[0]}<br>Dismissals: %{customdata[1]}<extra></extra>",
            customdata=[[b, d] for b, d in zip(balls, dismissals)],
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=inning_labels,
            y=strike_rate,
            name="Strike Rate",
            mode="lines+markers+text",
            text=[str(sr) for sr in strike_rate],
            textposition="top center",
            line={"color": ACCENT_ORANGE, "width": 3},
            marker={"size": 9, "color": ACCENT_ORANGE},
            hovertemplate="Inning: %{x}<br>Strike Rate: %{y}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Runs Scored", secondary_y=False, gridcolor="#e0e0e0")
    fig.update_yaxes(title_text="Strike Rate", secondary_y=True, showgrid=False)

    _apply_dark_layout(fig)
    return _render_fig(fig)


def create_chase_plot(df_chase: pd.DataFrame) -> Optional[str]:
    """
    Creates grouped comparison bars between Setting a Target (1st Innings) and Chasing (2nd Innings).
    """
    if df_chase.empty:
        return None

    # Convert to native Python types to avoid Plotly 6.x binary encoding bug
    contexts     = df_chase["innings_context"].tolist()
    strike_rate  = [round(float(v), 2) for v in df_chase["strike_rate"].tolist()]
    boundary_pct = [round(float(v), 1) for v in df_chase["boundary_pct"].tolist()]
    dot_pct      = [round(float(v), 1) for v in df_chase["dot_pct"].tolist()]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=contexts,
            y=strike_rate,
            name="Strike Rate",
            marker_color=ACCENT_BLUE,
            text=[str(sr) for sr in strike_rate],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            x=contexts,
            y=boundary_pct,
            name="Boundary %",
            marker_color=ACCENT_GREEN,
            text=[f"{v}%" for v in boundary_pct],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            x=contexts,
            y=dot_pct,
            name="Dot Ball %",
            marker_color=ACCENT_ORANGE,
            text=[f"{v}%" for v in dot_pct],
            textposition="auto",
        )
    )

    fig.update_layout(barmode="group")
    _apply_dark_layout(fig)
    return _render_fig(fig)


def create_opponent_team_plot(df_team: pd.DataFrame) -> Optional[str]:
    """
    Plots performance against Opponent Teams:
    - Bars: Runs scored against each team.
    - Line/Markers: Strike Rate against each team.
    """
    if df_team.empty:
        return None

    # Convert to native Python types to avoid Plotly 6.x binary encoding bug
    opponents   = df_team["opponent_team"].tolist()
    runs        = [int(v) for v in df_team["runs"].tolist()]
    matches     = [int(v) for v in df_team["matches"].tolist()]
    balls       = [int(v) for v in df_team["balls"].tolist()]
    strike_rate = [round(float(v), 2) for v in df_team["strike_rate"].tolist()]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=opponents,
            y=runs,
            name="Runs Scored",
            marker_color=ACCENT_BLUE,
            text=runs,
            textposition="auto",
            hovertemplate="Opponent: %{x}<br>Runs: %{y}<br>Matches: %{customdata[0]}<br>Balls: %{customdata[1]}<extra></extra>",
            customdata=[[m, b] for m, b in zip(matches, balls)],
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=opponents,
            y=strike_rate,
            name="Strike Rate",
            mode="lines+markers+text",
            text=[str(sr) for sr in strike_rate],
            textposition="top center",
            line={"color": ACCENT_ORANGE, "width": 3},
            marker={"size": 9, "color": ACCENT_ORANGE},
            hovertemplate="Opponent: %{x}<br>Strike Rate: %{y}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Runs Scored", secondary_y=False, gridcolor="#e0e0e0")
    fig.update_yaxes(title_text="Strike Rate", secondary_y=True, showgrid=False)

    _apply_dark_layout(fig)
    return _render_fig(fig)


def create_dismissal_plot(df_dismissal: pd.DataFrame) -> Optional[str]:
    """
    Creates an interactive Donut chart of dismissal modes (caught, bowled, lbw, etc.).
    """
    if df_dismissal.empty:
        return None

    # Convert to native Python types to avoid Plotly 6.x binary encoding bug
    labels = df_dismissal["dismissal_kind"].tolist()
    values = [int(v) for v in df_dismissal["dismissal_count"].tolist()]

    colors = [ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_PURPLE, ACCENT_RED]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker={"colors": colors[:len(df_dismissal)]},
                textinfo="label+percent",
                hovertemplate="Dismissal: %{label}<br>Count: %{value}<br>Share: %{percent}<extra></extra>",
            )
        ]
    )

    _apply_dark_layout(fig)
    return _render_fig(fig)


def create_venue_plot(df_venue: pd.DataFrame) -> Optional[str]:
    """
    Creates a horizontal bar chart showing top grounds by runs scored and colored by Strike Rate.
    """
    if df_venue.empty:
        return None

    df_sorted = df_venue.sort_values(by="runs", ascending=True)

    # Convert to native Python types to avoid Plotly 6.x binary encoding bug
    grounds     = df_sorted["ground_name"].tolist()
    runs        = [int(v) for v in df_sorted["runs"].tolist()]
    strike_rate = [round(float(v), 2) for v in df_sorted["strike_rate"].tolist()]
    text_labels = [f"{r} runs (SR {sr})" for r, sr in zip(runs, strike_rate)]

    fig = go.Figure(
        go.Bar(
            x=runs,
            y=grounds,
            orientation="h",
            marker={
                "color": strike_rate,
                "colorscale": "Tealgrn",
                "showscale": True,
                "colorbar": {"title": "Strike Rate"},
            },
            text=text_labels,
            textposition="auto",
            hovertemplate="Ground: %{y}<br>Runs: %{x}<br>Strike Rate: %{marker.color}<extra></extra>",
        )
    )

    _apply_dark_layout(fig, height=500)
    return _render_fig(fig)


