import os
import json
import sys

# Make sure analytics package is importable from app.py location
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify, abort, redirect
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots               

import analytics.batting_engine as bat_engine
import analytics.batting_plots as bat_plots
import analytics.bowling_engine as bowl_engine
import analytics.bowling_plots as bowl_plots
import analytics.matchup_engine as match_engine
import analytics.matchup_plots as match_plots
import analytics.venue_engine as venue_engine
import analytics.venue_plots as venue_plots
import analytics.simulator_engine as sim_engine
import analytics.simulator_plots as sim_plots
import analytics.insights_engine as ins_engine
import analytics.insights_plots as ins_plots

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _decode_bdata(obj):
    """Recursively convert Plotly typed-array dicts {dtype, bdata} to plain Python lists.
    Newer Plotly (>5.12) encodes numeric arrays as base64 binary for performance.
    Plotly.js handles it, but the conversion ensures compatibility across all versions.
    """
    import base64, struct
    if isinstance(obj, dict):
        if 'bdata' in obj and 'dtype' in obj:
            dtype_map = {'f8': ('d',8), 'f4': ('f',4), 'i4': ('i',4),
                         'i8': ('q',8), 'u1': ('B',1), 'u4': ('I',4)}
            raw          = base64.b64decode(obj['bdata'])
            fmt_ch, size = dtype_map.get(obj['dtype'], ('d', 8))
            count        = len(raw) // size
            return list(struct.unpack_from(f'{count}{fmt_ch}', raw))
        return {k: _decode_bdata(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_decode_bdata(i) for i in obj]
    return obj


def _fig_to_json(fig):
    """Convert a Plotly figure to a plain-Python dict for Jinja tojson.
    - Applies dark theme
    - Decodes binary-encoded numeric arrays to plain lists so Plotly.js
      renders data points correctly in all browser environments.
    """
    fig.update_layout(**_DARK)
    raw = json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    return _decode_bdata(raw)

import plotly

# Dark theme applied to every chart
_DARK = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(26,29,46,0.9)',
    font=dict(color='#e8eaf6'),
    xaxis=dict(gridcolor='#2e3250', tickfont=dict(color='#8891b2')),
    yaxis=dict(gridcolor='#2e3250', tickfont=dict(color='#8891b2')),
)


def zone_heatmap_json(zone_df, score_col, title, colorscale):
    if zone_df is None or zone_df.empty:
        return None
    pivot = zone_df.pivot_table(score_col, index='pitchLength', columns='pitchLine').fillna(0)
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values.tolist(),
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=colorscale,
        text=[[round(v, 2) for v in row] for row in pivot.values.tolist()],
        texttemplate='%{text}',
        showscale=True
    ))
    fig.update_layout(title=title, height=320, margin=dict(l=10,r=10,t=40,b=10),
                      xaxis_title='Pitch Line', yaxis_title='Pitch Length', **_DARK)
    return _fig_to_json(fig)


def bar_json(x, y, title, color=None, orientation='v', colorscale=None):
    if color and colorscale:
        fig = px.bar(x=x, y=y, color=color, color_continuous_scale=colorscale,
                     title=title, orientation=orientation)
    else:
        fig = px.bar(x=x, y=y, title=title, orientation=orientation)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340,
                      showlegend=False)
    return _fig_to_json(fig)


# ── Global Player Cache for Home Search ─────────────────────────────────────────
_CACHED_BATSMEN = None
_CACHED_BOWLERS = None

def _get_top_player_names():
    from analytics.db import get_db_connection
    try:
        con = get_db_connection()
        batsmen = [r[0] for r in con.execute("SELECT DISTINCT batter FROM fact_deliveries_master WHERE batter IS NOT NULL ORDER BY batter").fetchall()]
        bowlers = [r[0] for r in con.execute("SELECT DISTINCT bowler FROM fact_deliveries_master WHERE bowler IS NOT NULL ORDER BY bowler").fetchall()]
        con.close()
        return batsmen, bowlers
    except Exception:
        return [], []

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    """Home page with search bars."""
    global _CACHED_BATSMEN, _CACHED_BOWLERS
    if _CACHED_BATSMEN is None or _CACHED_BOWLERS is None:
        _CACHED_BATSMEN, _CACHED_BOWLERS = _get_top_player_names()
    return render_template('index.html', batsmen=_CACHED_BATSMEN, bowlers=_CACHED_BOWLERS)


@app.route('/batsman/<path:name>')
def batsman_report(name):
    """Seamless forward to master Batting Analytics."""
    return redirect(f"/batting?player={name.strip()}")


@app.route('/bowler/<path:name>')
def bowler_report(name):
    """Seamless forward to master Bowling Analytics."""
    return redirect(f"/bowling?player={name.strip()}")


@app.route('/matchup')
def matchup():
    """
    Matchup & Opponent Intelligence Route.
    Supports 3 confrontation modes:
    1. 'h2h'       : Batter vs Bowler head-to-head duel
    2. 'bat_team'  : Batter vs Opposition country
    3. 'bowl_team' : Bowler vs Opposition country
    """
    mode   = request.args.get('mode', 'h2h').strip()
    batter = request.args.get('batter', 'V Kohli').strip()
    bowler = request.args.get('bowler', 'JM Anderson').strip()
    team   = request.args.get('team', 'Australia').strip()

    all_teams = match_engine.get_all_teams()
    overview = None
    format_plot = None
    phase_plot = None
    shots_plot = None
    bowlers_plot = None
    batters_plot = None
    dismissal_plot = None
    dismissals_log = []

    if mode == 'h2h':
        overview = match_engine.get_h2h_overview(batter, bowler)
        if overview:
            df_format = match_engine.get_h2h_format_splits(batter, bowler)
            format_plot = match_plots.create_h2h_format_plot(df_format)
            df_phase = match_engine.get_h2h_phase_splits(batter, bowler)
            phase_plot = match_plots.create_h2h_phase_plot(df_phase)
            shots = match_engine.get_h2h_shots(batter, bowler)
            shots_plot = match_plots.create_h2h_shots_plot(shots)
            dismissals_log = match_engine.get_h2h_dismissals_log(batter, bowler)

    elif mode == 'bat_team':
        overview = match_engine.get_batter_vs_team_overview(batter, team)
        if overview:
            df_format = match_engine.get_batter_vs_team_format_splits(batter, team)
            format_plot = match_plots.create_bat_team_format_plot(df_format)
            df_bowlers = match_engine.get_batter_vs_team_bowlers(batter, team, limit=8)
            bowlers_plot = match_plots.create_bat_team_bowlers_plot(df_bowlers)
            df_phase = match_engine.get_batter_vs_team_phase_splits(batter, team)
            phase_plot = match_plots.create_bat_team_phase_plot(df_phase)

    elif mode == 'bowl_team':
        overview = match_engine.get_bowler_vs_team_overview(bowler, team)
        if overview:
            df_format = match_engine.get_bowler_vs_team_format_splits(bowler, team)
            format_plot = match_plots.create_bowl_team_format_plot(df_format)
            df_batters = match_engine.get_bowler_vs_team_batters(bowler, team, limit=8)
            batters_plot = match_plots.create_bowl_team_batters_plot(df_batters)
            df_dismissals = match_engine.get_bowler_vs_team_dismissals(bowler, team)
            dismissal_plot = match_plots.create_bowl_team_dismissal_plot(df_dismissals)

    famous_clashes = [
        {"title": "V Kohli vs JM Anderson", "batter": "V Kohli", "bowler": "JM Anderson", "mode": "h2h"},
        {"title": "RG Sharma vs TA Boult", "batter": "RG Sharma", "bowler": "TA Boult", "mode": "h2h"},
        {"title": "SPD Smith vs SCJ Broad", "batter": "SPD Smith", "bowler": "SCJ Broad", "mode": "h2h"},
        {"title": "DA Warner vs R Ashwin", "batter": "DA Warner", "bowler": "R Ashwin", "mode": "h2h"},
        {"title": "JC Buttler vs JJ Bumrah", "batter": "JC Buttler", "bowler": "JJ Bumrah", "mode": "h2h"},
        {"title": "V Kohli vs Australia", "batter": "V Kohli", "team": "Australia", "mode": "bat_team"},
        {"title": "RG Sharma vs Pakistan", "batter": "RG Sharma", "team": "Pakistan", "mode": "bat_team"},
        {"title": "JJ Bumrah vs Australia", "bowler": "JJ Bumrah", "team": "Australia", "mode": "bowl_team"},
        {"title": "R Ashwin vs England", "bowler": "R Ashwin", "team": "England", "mode": "bowl_team"},
    ]

    return render_template(
        'matchup.html',
        mode=mode,
        batter=batter,
        bowler=bowler,
        team=team,
        all_teams=all_teams,
        overview=overview,
        format_plot=format_plot,
        phase_plot=phase_plot,
        shots_plot=shots_plot,
        bowlers_plot=bowlers_plot,
        batters_plot=batters_plot,
        dismissal_plot=dismissal_plot,
        dismissals_log=dismissals_log,
        famous_clashes=famous_clashes,
    )


@app.route('/insights')
def insights():
    """Global Cricket Insights & Historical Evolution Dashboard."""
    # Format selection for deep-dive section (defaults to Test)
    fmt_raw = request.args.get('format', 'Test').strip()
    selected_format = fmt_raw if fmt_raw in ('Test', 'ODI', 'T20I') else 'Test'

    team = request.args.get('team', 'All Teams').strip()
    venue = request.args.get('venue', 'All Venues').strip()
    yr_min_str = request.args.get('year_min', '2001').strip()
    yr_max_str = request.args.get('year_max', '2026').strip()

    try:
        yr_min = int(yr_min_str)
    except (ValueError, TypeError):
        yr_min = 2001

    try:
        yr_max = int(yr_max_str)
    except (ValueError, TypeError):
        yr_max = 2026

    filter_opts = ins_engine.get_filter_options()

    # ── Top Section: Multi-Format Global Macro Comparisons ──────────
    kpis = ins_engine.get_executive_kpis("All Formats", team, venue, yr_min, yr_max)
    format_matrix = ins_engine.get_format_comparison_matrix(team, venue, yr_min, yr_max)
    fmt_bars_plot = ins_plots.create_format_comparison_bars(format_matrix)
    bat_acc_all = ins_engine.get_batting_accumulation_dna("All Formats", team, venue, yr_min, yr_max)
    acc_plot = ins_plots.create_batting_accumulation_stacked_plot(bat_acc_all['yearly'])
    toss_chase_all = ins_engine.get_toss_and_chasing_analytics("All Formats", team, venue, yr_min, yr_max)
    era_matrix = ins_engine.get_era_comparison("All Formats", team, venue)

    # ── Bottom Section: Format-Specific Visualizations & Dynamics ───
    fmt_scoring = ins_engine.get_yearly_scoring_evolution(selected_format, team, venue, yr_min, yr_max)
    rr_plot = ins_plots.create_run_rate_evolution_plot(fmt_scoring['by_format'], selected_format)
    bdry_plot = ins_plots.create_boundary_evolution_plot(fmt_scoring['by_format'])

    test_decay_plot = None
    test_draw_plot = None
    test_targets_plot = None
    target_plot = None

    if selected_format == 'Test':
        scoring_plot = ins_plots.create_historical_scoring_plot(fmt_scoring['by_innings'], 'Test')
        test_data = ins_engine.get_test_4innings_evolution(team, venue, yr_min, yr_max)
        test_decay_plot = ins_plots.create_test_innings_decay_plot(test_data['summary_innings'])
        test_draw_plot = ins_plots.create_test_draw_trajectory_plot(test_data['draw_trajectory'])
        test_targets_plot = ins_plots.create_test_chasing_brackets_plot(test_data['target_pursuits'])
    else:
        scoring_plot = ins_plots.create_historical_scoring_plot(fmt_scoring['by_format'], selected_format)
        bracket_key = 't20i_brackets' if selected_format == 'T20I' else 'odi_brackets'
        target_plot = ins_plots.create_target_chasing_plot(toss_chase_all[bracket_key], selected_format)

    # Format-specific bowling and dismissals
    bowl_data = ins_engine.get_bowling_evolution(selected_format, team, venue, yr_min, yr_max)
    bowling_plot = ins_plots.create_bowling_evolution_plot(bowl_data['yearly_bowling'])
    dismissals_plot = ins_plots.create_dismissal_types_donut(bowl_data['dismissal_types'])

    # Format-specific Top Performers for the team / era
    performers = ins_engine.get_top_performers(selected_format, team, venue, yr_min, yr_max)
    top_bat_plot = ins_plots.create_top_batters_plot(performers['top_batters'], selected_format)
    top_bowl_plot = ins_plots.create_top_bowlers_plot(performers['top_bowlers'], selected_format)

    return render_template(
        'insights.html',
        filter_options=filter_opts,
        selected_format=selected_format,
        selected_team=team,
        selected_venue=venue,
        selected_year_min=yr_min,
        selected_year_max=yr_max,
        kpis=kpis,
        format_matrix=format_matrix,
        dna_format=bat_acc_all['by_format'],
        toss_summary=toss_chase_all['toss_summary'],
        era_matrix=era_matrix,
        scoring_plot=scoring_plot,
        rr_plot=rr_plot,
        bdry_plot=bdry_plot,
        fmt_bars_plot=fmt_bars_plot,
        test_decay_plot=test_decay_plot,
        test_draw_plot=test_draw_plot,
        test_targets_plot=test_targets_plot,
        target_plot=target_plot,
        acc_plot=acc_plot,
        bowling_plot=bowling_plot,
        dismissals_plot=dismissals_plot,
        top_batters=performers['top_batters'],
        top_bowlers=performers['top_bowlers'],
        top_bat_plot=top_bat_plot,
        top_bowl_plot=top_bowl_plot,
    )


# ── Batting Analytics Route ───────────────────────────────────────────────────

@app.route('/batting')
def batting_analytics():
    player_name = request.args.get('player', 'V Kohli').strip()
    match_type = request.args.get('match_type', 'All').strip()
    phase = request.args.get('phase', 'All').strip()

    is_test_format = (match_type == 'Test')
    is_all_format  = (match_type == 'All')
    effective_phase = 'All' if is_test_format else phase

    overview = bat_engine.get_batter_overview(player_name, match_type=match_type, match_phase=effective_phase)

    # Generate Plotly charts
    phase_plot = None
    test_inning_plot = None
    chase_plot = None
    team_plot = None
    dismissal_plot = None
    venue_plot = None

    if overview:
        if is_test_format:
            # Test: innings 1-4 progression only — no phase chart, no chase matrix
            df_test = bat_engine.get_batter_test_inning_splits(player_name)
            test_inning_plot = bat_plots.create_test_inning_plot(df_test)

        elif is_all_format:
            # All Formats: mixing formats makes phase/chase meaningless — skip both
            pass

        else:
            # T20I or ODI: show phase profile AND chase matrix
            df_phase = bat_engine.get_batter_phase_splits(player_name, match_type=match_type)
            phase_plot = bat_plots.create_phase_plot(df_phase)

            df_chase = bat_engine.get_batter_chase_splits(player_name, match_type=match_type)
            chase_plot = bat_plots.create_chase_plot(df_chase)

        # Always show these three (filtered to whichever match_type was selected)
        df_team = bat_engine.get_batter_opponent_team_splits(player_name, match_type=match_type)
        team_plot = bat_plots.create_opponent_team_plot(df_team)

        df_dismissal = bat_engine.get_batter_dismissal_breakdown(player_name, match_type=match_type)
        dismissal_plot = bat_plots.create_dismissal_plot(df_dismissal)

        df_venue = bat_engine.get_batter_venue_splits(player_name, match_type=match_type)
        venue_plot = bat_plots.create_venue_plot(df_venue)

    popular_batters = [
        "V Kohli", "RG Sharma", "JC Buttler", "DA Warner", "Babar Azam",
        "SA Yadav", "H Klaasen", "KL Rahul", "MS Dhoni", "GJ Maxwell",
        "Q de Kock", "AB de Villiers", "CH Gayle", "AD Russell", "SPD Smith", "JE Root", "KS Williamson"
    ]

    return render_template(
        'batting_analytics.html',
        selected_batter=player_name,
        selected_format=match_type,
        selected_phase=effective_phase,
        is_test_format=is_test_format,
        is_all_format=is_all_format,
        overview=overview,
        phase_plot=phase_plot,
        test_inning_plot=test_inning_plot,
        chase_plot=chase_plot,
        team_plot=team_plot,
        dismissal_plot=dismissal_plot,
        venue_plot=venue_plot,
        popular_batters=popular_batters,
    )


@app.route('/api/search_batters')
def api_search_batters():
    q = request.args.get('q', '').strip()
    results = bat_engine.search_batsmen(q, limit=12)
    return jsonify(results)


# ── Bowling Analytics Route ───────────────────────────────────────────────────

@app.route('/bowling')
def bowling_analytics():
    player_name = request.args.get('player', 'JJ Bumrah').strip()
    match_type = request.args.get('match_type', 'All').strip()
    phase = request.args.get('phase', 'All').strip()

    is_test_format = (match_type == 'Test')
    is_all_format  = (match_type == 'All')
    effective_phase = 'All' if (is_test_format or is_all_format) else phase

    overview = bowl_engine.get_bowler_overview(
        player_name, match_type=match_type, match_phase=effective_phase
    )

    # Initialize Plotly charts
    phase_plot = None
    test_inning_plot = None
    defend_plot = None
    team_plot = None
    dismissal_plot = None
    venue_plot = None

    if overview:
        if is_test_format:
            # Test: Innings 1-4 progression only — no phase breakdown or defending matrix
            df_test = bowl_engine.get_bowler_test_inning_splits(player_name)
            test_inning_plot = bowl_plots.create_bowler_test_inning_plot(df_test)

        elif is_all_format:
            # All Formats: cross-format comparison only (no phase or defending split)
            pass

        else:
            # T20I or ODI: show phase impact profile AND defending matrix
            df_phase = bowl_engine.get_bowler_phase_splits(player_name, match_type=match_type)
            phase_plot = bowl_plots.create_bowler_phase_plot(df_phase)

            df_defend = bowl_engine.get_bowler_defending_splits(player_name, match_type=match_type)
            defend_plot = bowl_plots.create_bowler_defending_plot(df_defend)

        # Universal charts (filtered to selected match_type)
        df_dismissal = bowl_engine.get_bowler_dismissal_breakdown(player_name, match_type=match_type)
        dismissal_plot = bowl_plots.create_bowler_dismissal_plot(df_dismissal)

        df_team = bowl_engine.get_bowler_opponent_splits(player_name, match_type=match_type)
        team_plot = bowl_plots.create_bowler_opponent_plot(df_team)

        df_venue = bowl_engine.get_bowler_venue_splits(player_name, match_type=match_type)
        venue_plot = bowl_plots.create_bowler_venue_plot(df_venue)

    popular_bowlers = [
        "JJ Bumrah", "R Ashwin", "Rashid Khan", "MA Starc", "TA Boult",
        "K Rabada", "RA Jadeja", "Shaheen Shah Afridi", "PWA Cummins",
        "Kuldeep Yadav", "YS Chahal", "Mohammed Shami", "B Kumar",
        "SL Malinga", "DW Steyn", "JM Anderson", "SCJ Broad"
    ]

    return render_template(
        'bowling_analytics.html',
        selected_bowler=player_name,
        selected_format=match_type,
        selected_phase=effective_phase,
        is_test_format=is_test_format,
        is_all_format=is_all_format,
        overview=overview,
        phase_plot=phase_plot,
        test_inning_plot=test_inning_plot,
        defend_plot=defend_plot,
        team_plot=team_plot,
        dismissal_plot=dismissal_plot,
        venue_plot=venue_plot,
        popular_bowlers=popular_bowlers,
    )


@app.route('/api/search_bowlers')
def api_search_bowlers():
    q = request.args.get('q', '').strip()
    results = bowl_engine.search_bowlers(q, limit=12)
    return jsonify(results)


# ── Team & Venue Intelligence Route ──────────────────────────────────────────

@app.route('/venues')
def venues():
    """
    Team & Venue Intelligence Route.
    Supports 2 views:
    1. 'venue' : Stadium / Ground Profiler
    2. 'team'  : National Team Dynamics & Opponents Record
    """
    tab = request.args.get('tab', 'venue').strip()
    selected_venue = request.args.get('venue', 'Melbourne Cricket Ground').strip()
    selected_team = request.args.get('team', 'India').strip()
    match_type = request.args.get('format', 'All').strip()

    popular_venues = venue_engine.get_popular_venues()
    popular_teams = venue_engine.get_popular_teams()

    venue_overview = None
    venue_outcomes_plot = None
    venue_phase_plot = None
    test_decay_plot = None
    test_duration_plot = None
    format_comp_plot = None
    venue_top_batters = []
    venue_top_bowlers = []

    team_overview = None
    team_opponents_plot = None
    team_phase_plot = None
    team_fortress_plot = None

    if tab == 'venue':
        venue_overview = venue_engine.get_venue_overview(selected_venue, match_type=match_type)
        if venue_overview:
            if venue_overview.get("is_test"):
                df_test_inn = venue_engine.get_venue_test_innings(selected_venue)
                test_decay_plot = venue_plots.create_test_innings_decay_plot(df_test_inn)
                durations = venue_engine.get_venue_test_durations(selected_venue)
                test_duration_plot = venue_plots.create_test_duration_plot(durations)
                venue_top_batters = venue_engine.get_venue_top_batters(selected_venue, match_type="Test", limit=5)
                venue_top_bowlers = venue_engine.get_venue_top_bowlers(selected_venue, match_type="Test", limit=5)
            elif venue_overview.get("is_all"):
                format_comp_plot = venue_plots.create_venue_format_comparison_plot(venue_overview.get("format_cards", []))
                venue_top_batters = venue_engine.get_venue_top_batters(selected_venue, match_type="All", limit=5)
                venue_top_bowlers = venue_engine.get_venue_top_bowlers(selected_venue, match_type="All", limit=5)
            else:
                outcomes = venue_engine.get_venue_outcomes(selected_venue, match_type=match_type)
                venue_outcomes_plot = venue_plots.create_venue_outcomes_plot(outcomes)
                df_phase = venue_engine.get_venue_phase_splits(selected_venue, match_type=match_type)
                venue_phase_plot = venue_plots.create_venue_phase_plot(df_phase)
                venue_top_batters = venue_engine.get_venue_top_batters(selected_venue, match_type=match_type, limit=5)
                venue_top_bowlers = venue_engine.get_venue_top_bowlers(selected_venue, match_type=match_type, limit=5)

    elif tab == 'team':
        team_overview = venue_engine.get_team_overview(selected_team, match_type=match_type)
        if team_overview:
            df_opp = venue_engine.get_team_opponents_record(selected_team, match_type=match_type, limit=10)
            team_opponents_plot = venue_plots.create_team_opponents_plot(df_opp)
            df_phase = venue_engine.get_team_phase_dna(selected_team, match_type=match_type)
            team_phase_plot = venue_plots.create_team_phase_plot(df_phase)
            df_fortress = venue_engine.get_team_fortress_venues(selected_team, match_type=match_type, limit=8)
            team_fortress_plot = venue_plots.create_team_fortress_plot(df_fortress)

    return render_template(
        'venue_analytics.html',
        tab=tab,
        selected_venue=selected_venue,
        selected_team=selected_team,
        selected_format=match_type,
        popular_venues=popular_venues,
        popular_teams=popular_teams,
        venue_overview=venue_overview,
        venue_outcomes_plot=venue_outcomes_plot,
        venue_phase_plot=venue_phase_plot,
        test_decay_plot=test_decay_plot,
        test_duration_plot=test_duration_plot,
        format_comp_plot=format_comp_plot,
        venue_top_batters=venue_top_batters,
        venue_top_bowlers=venue_top_bowlers,
        team_overview=team_overview,
        team_opponents_plot=team_opponents_plot,
        team_phase_plot=team_phase_plot,
        team_fortress_plot=team_fortress_plot,
    )


# ── Pre-Game Strategy & Match Simulator Route ─────────────────────────────────

@app.route('/simulator')
@app.route('/strategy')
def simulator():
    """
    Pre-Game Strategy & Match Outcome Simulator.
    Simulates win probability, pitch behavior, and conditions-smart Ideal Playing XIs.
    """
    team1 = request.args.get('team1', 'India').strip()
    team2 = request.args.get('team2', 'Australia').strip()
    venue = request.args.get('venue', 'Melbourne Cricket Ground').strip()
    match_type = request.args.get('format', 'T20I').strip()
    era = request.args.get('era', 'current').strip()

    all_teams = sim_engine.get_all_popular_teams()
    all_venues = sim_engine.get_all_popular_venues()

    # Pitch & Conditions Analysis (era-filtered)
    pitch = sim_engine.get_venue_pitch_profile(venue, match_type=match_type, era=era)
    pitch_plot = sim_plots.create_pitch_distribution_plot(pitch)

    # Condition-smart Ideal Playing XI Selection for both teams
    team1_xi = sim_engine.select_ideal_xi(team1, team2, venue, match_type=match_type, era=era)
    team2_xi = sim_engine.select_ideal_xi(team2, team1, venue, match_type=match_type, era=era)

    # Matchup Outcome Simulation
    sim_data = sim_engine.simulate_matchup(team1, team2, team1_xi, team2_xi, venue, match_type=match_type, era=era)
    win_plot = sim_plots.create_win_probability_plot(sim_data)
    scores_plot = sim_plots.create_projected_scores_plot(sim_data)

    return render_template(
        'simulator.html',
        team1=team1,
        team2=team2,
        venue=venue,
        format=match_type,
        era=era,
        all_teams=all_teams,
        all_venues=all_venues,
        pitch=pitch,
        pitch_plot=pitch_plot,
        scores_plot=scores_plot,
        team1_xi=team1_xi,
        team2_xi=team2_xi,
        sim_data=sim_data,
        win_plot=win_plot,
    )


# ── API routes ────────────────────────────────────────────────────────────────

@app.route('/api/players')
def api_players():
    global _CACHED_BATSMEN, _CACHED_BOWLERS
    if _CACHED_BATSMEN is None or _CACHED_BOWLERS is None:
        _CACHED_BATSMEN, _CACHED_BOWLERS = _get_top_player_names()
    return jsonify({'batsmen': _CACHED_BATSMEN, 'bowlers': _CACHED_BOWLERS})


@app.route('/api/batsman/<path:name>')
def api_batsman(name):
    overview = bat_engine.get_batter_overview(name.strip())
    if not overview:
        return jsonify({'error': f'Batsman "{name}" not found'}), 404
    return jsonify(overview)


@app.route('/api/bowler/<path:name>')
def api_bowler(name):
    overview = bowl_engine.get_bowler_overview(name.strip())
    if not overview:
        return jsonify({'error': f'Bowler "{name}" not found'}), 404
    return jsonify(overview)


@app.errorhandler(404)
def not_found(e):
    # Inline HTML — no template file needed, never crashes
    msg  = str(e)
    html = (
        '<!DOCTYPE html><html><head><title>404</title>'
        '<style>'
        'body{background:#0f1117;color:#e8eaf6;font-family:sans-serif;text-align:center;padding:4rem}'
        'h1{font-size:5rem;color:#e74c3c}'
        'a{color:#4f8ef7}'
        '</style></head><body>'
        '<h1>404</h1><p>' + msg + '</p><a href=/>&larr; Home</a>'
        '</body></html>'
    )
    return html, 404


@app.route('/favicon.ico')
def favicon():
    # Return empty 204 to stop browsers requesting favicon.ico repeatedly
    return '', 204


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

