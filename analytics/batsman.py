"""
batsman.py
All batsman analysis functions used by notebooks AND Flask routes.
Single source of truth — notebooks import from here for consistency.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from analytics.data_loader import get_df, get_phase_sr, get_profiles, get_matchup


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_player_df(name):
    return get_df()[get_df()['Full Name'].str.lower().str.contains(
        name.lower(), na=False, regex=False)]

def _valid(bat_df):
    return bat_df[bat_df['is_valid'] == 1]

def _scaler_norm(series):
    s = series.values.reshape(-1, 1)
    if s.max() == s.min():
        return pd.Series([0.5] * len(series), index=series.index)
    return pd.Series(MinMaxScaler().fit_transform(s).ravel(), index=series.index)


# ── Core functions ────────────────────────────────────────────────────────────

def overall_stats(name):
    """Return dict of overall batting stats for a player."""
    bat   = _get_player_df(name)
    valid = _valid(bat)
    if valid.empty:
        return None
    runs  = int(valid['run'].sum())
    balls = len(valid)
    wkts  = int(valid['isWicket'].sum())
    bdry  = int(valid['isBoundary'].sum())
    dots  = int((valid['run'] == 0).sum())
    sr    = round(runs / balls * 100, 2) if balls > 0 else 0
    avg   = round(runs / wkts, 2)        if wkts > 0 else None
    b_pct = round(bdry / balls * 100, 2) if balls > 0 else 0
    d_pct = round(dots / balls * 100, 2) if balls > 0 else 0
    return {
        'full_name'   : bat['Full Name'].iloc[0],
        'batting_style': bat['Batsman_Batting_Style'].mode().iloc[0],
        'playing_role': bat['Batsman_Playing_Role'].mode().iloc[0],
        'total_runs'  : runs,
        'balls_faced' : balls,
        'wickets_lost': wkts,
        'boundaries'  : bdry,
        'strike_rate' : sr,
        'average'     : avg,
        'boundary_pct': b_pct,
        'dot_ball_pct': d_pct,
        'matches'     : bat['match_obj_id'].nunique()
    }


def strength_zones(name, min_balls=5):
    """
    Per pitch zone: SR, boundary%, wicket%.
    strength_score = 0.6 * norm_SR + 0.4 * norm_boundary_pct
    Returns DataFrame sorted descending by strength_score.
    """
    bat   = _get_player_df(name)
    valid = _valid(bat)
    if valid.empty:
        return pd.DataFrame()

    zone = valid.groupby(['pitchLine', 'pitchLength']).agg(
        balls      =('run', 'count'),
        runs       =('run', 'sum'),
        boundaries =('isBoundary', 'sum'),
        wickets    =('isWicket', 'sum'),
        dot_balls  =('run', lambda x: (x == 0).sum())
    ).reset_index()
    zone = zone[zone['balls'] >= min_balls].copy()
    zone['strike_rate']  = (zone['runs']  / zone['balls'] * 100).round(2)
    zone['boundary_pct'] = (zone['boundaries'] / zone['balls'] * 100).round(2)
    zone['wicket_pct']   = (zone['wickets'] / zone['balls'] * 100).round(2)
    zone['dot_pct']      = (zone['dot_balls'] / zone['balls'] * 100).round(2)

    if len(zone) > 1:
        zone['strength_score'] = (
            0.6 * _scaler_norm(zone['strike_rate']) +
            0.4 * _scaler_norm(zone['boundary_pct'])
        ).round(3)
    else:
        zone['strength_score'] = 0.5
    return zone.sort_values('strength_score', ascending=False).reset_index(drop=True)


def weakness_zones(name, min_balls=5):
    """
    weakness_score = 0.6 * (1 - norm_SR) + 0.4 * norm_wicket_pct
    Returns DataFrame sorted descending by weakness_score.
    """
    zone = strength_zones(name, min_balls)
    if zone.empty:
        return pd.DataFrame()

    if len(zone) > 1:
        zone['weakness_score'] = (
            0.6 * (1 - _scaler_norm(zone['strike_rate'])) +
            0.4 * _scaler_norm(zone['wicket_pct'])
        ).round(3)
    else:
        zone['weakness_score'] = 0.5
    return zone.sort_values('weakness_score', ascending=False).reset_index(drop=True)


def shot_risk_profile(name, min_balls=10):
    """
    Per shot type: boundary%, wicket%, avg_runs, risk_reward ratio.
    risk_reward = avg_runs / (wicket_pct/100 + epsilon)
    """
    bat   = _get_player_df(name)
    valid = _valid(bat)
    if valid.empty:
        return pd.DataFrame()

    shot = valid.groupby('shotType').agg(
        balls      =('run', 'count'),
        runs       =('run', 'sum'),
        boundaries =('isBoundary', 'sum'),
        wickets    =('isWicket', 'sum')
    ).reset_index()
    shot = shot[shot['balls'] >= min_balls].copy()
    shot['avg_runs']    = (shot['runs']       / shot['balls']).round(3)
    shot['boundary_pct']= (shot['boundaries'] / shot['balls'] * 100).round(2)
    shot['wicket_pct']  = (shot['wickets']    / shot['balls'] * 100).round(2)
    shot['risk_reward'] = (shot['avg_runs'] / (shot['wicket_pct'] / 100 + 1e-6)).round(2)
    shot['shot_label']  = shot['shotType'].str.replace('_', ' ').str.title()
    return shot.sort_values('boundary_pct', ascending=False).reset_index(drop=True)


def vs_bowling_style(name, min_balls=10):
    """SR, boundary%, wicket% against each bowling style."""
    bat   = _get_player_df(name)
    valid = _valid(bat)
    if valid.empty:
        return pd.DataFrame()

    vs = valid.groupby('Bowler_Bowling_Style').agg(
        balls      =('run', 'count'),
        runs       =('run', 'sum'),
        boundaries =('isBoundary', 'sum'),
        wickets    =('isWicket', 'sum')
    ).reset_index()
    vs = vs[vs['balls'] >= min_balls].copy()
    vs['strike_rate']  = (vs['runs']       / vs['balls'] * 100).round(2)
    vs['boundary_pct'] = (vs['boundaries'] / vs['balls'] * 100).round(2)
    vs['wicket_pct']   = (vs['wickets']    / vs['balls'] * 100).round(2)
    return vs.sort_values('strike_rate', ascending=False).reset_index(drop=True)


def chase_vs_set(name):
    """Compare 1st innings (setting) vs 2nd innings (chasing) performance."""
    bat   = _get_player_df(name)
    valid = _valid(bat)
    if valid.empty:
        return pd.DataFrame()

    perf = valid.groupby('inningNumber').agg(
        balls      =('run', 'count'),
        runs       =('run', 'sum'),
        boundaries =('isBoundary', 'sum'),
        wickets    =('isWicket', 'sum')
    ).reset_index()
    perf['strike_rate']  = (perf['runs']       / perf['balls'] * 100).round(2)
    perf['boundary_pct'] = (perf['boundaries'] / perf['balls'] * 100).round(2)
    perf['wicket_pct']   = (perf['wickets']    / perf['balls'] * 100).round(2)
    perf['label'] = perf['inningNumber'].map({1: 'Setting (1st Inn)', 2: 'Chasing (2nd Inn)'})
    return perf


def pressure_performance(name):
    """SR and boundary% under different wicket-fall pressure levels."""
    bat   = _get_player_df(name)
    valid = _valid(bat).copy()
    if valid.empty:
        return pd.DataFrame()

    valid['pressure_band'] = pd.cut(
        valid['totalWickets'],
        bins=[-1, 2, 5, 10],
        labels=['Low (0-2 wkts)', 'Medium (3-5 wkts)', 'High (6+ wkts)'])

    perf = valid.groupby('pressure_band', observed=True).agg(
        balls      =('run', 'count'),
        runs       =('run', 'sum'),
        boundaries =('isBoundary', 'sum'),
        wickets    =('isWicket', 'sum')
    ).reset_index()
    perf['strike_rate']  = (perf['runs']       / perf['balls'] * 100).round(2)
    perf['boundary_pct'] = (perf['boundaries'] / perf['balls'] * 100).round(2)
    return perf


def ground_performance(name, min_balls=20):
    """SR and boundary% at each ground."""
    bat   = _get_player_df(name)
    valid = _valid(bat)
    if valid.empty:
        return pd.DataFrame()

    gnd = valid.groupby('Ground Name').agg(
        balls      =('run', 'count'),
        runs       =('run', 'sum'),
        boundaries =('isBoundary', 'sum'),
        wickets    =('isWicket', 'sum')
    ).reset_index()
    gnd = gnd[gnd['balls'] >= min_balls].copy()
    gnd['strike_rate']  = (gnd['runs']       / gnd['balls'] * 100).round(2)
    gnd['boundary_pct'] = (gnd['boundaries'] / gnd['balls'] * 100).round(2)
    gnd['wicket_pct']   = (gnd['wickets']    / gnd['balls'] * 100).round(2)
    return gnd.sort_values('strike_rate', ascending=False).reset_index(drop=True)


def phase_stats(name):
    """Return phase_sr rows for a batsman (from the pre-computed CSV)."""
    ps  = get_phase_sr()
    key = name.lower().strip()
    return ps[ps['name_key'].str.contains(key, na=False, regex=False)].copy()


def head_to_head_vs_bowlers(name, bowler_names=None, top_n=10):
    """
    Best bowlers against this batsman by success_index_scaled.
    If bowler_names provided, filter to those names first.
    """
    matchup = get_matchup()
    key  = name.lower().strip()
    mask = matchup['Batsman_Name'].str.lower().str.contains(key, na=False, regex=False)
    sub  = matchup[mask].copy()
    if sub.empty:
        return pd.DataFrame()

    if bowler_names:
        b_lower = [b.lower() for b in bowler_names]
        bm = sub['Bowler_Name'].str.lower().apply(lambda n: any(b in n for b in b_lower))
        if bm.any():
            sub = sub[bm]

    return sub.nlargest(top_n, 'success_index_scaled').reset_index(drop=True)


def pregame_summary(name):
    """
    Return a concise dict summary for pre-game use.
    Contains: overall stats, top strength zone, top weakness zone,
    best bowling style (highest SR), worst bowling style (highest wicket%).
    """
    stats = overall_stats(name)
    if stats is None:
        return {'error': f"No data found for '{name}'"}

    sz  = strength_zones(name)
    wz  = weakness_zones(name)
    vbs = vs_bowling_style(name)

    top_strength = (sz.iloc[0][['pitchLine', 'pitchLength', 'strike_rate',
                                  'boundary_pct', 'strength_score']].to_dict()
                    if not sz.empty else {})
    top_weakness = (wz.iloc[0][['pitchLine', 'pitchLength', 'strike_rate',
                                  'wicket_pct', 'weakness_score']].to_dict()
                    if not wz.empty else {})
    best_style   = (vbs.iloc[0][['Bowler_Bowling_Style', 'strike_rate']].to_dict()
                    if not vbs.empty else {})
    worst_style  = (vbs.nsmallest(1, 'strike_rate').iloc[0]
                    [['Bowler_Bowling_Style', 'strike_rate', 'wicket_pct']].to_dict()
                    if not vbs.empty else {})

    return {
        'overall'        : stats,
        'top_strength'   : top_strength,
        'top_weakness'   : top_weakness,
        'best_vs_style'  : best_style,
        'worst_vs_style' : worst_style
    }
