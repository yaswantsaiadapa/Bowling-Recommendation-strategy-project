"""
bowler.py
All bowler analysis functions used by notebooks AND Flask routes.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from analytics.data_loader import get_df, get_bowler_stats, get_matchup


def _get_bowler_df(name):
    return get_df()[get_df()['Full Name_bowler'].str.lower().str.contains(
        name.lower(), na=False, regex=False)]

def _valid(bdf):
    return bdf[bdf['is_valid'] == 1]

def _scaler_norm(series):
    s = series.values.reshape(-1, 1)
    if s.max() == s.min():
        return pd.Series([0.5] * len(series), index=series.index)
    return pd.Series(MinMaxScaler().fit_transform(s).ravel(), index=series.index)


def overall_stats(name):
    bdf   = _get_bowler_df(name)
    valid = _valid(bdf)
    if valid.empty:
        return None

    balls = len(valid)
    runs  = int(valid['run'].sum())
    wkts  = int(valid['isWicket'].sum())
    bdry  = int(valid['isBoundary'].sum())
    dots  = int((valid['run'] == 0).sum())
    econ  = round(runs / (balls / 6), 2) if balls > 0 else 0
    bsr   = round(balls / wkts, 2)       if wkts > 0 else None
    dot_p = round(dots  / balls * 100, 2) if balls > 0 else 0
    bdry_p= round(bdry  / balls * 100, 2) if balls > 0 else 0
    wkt_p = round(wkts  / balls * 100, 2) if balls > 0 else 0

    return {
        'full_name'    : bdf['Full Name_bowler'].iloc[0],
        'bowling_style': bdf['Bowler_Bowling_Style'].mode().iloc[0],
        'playing_role' : bdf['Bowler_Playing_Role'].mode().iloc[0],
        'balls_bowled' : balls,
        'total_runs'   : runs,
        'total_wickets': wkts,
        'economy_rate' : econ,
        'bowling_sr'   : bsr,
        'dot_ball_pct' : dot_p,
        'boundary_pct' : bdry_p,
        'wicket_pct'   : wkt_p,
        'matches'      : bdf['match_obj_id'].nunique()
    }


def strength_zones(name, min_balls=5):
    """
    Per pitch zone: economy, wicket%, dot%.
    success_score = 0.5 * (1 - norm_economy) + 0.5 * norm_wicket_pct
    """
    bdf   = _get_bowler_df(name)
    valid = _valid(bdf)
    if valid.empty:
        return pd.DataFrame()

    zone = valid.groupby(['pitchLine', 'pitchLength']).agg(
        balls      =('run', 'count'),
        runs       =('run', 'sum'),
        wickets    =('isWicket', 'sum'),
        boundaries =('isBoundary', 'sum'),
        dot_balls  =('run', lambda x: (x == 0).sum())
    ).reset_index()
    zone = zone[zone['balls'] >= min_balls].copy()
    zone['economy']      = (zone['runs']       / (zone['balls'] / 6)).round(2)
    zone['wicket_pct']   = (zone['wickets']    / zone['balls'] * 100).round(2)
    zone['boundary_pct'] = (zone['boundaries'] / zone['balls'] * 100).round(2)
    zone['dot_pct']      = (zone['dot_balls']  / zone['balls'] * 100).round(2)

    if len(zone) > 1:
        zone['success_score'] = (
            0.5 * (1 - _scaler_norm(zone['economy'])) +
            0.5 * _scaler_norm(zone['wicket_pct'])
        ).round(3)
    else:
        zone['success_score'] = 0.5
    return zone.sort_values('success_score', ascending=False).reset_index(drop=True)


def weakness_zones(name, min_balls=5):
    """
    weakness_score = 0.6 * norm_economy + 0.4 * norm_boundary_pct
    High = zones where bowler leaks most runs.
    """
    zone = strength_zones(name, min_balls)
    if zone.empty:
        return pd.DataFrame()

    if len(zone) > 1:
        zone['weakness_score'] = (
            0.6 * _scaler_norm(zone['economy']) +
            0.4 * _scaler_norm(zone['boundary_pct'])
        ).round(3)
    else:
        zone['weakness_score'] = 0.5
    return zone.sort_values('weakness_score', ascending=False).reset_index(drop=True)


def phase_stats(name):
    """Economy, wicket%, dot%, boundary% per phase."""
    bdf   = _get_bowler_df(name)
    valid = _valid(bdf)
    if valid.empty:
        return pd.DataFrame()

    phase = valid.groupby('match_phase').agg(
        balls      =('run', 'count'),
        runs       =('run', 'sum'),
        wickets    =('isWicket', 'sum'),
        boundaries =('isBoundary', 'sum'),
        dot_balls  =('run', lambda x: (x == 0).sum())
    ).reset_index()
    phase['economy']      = (phase['runs']       / (phase['balls'] / 6)).round(2)
    phase['wicket_pct']   = (phase['wickets']    / phase['balls'] * 100).round(2)
    phase['boundary_pct'] = (phase['boundaries'] / phase['balls'] * 100).round(2)
    phase['dot_pct']      = (phase['dot_balls']  / phase['balls'] * 100).round(2)
    phase['match_phase']  = pd.Categorical(
        phase['match_phase'], ['Powerplay', 'Middle', 'Death'], ordered=True)
    return phase.sort_values('match_phase').reset_index(drop=True)


def vs_batting_style(name):
    """Economy, wicket%, boundary% against LHB and RHB."""
    bdf   = _get_bowler_df(name)
    valid = _valid(bdf)
    if valid.empty:
        return pd.DataFrame()

    vs = valid.groupby('Batsman_Batting_Style').agg(
        balls      =('run', 'count'),
        runs       =('run', 'sum'),
        wickets    =('isWicket', 'sum'),
        boundaries =('isBoundary', 'sum')
    ).reset_index()
    vs['economy']      = (vs['runs']       / (vs['balls'] / 6)).round(2)
    vs['wicket_pct']   = (vs['wickets']    / vs['balls'] * 100).round(2)
    vs['boundary_pct'] = (vs['boundaries'] / vs['balls'] * 100).round(2)
    return vs


def economy_trend(name):
    """Economy per over number (1–20)."""
    bdf   = _get_bowler_df(name)
    valid = _valid(bdf).copy()
    if valid.empty:
        return pd.DataFrame()

    valid['over_int'] = valid['oversActual'].astype(int)
    trend = valid.groupby('over_int').agg(
        balls=('run', 'count'), runs=('run', 'sum')
    ).reset_index()
    trend['economy'] = (trend['runs'] / (trend['balls'] / 6)).round(2)
    return trend


def wicket_delivery_profile(name, min_balls=3):
    """Which (line, length, phase) combination takes the most wickets?"""
    bdf   = _get_bowler_df(name)
    valid = _valid(bdf)
    if valid.empty:
        return pd.DataFrame()

    wkt = valid.groupby(['pitchLine', 'pitchLength', 'match_phase']).agg(
        balls  =('run', 'count'),
        wickets=('isWicket', 'sum')
    ).reset_index()
    wkt = wkt[wkt['balls'] >= min_balls].copy()
    wkt['wicket_pct'] = (wkt['wickets'] / wkt['balls'] * 100).round(2)
    return wkt.sort_values('wickets', ascending=False).reset_index(drop=True)


def best_batsmen_to_target(name, opposition_names=None, top_n=5):
    """
    Given opposition batsmen, return those the bowler has historically dismissed
    most effectively (highest success_index_scaled in matchup model).
    """
    matchup = get_matchup()
    key  = name.lower().strip()
    col  = 'Bowler_Full_Name' if 'Bowler_Full_Name' in matchup.columns else 'Bowler_Name'
    mask = matchup[col].str.lower().str.contains(key, na=False, regex=False)
    sub  = matchup[mask].copy()
    if sub.empty:
        return pd.DataFrame()

    if opposition_names:
        o_lower = [o.lower() for o in opposition_names]
        ocol = 'Batsman_Full_Name' if 'Batsman_Full_Name' in sub.columns else 'Batsman_Name'
        om = sub[ocol].str.lower().apply(lambda n: any(o in n for o in o_lower))
        if om.any():
            sub = sub[om]

    return sub.nlargest(top_n, 'success_index_scaled').reset_index(drop=True)


def pregame_summary(name):
    """Concise pre-game summary dict for Flask and notebooks."""
    stats = overall_stats(name)
    if stats is None:
        return {'error': f"No data found for bowler '{name}'"}

    sz  = strength_zones(name)
    wz  = weakness_zones(name)
    ps  = phase_stats(name)
    vbs = vs_batting_style(name)

    top_strength = (sz.iloc[0][['pitchLine', 'pitchLength', 'economy',
                                  'wicket_pct', 'success_score']].to_dict()
                    if not sz.empty else {})
    top_weakness = (wz.iloc[0][['pitchLine', 'pitchLength', 'economy',
                                  'boundary_pct', 'weakness_score']].to_dict()
                    if not wz.empty else {})
    best_phase   = (ps.loc[ps['wicket_pct'].idxmax(), ['match_phase', 'wicket_pct',
                                                         'economy']].to_dict()
                    if not ps.empty else {})
    worst_phase  = (ps.loc[ps['economy'].idxmax(), ['match_phase', 'economy',
                                                      'boundary_pct']].to_dict()
                    if not ps.empty else {})

    return {
        'overall'       : stats,
        'top_strength'  : top_strength,
        'top_weakness'  : top_weakness,
        'best_phase'    : best_phase,
        'worst_phase'   : worst_phase,
        'vs_lhb_rhb'    : vbs.to_dict('records') if not vbs.empty else []
    }
