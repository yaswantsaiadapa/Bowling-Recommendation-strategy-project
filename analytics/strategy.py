"""
strategy.py
Bowling strategy and field placement recommendation engine.
"""
import pandas as pd
from analytics.data_loader import get_df, get_phase_sr, get_profiles, get_matchup

# ── Strategy lookup tables ────────────────────────────────────────────────────

STRATEGY_RULES = {
    ('High Threat',   'fast')  : {'line': 'OUTSIDE_OFFSTUMP', 'length': 'SHORT_OF_A_GOOD_LENGTH',
        'plan': 'Back-of-a-length outside off. Attack with bouncers. Avoid full deliveries.'},
    ('High Threat',   'spin')  : {'line': 'ON_THE_STUMPS', 'length': 'FULL',
        'plan': 'Flight and drift. Invite the drive. Mid-on and long-off deep.'},
    ('High Threat',   'medium'): {'line': 'OUTSIDE_OFFSTUMP', 'length': 'GOOD_LENGTH',
        'plan': 'Off-stump channel with swing. No width. Mix cutters.'},
    ('Medium Threat', 'fast')  : {'line': 'ON_THE_STUMPS', 'length': 'GOOD_LENGTH',
        'plan': 'Good length on and outside off. Mix yorkers and slower balls.'},
    ('Medium Threat', 'spin')  : {'line': 'OUTSIDE_OFFSTUMP', 'length': 'GOOD_LENGTH',
        'plan': 'Disciplined off-stump line. Vary flight. Build dot-ball pressure.'},
    ('Medium Threat', 'medium'): {'line': 'OUTSIDE_OFFSTUMP', 'length': 'FULL',
        'plan': 'Full and swing. Mix slower cutters. Keep boundary riders set.'},
    ('Low Threat',    'fast')  : {'line': 'ON_THE_STUMPS', 'length': 'YORKER',
        'plan': 'Attack with yorkers. Straight and full. Batsman under pressure.'},
    ('Low Threat',    'spin')  : {'line': 'ON_THE_STUMPS', 'length': 'FULL',
        'plan': 'Toss it up. Attack with close catchers. Go for the wicket.'},
    ('Low Threat',    'medium'): {'line': 'ON_THE_STUMPS', 'length': 'GOOD_LENGTH',
        'plan': 'Consistent lines. Build dot-ball pressure.'},
}

PHASE_NOTE = {
    'Powerplay': ' [PP] Slip cordon; fielders up; restrict boundaries.',
    'Middle'   : ' [Mid] Rotate bowlers; vary pace; compound dot pressure.',
    'Death'    : ' [Death] Wide yorkers; all boundary riders back.'
}

FIELD_TEMPLATES = {
    ('Powerplay', 'fast')  : ['2nd Slip','Gully','Cover Point','Mid-Off','Mid-On','Square Leg','Fine Leg','Third Man','Mid-Wicket'],
    ('Powerplay', 'spin')  : ['Slip','Cover','Mid-Off','Mid-On','Mid-Wicket','Square Leg','Deep Mid-Wicket','Long-On','Fine Leg'],
    ('Powerplay', 'medium'): ['Slip','Gully','Cover Point','Mid-Off','Mid-On','Square Leg','Fine Leg','Third Man','Mid-Wicket'],
    ('Middle',    'fast')  : ['Slip','Cover Point','Deep Extra Cover','Long-Off','Mid-On','Deep Mid-Wicket','Square Leg','Fine Leg','Third Man'],
    ('Middle',    'spin')  : ['Slip','Short Mid-Wicket','Cover','Long-Off','Long-On','Deep Mid-Wicket','Square Leg','Fine Leg','Third Man'],
    ('Middle',    'medium'): ['Slip','Cover Point','Mid-Off','Long-On','Deep Mid-Wicket','Square Leg','Fine Leg','Third Man','Deep Point'],
    ('Death',     'fast')  : ['Deep Extra Cover','Long-Off','Long-On','Deep Mid-Wicket','Deep Square Leg','Fine Leg','Third Man','Mid-Off','Mid-On'],
    ('Death',     'spin')  : ['Long-Off','Long-On','Deep Mid-Wicket','Deep Square Leg','Fine Leg','Third Man','Extra Cover','Mid-Off','Mid-On'],
    ('Death',     'medium'): ['Long-Off','Long-On','Deep Extra Cover','Deep Mid-Wicket','Deep Square Leg','Fine Leg','Third Man','Mid-Off','Mid-On'],
}

LHB_SWAP = {
    'Cover Point':'Mid-Wicket','Deep Extra Cover':'Deep Mid-Wicket','Extra Cover':'Mid-Wicket',
    'Cover':'Mid-Wicket','Gully':'Square Leg','Mid-Off':'Mid-On','Mid-On':'Mid-Off',
    'Long-Off':'Long-On','Long-On':'Long-Off','Deep Mid-Wicket':'Deep Extra Cover',
    'Square Leg':'Cover Point','Deep Square Leg':'Deep Point','Fine Leg':'Third Man',
    'Third Man':'Fine Leg','Short Mid-Wicket':'Short Cover','Mid-Wicket':'Cover',
    '2nd Slip':'2nd Slip','Slip':'Slip','Deep Point':'Deep Square Leg'
}


def classify_bowler_type(style):
    s = style.lower()
    if 'fast' in s:   return 'fast'
    if 'medium' in s: return 'medium'
    return 'spin'


def get_batsman_threat(batsman_name, phase='Middle'):
    """
    Returns threat level dict from phase_sr data (Full Name based lookup).
    """
    key      = batsman_name.lower().strip()
    phase_sr = get_phase_sr()
    profiles = get_profiles()

    phase_row = phase_sr[
        (phase_sr['name_key'].str.contains(key, na=False, regex=False)) &
        (phase_sr['match_phase'].str.lower() == phase.lower())
    ]
    prof_row = profiles[profiles['name_key'].str.contains(key, na=False, regex=False)]

    if phase_row.empty:
        return {'error': f"No phase data for '{batsman_name}' in {phase}."}

    sr       = float(phase_row['strike_rate'].iloc[0])
    bdry_pct = float(phase_row['boundary_pct'].iloc[0])
    adapt    = float(phase_row['adaptability_index'].iloc[0])
    cons     = (float(prof_row['consistency_index'].iloc[0])
                if not prof_row.empty and 'consistency_index' in prof_row.columns else 0.0)
    display  = (prof_row['Full Name'].iloc[0] if not prof_row.empty
                else phase_row['Full Name'].iloc[0])

    agg = (sr / 200) * 0.5 + (bdry_pct / 50) * 0.3 + adapt * 0.2
    threat = ('High Threat' if agg > 0.6 else ('Medium Threat' if agg > 0.35 else 'Low Threat'))

    return {
        'batsman'           : display,
        'phase'             : phase,
        'strike_rate'       : round(sr, 1),
        'boundary_pct'      : round(bdry_pct, 1),
        'adaptability_index': round(adapt, 3),
        'consistency_index' : round(cons, 3),
        'aggressiveness'    : round(agg, 3),
        'threat_level'      : threat
    }


def recommend_bowling_strategy(batsman_name, phase, bowler_bowling_style):
    """Full bowling strategy recommendation."""
    threat = get_batsman_threat(batsman_name, phase)
    if 'error' in threat:
        return threat

    btype = classify_bowler_type(bowler_bowling_style)
    rule  = STRATEGY_RULES.get(
        (threat['threat_level'], btype),
        STRATEGY_RULES[('Medium Threat', 'medium')])
    plan  = rule['plan'] + PHASE_NOTE.get(phase.capitalize(), '')

    return {
        'batsman'            : threat['batsman'],
        'phase'              : phase,
        'bowler_style'       : bowler_bowling_style,
        'strike_rate'        : threat['strike_rate'],
        'adaptability'       : threat['adaptability_index'],
        'threat_level'       : threat['threat_level'],
        'aggressiveness'     : threat['aggressiveness'],
        'recommended_line'   : rule['line'],
        'recommended_length' : rule['length'],
        'tactical_plan'      : plan
    }


def suggest_field_placement(batsman_name, phase, bowler_bowling_style,
                              pitch_line='OUTSIDE_OFFSTUMP',
                              pitch_length='GOOD_LENGTH'):
    """Returns 9 fielder positions with dynamic adjustments."""
    btype  = classify_bowler_type(bowler_bowling_style)
    ph_key = phase.capitalize()
    if ph_key not in ('Powerplay', 'Middle', 'Death'):
        ph_key = 'Middle'

    positions = list(FIELD_TEMPLATES.get((ph_key, btype),
                                          FIELD_TEMPLATES[('Middle', 'medium')]))

    # LHB adjustment
    df       = get_df()
    bat_rows = df[df['Full Name'].str.lower().str.contains(
        batsman_name.lower(), na=False, regex=False)]
    if not bat_rows.empty:
        style = bat_rows['Batsman_Batting_Style'].mode().iloc[0].lower()
        if 'left' in style:
            positions = [LHB_SWAP.get(p, p) for p in positions]

    notes = []
    if pitch_length in ('SHORT', 'SHORT_OF_A_GOOD_LENGTH') and 'Deep Square Leg' not in positions:
        positions[-1] = 'Deep Square Leg'
        notes.append('Deep Square Leg added for short-pitch bowling.')
    if pitch_line == 'DOWN_LEG' and 'Fine Leg' not in positions:
        positions.append('Fine Leg')
        notes.append('Fine Leg added for leg-side deliveries.')
    if not notes:
        notes.append('Standard template applied.')

    return {
        'batsman'        : batsman_name,
        'phase'          : phase,
        'bowler_style'   : bowler_bowling_style,
        'field_positions': positions[:9],
        'notes'          : notes
    }


def full_matchup_report(batsman_name, phase, bowler_bowling_style,
                         pitch_line='OUTSIDE_OFFSTUMP',
                         pitch_length='GOOD_LENGTH'):
    """
    Combines threat assessment, bowling strategy, field placement,
    and historical matchup data into one structured report.
    Always returns a dict-of-dicts (safe to iterate at every level).
    """
    strategy = recommend_bowling_strategy(batsman_name, phase, bowler_bowling_style)
    if 'error' in strategy:
        return {'ERROR': {'message': strategy['error']}}

    field = suggest_field_placement(
        batsman_name, phase, bowler_bowling_style, pitch_line, pitch_length)

    matchup = get_matchup()
    # Filter by batsman only — show actual individual bowler records vs this batsman.
    # Do NOT filter by style type: that would mix all fast/spin/medium bowlers together
    # regardless of who is actually bowling in this match.
    hist = matchup[
        matchup['Batsman_Name'].str.lower().str.contains(
            batsman_name.lower(), na=False, regex=False)
    ].nlargest(5, 'success_index_scaled')

    hist_info = (
        {'best_bowler'    : hist.iloc[0]['Bowler_Name'],
         'best_style'     : hist.iloc[0]['Bowler_Bowling_Style'],
         'success_index'  : round(hist.iloc[0]['success_index_scaled'], 3),
         'avg_economy'    : round(hist['economy_rate'].mean(), 2),
         'avg_wicket_pct' : round(hist['wicket_pct'].mean(), 2),
         'total_records'  : len(matchup[matchup['Batsman_Name'].str.lower()
                                        .str.contains(batsman_name.lower(), na=False, regex=False)])}
        if not hist.empty else {'note': 'No historical matchup data.'}
    )

    return {
        'THREAT ASSESSMENT': {
            'Batsman'     : strategy['batsman'],
            'Phase'       : phase,
            'Strike Rate' : str(strategy['strike_rate']),
            'Adaptability': str(strategy['adaptability']),
            'Aggressiveness': str(strategy['aggressiveness']),
            'Threat Level': strategy['threat_level']
        },
        'BOWLING PLAN': {
            'Style'  : bowler_bowling_style,
            'Line'   : strategy['recommended_line'],
            'Length' : strategy['recommended_length'],
            'Plan'   : strategy['tactical_plan']
        },
        'FIELD PLACEMENT': {
            'Positions': ', '.join(field['field_positions']),
            'Notes'    : '; '.join(field['notes'])
        },
        'HISTORICAL MATCHUP': hist_info
    }


def find_best_bowler_vs(batsman_name, squad_names=None, top_n=5):
    """Rank squad bowlers by historical success index against this batsman."""
    matchup = get_matchup()
    key     = batsman_name.lower().strip()
    col     = 'Batsman_Full_Name' if 'Batsman_Full_Name' in matchup.columns else 'Batsman_Name'
    mask    = matchup[col].str.lower().str.contains(key, na=False, regex=False)
    sub     = matchup[mask].copy()

    if not sub.empty and squad_names:
        s_lower = [s.lower() for s in squad_names]
        sm      = sub['Bowler_Name'].str.lower().apply(
            lambda n: any(s in n for s in s_lower))
        if sm.any():
            sub = sub[sm]

    return sub.nlargest(top_n, 'success_index_scaled').reset_index(drop=True)
