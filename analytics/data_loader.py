"""
data_loader.py
Loads and caches all project datasets once.
All other modules import from here — no duplicate pd.read_csv calls anywhere.
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ── Path configuration ────────────────────────────────────────────────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'models')


def _load_master():
    df = pd.read_csv(os.path.join(DATA_DIR, 'final_processed_data.csv'))
    for c in ['isFour', 'isSix', 'isWicket']:
        df[c] = df[c].astype(int)
    df['isBoundary'] = ((df['isFour'] == 1) | (df['isSix'] == 1)).astype(int)
    df['is_valid']   = ((df['wides'] == 0) & (df['noballs'] == 0)).astype(int)
    df['match_phase'] = df['oversActual'].apply(
        lambda o: 'Powerplay' if o <= 6 else ('Middle' if o <= 15 else 'Death'))
    return df


def _load_phase_sr():
    """
    Normalise phase_sr.csv to consistent schema regardless of which
    notebook version produced it (old = Batsman_Name short codes;
    new = Full Name).  Always returns a df with:
    Full Name, name_key, boundary_pct, dot_ball_pct, adaptability_index.
    """
    path = os.path.join(DATA_DIR, 'phase_sr.csv')
    ps   = pd.read_csv(path)

    # Drop unnamed index columns
    ps.drop(columns=[c for c in ps.columns if 'Unnamed' in str(c)], inplace=True)

    # Old format: has Batsman_Name short codes, no Full Name column
    if 'Full Name' not in ps.columns:
        df = get_df()
        name_map = (df[['Batsman_Name', 'Full Name']]
                    .drop_duplicates()
                    .set_index('Batsman_Name')['Full Name']
                    .to_dict())
        ps['Full Name'] = ps['Batsman_Name'].map(name_map).fillna(ps['Batsman_Name'])

    # Normalise column names (old: boundary_percentage, new: boundary_pct)
    ps.rename(columns={
        'boundary_percentage': 'boundary_pct',
        'dot_ball_percentage': 'dot_ball_pct',
        'wicket_percentage'  : 'wicket_pct'
    }, inplace=True, errors='ignore')

    ps['name_key'] = ps['Full Name'].str.lower().str.strip()
    return ps


def _load_profiles():
    path = os.path.join(DATA_DIR, 'batsman_profiles.csv')
    bp = pd.read_csv(path)
    bp['name_key'] = bp['Full Name'].str.lower().str.strip()
    return bp


def _load_bowler_stats():
    path = os.path.join(DATA_DIR, 'bowler_stats.csv')
    bs   = pd.read_csv(path)
    # Normalise column names to consistent short form
    bs.rename(columns={
        'wicket_percentage'  : 'wicket_pct',
        'boundary_percentage': 'boundary_pct',
        'dot_ball_percentage': 'dot_ball_pct',
    }, inplace=True, errors='ignore')
    bs['name_key'] = bs['Bowler_Name'].str.lower().str.strip()
    return bs


def _load_matchup():
    """Load bowling_success_model.csv.
    Batsman_Name and Bowler_Name are Full Names after notebook 05 is re-run.
    For backwards compatibility with the old CSV (short codes), add Full Name
    columns from the master df if they are not already full names.
    """
    ms = pd.read_csv(os.path.join(DATA_DIR, 'bowling_success_model.csv'))
    # Detect old format: short codes are typically <= 15 chars like 'V Kohli'
    # Full names are longer like 'Virat Kohli'. Use the presence of spaces + length.
    sample = ms['Batsman_Name'].dropna().iloc[0] if len(ms) > 0 else ''
    is_short_code = len(sample.split()) <= 2 and len(sample) <= 15 and sample == sample.split()[0][0] + ' ' + sample.split()[-1] if ' ' in sample else len(sample) <= 10
    if is_short_code and 'Batsman_Full_Name' not in ms.columns:
        df = get_df()
        bat_map  = (df[['Batsman_Name','Full Name']]
                    .drop_duplicates().set_index('Batsman_Name')['Full Name'].to_dict())
        bowl_map = (df[['Bowler_Name','Full Name_bowler']]
                    .drop_duplicates().set_index('Bowler_Name')['Full Name_bowler'].to_dict())
        ms['Batsman_Name'] = ms['Batsman_Name'].map(bat_map).fillna(ms['Batsman_Name'])
        ms['Bowler_Name']  = ms['Bowler_Name'].map(bowl_map).fillna(ms['Bowler_Name'])
    # Normalise metric column names to consistent short form
    # Works whether CSV has 'wicket_percentage' or 'wicket_pct' etc.
    ms.rename(columns={
        'wicket_percentage'  : 'wicket_pct',
        'boundary_percentage': 'boundary_pct',
        'dot_ball_percentage': 'dot_ball_pct',
    }, inplace=True, errors='ignore')
    return ms


# ── Singleton cache ───────────────────────────────────────────────────────────
_cache = {}

def get_df():
    if 'df' not in _cache:
        _cache['df'] = _load_master()
    return _cache['df']

def get_phase_sr():
    if 'phase_sr' not in _cache:
        _cache['phase_sr'] = _load_phase_sr()
    return _cache['phase_sr']

def get_profiles():
    if 'profiles' not in _cache:
        _cache['profiles'] = _load_profiles()
    return _cache['profiles']

def get_bowler_stats():
    if 'bowler_stats' not in _cache:
        _cache['bowler_stats'] = _load_bowler_stats()
    return _cache['bowler_stats']

def get_matchup():
    if 'matchup' not in _cache:
        _cache['matchup'] = _load_matchup()
    return _cache['matchup']


# ── Helper: all batsman Full Names ────────────────────────────────────────────
def all_batsman_names():
    return sorted(get_df()['Full Name'].dropna().unique().tolist())

def all_bowler_names():
    return sorted(get_df()['Full Name_bowler'].dropna().unique().tolist())

def all_grounds():
    return sorted(get_df()['Ground Name'].dropna().unique().tolist())
