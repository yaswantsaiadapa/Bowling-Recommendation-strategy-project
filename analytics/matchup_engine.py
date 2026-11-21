"""
Matchup & Opponent Intelligence Query Engine.

Provides deep historical confrontation queries across 4.81M multi-format deliveries:
1. Mode 1: ⚔️ Batter vs Bowler (Head-to-head confrontation duel)
2. Mode 2: 🏏 Batter vs Team (Batsman record against an entire national team)
3. Mode 3: 🎯 Bowler vs Team (Bowler record against an entire national batting lineup)
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import pandas as pd
from analytics.db import get_db_connection, DB_PATH

BOWLER_WICKET_KINDS = (
    "'bowled', 'caught', 'lbw', 'stumped', 'caught and bowled', 'hit wicket'"
)


def get_all_teams() -> List[str]:
    """
    Returns a clean list of top international cricket teams sorted by match volume.
    """
    con = get_db_connection()
    sql = """
        SELECT batting_team 
        FROM fact_deliveries_master 
        WHERE batting_team IS NOT NULL
        GROUP BY batting_team 
        HAVING COUNT(*) > 5000
        ORDER BY COUNT(*) DESC
    """
    res = con.execute(sql).fetchall()
    con.close()
    return [r[0] for r in res]


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: BATTER VS BOWLER (HEAD-TO-HEAD DUEL)
# ══════════════════════════════════════════════════════════════════════════════

def get_h2h_overview(batter: str, bowler: str) -> Optional[Dict[str, Any]]:
    """
    Computes career executive KPIs for a specific Batsman vs Bowler head-to-head duel.
    """
    con = get_db_connection()
    c_bat = batter.strip().replace("'", "''")
    c_bowl = bowler.strip().replace("'", "''")

    sql = f"""
        SELECT 
            COUNT(DISTINCT match_id) as matches,
            COUNT(DISTINCT match_id || '_' || inning) as innings,
            COUNT(*) as balls_faced,
            SUM(runs_batter) as total_runs,
            SUM(CASE WHEN is_wicket AND dismissal_kind IN ({BOWLER_WICKET_KINDS}) THEN 1 ELSE 0 END) as dismissals,
            SUM(is_dot::INT) as dot_balls,
            SUM(is_boundary::INT) as boundaries,
            SUM(is_four::INT) as fours,
            SUM(is_six::INT) as sixes
        FROM fact_deliveries_master
        WHERE batter = '{c_bat}' AND bowler = '{c_bowl}'
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty or int(df["balls_faced"].iloc[0] or 0) == 0:
        return None

    row = df.iloc[0]
    balls = int(row["balls_faced"] or 0)
    runs = int(row["total_runs"] or 0)
    dismissals = int(row["dismissals"] or 0)
    dots = int(row["dot_balls"] or 0)
    boundaries = int(row["boundaries"] or 0)

    sr = round((runs * 100.0) / balls, 1) if balls > 0 else 0.0
    avg = round(runs / dismissals, 2) if dismissals > 0 else float(runs)
    dot_pct = round((dots * 100.0) / balls, 1) if balls > 0 else 0.0
    boundary_pct = round((boundaries * 100.0) / balls, 1) if balls > 0 else 0.0

    # Data-driven verdict badge
    if dismissals >= 4 and avg <= 25.0:
        verdict = "🎯 Bowler Dominance"
        verdict_color = "red"
        verdict_desc = f"{bowler} has dismissed {batter} {dismissals} times, keeping him to an average of {avg}."
    elif sr >= 135.0 and dismissals <= 1:
        verdict = "🏏 Batter Dominance"
        verdict_color = "green"
        verdict_desc = f"{batter} attacks {bowler} freely at a high strike rate of {sr} with only {dismissals} dismissals."
    elif dismissals >= 2 and sr >= 110.0:
        verdict = "⚔️ High-Octane Rivalry"
        verdict_color = "orange"
        verdict_desc = f"Both players take the attack to each other: {runs} runs scored but with {dismissals} dismissals."
    else:
        verdict = "⚖️ Contested Battle"
        verdict_color = "blue"
        verdict_desc = f"Balanced confrontation across {balls} balls faced."

    return {
        "batter": batter,
        "bowler": bowler,
        "matches": int(row["matches"] or 0),
        "innings": int(row["innings"] or 0),
        "balls_faced": balls,
        "total_runs": runs,
        "dismissals": dismissals,
        "strike_rate": sr,
        "batting_average": avg,
        "dot_balls": dots,
        "dot_pct": dot_pct,
        "boundaries": boundaries,
        "boundary_pct": boundary_pct,
        "fours": int(row["fours"] or 0),
        "sixes": int(row["sixes"] or 0),
        "verdict": verdict,
        "verdict_color": verdict_color,
        "verdict_desc": verdict_desc,
    }


def get_h2h_format_splits(batter: str, bowler: str) -> pd.DataFrame:
    """
    Compares the head-to-head confrontation across formats (Test, ODI, T20I).
    """
    con = get_db_connection()
    c_bat = batter.strip().replace("'", "''")
    c_bowl = bowler.strip().replace("'", "''")

    sql = f"""
        SELECT 
            match_type as format,
            COUNT(DISTINCT match_id) as matches,
            COUNT(*) as balls_faced,
            SUM(runs_batter) as total_runs,
            SUM(CASE WHEN is_wicket AND dismissal_kind IN ({BOWLER_WICKET_KINDS}) THEN 1 ELSE 0 END) as dismissals
        FROM fact_deliveries_master
        WHERE batter = '{c_bat}' AND bowler = '{c_bowl}'
        GROUP BY match_type
        ORDER BY 
            CASE match_type 
                WHEN 'Test' THEN 1 
                WHEN 'ODI' THEN 2 
                WHEN 'T20I' THEN 3 
                ELSE 4 
            END
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty:
        return pd.DataFrame()

    df["balls_faced"] = df["balls_faced"].astype(int)
    df["total_runs"] = df["total_runs"].astype(int)
    df["dismissals"] = df["dismissals"].astype(int)

    df["strike_rate"] = df.apply(
        lambda r: round((r["total_runs"] * 100.0) / r["balls_faced"], 1)
        if r["balls_faced"] > 0
        else 0.0,
        axis=1,
    )
    df["batting_average"] = df.apply(
        lambda r: round(r["total_runs"] / r["dismissals"], 2)
        if r["dismissals"] > 0
        else float(r["total_runs"]),
        axis=1,
    )
    return df


def get_h2h_phase_splits(batter: str, bowler: str) -> pd.DataFrame:
    """
    Analyzes face-off performance across match phases (Powerplay, Middle, Death).
    """
    con = get_db_connection()
    c_bat = batter.strip().replace("'", "''")
    c_bowl = bowler.strip().replace("'", "''")

    sql = f"""
        SELECT 
            match_phase,
            COUNT(*) as balls_faced,
            SUM(runs_batter) as total_runs,
            SUM(CASE WHEN is_wicket AND dismissal_kind IN ({BOWLER_WICKET_KINDS}) THEN 1 ELSE 0 END) as dismissals
        FROM fact_deliveries_master
        WHERE batter = '{c_bat}' AND bowler = '{c_bowl}'
          AND match_phase IN ('Powerplay', 'Middle', 'Death')
        GROUP BY match_phase
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty:
        return pd.DataFrame()

    df["balls_faced"] = df["balls_faced"].astype(int)
    df["total_runs"] = df["total_runs"].astype(int)
    df["dismissals"] = df["dismissals"].astype(int)

    df["strike_rate"] = df.apply(
        lambda r: round((r["total_runs"] * 100.0) / r["balls_faced"], 1)
        if r["balls_faced"] > 0
        else 0.0,
        axis=1,
    )

    phase_order = {"Powerplay": 1, "Middle": 2, "Death": 3}
    df["sort_order"] = df["match_phase"].map(phase_order).fillna(99)
    df = df.sort_values(by="sort_order").drop(columns=["sort_order"])
    return df


def get_h2h_shots(batter: str, bowler: str) -> Dict[str, int]:
    """
    Breaks down all balls into Dots, 1s, 2s, 3s, 4s, 6s.
    """
    con = get_db_connection()
    c_bat = batter.strip().replace("'", "''")
    c_bowl = bowler.strip().replace("'", "''")

    sql = f"""
        SELECT 
            SUM(CASE WHEN runs_batter = 0 THEN 1 ELSE 0 END) as dots,
            SUM(CASE WHEN runs_batter = 1 THEN 1 ELSE 0 END) as singles,
            SUM(CASE WHEN runs_batter = 2 THEN 1 ELSE 0 END) as twos,
            SUM(CASE WHEN runs_batter = 3 THEN 1 ELSE 0 END) as threes,
            SUM(CASE WHEN runs_batter = 4 THEN 1 ELSE 0 END) as fours,
            SUM(CASE WHEN runs_batter >= 6 THEN 1 ELSE 0 END) as sixes
        FROM fact_deliveries_master
        WHERE batter = '{c_bat}' AND bowler = '{c_bowl}'
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty:
        return {"Dots (0)": 0, "Singles (1)": 0, "Twos (2)": 0, "Threes (3)": 0, "Fours (4)": 0, "Sixes (6)": 0}

    row = df.iloc[0]
    return {
        "Dots (0)": int(row["dots"] or 0),
        "Singles (1)": int(row["singles"] or 0),
        "Twos (2)": int(row["twos"] or 0),
        "Threes (3)": int(row["threes"] or 0),
        "Fours (4)": int(row["fours"] or 0),
        "Sixes (6)": int(row["sixes"] or 0),
    }


def get_h2h_dismissals_log(batter: str, bowler: str) -> List[Dict[str, Any]]:
    """
    Chronological list of every dismissal when this bowler dismissed this batter.
    """
    con = get_db_connection()
    c_bat = batter.strip().replace("'", "''")
    c_bowl = bowler.strip().replace("'", "''")

    sql = f"""
        SELECT 
            match_date,
            ground_name,
            match_type,
            inning,
            over,
            ball,
            dismissal_kind
        FROM fact_deliveries_master
        WHERE batter = '{c_bat}' AND bowler = '{c_bowl}'
          AND is_wicket = TRUE
          AND dismissal_kind IN ({BOWLER_WICKET_KINDS})
        ORDER BY match_date DESC
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty:
        return []

    results = []
    for _, r in df.iterrows():
        results.append({
            "date": str(r["match_date"]),
            "venue": str(r["ground_name"]),
            "format": str(r["match_type"]),
            "inning": int(r["inning"]),
            "over": f"{int(r['over'])}.{int(r['ball'])}",
            "dismissal": str(r["dismissal_kind"]).title(),
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: BATTER VS OPPOSITION TEAM
# ══════════════════════════════════════════════════════════════════════════════

def get_batter_vs_team_overview(batter: str, team: str) -> Optional[Dict[str, Any]]:
    """
    Career executive KPIs for a Batsman against a specific country/team.
    """
    con = get_db_connection()
    c_bat = batter.strip().replace("'", "''")
    c_team = team.strip().replace("'", "''")

    sql = f"""
        SELECT 
            COUNT(DISTINCT match_id) as matches,
            COUNT(DISTINCT match_id || '_' || inning) as innings,
            COUNT(*) as balls_faced,
            SUM(runs_batter) as total_runs,
            SUM(is_wicket::INT) as dismissals,
            SUM(is_dot::INT) as dot_balls,
            SUM(is_boundary::INT) as boundaries,
            SUM(is_four::INT) as fours,
            SUM(is_six::INT) as sixes
        FROM fact_deliveries_master
        WHERE batter = '{c_bat}' AND bowling_team = '{c_team}'
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty or int(df["balls_faced"].iloc[0] or 0) == 0:
        return None

    row = df.iloc[0]
    balls = int(row["balls_faced"] or 0)
    runs = int(row["total_runs"] or 0)
    dismissals = int(row["dismissals"] or 0)
    dots = int(row["dot_balls"] or 0)
    boundaries = int(row["boundaries"] or 0)

    sr = round((runs * 100.0) / balls, 1) if balls > 0 else 0.0
    avg = round(runs / dismissals, 2) if dismissals > 0 else float(runs)
    dot_pct = round((dots * 100.0) / balls, 1) if balls > 0 else 0.0
    boundary_pct = round((boundaries * 100.0) / balls, 1) if balls > 0 else 0.0

    return {
        "batter": batter,
        "team": team,
        "matches": int(row["matches"] or 0),
        "innings": int(row["innings"] or 0),
        "balls_faced": balls,
        "total_runs": runs,
        "dismissals": dismissals,
        "strike_rate": sr,
        "batting_average": avg,
        "dot_balls": dots,
        "dot_pct": dot_pct,
        "boundaries": boundaries,
        "boundary_pct": boundary_pct,
        "fours": int(row["fours"] or 0),
        "sixes": int(row["sixes"] or 0),
    }


def get_batter_vs_team_format_splits(batter: str, team: str) -> pd.DataFrame:
    """
    Batsman performance against this team broken down by format.
    """
    con = get_db_connection()
    c_bat = batter.strip().replace("'", "''")
    c_team = team.strip().replace("'", "''")

    sql = f"""
        SELECT 
            match_type as format,
            COUNT(DISTINCT match_id) as matches,
            COUNT(DISTINCT match_id || '_' || inning) as innings,
            COUNT(*) as balls_faced,
            SUM(runs_batter) as total_runs,
            SUM(is_wicket::INT) as dismissals
        FROM fact_deliveries_master
        WHERE batter = '{c_bat}' AND bowling_team = '{c_team}'
        GROUP BY match_type
        ORDER BY 
            CASE match_type 
                WHEN 'Test' THEN 1 
                WHEN 'ODI' THEN 2 
                WHEN 'T20I' THEN 3 
                ELSE 4 
            END
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty:
        return pd.DataFrame()

    df["balls_faced"] = df["balls_faced"].astype(int)
    df["total_runs"] = df["total_runs"].astype(int)
    df["dismissals"] = df["dismissals"].astype(int)

    df["strike_rate"] = df.apply(
        lambda r: round((r["total_runs"] * 100.0) / r["balls_faced"], 1)
        if r["balls_faced"] > 0
        else 0.0,
        axis=1,
    )
    df["batting_average"] = df.apply(
        lambda r: round(r["total_runs"] / r["dismissals"], 2)
        if r["dismissals"] > 0
        else float(r["total_runs"]),
        axis=1,
    )
    return df


def get_batter_vs_team_bowlers(batter: str, team: str, limit: int = 10) -> pd.DataFrame:
    """
    Which bowlers from this opposition team have contested this batter the most?
    """
    con = get_db_connection()
    c_bat = batter.strip().replace("'", "''")
    c_team = team.strip().replace("'", "''")

    sql = f"""
        SELECT 
            bowler,
            COUNT(*) as balls_faced,
            SUM(runs_batter) as runs_scored,
            SUM(CASE WHEN is_wicket AND dismissal_kind IN ({BOWLER_WICKET_KINDS}) THEN 1 ELSE 0 END) as dismissals
        FROM fact_deliveries_master
        WHERE batter = '{c_bat}' AND bowling_team = '{c_team}'
        GROUP BY bowler
        ORDER BY dismissals DESC, runs_scored DESC
        LIMIT {limit}
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty:
        return pd.DataFrame()

    df["balls_faced"] = df["balls_faced"].astype(int)
    df["runs_scored"] = df["runs_scored"].astype(int)
    df["dismissals"] = df["dismissals"].astype(int)

    df["strike_rate"] = df.apply(
        lambda r: round((r["runs_scored"] * 100.0) / r["balls_faced"], 1)
        if r["balls_faced"] > 0
        else 0.0,
        axis=1,
    )
    return df


def get_batter_vs_team_phase_splits(batter: str, team: str) -> pd.DataFrame:
    """
    Phase scoring against this team (Powerplay, Middle, Death).
    """
    con = get_db_connection()
    c_bat = batter.strip().replace("'", "''")
    c_team = team.strip().replace("'", "''")

    sql = f"""
        SELECT 
            match_phase,
            COUNT(*) as balls_faced,
            SUM(runs_batter) as total_runs,
            SUM(is_wicket::INT) as dismissals
        FROM fact_deliveries_master
        WHERE batter = '{c_bat}' AND bowling_team = '{c_team}'
          AND match_phase IN ('Powerplay', 'Middle', 'Death')
        GROUP BY match_phase
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty:
        return pd.DataFrame()

    df["balls_faced"] = df["balls_faced"].astype(int)
    df["total_runs"] = df["total_runs"].astype(int)
    df["dismissals"] = df["dismissals"].astype(int)

    df["strike_rate"] = df.apply(
        lambda r: round((r["total_runs"] * 100.0) / r["balls_faced"], 1)
        if r["balls_faced"] > 0
        else 0.0,
        axis=1,
    )

    phase_order = {"Powerplay": 1, "Middle": 2, "Death": 3}
    df["sort_order"] = df["match_phase"].map(phase_order).fillna(99)
    df = df.sort_values(by="sort_order").drop(columns=["sort_order"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODE 3: BOWLER VS OPPOSITION TEAM
# ══════════════════════════════════════════════════════════════════════════════

def get_bowler_vs_team_overview(bowler: str, team: str) -> Optional[Dict[str, Any]]:
    """
    Career executive KPIs for a Bowler against a specific country/team.
    """
    con = get_db_connection()
    c_bowl = bowler.strip().replace("'", "''")
    c_team = team.strip().replace("'", "''")

    sql = f"""
        SELECT 
            COUNT(DISTINCT match_id) as matches,
            COUNT(DISTINCT match_id || '_' || inning) as innings,
            COUNT(*) as total_deliveries,
            SUM(CASE WHEN extra_wides = 0 AND extra_noballs = 0 THEN 1 ELSE 0 END) as legal_balls,
            SUM(runs_batter + extra_wides + extra_noballs) as runs_conceded,
            SUM(CASE WHEN is_wicket AND dismissal_kind IN ({BOWLER_WICKET_KINDS}) THEN 1 ELSE 0 END) as wickets,
            SUM(CASE WHEN (runs_batter + extra_wides + extra_noballs) = 0 THEN 1 ELSE 0 END) as dot_balls,
            SUM(is_boundary::INT) as boundaries,
            SUM(extra_wides) as wides,
            SUM(extra_noballs) as noballs
        FROM fact_deliveries_master
        WHERE bowler = '{c_bowl}' AND batting_team = '{c_team}'
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty or int(df["total_deliveries"].iloc[0] or 0) == 0:
        return None

    row = df.iloc[0]
    total_deliveries = int(row["total_deliveries"] or 0)
    legal_balls = int(row["legal_balls"] or 0)
    runs = int(row["runs_conceded"] or 0)
    wkts = int(row["wickets"] or 0)
    dots = int(row["dot_balls"] or 0)

    overs_display = f"{legal_balls // 6}.{legal_balls % 6}"
    econ = round(runs / (legal_balls / 6.0), 2) if legal_balls > 0 else 0.0
    avg = round(runs / wkts, 2) if wkts > 0 else 0.0
    sr = round(legal_balls / wkts, 1) if wkts > 0 else 0.0
    dot_pct = round((dots * 100.0) / total_deliveries, 1) if total_deliveries > 0 else 0.0

    return {
        "bowler": bowler,
        "team": team,
        "matches": int(row["matches"] or 0),
        "innings": int(row["innings"] or 0),
        "legal_balls": legal_balls,
        "overs_display": overs_display,
        "runs_conceded": runs,
        "wickets": wkts,
        "economy_rate": econ,
        "bowling_average": avg,
        "strike_rate": sr,
        "dot_balls": dots,
        "dot_pct": dot_pct,
        "extras_conceded": int(row["wides"] or 0) + int(row["noballs"] or 0),
    }


def get_bowler_vs_team_format_splits(bowler: str, team: str) -> pd.DataFrame:
    """
    Bowler performance against this team broken down by format.
    """
    con = get_db_connection()
    c_bowl = bowler.strip().replace("'", "''")
    c_team = team.strip().replace("'", "''")

    sql = f"""
        SELECT 
            match_type as format,
            COUNT(DISTINCT match_id) as matches,
            SUM(CASE WHEN extra_wides = 0 AND extra_noballs = 0 THEN 1 ELSE 0 END) as legal_balls,
            SUM(runs_batter + extra_wides + extra_noballs) as runs_conceded,
            SUM(CASE WHEN is_wicket AND dismissal_kind IN ({BOWLER_WICKET_KINDS}) THEN 1 ELSE 0 END) as wickets
        FROM fact_deliveries_master
        WHERE bowler = '{c_bowl}' AND batting_team = '{c_team}'
        GROUP BY match_type
        ORDER BY 
            CASE match_type 
                WHEN 'Test' THEN 1 
                WHEN 'ODI' THEN 2 
                WHEN 'T20I' THEN 3 
                ELSE 4 
            END
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty:
        return pd.DataFrame()

    df["legal_balls"] = df["legal_balls"].astype(int)
    df["runs_conceded"] = df["runs_conceded"].astype(int)
    df["wickets"] = df["wickets"].astype(int)

    df["overs"] = df["legal_balls"].apply(lambda b: f"{b // 6}.{b % 6}")
    df["economy_rate"] = df.apply(
        lambda r: round(r["runs_conceded"] / (r["legal_balls"] / 6.0), 2)
        if r["legal_balls"] > 0
        else 0.0,
        axis=1,
    )
    df["bowling_average"] = df.apply(
        lambda r: round(r["runs_conceded"] / r["wickets"], 2) if r["wickets"] > 0 else 0.0,
        axis=1,
    )
    return df


def get_bowler_vs_team_batters(bowler: str, team: str, limit: int = 10) -> pd.DataFrame:
    """
    Which batters from this team did this bowler dismiss most frequently?
    """
    con = get_db_connection()
    c_bowl = bowler.strip().replace("'", "''")
    c_team = team.strip().replace("'", "''")

    sql = f"""
        SELECT 
            batter,
            COUNT(*) as balls_bowled,
            SUM(runs_batter) as runs_conceded,
            SUM(CASE WHEN is_wicket AND dismissal_kind IN ({BOWLER_WICKET_KINDS}) THEN 1 ELSE 0 END) as wickets
        FROM fact_deliveries_master
        WHERE bowler = '{c_bowl}' AND batting_team = '{c_team}'
        GROUP BY batter
        ORDER BY wickets DESC, balls_bowled DESC
        LIMIT {limit}
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty:
        return pd.DataFrame()

    df["balls_bowled"] = df["balls_bowled"].astype(int)
    df["runs_conceded"] = df["runs_conceded"].astype(int)
    df["wickets"] = df["wickets"].astype(int)

    df["strike_rate"] = df.apply(
        lambda r: round((r["runs_conceded"] * 100.0) / r["balls_bowled"], 1)
        if r["balls_bowled"] > 0
        else 0.0,
        axis=1,
    )
    return df


def get_bowler_vs_team_dismissals(bowler: str, team: str) -> pd.DataFrame:
    """
    Distribution of dismissal kinds against this team.
    """
    con = get_db_connection()
    c_bowl = bowler.strip().replace("'", "''")
    c_team = team.strip().replace("'", "''")

    sql = f"""
        SELECT 
            dismissal_kind,
            COUNT(*) as dismissal_count
        FROM fact_deliveries_master
        WHERE bowler = '{c_bowl}' AND batting_team = '{c_team}'
          AND is_wicket = TRUE
          AND dismissal_kind IN ({BOWLER_WICKET_KINDS})
        GROUP BY dismissal_kind
        ORDER BY dismissal_count DESC
    """
    df = con.execute(sql).df()
    con.close()
    return df
