"""
Batting Analytics Query Engine.

Provides deep, high-speed analytical extractions for any batsman across all 4.81M deliveries (T20, ODI, Test):
1. Overview KPIs: Career runs, balls, strike rate, boundaries, dot ball %, dismissals.
2. Phase Breakdown: Powerplay vs Middle vs Death scoring, aggression & strike rates (for T20/ODI).
3. Test Innings Splits: Inning 1, Inning 2, Inning 3, Inning 4 performance (for Tests).
4. Chase Matrix: 1st Batting (Setting Target) vs 2nd Innings Chasing performance.
5. Opponent Team Dominance: Teams scored the most against and strike rate per team.
6. Dismissal Anatomy: Distribution of dismissal modes (caught, bowled, lbw, run out, stumped).
7. Venue & Ground Intelligence: Top scoring grounds and strike rates per stadium.
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


def search_batsmen(query: str, limit: int = 10) -> List[str]:
    """
    Finds player names matching a search query for search autocomplete.
    """
    if not query or len(query.strip()) < 2:
        return []

    con = get_db_connection()
    clean_q = query.strip().replace("'", "''")
    sql = f"""
        SELECT player_name 
        FROM dim_players 
        WHERE LOWER(player_name) LIKE '%{clean_q.lower()}%'
          AND balls_faced >= 20
        ORDER BY balls_faced DESC
        LIMIT {limit}
    """
    res = con.execute(sql).fetchall()
    con.close()
    return [r[0] for r in res]


def get_batter_overview(
    player_name: str, 
    match_type: Optional[str] = None, 
    match_phase: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Computes career executive KPIs for a batsman across selected format and phase.
    """
    con = get_db_connection()
    clean_name = player_name.strip().replace("'", "''")

    where_clauses = [f"batter = '{clean_name}'"]
    if match_type and match_type != "All":
        clean_mt = match_type.strip().replace("'", "''")
        where_clauses.append(f"match_type = '{clean_mt}'")
    if match_phase and match_phase != "All":
        clean_mp = match_phase.strip().replace("'", "''")
        where_clauses.append(f"match_phase = '{clean_mp}'")

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT 
            COUNT(DISTINCT match_id) as matches_played,
            COUNT(DISTINCT match_id || '_' || inning) as innings_batted,
            SUM(runs_batter) as total_runs,
            COUNT(*) as balls_faced,
            SUM(is_dot::INT) as dot_balls,
            SUM(is_four::INT) as fours,
            SUM(is_six::INT) as sixes,
            SUM(is_boundary::INT) as boundaries,
            SUM(is_wicket::INT) as dismissals
        FROM fact_deliveries_master
        WHERE {where_sql}
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty or df["balls_faced"].iloc[0] == 0:
        return None

    row = df.iloc[0]
    balls = int(row["balls_faced"])
    runs = int(row["total_runs"] or 0)
    dismissals = int(row["dismissals"] or 0)
    dots = int(row["dot_balls"] or 0)
    boundaries = int(row["boundaries"] or 0)

    sr = round((runs * 100.0) / balls, 1) if balls > 0 else 0.0
    dot_pct = round((dots * 100.0) / balls, 1) if balls > 0 else 0.0
    boundary_pct = round((boundaries * 100.0) / balls, 1) if balls > 0 else 0.0
    balls_per_dismissal = round(balls / dismissals, 1) if dismissals > 0 else float(balls)

    return {
        "player_name": player_name,
        "matches_played": int(row["matches_played"]),
        "innings_batted": int(row["innings_batted"]),
        "total_runs": runs,
        "balls_faced": balls,
        "strike_rate": sr,
        "fours": int(row["fours"] or 0),
        "sixes": int(row["sixes"] or 0),
        "boundaries": boundaries,
        "boundary_pct": boundary_pct,
        "dot_balls": dots,
        "dot_pct": dot_pct,
        "dismissals": dismissals,
        "balls_per_dismissal": balls_per_dismissal,
    }


def get_batter_phase_splits(player_name: str, match_type: Optional[str] = None) -> pd.DataFrame:
    """
    Computes performance broken down by match phase: Powerplay, Middle, Death.
    Relevant for limited-overs cricket (T20I, ODI, All).
    """
    con = get_db_connection()
    clean_name = player_name.strip().replace("'", "''")

    where_clauses = [f"batter = '{clean_name}'", "match_phase IN ('Powerplay', 'Middle', 'Death')"]
    if match_type and match_type != "All":
        clean_mt = match_type.strip().replace("'", "''")
        where_clauses.append(f"match_type = '{clean_mt}'")

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT 
            match_phase,
            SUM(runs_batter) as runs,
            COUNT(*) as balls,
            ROUND(SUM(runs_batter) * 100.0 / COUNT(*), 1) as strike_rate,
            ROUND(SUM(is_dot::INT) * 100.0 / COUNT(*), 1) as dot_pct,
            ROUND(SUM(is_boundary::INT) * 100.0 / COUNT(*), 1) as boundary_pct,
            SUM(is_wicket::INT) as dismissals
        FROM fact_deliveries_master
        WHERE {where_sql}
        GROUP BY match_phase
        ORDER BY CASE 
            WHEN match_phase = 'Powerplay' THEN 1
            WHEN match_phase = 'Middle' THEN 2
            WHEN match_phase = 'Death' THEN 3
            ELSE 4 END
    """
    df = con.execute(sql).df()
    con.close()
    return df


def get_batter_test_inning_splits(player_name: str) -> pd.DataFrame:
    """
    Computes performance in Test cricket broken down by innings:
    Inning 1, Inning 2, Inning 3, Inning 4.
    """
    con = get_db_connection()
    clean_name = player_name.strip().replace("'", "''")

    sql = f"""
        SELECT 
            'Inning ' || inning as inning_label,
            inning,
            COUNT(DISTINCT match_id) as matches,
            SUM(runs_batter) as runs,
            COUNT(*) as balls,
            ROUND(SUM(runs_batter) * 100.0 / COUNT(*), 1) as strike_rate,
            ROUND(SUM(is_dot::INT) * 100.0 / COUNT(*), 1) as dot_pct,
            ROUND(SUM(is_boundary::INT) * 100.0 / COUNT(*), 1) as boundary_pct,
            SUM(is_wicket::INT) as dismissals
        FROM fact_deliveries_master
        WHERE batter = '{clean_name}' AND match_type = 'Test' AND inning IN (1, 2, 3, 4)
        GROUP BY inning
        ORDER BY inning
    """
    df = con.execute(sql).df()
    con.close()
    return df


def get_batter_chase_splits(player_name: str, match_type: Optional[str] = None) -> pd.DataFrame:
    """
    Computes comparison between Inning 1 (Setting Target) and Inning 2 (Chasing).
    """
    con = get_db_connection()
    clean_name = player_name.strip().replace("'", "''")

    where_clauses = [f"batter = '{clean_name}'", "inning IN (1, 2)"]
    if match_type and match_type != "All":
        clean_mt = match_type.strip().replace("'", "''")
        where_clauses.append(f"match_type = '{clean_mt}'")

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT 
            CASE WHEN inning = 1 THEN '1st Batting (Setting Target)' ELSE '2nd Batting (Chasing)' END as innings_context,
            SUM(runs_batter) as runs,
            COUNT(*) as balls,
            ROUND(SUM(runs_batter) * 100.0 / COUNT(*), 1) as strike_rate,
            ROUND(SUM(is_dot::INT) * 100.0 / COUNT(*), 1) as dot_pct,
            ROUND(SUM(is_boundary::INT) * 100.0 / COUNT(*), 1) as boundary_pct,
            SUM(is_wicket::INT) as dismissals
        FROM fact_deliveries_master
        WHERE {where_sql}
        GROUP BY innings_context
        ORDER BY innings_context DESC
    """
    df = con.execute(sql).df()
    con.close()
    return df


def get_batter_opponent_team_splits(
    player_name: str, match_type: Optional[str] = None, limit: int = 8
) -> pd.DataFrame:
    """
    Computes performance against Opponent Bowling Teams:
    Top teams scored against with strike rates, boundary rates, and dismissals.
    """
    con = get_db_connection()
    clean_name = player_name.strip().replace("'", "''")

    where_clauses = [f"batter = '{clean_name}'"]
    if match_type and match_type != "All":
        clean_mt = match_type.strip().replace("'", "''")
        where_clauses.append(f"match_type = '{clean_mt}'")

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT 
            bowling_team as opponent_team,
            COUNT(DISTINCT match_id) as matches,
            SUM(runs_batter) as runs,
            COUNT(*) as balls,
            ROUND(SUM(runs_batter) * 100.0 / COUNT(*), 1) as strike_rate,
            ROUND(SUM(is_dot::INT) * 100.0 / COUNT(*), 1) as dot_pct,
            ROUND(SUM(is_boundary::INT) * 100.0 / COUNT(*), 1) as boundary_pct,
            SUM(is_wicket::INT) as dismissals
        FROM fact_deliveries_master
        WHERE {where_sql} AND bowling_team IS NOT NULL AND bowling_team != ''
        GROUP BY bowling_team
        HAVING COUNT(*) >= 20
        ORDER BY runs DESC
        LIMIT {limit}
    """
    df = con.execute(sql).df()
    con.close()
    return df


def get_batter_dismissal_breakdown(player_name: str, match_type: Optional[str] = None) -> pd.DataFrame:
    """
    Computes breakdown of dismissal modes (caught, bowled, lbw, run out, stumped).
    """
    con = get_db_connection()
    clean_name = player_name.strip().replace("'", "''")

    where_clauses = [
        f"player_out = '{clean_name}'", 
        "dismissal_kind IS NOT NULL", 
        "dismissal_kind != ''"
    ]
    if match_type and match_type != "All":
        clean_mt = match_type.strip().replace("'", "''")
        where_clauses.append(f"match_type = '{clean_mt}'")

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT 
            dismissal_kind,
            COUNT(*) as dismissal_count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as pct
        FROM fact_deliveries_master
        WHERE {where_sql}
        GROUP BY dismissal_kind
        ORDER BY dismissal_count DESC
    """
    df = con.execute(sql).df()
    con.close()
    return df


def get_batter_venue_splits(
    player_name: str, match_type: Optional[str] = None, limit: int = 8
) -> pd.DataFrame:
    """
    Computes top grounds by runs scored, along with strike rate and boundary rates.
    """
    con = get_db_connection()
    clean_name = player_name.strip().replace("'", "''")

    where_clauses = [f"batter = '{clean_name}'", "ground_name != 'Unknown Venue'"]
    if match_type and match_type != "All":
        clean_mt = match_type.strip().replace("'", "''")
        where_clauses.append(f"match_type = '{clean_mt}'")

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT 
            ground_name,
            COUNT(DISTINCT match_id) as matches,
            SUM(runs_batter) as runs,
            COUNT(*) as balls,
            ROUND(SUM(runs_batter) * 100.0 / COUNT(*), 1) as strike_rate,
            ROUND(SUM(is_boundary::INT) * 100.0 / COUNT(*), 1) as boundary_pct
        FROM fact_deliveries_master
        WHERE {where_sql}
        GROUP BY ground_name
        HAVING COUNT(*) >= 25
        ORDER BY runs DESC
        LIMIT {limit}
    """
    df = con.execute(sql).df()
    con.close()
    return df
