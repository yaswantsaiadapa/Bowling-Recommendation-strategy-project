"""
Bowling Analytics Query Engine.

Provides deep, high-speed analytical extractions for any bowler across all 4.81M deliveries (T20I, ODI, Test):
1. Overview KPIs: Career wickets, average, economy rate, strike rate, dot ball %, extras.
2. Phase Breakdown: Powerplay vs Middle vs Death wickets and economy rates (T20I / ODI only).
3. Test Innings Splits: Inning 1, Inning 2, Inning 3, Inning 4 progression (Test only).
4. Defending Matrix: 1st Innings (Restricting) vs 2nd Innings (Defending target) (T20I / ODI only).
5. Wicket Anatomy: Distribution of bowler dismissal modes (bowled, caught, lbw, stumped, caught and bowled).
6. Opponent Team Mastery: Dominance over opposing international batting lineups.
7. Venue Mastery: Top grounds by wickets taken and economy rate.
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

DB_PATH = PROJECT_ROOT / "data" / "processed" / "cricket_master.duckdb"

# Official bowler-credited dismissal kinds per cricket laws
BOWLER_WICKET_KINDS = (
    "'bowled', 'caught', 'lbw', 'stumped', 'caught and bowled', 'hit wicket'"
)


def get_db_connection() -> duckdb.DuckDBPyConnection:
    """Returns a read-only DuckDB connection to the local analytical database."""
    return duckdb.connect(str(DB_PATH), read_only=True)


def search_bowlers(query: str, limit: int = 10) -> List[str]:
    """
    Finds bowler names matching a search query for autocomplete.
    """
    if not query or len(query.strip()) < 2:
        return []

    con = get_db_connection()
    clean_q = query.strip().replace("'", "''")
    sql = f"""
        SELECT player_name 
        FROM dim_players 
        WHERE LOWER(player_name) LIKE '%{clean_q.lower()}%'
          AND balls_bowled >= 20
        ORDER BY balls_bowled DESC
        LIMIT {limit}
    """
    res = con.execute(sql).fetchall()
    con.close()
    return [r[0] for r in res]


def get_bowler_overview(
    player_name: str,
    match_type: Optional[str] = None,
    match_phase: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Computes career executive KPIs for a bowler across selected format and phase.
    """
    con = get_db_connection()
    clean_name = player_name.strip().replace("'", "''")

    where_clauses = [f"bowler = '{clean_name}'"]
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
            COUNT(DISTINCT match_id || '_' || inning) as innings_bowled,
            COUNT(*) as total_deliveries,
            SUM(CASE WHEN extra_wides = 0 AND extra_noballs = 0 THEN 1 ELSE 0 END) as legal_balls,
            SUM(runs_batter + extra_wides + extra_noballs) as runs_conceded,
            SUM(CASE WHEN is_wicket AND dismissal_kind IN ({BOWLER_WICKET_KINDS}) THEN 1 ELSE 0 END) as wickets,
            SUM(CASE WHEN (runs_batter + extra_wides + extra_noballs) = 0 THEN 1 ELSE 0 END) as dot_balls,
            SUM(is_boundary::INT) as boundaries_conceded,
            SUM(is_four::INT) as fours_conceded,
            SUM(is_six::INT) as sixes_conceded,
            SUM(extra_wides) as total_wides,
            SUM(extra_noballs) as total_noballs
        FROM fact_deliveries_master
        WHERE {where_sql}
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
    boundaries = int(row["boundaries_conceded"] or 0)
    wides = int(row["total_wides"] or 0)
    noballs = int(row["total_noballs"] or 0)

    overs_bowled = round(legal_balls / 6.0, 1)
    overs_display = f"{legal_balls // 6}.{legal_balls % 6}"

    econ = round(runs / (legal_balls / 6.0), 2) if legal_balls > 0 else 0.0
    avg = round(runs / wkts, 2) if wkts > 0 else 0.0
    sr = round(legal_balls / wkts, 1) if wkts > 0 else 0.0
    dot_pct = round((dots * 100.0) / total_deliveries, 1) if total_deliveries > 0 else 0.0
    boundary_pct = round((boundaries * 100.0) / total_deliveries, 1) if total_deliveries > 0 else 0.0

    return {
        "player_name": player_name,
        "matches_played": int(row["matches_played"] or 0),
        "innings_bowled": int(row["innings_bowled"] or 0),
        "total_deliveries": total_deliveries,
        "legal_balls": legal_balls,
        "overs_bowled": overs_bowled,
        "overs_display": overs_display,
        "runs_conceded": runs,
        "wickets": wkts,
        "economy_rate": econ,
        "bowling_average": avg,
        "strike_rate": sr,
        "dot_balls": dots,
        "dot_pct": dot_pct,
        "boundaries_conceded": boundaries,
        "boundary_pct": boundary_pct,
        "fours_conceded": int(row["fours_conceded"] or 0),
        "sixes_conceded": int(row["sixes_conceded"] or 0),
        "wides": wides,
        "noballs": noballs,
        "extras_conceded": wides + noballs,
    }


def get_bowler_phase_splits(
    player_name: str, match_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Computes performance broken down by match phase: Powerplay, Middle, Death.
    Exclusively relevant for limited-overs cricket (T20I, ODI).
    """
    con = get_db_connection()
    clean_name = player_name.strip().replace("'", "''")

    where_clauses = [
        f"bowler = '{clean_name}'",
        "match_phase IN ('Powerplay', 'Middle', 'Death')",
    ]
    if match_type and match_type != "All":
        clean_mt = match_type.strip().replace("'", "''")
        where_clauses.append(f"match_type = '{clean_mt}'")

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT 
            match_phase,
            SUM(CASE WHEN extra_wides = 0 AND extra_noballs = 0 THEN 1 ELSE 0 END) as legal_balls,
            SUM(runs_batter + extra_wides + extra_noballs) as runs_conceded,
            SUM(CASE WHEN is_wicket AND dismissal_kind IN ({BOWLER_WICKET_KINDS}) THEN 1 ELSE 0 END) as wickets,
            SUM(CASE WHEN (runs_batter + extra_wides + extra_noballs) = 0 THEN 1 ELSE 0 END) as dot_balls,
            COUNT(*) as total_deliveries
        FROM fact_deliveries_master
        WHERE {where_sql}
        GROUP BY match_phase
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty:
        return pd.DataFrame()

    df["legal_balls"] = df["legal_balls"].astype(int)
    df["runs_conceded"] = df["runs_conceded"].astype(int)
    df["wickets"] = df["wickets"].astype(int)
    df["dot_balls"] = df["dot_balls"].astype(int)

    df["economy_rate"] = df.apply(
        lambda r: round(r["runs_conceded"] / (r["legal_balls"] / 6.0), 2)
        if r["legal_balls"] > 0
        else 0.0,
        axis=1,
    )
    df["strike_rate"] = df.apply(
        lambda r: round(r["legal_balls"] / r["wickets"], 1) if r["wickets"] > 0 else 0.0,
        axis=1,
    )
    df["dot_pct"] = df.apply(
        lambda r: round((r["dot_balls"] * 100.0) / r["total_deliveries"], 1)
        if r["total_deliveries"] > 0
        else 0.0,
        axis=1,
    )

    # Sort in chronological order
    phase_order = {"Powerplay": 1, "Middle": 2, "Death": 3}
    df["sort_order"] = df["match_phase"].map(phase_order).fillna(99)
    df = df.sort_values(by="sort_order").drop(columns=["sort_order"])

    return df


def get_bowler_test_inning_splits(player_name: str) -> pd.DataFrame:
    """
    Computes performance across the 4 innings of Test matches.
    """
    con = get_db_connection()
    clean_name = player_name.strip().replace("'", "''")

    sql = f"""
        SELECT 
            inning,
            SUM(CASE WHEN extra_wides = 0 AND extra_noballs = 0 THEN 1 ELSE 0 END) as legal_balls,
            SUM(runs_batter + extra_wides + extra_noballs) as runs_conceded,
            SUM(CASE WHEN is_wicket AND dismissal_kind IN ({BOWLER_WICKET_KINDS}) THEN 1 ELSE 0 END) as wickets,
            SUM(CASE WHEN (runs_batter + extra_wides + extra_noballs) = 0 THEN 1 ELSE 0 END) as dot_balls,
            COUNT(*) as total_deliveries
        FROM fact_deliveries_master
        WHERE bowler = '{clean_name}'
          AND match_type = 'Test'
          AND inning IN (1, 2, 3, 4)
        GROUP BY inning
        ORDER BY inning ASC
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty:
        return pd.DataFrame()

    labels = {
        1: "Inning 1 (Setting Tone)",
        2: "Inning 2 (First Response)",
        3: "Inning 3 (Turnaround)",
        4: "Inning 4 (Day 5 Pressure)",
    }
    df["inning_label"] = df["inning"].map(labels)
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
    df["strike_rate"] = df.apply(
        lambda r: round(r["legal_balls"] / r["wickets"], 1) if r["wickets"] > 0 else 0.0,
        axis=1,
    )

    return df


def get_bowler_defending_splits(
    player_name: str, match_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Compares Bowling 1st (restricting total) vs Bowling 2nd (defending target).
    Exclusively relevant for limited-overs cricket (T20I, ODI).
    """
    con = get_db_connection()
    clean_name = player_name.strip().replace("'", "''")

    where_clauses = [
        f"bowler = '{clean_name}'",
        "inning IN (1, 2)",
    ]
    if match_type and match_type != "All":
        clean_mt = match_type.strip().replace("'", "''")
        where_clauses.append(f"match_type = '{clean_mt}'")

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT 
            inning,
            SUM(CASE WHEN extra_wides = 0 AND extra_noballs = 0 THEN 1 ELSE 0 END) as legal_balls,
            SUM(runs_batter + extra_wides + extra_noballs) as runs_conceded,
            SUM(CASE WHEN is_wicket AND dismissal_kind IN ({BOWLER_WICKET_KINDS}) THEN 1 ELSE 0 END) as wickets,
            SUM(CASE WHEN (runs_batter + extra_wides + extra_noballs) = 0 THEN 1 ELSE 0 END) as dot_balls,
            SUM(is_boundary::INT) as boundaries,
            COUNT(*) as total_deliveries
        FROM fact_deliveries_master
        WHERE {where_sql}
        GROUP BY inning
        ORDER BY inning ASC
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty:
        return pd.DataFrame()

    labels = {
        1: "1st Innings (Restricting)",
        2: "2nd Innings (Defending Target)",
    }
    df["innings_context"] = df["inning"].map(labels)
    df["legal_balls"] = df["legal_balls"].astype(int)
    df["runs_conceded"] = df["runs_conceded"].astype(int)
    df["wickets"] = df["wickets"].astype(int)

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
    df["strike_rate"] = df.apply(
        lambda r: round(r["legal_balls"] / r["wickets"], 1) if r["wickets"] > 0 else 0.0,
        axis=1,
    )
    df["dot_pct"] = df.apply(
        lambda r: round((r["dot_balls"] * 100.0) / r["total_deliveries"], 1)
        if r["total_deliveries"] > 0
        else 0.0,
        axis=1,
    )
    df["boundary_pct"] = df.apply(
        lambda r: round((r["boundaries"] * 100.0) / r["total_deliveries"], 1)
        if r["total_deliveries"] > 0
        else 0.0,
        axis=1,
    )

    return df


def get_bowler_dismissal_breakdown(
    player_name: str, match_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Breaks down every career dismissal credited to the bowler by mode.
    """
    con = get_db_connection()
    clean_name = player_name.strip().replace("'", "''")

    where_clauses = [
        f"bowler = '{clean_name}'",
        f"is_wicket = TRUE",
        f"dismissal_kind IN ({BOWLER_WICKET_KINDS})",
    ]
    if match_type and match_type != "All":
        clean_mt = match_type.strip().replace("'", "''")
        where_clauses.append(f"match_type = '{clean_mt}'")

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT 
            dismissal_kind,
            COUNT(*) as dismissal_count
        FROM fact_deliveries_master
        WHERE {where_sql}
        GROUP BY dismissal_kind
        ORDER BY dismissal_count DESC
    """
    df = con.execute(sql).df()
    con.close()
    return df


def get_bowler_opponent_splits(
    player_name: str, match_type: Optional[str] = None, limit: int = 10
) -> pd.DataFrame:
    """
    Extracts top international opponent teams bowled against, ranked by wickets taken.
    """
    con = get_db_connection()
    clean_name = player_name.strip().replace("'", "''")

    where_clauses = [
        f"bowler = '{clean_name}'",
        "batting_team IS NOT NULL",
    ]
    if match_type and match_type != "All":
        clean_mt = match_type.strip().replace("'", "''")
        where_clauses.append(f"match_type = '{clean_mt}'")

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT 
            batting_team as opponent_team,
            COUNT(DISTINCT match_id) as matches,
            SUM(CASE WHEN extra_wides = 0 AND extra_noballs = 0 THEN 1 ELSE 0 END) as legal_balls,
            SUM(runs_batter + extra_wides + extra_noballs) as runs_conceded,
            SUM(CASE WHEN is_wicket AND dismissal_kind IN ({BOWLER_WICKET_KINDS}) THEN 1 ELSE 0 END) as wickets
        FROM fact_deliveries_master
        WHERE {where_sql}
        GROUP BY batting_team
        HAVING SUM(CASE WHEN is_wicket AND dismissal_kind IN ({BOWLER_WICKET_KINDS}) THEN 1 ELSE 0 END) > 0
        ORDER BY wickets DESC, runs_conceded ASC
        LIMIT {limit}
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty:
        return pd.DataFrame()

    df["legal_balls"] = df["legal_balls"].astype(int)
    df["runs_conceded"] = df["runs_conceded"].astype(int)
    df["wickets"] = df["wickets"].astype(int)
    df["matches"] = df["matches"].astype(int)

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


def get_bowler_venue_splits(
    player_name: str, match_type: Optional[str] = None, limit: int = 10
) -> pd.DataFrame:
    """
    Extracts top venues bowled at, ranked by wickets taken.
    """
    con = get_db_connection()
    clean_name = player_name.strip().replace("'", "''")

    where_clauses = [
        f"bowler = '{clean_name}'",
        "ground_name IS NOT NULL",
    ]
    if match_type and match_type != "All":
        clean_mt = match_type.strip().replace("'", "''")
        where_clauses.append(f"match_type = '{clean_mt}'")

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT 
            ground_name,
            COUNT(DISTINCT match_id) as matches,
            SUM(CASE WHEN extra_wides = 0 AND extra_noballs = 0 THEN 1 ELSE 0 END) as legal_balls,
            SUM(runs_batter + extra_wides + extra_noballs) as runs_conceded,
            SUM(CASE WHEN is_wicket AND dismissal_kind IN ({BOWLER_WICKET_KINDS}) THEN 1 ELSE 0 END) as wickets
        FROM fact_deliveries_master
        WHERE {where_sql}
        GROUP BY ground_name
        HAVING SUM(CASE WHEN is_wicket AND dismissal_kind IN ({BOWLER_WICKET_KINDS}) THEN 1 ELSE 0 END) > 0
        ORDER BY wickets DESC, runs_conceded ASC
        LIMIT {limit}
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty:
        return pd.DataFrame()

    df["legal_balls"] = df["legal_balls"].astype(int)
    df["runs_conceded"] = df["runs_conceded"].astype(int)
    df["wickets"] = df["wickets"].astype(int)
    df["matches"] = df["matches"].astype(int)

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
