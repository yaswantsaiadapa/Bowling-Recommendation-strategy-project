"""
Global Cricket Insights & Historical Evolution Analytics Engine.

Directly powers the /insights dashboard using the master 4.81M delivery DuckDB database.
Extracts empirical macro-trends from 2001 to 2026:
1. Historical Scoring & Run Rate Evolution (Format-pure, Test 4-innings lines, legal balls denominator).
2. Format Comparison Matrix (Test vs ODI vs T20I side-by-side benchmark).
3. Test Cricket Four-Innings Progression & Draw Rate Trajectory (No limited-overs fluff).
4. Batting Accumulation DNA (Dot %, running runs %, boundary runs % stacked over time).
5. Toss Impact & Batting 1st vs Chasing Target Brackets (Zero dew/day-night claims).
6. Bowling & Dismissal Evolution (Wickets per 100 balls, economy, dismissal mode shares).
7. Three-Era Comparative Matrix (2000-2009 vs 2010-2019 vs 2020-2026 YTD).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import pandas as pd

DB_PATH = PROJECT_ROOT / "data" / "processed" / "cricket_master.duckdb"


def _get_connection():
    """Return a read-only connection to the master DuckDB database."""
    return duckdb.connect(str(DB_PATH), read_only=True)


def get_filter_options() -> Dict[str, Any]:
    """Retrieve available filter options from DuckDB."""
    con = _get_connection()
    try:
        formats = ["All Formats", "Test", "ODI", "T20I"]

        teams_res = con.execute("""
            SELECT DISTINCT team FROM (
                SELECT team_1 as team FROM dim_matches WHERE team_1 IS NOT NULL AND team_1 != ''
                UNION
                SELECT team_2 as team FROM dim_matches WHERE team_2 IS NOT NULL AND team_2 != ''
            )
            ORDER BY team
        """).fetchall()
        teams = ["All Teams"] + [r[0] for r in teams_res]

        venues_res = con.execute("""
            SELECT venue, COUNT(*) as cnt
            FROM dim_matches
            WHERE venue IS NOT NULL AND venue != ''
            GROUP BY venue
            HAVING cnt >= 10
            ORDER BY cnt DESC
        """).fetchall()
        venues = ["All Venues"] + [r[0] for r in venues_res]

        years_res = con.execute("""
            SELECT MIN(SUBSTR(match_date, 1, 4)::INT) as min_yr,
                   MAX(SUBSTR(match_date, 1, 4)::INT) as max_yr
            FROM dim_matches
            WHERE match_date IS NOT NULL
        """).fetchone()

        min_year = int(years_res[0]) if years_res and years_res[0] else 2001
        max_year = int(years_res[1]) if years_res and years_res[1] else 2026

        return {
            "formats": formats,
            "teams": teams,
            "venues": venues,
            "min_year": max(2001, min_year),
            "max_year": max_year,
        }
    finally:
        con.close()


def _build_where_clause(
    match_type: Optional[str] = None,
    team: Optional[str] = None,
    venue: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    table_alias: str = "m"
) -> Tuple[str, List[Any]]:
    """Build parameterized SQL WHERE conditions for matches or joins."""
    clauses = [f"{table_alias}.match_date IS NOT NULL"]
    params = []

    if match_type and match_type not in ("All Formats", "All", ""):
        clauses.append(f"{table_alias}.match_type = ?")
        params.append(match_type)

    if team and team not in ("All Teams", "All", ""):
        clauses.append(f"({table_alias}.team_1 = ? OR {table_alias}.team_2 = ?)")
        params.extend([team, team])

    if venue and venue not in ("All Venues", "All", ""):
        clauses.append(f"{table_alias}.venue = ?")
        params.append(venue)

    if year_min:
        clauses.append(f"SUBSTR({table_alias}.match_date, 1, 4)::INT >= ?")
        params.append(int(year_min))

    if year_max:
        clauses.append(f"SUBSTR({table_alias}.match_date, 1, 4)::INT <= ?")
        params.append(int(year_max))

    where_sql = " AND ".join(clauses)
    return where_sql, params


def get_executive_kpis(
    match_type: Optional[str] = None,
    team: Optional[str] = None,
    venue: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None
) -> Dict[str, Any]:
    """Calculate top-level aggregate KPIs across the selected slice."""
    con = _get_connection()
    try:
        where_sql, params = _build_where_clause(match_type, team, venue, year_min, year_max, "m")

        query = f"""
            SELECT 
                COUNT(DISTINCT m.match_id) as total_matches,
                COUNT(f.delivery_key) as total_deliveries,
                SUM(f.total_runs) as total_runs,
                SUM(CASE WHEN f.is_wicket = true AND f.dismissal_kind IN ('bowled', 'caught', 'lbw', 'stumped', 'caught and bowled', 'hit wicket') THEN 1 ELSE 0 END) as bowler_wickets,
                SUM(CASE WHEN f.is_wicket = true THEN 1 ELSE 0 END) as all_wickets,
                SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END) as legal_balls,
                SUM(CASE WHEN f.is_four = true THEN 1 ELSE 0 END) as total_fours,
                SUM(CASE WHEN f.is_six = true THEN 1 ELSE 0 END) as total_sixes,
                SUM(CASE WHEN f.is_dot = true THEN 1 ELSE 0 END) as dot_balls
            FROM dim_matches m
            JOIN fact_deliveries_master f ON m.match_id = f.match_id
            WHERE {where_sql}
        """
        row = con.execute(query, params).fetchone()

        total_matches = int(row[0] or 0)
        total_balls = int(row[1] or 0)
        total_runs = int(row[2] or 0)
        bowler_wkts = int(row[3] or 0)
        all_wkts = int(row[4] or 0)
        legal_balls = int(row[5] or 0)
        total_fours = int(row[6] or 0)
        total_sixes = int(row[7] or 0)
        dot_balls = int(row[8] or 0)

        rpo = round((total_runs / (legal_balls / 6.0)), 2) if legal_balls > 0 else 0.0
        bdry_runs = (total_fours * 4) + (total_sixes * 6)
        bdry_run_pct = round((bdry_runs / total_runs * 100.0), 1) if total_runs > 0 else 0.0
        dot_pct = round((dot_balls / legal_balls * 100.0), 1) if legal_balls > 0 else 0.0
        bdry_freq = round((legal_balls / (total_fours + total_sixes)), 1) if (total_fours + total_sixes) > 0 else 0.0
        wkts_per_100 = round((bowler_wkts / (legal_balls / 100.0)), 2) if legal_balls > 0 else 0.0

        return {
            "total_matches": total_matches,
            "total_deliveries": total_balls,
            "total_runs": total_runs,
            "total_wickets": all_wkts,
            "bowler_wickets": bowler_wkts,
            "legal_balls": legal_balls,
            "run_rate": rpo,
            "boundary_run_pct": bdry_run_pct,
            "dot_ball_pct": dot_pct,
            "balls_per_boundary": bdry_freq,
            "wickets_per_100_balls": wkts_per_100,
        }
    finally:
        con.close()


def get_yearly_scoring_evolution(
    match_type: Optional[str] = None,
    team: Optional[str] = None,
    venue: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None
) -> Dict[str, Any]:
    """
    Extracts annual scoring evolution:
    - If format is 'Test': Innings 1, 2, 3, 4 independent average totals over time.
    - If format is 'ODI' or 'T20I': Overall average innings score, RPO, and boundaries over time.
    - If format is 'All Formats': Yearly metrics broken down by format.
    """
    con = _get_connection()
    try:
        where_sql, params = _build_where_clause(match_type, team, venue, year_min, year_max, "m")

        # Innings-level yearly aggregation
        inn_query = f"""
            WITH innings_totals AS (
                SELECT 
                    m.match_id,
                    m.match_type,
                    SUBSTR(m.match_date, 1, 4)::INT as yr,
                    f.inning as innings,
                    SUM(f.total_runs) as inn_runs,
                    SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END) as inn_legal_balls,
                    SUM(CASE WHEN f.is_four = true THEN 1 ELSE 0 END) as inn_fours,
                    SUM(CASE WHEN f.is_six = true THEN 1 ELSE 0 END) as inn_sixes,
                    SUM(CASE WHEN f.is_dot = true THEN 1 ELSE 0 END) as inn_dots
                FROM dim_matches m
                JOIN fact_deliveries_master f ON m.match_id = f.match_id
                WHERE {where_sql} AND f.inning BETWEEN 1 AND 4
                GROUP BY m.match_id, m.match_type, yr, f.inning
            )
            SELECT 
                yr,
                match_type,
                innings,
                COUNT(*) as innings_count,
                ROUND(AVG(inn_runs), 1) as avg_score,
                ROUND(SUM(inn_runs) / NULLIF(SUM(inn_legal_balls) / 6.0, 0), 2) as run_rate,
                ROUND(AVG(inn_fours), 1) as avg_fours,
                ROUND(AVG(inn_sixes), 1) as avg_sixes,
                ROUND(SUM(inn_fours * 4 + inn_sixes * 6) / NULLIF(SUM(inn_runs), 0) * 100.0, 1) as bdry_run_pct,
                ROUND(SUM(inn_legal_balls) / NULLIF(SUM(inn_fours + inn_sixes), 0), 1) as balls_per_bdry,
                ROUND(SUM(inn_dots) / NULLIF(SUM(inn_legal_balls), 0) * 100.0, 1) as dot_pct
            FROM innings_totals
            GROUP BY yr, match_type, innings
            ORDER BY yr ASC, innings ASC
        """
        df_inn = con.execute(inn_query, params).df()

        # Overall yearly aggregation per format
        fmt_query = f"""
            WITH innings_totals AS (
                SELECT 
                    m.match_id,
                    m.match_type,
                    SUBSTR(m.match_date, 1, 4)::INT as yr,
                    f.inning as innings,
                    SUM(f.total_runs) as inn_runs,
                    SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END) as inn_legal_balls,
                    SUM(CASE WHEN f.is_four = true THEN 1 ELSE 0 END) as inn_fours,
                    SUM(CASE WHEN f.is_six = true THEN 1 ELSE 0 END) as inn_sixes,
                    SUM(CASE WHEN f.is_dot = true THEN 1 ELSE 0 END) as inn_dots
                FROM dim_matches m
                JOIN fact_deliveries_master f ON m.match_id = f.match_id
                WHERE {where_sql} AND f.inning BETWEEN 1 AND 4
                GROUP BY m.match_id, m.match_type, yr, f.inning
            )
            SELECT 
                yr,
                match_type,
                COUNT(*) as innings_count,
                ROUND(AVG(inn_runs), 1) as avg_score,
                ROUND(SUM(inn_runs) / NULLIF(SUM(inn_legal_balls) / 6.0, 0), 2) as run_rate,
                ROUND(AVG(inn_fours), 1) as avg_fours,
                ROUND(AVG(inn_sixes), 1) as avg_sixes,
                ROUND(SUM(inn_fours * 4 + inn_sixes * 6) / NULLIF(SUM(inn_runs), 0) * 100.0, 1) as bdry_run_pct,
                ROUND(SUM(inn_legal_balls) / NULLIF(SUM(inn_fours + inn_sixes), 0), 1) as balls_per_bdry,
                ROUND(SUM(inn_dots) / NULLIF(SUM(inn_legal_balls), 0) * 100.0, 1) as dot_pct
            FROM innings_totals
            GROUP BY yr, match_type
            ORDER BY yr ASC
        """
        df_fmt = con.execute(fmt_query, params).df()

        return {
            "by_innings": df_inn,
            "by_format": df_fmt,
        }
    finally:
        con.close()


def get_format_comparison_matrix(
    team: Optional[str] = None,
    venue: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None
) -> Dict[str, Any]:
    """
    Computes a clean side-by-side format benchmark table:
    Test vs ODI vs T20I across standard empirical metrics.
    """
    con = _get_connection()
    try:
        where_sql, params = _build_where_clause(None, team, venue, year_min, year_max, "m")

        query = f"""
            WITH inn_stats AS (
                SELECT 
                    m.match_id,
                    m.match_type,
                    f.inning as innings,
                    SUM(f.total_runs) as inn_runs,
                    SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END) as inn_legal,
                    SUM(CASE WHEN f.is_four = true THEN 1 ELSE 0 END) as inn_fours,
                    SUM(CASE WHEN f.is_six = true THEN 1 ELSE 0 END) as inn_sixes,
                    SUM(CASE WHEN f.is_dot = true THEN 1 ELSE 0 END) as inn_dots,
                    SUM(CASE WHEN f.is_wicket = true AND f.dismissal_kind IN ('bowled', 'caught', 'lbw', 'stumped', 'caught and bowled', 'hit wicket') THEN 1 ELSE 0 END) as inn_wkts
                FROM dim_matches m
                JOIN fact_deliveries_master f ON m.match_id = f.match_id
                WHERE {where_sql} AND m.match_type IN ('Test', 'ODI', 'T20I') AND f.inning BETWEEN 1 AND 4
                GROUP BY m.match_id, m.match_type, f.inning
            )
            SELECT 
                match_type,
                COUNT(*) as innings_count,
                ROUND(AVG(inn_runs), 1) as avg_score,
                ROUND(AVG(CASE WHEN innings = 1 THEN inn_runs ELSE NULL END), 1) as avg_1st_inn_score,
                ROUND(SUM(inn_runs) / NULLIF(SUM(inn_legal) / 6.0, 0), 2) as run_rate,
                ROUND(AVG(inn_fours), 1) as avg_fours,
                ROUND(AVG(inn_sixes), 1) as avg_sixes,
                ROUND(SUM(inn_dots) / NULLIF(SUM(inn_legal), 0) * 100.0, 1) as dot_pct,
                ROUND(SUM(inn_fours + inn_sixes) / NULLIF(SUM(inn_legal), 0) * 100.0, 1) as boundary_ball_pct,
                ROUND(SUM(inn_fours * 4 + inn_sixes * 6) / NULLIF(SUM(inn_runs), 0) * 100.0, 1) as boundary_run_pct,
                ROUND(SUM(inn_wkts) / NULLIF(SUM(inn_legal) / 100.0, 0), 2) as wkts_per_100_balls,
                ROUND(AVG(inn_wkts), 1) as avg_wkts_per_inn
            FROM inn_stats
            GROUP BY match_type
            ORDER BY CASE match_type WHEN 'Test' THEN 1 WHEN 'ODI' THEN 2 WHEN 'T20I' THEN 3 ELSE 4 END
        """
        rows = con.execute(query, params).fetchall()

        matrix = {}
        for r in rows:
            m_type = str(r[0])
            matrix[m_type] = {
                "innings_count": int(r[1] or 0),
                "avg_score": float(r[2] or 0.0),
                "avg_1st_inn_score": float(r[3] or 0.0),
                "run_rate": float(r[4] or 0.0),
                "avg_fours": float(r[5] or 0.0),
                "avg_sixes": float(r[6] or 0.0),
                "dot_pct": float(r[7] or 0.0),
                "boundary_ball_pct": float(r[8] or 0.0),
                "boundary_run_pct": float(r[9] or 0.0),
                "wkts_per_100_balls": float(r[10] or 0.0),
                "avg_wkts_per_inn": float(r[11] or 0.0),
            }

        return matrix
    finally:
        con.close()


def get_test_4innings_evolution(
    team: Optional[str] = None,
    venue: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None
) -> Dict[str, Any]:
    """
    Dedicated Test Cricket Four-Innings Progression and Outright vs Draw analytics.
    Excludes limited-overs fluff; focuses on:
    - Innings 1, 2, 3, 4 yearly average scores.
    - Run rate decay by innings number.
    - Wickets lost & All-out % by innings.
    - Historical Win/Loss vs Draw percentage evolution over time.
    - 4th innings target pursuit outcomes (Target brackets vs result).
    """
    con = _get_connection()
    try:
        where_sql, params = _build_where_clause("Test", team, venue, year_min, year_max, "m")

        # Yearly average by innings 1-4
        yearly_sql = f"""
            WITH inn_stats AS (
                SELECT 
                    m.match_id,
                    SUBSTR(m.match_date, 1, 4)::INT as yr,
                    f.inning as innings,
                    SUM(f.total_runs) as inn_runs,
                    SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END) as inn_legal,
                    SUM(CASE WHEN f.is_wicket = true THEN 1 ELSE 0 END) as inn_wkts
                FROM dim_matches m
                JOIN fact_deliveries_master f ON m.match_id = f.match_id
                WHERE {where_sql} AND f.inning BETWEEN 1 AND 4
                GROUP BY m.match_id, yr, f.inning
            )
            SELECT 
                yr,
                innings,
                COUNT(*) as inn_count,
                ROUND(AVG(inn_runs), 1) as avg_runs,
                ROUND(SUM(inn_runs) / NULLIF(SUM(inn_legal) / 6.0, 0), 2) as run_rate,
                ROUND(AVG(inn_wkts), 1) as avg_wkts
            FROM inn_stats
            GROUP BY yr, innings
            ORDER BY yr ASC, innings ASC
        """
        df_yearly = con.execute(yearly_sql, params).df()

        # Overall innings 1-4 summary
        summary_sql = f"""
            WITH inn_stats AS (
                SELECT 
                    m.match_id,
                    f.inning as innings,
                    SUM(f.total_runs) as inn_runs,
                    SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END) as inn_legal,
                    SUM(CASE WHEN f.is_wicket = true THEN 1 ELSE 0 END) as inn_wkts
                FROM dim_matches m
                JOIN fact_deliveries_master f ON m.match_id = f.match_id
                WHERE {where_sql} AND f.inning BETWEEN 1 AND 4
                GROUP BY m.match_id, f.inning
            )
            SELECT 
                innings,
                COUNT(*) as inn_count,
                ROUND(AVG(inn_runs), 1) as avg_runs,
                ROUND(SUM(inn_runs) / NULLIF(SUM(inn_legal) / 6.0, 0), 2) as run_rate,
                ROUND(AVG(inn_wkts), 1) as avg_wkts,
                ROUND(SUM(CASE WHEN inn_wkts >= 10 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as all_out_pct,
                ROUND(SUM(inn_wkts) / NULLIF(SUM(inn_legal) / 100.0, 0), 2) as wkts_per_100_balls
            FROM inn_stats
            GROUP BY innings
            ORDER BY innings ASC
        """
        df_summary = con.execute(summary_sql, params).df()

        # Draw % vs Result % over time
        draw_sql = f"""
            SELECT 
                SUBSTR(match_date, 1, 4)::INT as yr,
                COUNT(*) as total_matches,
                SUM(CASE WHEN outcome_result = 'draw' OR winner IS NULL OR winner = '' THEN 1 ELSE 0 END) as draw_matches,
                ROUND(SUM(CASE WHEN outcome_result = 'draw' OR winner IS NULL OR winner = '' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as draw_pct,
                ROUND(SUM(CASE WHEN winner IS NOT NULL AND winner != '' AND (outcome_result IS NULL OR outcome_result != 'draw') THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as result_pct
            FROM dim_matches m
            WHERE {where_sql}
            GROUP BY yr
            ORDER BY yr ASC
        """
        df_draws = con.execute(draw_sql, params).df()

        # 4th Innings Target Pursuits
        target_sql = f"""
            WITH test_chases AS (
                SELECT 
                    m.match_id,
                    m.winner,
                    m.outcome_result,
                    SUM(CASE WHEN f.inning = 1 THEN f.total_runs ELSE 0 END) as inn1,
                    SUM(CASE WHEN f.inning = 2 THEN f.total_runs ELSE 0 END) as inn2,
                    SUM(CASE WHEN f.inning = 3 THEN f.total_runs ELSE 0 END) as inn3,
                    SUM(CASE WHEN f.inning = 4 THEN f.total_runs ELSE 0 END) as inn4,
                    SUM(CASE WHEN f.inning = 4 AND f.is_wicket = true THEN 1 ELSE 0 END) as inn4_wkts
                FROM dim_matches m
                JOIN fact_deliveries_master f ON m.match_id = f.match_id
                WHERE {where_sql}
                GROUP BY m.match_id, m.winner, m.outcome_result
                HAVING inn4 > 0
            ),
            chase_eval AS (
                SELECT 
                    match_id,
                    (inn1 + inn3 - inn2 + 1) as target,
                    inn4,
                    inn4_wkts,
                    outcome_result,
                    winner,
                    CASE 
                        WHEN outcome_result = 'draw' OR winner IS NULL OR winner = '' THEN 'Draw (Survived)'
                        WHEN inn4 >= (inn1 + inn3 - inn2 + 1) THEN 'Successful Chase'
                        ELSE 'All-Out / Defeat'
                    END as chase_outcome
                FROM test_chases
                WHERE (inn1 + inn3 - inn2 + 1) > 0
            )
            SELECT 
                CASE 
                    WHEN target < 150 THEN '< 150'
                    WHEN target BETWEEN 150 AND 249 THEN '150–249'
                    WHEN target BETWEEN 250 AND 349 THEN '250–349'
                    ELSE '350+'
                END as target_range,
                COUNT(*) as match_count,
                ROUND(SUM(CASE WHEN chase_outcome = 'Successful Chase' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as chase_win_pct,
                ROUND(SUM(CASE WHEN chase_outcome = 'Draw (Survived)' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as draw_pct,
                ROUND(SUM(CASE WHEN chase_outcome = 'All-Out / Defeat' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as defeat_pct
            FROM chase_eval
            GROUP BY target_range
            ORDER BY MIN(target) ASC
        """
        df_targets = con.execute(target_sql, params).df()

        return {
            "yearly_innings": df_yearly,
            "summary_innings": df_summary,
            "draw_trajectory": df_draws,
            "target_pursuits": df_targets,
        }
    finally:
        con.close()


def get_batting_accumulation_dna(
    match_type: Optional[str] = None,
    team: Optional[str] = None,
    venue: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None
) -> Dict[str, Any]:
    """
    Computes batting run accumulation mechanics:
    - Yearly % of Dots, Singles, Twos, Threes, Fours, Sixes.
    - Format-wise breakdown of accumulation DNA.
    - Denominator is strictly legal deliveries for ball percentages.
    """
    con = _get_connection()
    try:
        where_sql, params = _build_where_clause(match_type, team, venue, year_min, year_max, "m")

        yearly_sql = f"""
            SELECT 
                SUBSTR(m.match_date, 1, 4)::INT as yr,
                COUNT(*) as total_deliveries,
                SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END) as legal_balls,
                ROUND(SUM(CASE WHEN f.is_dot = true THEN 1 ELSE 0 END) * 100.0 / NULLIF(SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END), 0), 1) as dot_pct,
                ROUND(SUM(CASE WHEN f.runs_batter = 1 THEN 1 ELSE 0 END) * 100.0 / NULLIF(SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END), 0), 1) as single_pct,
                ROUND(SUM(CASE WHEN f.runs_batter = 2 THEN 1 ELSE 0 END) * 100.0 / NULLIF(SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END), 0), 1) as two_pct,
                ROUND(SUM(CASE WHEN f.runs_batter = 3 THEN 1 ELSE 0 END) * 100.0 / NULLIF(SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END), 0), 1) as three_pct,
                ROUND(SUM(CASE WHEN f.is_four = true THEN 1 ELSE 0 END) * 100.0 / NULLIF(SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END), 0), 1) as four_pct,
                ROUND(SUM(CASE WHEN f.is_six = true THEN 1 ELSE 0 END) * 100.0 / NULLIF(SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END), 0), 1) as six_pct,
                ROUND(SUM(CASE WHEN f.is_four = true OR f.is_six = true THEN 1 ELSE 0 END) * 100.0 / NULLIF(SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END), 0), 1) as boundary_pct,
                ROUND(SUM(f.runs_batter) / NULLIF(SUM(CASE WHEN f.runs_batter > 0 THEN 1 ELSE 0 END), 0), 2) as runs_per_scoring_ball
            FROM dim_matches m
            JOIN fact_deliveries_master f ON m.match_id = f.match_id
            WHERE {where_sql}
            GROUP BY yr
            ORDER BY yr ASC
        """
        df_yearly = con.execute(yearly_sql, params).df()

        # Format breakdown
        fmt_sql = f"""
            SELECT 
                m.match_type,
                SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END) as legal_balls,
                ROUND(SUM(CASE WHEN f.is_dot = true THEN 1 ELSE 0 END) * 100.0 / NULLIF(SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END), 0), 1) as dot_pct,
                ROUND(SUM(CASE WHEN f.runs_batter = 1 THEN 1 ELSE 0 END) * 100.0 / NULLIF(SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END), 0), 1) as single_pct,
                ROUND(SUM(CASE WHEN f.runs_batter = 2 THEN 1 ELSE 0 END) * 100.0 / NULLIF(SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END), 0), 1) as two_pct,
                ROUND(SUM(CASE WHEN f.runs_batter = 3 THEN 1 ELSE 0 END) * 100.0 / NULLIF(SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END), 0), 1) as three_pct,
                ROUND(SUM(CASE WHEN f.is_four = true THEN 1 ELSE 0 END) * 100.0 / NULLIF(SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END), 0), 1) as four_pct,
                ROUND(SUM(CASE WHEN f.is_six = true THEN 1 ELSE 0 END) * 100.0 / NULLIF(SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END), 0), 1) as six_pct,
                ROUND(SUM(CASE WHEN f.is_four = true THEN 4 WHEN f.is_six = true THEN 6 ELSE 0 END) * 100.0 / NULLIF(SUM(f.total_runs), 0), 1) as runs_from_boundaries_pct,
                ROUND(SUM(CASE WHEN f.runs_batter BETWEEN 1 AND 3 THEN f.runs_batter ELSE 0 END) * 100.0 / NULLIF(SUM(f.total_runs), 0), 1) as runs_from_running_pct
            FROM dim_matches m
            JOIN fact_deliveries_master f ON m.match_id = f.match_id
            WHERE {where_sql} AND m.match_type IN ('Test', 'ODI', 'T20I')
            GROUP BY m.match_type
            ORDER BY CASE m.match_type WHEN 'Test' THEN 1 WHEN 'ODI' THEN 2 WHEN 'T20I' THEN 3 ELSE 4 END
        """
        df_format = con.execute(fmt_sql, params).df()

        return {
            "yearly": df_yearly,
            "by_format": df_format,
        }
    finally:
        con.close()


def get_toss_and_chasing_analytics(
    match_type: Optional[str] = None,
    team: Optional[str] = None,
    venue: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None
) -> Dict[str, Any]:
    """
    Computes Toss and Batting 1st vs Chasing outcomes:
    - Toss winning team win % and decision split (bat vs field).
    - Win % choosing bat vs field.
    - Format-wise differences (Test vs ODI vs T20I).
    - Limited-Overs Target Chasing Brackets (<140, 140-169, 170-199, 200+ for T20I; <200, 200-249, 250-299, 300+ for ODI).
    - OMITTING completely all dew and day/night claims.
    """
    con = _get_connection()
    try:
        where_sql, params = _build_where_clause(match_type, team, venue, year_min, year_max, "m")

        # Toss Conversion Overall & By Format
        toss_sql = f"""
            SELECT 
                m.match_type,
                COUNT(*) as completed_matches,
                ROUND(SUM(CASE WHEN m.toss_winner = m.winner THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as toss_win_match_win_pct,
                ROUND(SUM(CASE WHEN m.toss_decision = 'bat' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as choose_bat_pct,
                ROUND(SUM(CASE WHEN m.toss_decision = 'field' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as choose_field_pct,
                ROUND(SUM(CASE WHEN m.toss_decision = 'bat' AND m.toss_winner = m.winner THEN 1 ELSE 0 END) * 100.0 / NULLIF(SUM(CASE WHEN m.toss_decision = 'bat' THEN 1 ELSE 0 END), 0), 1) as bat_first_conversion_pct,
                ROUND(SUM(CASE WHEN m.toss_decision = 'field' AND m.toss_winner = m.winner THEN 1 ELSE 0 END) * 100.0 / NULLIF(SUM(CASE WHEN m.toss_decision = 'field' THEN 1 ELSE 0 END), 0), 1) as field_first_conversion_pct
            FROM dim_matches m
            WHERE {where_sql} AND m.toss_winner IS NOT NULL AND m.toss_winner != '' 
              AND m.winner IS NOT NULL AND m.winner != '' AND m.toss_decision IN ('bat', 'field')
            GROUP BY m.match_type
            ORDER BY CASE m.match_type WHEN 'Test' THEN 1 WHEN 'ODI' THEN 2 WHEN 'T20I' THEN 3 ELSE 4 END
        """
        df_toss = con.execute(toss_sql, params).df()

        # Limited-Overs Batting 1st vs Chasing Win Rates & Scores
        lo_sql = f"""
            WITH inn_totals AS (
                SELECT 
                    m.match_id,
                    m.match_type,
                    m.winner,
                    m.win_margin_runs,
                    m.win_margin_wkts,
                    SUM(CASE WHEN f.inning = 1 THEN f.total_runs ELSE 0 END) as inn1_runs,
                    SUM(CASE WHEN f.inning = 2 THEN f.total_runs ELSE 0 END) as inn2_runs
                FROM dim_matches m
                JOIN fact_deliveries_master f ON m.match_id = f.match_id
                WHERE {where_sql} AND m.match_type IN ('ODI', 'T20I') AND m.winner IS NOT NULL AND m.winner != ''
                GROUP BY m.match_id, m.match_type, m.winner, m.win_margin_runs, m.win_margin_wkts
                HAVING inn1_runs > 0 AND inn2_runs > 0
            )
            SELECT 
                match_type,
                COUNT(*) as matches_analyzed,
                ROUND(AVG(inn1_runs), 1) as avg_1st_innings,
                ROUND(SUM(CASE WHEN win_margin_runs > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as bat_first_win_pct,
                ROUND(SUM(CASE WHEN win_margin_wkts > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as chase_win_pct,
                ROUND(AVG(CASE WHEN win_margin_wkts > 0 THEN inn2_runs ELSE NULL END), 1) as avg_successful_chase_score,
                ROUND(AVG(CASE WHEN win_margin_runs > 0 THEN inn2_runs ELSE NULL END), 1) as avg_unsuccessful_chase_score
            FROM inn_totals
            GROUP BY match_type
            ORDER BY match_type DESC
        """
        df_lo = con.execute(lo_sql, params).df()

        # Chasing Brackets for T20I
        t20_brackets_sql = f"""
            WITH inn_totals AS (
                SELECT 
                    m.match_id,
                    m.win_margin_runs,
                    m.win_margin_wkts,
                    SUM(CASE WHEN f.inning = 1 THEN f.total_runs ELSE 0 END) as inn1_runs
                FROM dim_matches m
                JOIN fact_deliveries_master f ON m.match_id = f.match_id
                WHERE {where_sql} AND m.match_type = 'T20I' AND m.winner IS NOT NULL AND m.winner != ''
                GROUP BY m.match_id, m.win_margin_runs, m.win_margin_wkts
                HAVING inn1_runs > 0
            )
            SELECT 
                CASE 
                    WHEN inn1_runs < 140 THEN '< 140'
                    WHEN inn1_runs BETWEEN 140 AND 169 THEN '140–169'
                    WHEN inn1_runs BETWEEN 170 AND 199 THEN '170–199'
                    ELSE '200+'
                END as target_bracket,
                COUNT(*) as match_count,
                ROUND(SUM(CASE WHEN win_margin_wkts > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as chase_success_pct,
                ROUND(SUM(CASE WHEN win_margin_runs > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as bat_first_defended_pct
            FROM inn_totals
            GROUP BY target_bracket
            ORDER BY MIN(inn1_runs) ASC
        """
        df_t20_brackets = con.execute(t20_brackets_sql, params).df()

        # Chasing Brackets for ODI
        odi_brackets_sql = f"""
            WITH inn_totals AS (
                SELECT 
                    m.match_id,
                    m.win_margin_runs,
                    m.win_margin_wkts,
                    SUM(CASE WHEN f.inning = 1 THEN f.total_runs ELSE 0 END) as inn1_runs
                FROM dim_matches m
                JOIN fact_deliveries_master f ON m.match_id = f.match_id
                WHERE {where_sql} AND m.match_type = 'ODI' AND m.winner IS NOT NULL AND m.winner != ''
                GROUP BY m.match_id, m.win_margin_runs, m.win_margin_wkts
                HAVING inn1_runs > 0
            )
            SELECT 
                CASE 
                    WHEN inn1_runs < 200 THEN '< 200'
                    WHEN inn1_runs BETWEEN 200 AND 249 THEN '200–249'
                    WHEN inn1_runs BETWEEN 250 AND 299 THEN '250–299'
                    ELSE '300+'
                END as target_bracket,
                COUNT(*) as match_count,
                ROUND(SUM(CASE WHEN win_margin_wkts > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as chase_success_pct,
                ROUND(SUM(CASE WHEN win_margin_runs > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as bat_first_defended_pct
            FROM inn_totals
            GROUP BY target_bracket
            ORDER BY MIN(inn1_runs) ASC
        """
        df_odi_brackets = con.execute(odi_brackets_sql, params).df()

        return {
            "toss_summary": df_toss,
            "limited_overs_outcomes": df_lo,
            "t20i_brackets": df_t20_brackets,
            "odi_brackets": df_odi_brackets,
        }
    finally:
        con.close()


def get_bowling_evolution(
    match_type: Optional[str] = None,
    team: Optional[str] = None,
    venue: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None
) -> Dict[str, Any]:
    """
    Computes historical bowling metrics and dismissal distributions:
    - Yearly trend of wickets per 100 legal deliveries, bowling average, and economy.
    - Distribution of dismissal types: caught, bowled, lbw, stumped, run out, caught behind.
    - Format-wise dismissal comparison.
    """
    con = _get_connection()
    try:
        where_sql, params = _build_where_clause(match_type, team, venue, year_min, year_max, "m")

        yearly_sql = f"""
            SELECT 
                SUBSTR(m.match_date, 1, 4)::INT as yr,
                SUM(f.total_runs) as total_runs,
                SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END) as legal_balls,
                SUM(CASE WHEN f.is_wicket = true AND f.dismissal_kind IN ('bowled', 'caught', 'lbw', 'stumped', 'caught and bowled', 'hit wicket') THEN 1 ELSE 0 END) as bowler_wkts,
                SUM(CASE WHEN f.is_wicket = true THEN 1 ELSE 0 END) as all_wkts,
                ROUND(SUM(CASE WHEN f.is_wicket = true AND f.dismissal_kind IN ('bowled', 'caught', 'lbw', 'stumped', 'caught and bowled', 'hit wicket') THEN 1 ELSE 0 END) / NULLIF(SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END) / 100.0, 0), 2) as wkts_per_100_balls,
                ROUND(SUM(f.total_runs) / NULLIF(SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END) / 6.0, 0), 2) as economy_rate,
                ROUND(SUM(f.total_runs) / NULLIF(SUM(CASE WHEN f.is_wicket = true AND f.dismissal_kind IN ('bowled', 'caught', 'lbw', 'stumped', 'caught and bowled', 'hit wicket') THEN 1 ELSE 0 END), 0), 2) as bowling_avg
            FROM dim_matches m
            JOIN fact_deliveries_master f ON m.match_id = f.match_id
            WHERE {where_sql}
            GROUP BY yr
            ORDER BY yr ASC
        """
        df_yearly = con.execute(yearly_sql, params).df()

        # Dismissal types distribution
        dismissals_sql = f"""
            SELECT 
                CASE 
                    WHEN f.dismissal_kind IN ('caught', 'caught and bowled') THEN 'Caught (Field)'
                    WHEN f.dismissal_kind = 'bowled' THEN 'Bowled'
                    WHEN f.dismissal_kind = 'lbw' THEN 'LBW'
                    WHEN f.dismissal_kind = 'stumped' THEN 'Stumped'
                    WHEN f.dismissal_kind = 'run out' THEN 'Run Out'
                    WHEN f.dismissal_kind = 'hit wicket' THEN 'Hit Wicket'
                    ELSE 'Other / Retired'
                END as dismissal_group,
                COUNT(*) as wkt_count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as share_pct
            FROM dim_matches m
            JOIN fact_deliveries_master f ON m.match_id = f.match_id
            WHERE {where_sql} AND f.is_wicket = true AND f.dismissal_kind IS NOT NULL AND f.dismissal_kind != ''
            GROUP BY dismissal_group
            ORDER BY wkt_count DESC
        """
        df_dismissals = con.execute(dismissals_sql, params).df()

        return {
            "yearly_bowling": df_yearly,
            "dismissal_types": df_dismissals,
        }
    finally:
        con.close()


def get_era_comparison(
    match_type: Optional[str] = None,
    team: Optional[str] = None,
    venue: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Computes comparative metrics across three distinct eras:
    - 2000–2009: Classical Pre-Modern Era
    - 2010–2019: Analytical & White-Ball Acceleration Era
    - 2020–2026: Modern High-Impact Ultra-Aggression Era
    """
    con = _get_connection()
    try:
        where_sql, params = _build_where_clause(match_type, team, venue, None, None, "m")

        query = f"""
            WITH era_matches AS (
                SELECT 
                    m.match_id,
                    m.match_type,
                    m.win_margin_runs,
                    m.win_margin_wkts,
                    CASE 
                        WHEN SUBSTR(m.match_date, 1, 4)::INT BETWEEN 2000 AND 2009 THEN '2000–2009 (Classical)'
                        WHEN SUBSTR(m.match_date, 1, 4)::INT BETWEEN 2010 AND 2019 THEN '2010–2019 (White-Ball Surge)'
                        WHEN SUBSTR(m.match_date, 1, 4)::INT >= 2020 THEN '2020–2026 (Modern High-Tempo)'
                        ELSE 'Pre-2000'
                    END as era_label,
                    f.total_runs,
                    f.runs_batter,
                    f.extra_wides,
                    f.extra_noballs,
                    f.is_four,
                    f.is_six,
                    f.is_wicket,
                    f.dismissal_kind
                FROM dim_matches m
                JOIN fact_deliveries_master f ON m.match_id = f.match_id
                WHERE {where_sql}
            )
            SELECT 
                era_label,
                COUNT(DISTINCT match_id) as matches,
                ROUND(SUM(total_runs) / NULLIF(SUM(CASE WHEN extra_wides = 0 AND extra_noballs = 0 THEN 1 ELSE 0 END) / 6.0, 0), 2) as run_rate,
                ROUND(SUM(CASE WHEN total_runs = 0 AND extra_wides = 0 AND extra_noballs = 0 THEN 1 ELSE 0 END) * 100.0 / NULLIF(SUM(CASE WHEN extra_wides = 0 AND extra_noballs = 0 THEN 1 ELSE 0 END), 0), 1) as dot_pct,
                ROUND(SUM(CASE WHEN is_four = true OR is_six = true THEN 1 ELSE 0 END) * 100.0 / NULLIF(SUM(CASE WHEN extra_wides = 0 AND extra_noballs = 0 THEN 1 ELSE 0 END), 0), 1) as boundary_ball_pct,
                ROUND(SUM(CASE WHEN is_four = true THEN 4 WHEN is_six = true THEN 6 ELSE 0 END) * 100.0 / NULLIF(SUM(total_runs), 0), 1) as boundary_run_pct,
                ROUND(SUM(CASE WHEN is_wicket = true AND dismissal_kind IN ('bowled', 'caught', 'lbw', 'stumped', 'caught and bowled', 'hit wicket') THEN 1 ELSE 0 END) / NULLIF(SUM(CASE WHEN extra_wides = 0 AND extra_noballs = 0 THEN 1 ELSE 0 END) / 100.0, 0), 2) as wkts_per_100_balls,
                ROUND(SUM(total_runs) / NULLIF(SUM(CASE WHEN is_wicket = true AND dismissal_kind IN ('bowled', 'caught', 'lbw', 'stumped', 'caught and bowled', 'hit wicket') THEN 1 ELSE 0 END), 0), 1) as bowling_avg
            FROM era_matches
            WHERE era_label != 'Pre-2000'
            GROUP BY era_label
            ORDER BY MIN(match_id) ASC
        """
        rows = con.execute(query, params).fetchall()

        eras = []
        for r in rows:
            eras.append({
                "era": str(r[0]),
                "matches": int(r[1] or 0),
                "run_rate": float(r[2] or 0.0),
                "dot_pct": float(r[3] or 0.0),
                "boundary_ball_pct": float(r[4] or 0.0),
                "boundary_run_pct": float(r[5] or 0.0),
                "wkts_per_100_balls": float(r[6] or 0.0),
                "bowling_avg": float(r[7] or 0.0),
            })
        return eras
    finally:
        con.close()


def get_top_performers(
    match_type: str,
    team: Optional[str] = None,
    venue: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    limit: int = 8
) -> Dict[str, pd.DataFrame]:
    """
    Extracts top performing batters (by runs, avg, SR) and bowlers (by wickets, avg, economy)
    for the selected format, team, venue, and era window.
    """
    con = _get_connection()
    try:
        where_sql, params = _build_where_clause(match_type, team, venue, year_min, year_max, "m")

        bat_team_cond = ""
        bowl_team_cond = ""
        bat_params = list(params)
        bowl_params = list(params)

        if team and team not in ("All Teams", "All", ""):
            bat_team_cond = " AND f.batting_team = ?"
            bowl_team_cond = " AND f.bowling_team = ?"
            bat_params.append(team)
            bowl_params.append(team)

        # Top Batters
        q_bat = f"""
            SELECT 
                f.batter as player_name,
                COUNT(DISTINCT f.match_id) as matches,
                SUM(f.runs_batter) as runs,
                COUNT(CASE WHEN f.is_wicket = true AND f.player_out = f.batter THEN 1 END) as dismissals,
                COUNT(f.ball) as balls_faced,
                ROUND(SUM(f.runs_batter) * 100.0 / NULLIF(COUNT(f.ball), 0), 1) as strike_rate,
                ROUND(SUM(f.runs_batter) / NULLIF(COUNT(CASE WHEN f.is_wicket = true AND f.player_out = f.batter THEN 1 END), 0), 1) as batting_avg,
                SUM(CASE WHEN f.is_four = true THEN 1 ELSE 0 END) as fours,
                SUM(CASE WHEN f.is_six = true THEN 1 ELSE 0 END) as sixes
            FROM dim_matches m
            JOIN fact_deliveries_master f ON m.match_id = f.match_id
            WHERE {where_sql} {bat_team_cond}
            GROUP BY f.batter
            HAVING runs >= 30
            ORDER BY runs DESC, batting_avg DESC
            LIMIT {limit}
        """
        df_bat = con.execute(q_bat, bat_params).df()

        # Top Bowlers
        q_bowl = f"""
            SELECT 
                f.bowler as player_name,
                COUNT(DISTINCT f.match_id) as matches,
                SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END) as legal_balls,
                SUM(f.total_runs) as runs_conceded,
                SUM(CASE WHEN f.is_wicket = true AND f.dismissal_kind IN ('bowled', 'caught', 'lbw', 'stumped', 'caught and bowled', 'hit wicket') THEN 1 ELSE 0 END) as wickets,
                ROUND(SUM(f.total_runs) / NULLIF(SUM(CASE WHEN f.is_wicket = true AND f.dismissal_kind IN ('bowled', 'caught', 'lbw', 'stumped', 'caught and bowled', 'hit wicket') THEN 1 ELSE 0 END), 0), 2) as bowling_avg,
                ROUND(SUM(f.total_runs) / NULLIF(SUM(CASE WHEN f.extra_wides = 0 AND f.extra_noballs = 0 THEN 1 ELSE 0 END) / 6.0, 0), 2) as economy_rate
            FROM dim_matches m
            JOIN fact_deliveries_master f ON m.match_id = f.match_id
            WHERE {where_sql} {bowl_team_cond}
            GROUP BY f.bowler
            HAVING wickets >= 2
            ORDER BY wickets DESC, bowling_avg ASC
            LIMIT {limit}
        """
        df_bowl = con.execute(q_bowl, bowl_params).df()

        return {
            "top_batters": df_bat,
            "top_bowlers": df_bowl,
        }
    finally:
        con.close()

