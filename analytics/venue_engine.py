"""
Team & Venue Intelligence Query Engine.

Provides deep analytical extractions for:
1. 🏟️ Ground Profiler:
   - Format-conditional stadium intelligence:
     * Test Cricket: 4 Innings progression, match completion days (Day 3 / 4 / 5 / Draw), 4th innings target pursuit success.
     * T20I & ODI: Format-pure 1st vs 2nd innings totals, Bat 1st vs Chase win bias, Phase scoring run rates.
     * All Formats: Independent format benchmark cards (no blended averages).
   - Venue Hall of Fame (All-time top run scorers and wicket takers at the ground)

2. 🏏 Team Dynamics:
   - Win/Loss records and winning percentages
   - Batting 1st vs Chasing win efficiency
   - Head-to-Head win rate against international opponents
   - Team phase scoring DNA (Powerplay vs Middle vs Death)
   - Fortress grounds (Highest win percentage venues)
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import pandas as pd
from analytics.db import get_db_connection, DB_PATH

BOWLER_WICKET_KINDS = (
    "'bowled', 'caught', 'lbw', 'stumped', 'caught and bowled', 'hit wicket'"
)


def get_popular_venues() -> List[str]:
    """Returns the top venues hosted sorted by match count."""
    con = get_db_connection()
    sql = """
        SELECT ground_name 
        FROM dim_matches 
        WHERE ground_name IS NOT NULL
        GROUP BY ground_name 
        HAVING COUNT(*) >= 15
        ORDER BY COUNT(*) DESC
    """
    res = con.execute(sql).fetchall()
    con.close()
    return [r[0] for r in res]


def get_popular_teams() -> List[str]:
    """Returns all primary international cricket teams."""
    con = get_db_connection()
    sql = """
        SELECT team_1 as team
        FROM dim_matches
        WHERE team_1 IS NOT NULL
        GROUP BY team_1
        HAVING COUNT(*) >= 40
        ORDER BY COUNT(*) DESC
    """
    res = con.execute(sql).fetchall()
    con.close()
    return [r[0] for r in res]


# ══════════════════════════════════════════════════════════════════════════════
# VENUE INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

def get_venue_overview(venue_name: str, match_type: str = "All") -> Optional[Dict[str, Any]]:
    """Computes format-aware executive stadium KPIs."""
    con = get_db_connection()
    c_venue = venue_name.strip().replace("'", "''")

    where_match = [f"ground_name = '{c_venue}'"]
    where_fact = [f"ground_name = '{c_venue}'"]

    if match_type != "All":
        c_mt = match_type.strip().replace("'", "''")
        where_match.append(f"match_type = '{c_mt}'")
        where_fact.append(f"match_type = '{c_mt}'")

    sql_matches = f"""
        SELECT 
            COUNT(*) as total_matches,
            SUM(CASE WHEN winner IS NOT NULL AND winner != '' THEN 1 ELSE 0 END) as decided_matches,
            SUM(CASE WHEN win_margin_runs IS NOT NULL AND win_margin_runs > 0 THEN 1 ELSE 0 END) as bat_first_wins,
            SUM(CASE WHEN win_margin_wkts IS NOT NULL AND win_margin_wkts > 0 THEN 1 ELSE 0 END) as chase_wins,
            SUM(CASE WHEN toss_winner = winner THEN 1 ELSE 0 END) as toss_match_wins,
            SUM(CASE WHEN winner IS NULL OR winner = '' OR outcome_result = 'draw' THEN 1 ELSE 0 END) as draws
        FROM dim_matches
        WHERE {" AND ".join(where_match)}
    """
    row_m = con.execute(sql_matches).fetchone()

    if not row_m or row_m[0] == 0:
        con.close()
        return None

    total_matches = row_m[0]
    decided = row_m[1] or 0
    bat_1st_wins = row_m[2] or 0
    chase_wins = row_m[3] or 0
    toss_wins = row_m[4] or 0
    draws = row_m[5] or 0

    decided_pct = round((decided * 100.0) / total_matches, 1) if total_matches > 0 else 0.0
    draw_pct = round((draws * 100.0) / total_matches, 1) if total_matches > 0 else 0.0
    bat_1st_pct = round((bat_1st_wins * 100.0) / decided, 1) if decided > 0 else 0.0
    chase_pct = round((chase_wins * 100.0) / decided, 1) if decided > 0 else 0.0
    toss_pct = round((toss_wins * 100.0) / total_matches, 1) if total_matches > 0 else 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # CASE 1: TEST CRICKET VENUE OVERVIEW
    # ──────────────────────────────────────────────────────────────────────────
    if match_type == "Test":
        # 4 Innings scores
        sql_4inn = f"""
            SELECT 
                inning,
                ROUND(AVG(total_runs), 0) as avg_score,
                MAX(total_runs) as max_score,
                MIN(total_runs) as min_score,
                COUNT(DISTINCT match_id) as sample_count
            FROM (
                SELECT match_id, inning, SUM(total_runs) as total_runs
                FROM fact_deliveries_master
                WHERE ground_name = '{c_venue}' AND match_type = 'Test' AND inning IN (1, 2, 3, 4)
                GROUP BY match_id, inning
            )
            GROUP BY inning
            ORDER BY inning
        """
        inn_df = con.execute(sql_4inn).df()
        avg_1st = avg_2nd = avg_3rd = avg_4th = 0
        highest_total = 0
        lowest_total = 999

        for _, r in inn_df.iterrows():
            inn = int(r["inning"])
            score = int(r["avg_score"] or 0)
            if inn == 1:
                avg_1st = score
            elif inn == 2:
                avg_2nd = score
            elif inn == 3:
                avg_3rd = score
            elif inn == 4:
                avg_4th = score
            highest_total = max(highest_total, int(r["max_score"] or 0))
            if r["min_score"] and r["min_score"] > 20:
                lowest_total = min(lowest_total, int(r["min_score"]))

        if lowest_total == 999:
            lowest_total = 0

        # Match Duration & 4th Innings Chase statistics
        sql_test_meta = f"""
            WITH match_stats AS (
                SELECT 
                    m.match_id,
                    m.winner,
                    m.win_margin_runs,
                    m.win_margin_wkts,
                    m.outcome_result,
                    MAX(f.inning) as num_innings,
                    COUNT(f.ball) as total_balls
                FROM dim_matches m
                JOIN fact_deliveries_master f ON m.match_id = f.match_id
                WHERE m.ground_name = '{c_venue}' AND m.match_type = 'Test'
                GROUP BY m.match_id, m.winner, m.win_margin_runs, m.win_margin_wkts, m.outcome_result
            )
            SELECT 
                SUM(CASE WHEN outcome_result = 'draw' OR winner IS NULL OR winner = '' THEN 0
                         WHEN total_balls <= 1620 THEN 1 ELSE 0 END) as day_3_finishes,
                SUM(CASE WHEN outcome_result != 'draw' AND winner IS NOT NULL AND winner != ''
                         AND total_balls > 1620 AND total_balls <= 2160 THEN 1 ELSE 0 END) as day_4_finishes,
                SUM(CASE WHEN outcome_result != 'draw' AND winner IS NOT NULL AND winner != ''
                         AND total_balls > 2160 THEN 1 ELSE 0 END) as day_5_finishes,
                SUM(CASE WHEN outcome_result = 'draw' OR winner IS NULL OR winner = '' THEN 1 ELSE 0 END) as draw_finishes,
                SUM(CASE WHEN num_innings = 4 THEN 1 ELSE 0 END) as reached_4th_inn,
                SUM(CASE WHEN num_innings = 4 AND win_margin_wkts > 0 THEN 1 ELSE 0 END) as chased_4th_inn,
                SUM(CASE WHEN num_innings = 4 AND win_margin_runs > 0 THEN 1 ELSE 0 END) as defended_4th_inn
            FROM match_stats
        """
        meta_row = con.execute(sql_test_meta).fetchone()
        con.close()

        day_3 = int(meta_row[0] or 0) if meta_row else 0
        day_4 = int(meta_row[1] or 0) if meta_row else 0
        day_5 = int(meta_row[2] or 0) if meta_row else 0
        draw_count = int(meta_row[3] or 0) if meta_row else 0
        reached_4th = int(meta_row[4] or 0) if meta_row else 0
        chased_4th = int(meta_row[5] or 0) if meta_row else 0
        defended_4th = int(meta_row[6] or 0) if meta_row else 0

        chase_4th_pct = round((chased_4th * 100.0) / reached_4th, 1) if reached_4th > 0 else 0.0

        # Typical duration label
        if day_5 >= max(day_3, day_4, 1):
            typical_duration = "Day 5 Battle"
        elif day_4 >= max(day_3, 1):
            typical_duration = "Day 4 Decider"
        else:
            typical_duration = "Day 3 or Earlier"

        verdict = f"🏛️ Classic Test Pitch ({decided_pct}% Result Rate)"
        verdict_color = "red"
        verdict_desc = f"{decided} of {total_matches} Tests produced an outright winner ({decided_pct}% result rate, {draw_pct}% draws). Teams batting 4th successfully chase targets {chase_4th_pct}% of the time."

        return {
            "venue_name": venue_name,
            "format": "Test",
            "is_test": True,
            "is_all": False,
            "total_matches": total_matches,
            "decided_matches": decided,
            "draws": draws,
            "decided_pct": decided_pct,
            "draw_pct": draw_pct,
            "avg_1st_innings": avg_1st,
            "avg_2nd_innings": avg_2nd,
            "avg_3rd_innings": avg_3rd,
            "avg_4th_innings": avg_4th,
            "highest_total": highest_total,
            "lowest_total": lowest_total,
            "reached_4th_inn": reached_4th,
            "chased_4th_inn": chased_4th,
            "defended_4th_inn": defended_4th,
            "fourth_inn_chase_success_pct": chase_4th_pct,
            "day_3_finishes": day_3,
            "day_4_finishes": day_4,
            "day_5_finishes": day_5,
            "draw_finishes": draw_count,
            "typical_duration": typical_duration,
            "verdict": verdict,
            "verdict_color": verdict_color,
            "verdict_desc": verdict_desc,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # CASE 2: ALL FORMATS VENUE OVERVIEW (Never blend format scores!)
    # ──────────────────────────────────────────────────────────────────────────
    if match_type == "All":
        # Format-specific breakdown cards
        sql_breakdown = f"""
            WITH match_summary AS (
                SELECT 
                    match_type,
                    COUNT(*) as total_matches,
                    SUM(CASE WHEN winner IS NOT NULL AND winner != '' THEN 1 ELSE 0 END) as decided,
                    SUM(CASE WHEN win_margin_runs > 0 THEN 1 ELSE 0 END) as bat_first_wins,
                    SUM(CASE WHEN win_margin_wkts > 0 THEN 1 ELSE 0 END) as chase_wins,
                    SUM(CASE WHEN winner IS NULL OR winner = '' OR outcome_result = 'draw' THEN 1 ELSE 0 END) as draws
                FROM dim_matches
                WHERE ground_name = '{c_venue}'
                GROUP BY match_type
            ),
            score_summary AS (
                SELECT 
                    match_type,
                    inning,
                    ROUND(AVG(total_runs), 0) as avg_score
                FROM (
                    SELECT match_id, match_type, inning, SUM(total_runs) as total_runs
                    FROM fact_deliveries_master
                    WHERE ground_name = '{c_venue}' AND inning IN (1, 2)
                    GROUP BY match_id, match_type, inning
                )
                GROUP BY match_type, inning
            )
            SELECT 
                m.match_type,
                m.total_matches,
                m.decided,
                m.bat_first_wins,
                m.chase_wins,
                m.draws,
                s1.avg_score as avg_1st,
                s2.avg_score as avg_2nd
            FROM match_summary m
            LEFT JOIN score_summary s1 ON m.match_type = s1.match_type AND s1.inning = 1
            LEFT JOIN score_summary s2 ON m.match_type = s2.match_type AND s2.inning = 2
            ORDER BY m.total_matches DESC
        """
        breakdown_df = con.execute(sql_breakdown).df()
        con.close()

        format_cards = []
        for _, r in breakdown_df.iterrows():
            fmt = str(r["match_type"])
            t_m = int(r["total_matches"])
            dec = int(r["decided"] or 0)
            b1 = int(r["bat_first_wins"] or 0)
            ch = int(r["chase_wins"] or 0)
            dr = int(r["draws"] or 0)
            a1 = int(r["avg_1st"] or 0)
            a2 = int(r["avg_2nd"] or 0)

            card = {
                "format": fmt,
                "matches": t_m,
                "decided": dec,
                "draws": dr,
                "avg_1st": a1,
                "avg_2nd": a2,
            }
            if fmt == "Test":
                card["decided_pct"] = round((dec * 100.0) / t_m, 1) if t_m > 0 else 0.0
                card["draw_pct"] = round((dr * 100.0) / t_m, 1) if t_m > 0 else 0.0
            else:
                card["bat_first_win_pct"] = round((b1 * 100.0) / dec, 1) if dec > 0 else 0.0
                card["chase_win_pct"] = round((ch * 100.0) / dec, 1) if dec > 0 else 0.0
            format_cards.append(card)

        verdict = f"🌐 Multi-Format International Stadium ({total_matches} Matches)"
        verdict_color = "purple"
        verdict_desc = f"Hosted {total_matches} total international matches across Test, ODI, and T20I cricket. Format benchmarks shown below independently."

        return {
            "venue_name": venue_name,
            "format": "All",
            "is_test": False,
            "is_all": True,
            "total_matches": total_matches,
            "decided_matches": decided,
            "ties_draws": draws,
            "format_cards": format_cards,
            "verdict": verdict,
            "verdict_color": verdict_color,
            "verdict_desc": verdict_desc,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # CASE 3: LIMITED OVERS (T20I / ODI) VENUE OVERVIEW
    # ──────────────────────────────────────────────────────────────────────────
    sql_scores = f"""
        SELECT 
            inning,
            ROUND(AVG(inning_runs), 0) as avg_score,
            MAX(inning_runs) as max_score,
            MIN(inning_runs) as min_score
        FROM (
            SELECT match_id, inning, SUM(total_runs) as inning_runs
            FROM fact_deliveries_master
            WHERE {" AND ".join(where_fact)}
              AND inning IN (1, 2)
            GROUP BY match_id, inning
            HAVING COUNT(*) >= 60
        )
        GROUP BY inning
    """
    scores_df = con.execute(sql_scores).df()
    con.close()

    avg_1st = 0
    avg_2nd = 0
    highest_total = 0
    lowest_total = 999

    for _, r in scores_df.iterrows():
        if r["inning"] == 1:
            avg_1st = int(r["avg_score"] or 0)
        elif r["inning"] == 2:
            avg_2nd = int(r["avg_score"] or 0)
        highest_total = max(highest_total, int(r["max_score"] or 0))
        if r["min_score"] and r["min_score"] > 20:
            lowest_total = min(lowest_total, int(r["min_score"]))

    if lowest_total == 999:
        lowest_total = 0

    if bat_1st_pct >= 56.0:
        verdict = f"🏏 Batting First Fortress ({bat_1st_pct}% Win Rate)"
        verdict_color = "blue"
        verdict_desc = f"Teams batting first win {bat_1st_pct}% of matches. Defending totals is statistically favored here."
    elif chase_pct >= 56.0:
        verdict = f"🎯 Chasing Heaven ({chase_pct}% Win Rate)"
        verdict_color = "green"
        verdict_desc = f"Teams chasing win {chase_pct}% of matches. Target hunting is highly effective here."
    else:
        verdict = "⚖️ Balanced Pitch"
        verdict_color = "orange"
        verdict_desc = f"Evenly poised track: {bat_1st_pct}% bat first wins vs {chase_pct}% chase wins."

    return {
        "venue_name": venue_name,
        "format": match_type,
        "is_test": False,
        "is_all": False,
        "total_matches": total_matches,
        "decided_matches": decided,
        "bat_first_wins": bat_1st_wins,
        "chase_wins": chase_wins,
        "ties_draws": draws,
        "bat_first_win_pct": bat_1st_pct,
        "chase_win_pct": chase_pct,
        "toss_match_win_pct": toss_pct,
        "avg_1st_innings": avg_1st,
        "avg_2nd_innings": avg_2nd,
        "highest_total": highest_total,
        "lowest_total": lowest_total,
        "verdict": verdict,
        "verdict_color": verdict_color,
        "verdict_desc": verdict_desc,
    }


def get_venue_test_innings(venue_name: str) -> pd.DataFrame:
    """Returns 4 innings score decay progression for Test cricket."""
    con = get_db_connection()
    c_venue = venue_name.strip().replace("'", "''")

    sql = f"""
        SELECT 
            inning,
            ROUND(AVG(total_runs), 1) as avg_score,
            COUNT(DISTINCT match_id) as sample_size,
            MAX(total_runs) as max_score,
            MIN(total_runs) as min_score
        FROM (
            SELECT match_id, inning, SUM(total_runs) as total_runs
            FROM fact_deliveries_master
            WHERE ground_name = '{c_venue}' AND match_type = 'Test' AND inning IN (1, 2, 3, 4)
            GROUP BY match_id, inning
        )
        GROUP BY inning
        ORDER BY inning
    """
    df = con.execute(sql).df()
    con.close()
    return df


def get_venue_test_durations(venue_name: str) -> Dict[str, int]:
    """Returns match completion duration breakdown for Test cricket."""
    con = get_db_connection()
    c_venue = venue_name.strip().replace("'", "''")

    sql = f"""
        WITH match_stats AS (
            SELECT 
                m.match_id,
                m.winner,
                m.outcome_result,
                COUNT(f.ball) as total_balls
            FROM dim_matches m
            JOIN fact_deliveries_master f ON m.match_id = f.match_id
            WHERE m.ground_name = '{c_venue}' AND m.match_type = 'Test'
            GROUP BY m.match_id, m.winner, m.outcome_result
        )
        SELECT 
            CASE 
                WHEN outcome_result = 'draw' OR winner IS NULL OR winner = '' THEN 'Draw'
                WHEN total_balls <= 1620 THEN 'Day 3 or earlier'
                WHEN total_balls <= 2160 THEN 'Day 4'
                ELSE 'Day 5'
            END as duration_category,
            COUNT(*) as match_count
        FROM match_stats
        GROUP BY 1
    """
    df = con.execute(sql).df()
    con.close()

    durations = {"Day 3 or earlier": 0, "Day 4": 0, "Day 5": 0, "Draw": 0}
    for _, r in df.iterrows():
        cat = str(r["duration_category"])
        if cat in durations:
            durations[cat] = int(r["match_count"])
    return durations


def get_venue_outcomes(venue_name: str, match_type: str = "All") -> Dict[str, int]:
    """Returns outcome distribution for limited-overs or general match views."""
    con = get_db_connection()
    c_venue = venue_name.strip().replace("'", "''")

    where_match = [f"ground_name = '{c_venue}'"]
    if match_type != "All":
        where_match.append(f"match_type = '{match_type.strip().replace('\'', '\'\'')}'")

    sql = f"""
        SELECT 
            SUM(CASE WHEN win_margin_runs IS NOT NULL AND win_margin_runs > 0 THEN 1 ELSE 0 END) as bat_first,
            SUM(CASE WHEN win_margin_wkts IS NOT NULL AND win_margin_wkts > 0 THEN 1 ELSE 0 END) as chase,
            SUM(CASE WHEN winner IS NULL OR winner = '' OR outcome_result = 'draw' THEN 1 ELSE 0 END) as ties_draws
        FROM dim_matches
        WHERE {" AND ".join(where_match)}
    """
    row = con.execute(sql).fetchone()
    con.close()

    if not row:
        return {"Batting 1st Wins": 0, "Chasing Wins": 0, "Ties / Draws": 0}

    return {
        "Batting 1st Wins": int(row[0] or 0),
        "Chasing Wins": int(row[1] or 0),
        "Ties / Draws": int(row[2] or 0),
    }


def get_venue_phase_splits(venue_name: str, match_type: str = "All") -> pd.DataFrame:
    """Scoring run rates and containment across phases at this venue."""
    con = get_db_connection()
    c_venue = venue_name.strip().replace("'", "''")

    where_fact = [
        f"ground_name = '{c_venue}'",
        "match_phase IN ('Powerplay', 'Middle', 'Death')",
    ]
    if match_type != "All":
        where_fact.append(f"match_type = '{match_type.strip().replace('\'', '\'\'')}'")

    sql = f"""
        SELECT 
            match_phase,
            SUM(total_runs) as runs,
            COUNT(*) as balls,
            ROUND((SUM(total_runs) * 6.0) / COUNT(*), 2) as run_rate,
            ROUND(SUM(is_dot::INT) * 100.0 / COUNT(*), 1) as dot_pct,
            ROUND(SUM(is_boundary::INT) * 100.0 / COUNT(*), 1) as boundary_pct
        FROM fact_deliveries_master
        WHERE {" AND ".join(where_fact)}
        GROUP BY match_phase
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty:
        return pd.DataFrame()

    phase_order = {"Powerplay": 1, "Middle": 2, "Death": 3}
    df["sort_order"] = df["match_phase"].map(phase_order).fillna(99)
    df = df.sort_values(by="sort_order").drop(columns=["sort_order"])
    return df


def get_venue_top_batters(venue_name: str, match_type: str = "All", limit: int = 5) -> List[Dict[str, Any]]:
    """All-time top run scorers at this ground."""
    con = get_db_connection()
    c_venue = venue_name.strip().replace("'", "''")

    where_fact = [f"ground_name = '{c_venue}'"]
    if match_type != "All":
        where_fact.append(f"match_type = '{match_type.strip().replace('\'', '\'\'')}'")

    sql = f"""
        SELECT 
            batter,
            COUNT(DISTINCT match_id) as matches,
            SUM(runs_batter) as total_runs,
            COUNT(*) as balls_faced,
            SUM(is_wicket::INT) as dismissals
        FROM fact_deliveries_master
        WHERE {" AND ".join(where_fact)}
        GROUP BY batter
        ORDER BY total_runs DESC
        LIMIT {limit}
    """
    df = con.execute(sql).df()
    con.close()

    results = []
    for _, r in df.iterrows():
        b = int(r["balls_faced"])
        runs = int(r["total_runs"])
        d = int(r["dismissals"])
        sr = round((runs * 100.0) / b, 1) if b > 0 else 0.0
        avg = round(runs / d, 1) if d > 0 else float(runs)
        results.append({
            "player": str(r["batter"]),
            "matches": int(r["matches"]),
            "runs": runs,
            "balls": b,
            "strike_rate": sr,
            "average": avg,
        })
    return results


def get_venue_top_bowlers(venue_name: str, match_type: str = "All", limit: int = 5) -> List[Dict[str, Any]]:
    """All-time top wicket takers at this ground."""
    con = get_db_connection()
    c_venue = venue_name.strip().replace("'", "''")

    where_fact = [f"ground_name = '{c_venue}'"]
    if match_type != "All":
        where_fact.append(f"match_type = '{match_type.strip().replace('\'', '\'\'')}'")

    sql = f"""
        SELECT 
            bowler,
            COUNT(DISTINCT match_id) as matches,
            SUM(CASE WHEN extra_wides = 0 AND extra_noballs = 0 THEN 1 ELSE 0 END) as legal_balls,
            SUM(runs_batter + extra_wides + extra_noballs) as runs_conceded,
            SUM(CASE WHEN is_wicket AND dismissal_kind IN ({BOWLER_WICKET_KINDS}) THEN 1 ELSE 0 END) as wickets
        FROM fact_deliveries_master
        WHERE {" AND ".join(where_fact)}
        GROUP BY bowler
        ORDER BY wickets DESC, runs_conceded ASC
        LIMIT {limit}
    """
    df = con.execute(sql).df()
    con.close()

    results = []
    for _, r in df.iterrows():
        b = int(r["legal_balls"])
        w = int(r["wickets"])
        rc = int(r["runs_conceded"])
        overs = f"{b // 6}.{b % 6}"
        econ = round(rc / (b / 6.0), 2) if b > 0 else 0.0
        avg = round(rc / w, 1) if w > 0 else 0.0
        results.append({
            "player": str(r["bowler"]),
            "matches": int(r["matches"]),
            "wickets": w,
            "overs": overs,
            "economy": econ,
            "average": avg,
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# TEAM DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════

def get_team_overview(team_name: str, match_type: str = "All") -> Optional[Dict[str, Any]]:
    """Computes executive national team KPIs."""
    con = get_db_connection()
    c_team = team_name.strip().replace("'", "''")

    where_match = [f"(team_1 = '{c_team}' OR team_2 = '{c_team}')"]
    if match_type != "All":
        where_match.append(f"match_type = '{match_type.strip().replace('\'', '\'\'')}'")

    sql = f"""
        SELECT 
            COUNT(*) as total_matches,
            SUM(CASE WHEN winner = '{c_team}' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN winner IS NOT NULL AND winner != '' AND winner != '{c_team}' THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN winner IS NULL OR winner = '' OR outcome_result = 'draw' THEN 1 ELSE 0 END) as ties_draws,
            SUM(CASE WHEN winner = '{c_team}' AND win_margin_runs > 0 THEN 1 ELSE 0 END) as bat_first_wins,
            SUM(CASE WHEN winner = '{c_team}' AND win_margin_wkts > 0 THEN 1 ELSE 0 END) as chase_wins,
            SUM(CASE WHEN toss_winner = '{c_team}' THEN 1 ELSE 0 END) as toss_won,
            SUM(CASE WHEN toss_winner = '{c_team}' AND winner = '{c_team}' THEN 1 ELSE 0 END) as toss_match_wins
        FROM dim_matches
        WHERE {" AND ".join(where_match)}
    """
    row = con.execute(sql).fetchone()
    con.close()

    if not row or row[0] == 0:
        return None

    total = row[0]
    wins = row[1] or 0
    losses = row[2] or 0
    ties_draws = row[3] or 0
    bat_1st = row[4] or 0
    chase = row[5] or 0
    toss_won = row[6] or 0
    toss_match_wins = row[7] or 0

    win_pct = round((wins * 100.0) / total, 1) if total > 0 else 0.0
    bat_first_pct = round((bat_1st * 100.0) / wins, 1) if wins > 0 else 0.0
    chase_pct = round((chase * 100.0) / wins, 1) if wins > 0 else 0.0
    toss_conv_pct = round((toss_match_wins * 100.0) / toss_won, 1) if toss_won > 0 else 0.0

    return {
        "team_name": team_name,
        "format": match_type,
        "total_matches": total,
        "wins": wins,
        "losses": losses,
        "ties_draws": ties_draws,
        "win_pct": win_pct,
        "bat_first_wins": bat_1st,
        "chase_wins": chase,
        "bat_first_win_pct": bat_first_pct,
        "chase_win_pct": chase_pct,
        "toss_conversion_pct": toss_conv_pct,
    }


def get_team_opponents_record(team_name: str, match_type: str = "All", limit: int = 10) -> pd.DataFrame:
    """Head-to-Head record against every international opponent."""
    con = get_db_connection()
    c_team = team_name.strip().replace("'", "''")

    where_match = [f"(team_1 = '{c_team}' OR team_2 = '{c_team}')"]
    if match_type != "All":
        where_match.append(f"match_type = '{match_type.strip().replace('\'', '\'\'')}'")

    sql = f"""
        SELECT 
            CASE WHEN team_1 = '{c_team}' THEN team_2 ELSE team_1 END as opponent,
            COUNT(*) as matches,
            SUM(CASE WHEN winner = '{c_team}' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN winner IS NOT NULL AND winner != '' AND winner != '{c_team}' THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN winner IS NULL OR winner = '' OR outcome_result = 'draw' THEN 1 ELSE 0 END) as ties_draws
        FROM dim_matches
        WHERE {" AND ".join(where_match)}
        GROUP BY opponent
        HAVING COUNT(*) >= 5
        ORDER BY matches DESC
        LIMIT {limit}
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty:
        return pd.DataFrame()

    df["matches"] = df["matches"].astype(int)
    df["wins"] = df["wins"].astype(int)
    df["losses"] = df["losses"].astype(int)
    df["win_pct"] = df.apply(
        lambda r: round((r["wins"] * 100.0) / r["matches"], 1) if r["matches"] > 0 else 0.0,
        axis=1,
    )
    return df


def get_team_phase_dna(team_name: str, match_type: str = "All") -> pd.DataFrame:
    """Team batting scoring trajectory across phases."""
    con = get_db_connection()
    c_team = team_name.strip().replace("'", "''")

    where_fact = [
        f"batting_team = '{c_team}'",
        "match_phase IN ('Powerplay', 'Middle', 'Death')",
    ]
    if match_type != "All":
        where_fact.append(f"match_type = '{match_type.strip().replace('\'', '\'\'')}'")

    sql = f"""
        SELECT 
            match_phase,
            SUM(total_runs) as runs,
            COUNT(*) as balls,
            ROUND((SUM(total_runs) * 6.0) / COUNT(*), 2) as run_rate,
            ROUND(SUM(is_boundary::INT) * 100.0 / COUNT(*), 1) as boundary_pct,
            ROUND(SUM(is_dot::INT) * 100.0 / COUNT(*), 1) as dot_pct
        FROM fact_deliveries_master
        WHERE {" AND ".join(where_fact)}
        GROUP BY match_phase
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty:
        return pd.DataFrame()

    phase_order = {"Powerplay": 1, "Middle": 2, "Death": 3}
    df["sort_order"] = df["match_phase"].map(phase_order).fillna(99)
    df = df.sort_values(by="sort_order").drop(columns=["sort_order"])
    return df


def get_team_fortress_venues(team_name: str, match_type: str = "All", limit: int = 8) -> pd.DataFrame:
    """Grounds with highest win percentage for this team."""
    con = get_db_connection()
    c_team = team_name.strip().replace("'", "''")

    where_match = [f"(team_1 = '{c_team}' OR team_2 = '{c_team}')"]
    if match_type != "All":
        where_match.append(f"match_type = '{match_type.strip().replace('\'', '\'\'')}'")

    sql = f"""
        SELECT 
            ground_name,
            COUNT(*) as matches,
            SUM(CASE WHEN winner = '{c_team}' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN winner != '{c_team}' AND winner IS NOT NULL AND winner != '' THEN 1 ELSE 0 END) as losses
        FROM dim_matches
        WHERE {" AND ".join(where_match)}
        GROUP BY ground_name
        HAVING COUNT(*) >= 5
        ORDER BY (SUM(CASE WHEN winner = '{c_team}' THEN 1 ELSE 0 END) * 1.0 / COUNT(*)) DESC, matches DESC
        LIMIT {limit}
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty:
        return pd.DataFrame()

    df["matches"] = df["matches"].astype(int)
    df["wins"] = df["wins"].astype(int)
    df["win_pct"] = df.apply(
        lambda r: round((r["wins"] * 100.0) / r["matches"], 1) if r["matches"] > 0 else 0.0,
        axis=1,
    )
    return df
