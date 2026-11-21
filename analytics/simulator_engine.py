"""
Pre-Game Strategy & Match Simulator Engine.

Provides deep analytical simulation:
1. 🏟️ Ground & Pitch Profiler:
   - Evaluates Pace vs Spin wicket share, bowling averages, and economy rates.
   - Evaluates Batting scoring rates in the chosen era (and prior era window).
   - Test Cricket: Computes 4 Innings progression averages (Inning 1, 2, 3, 4) and Outright vs Draw rate.
   - Limited Overs: Computes 1st & 2nd Innings totals, RPO, boundary %, and Bat 1st vs Chase outcomes.
   - Strict data-grounded proof without pre-meditated fluff text.

2. 🏏 Era-Aware Squad Filtering & Ideal Playing XI Selection:
   - Restricts player pool by Era (Current Active 2023–2026, 2019, 2015, 2011, All-Time).
   - Automatically excludes retired players outside the selected era.
   - Condition-adapted 11 (pitch pace/spin favorability).
   - Strict role balance (1 WK, 4-5 Batters, 1-2 All-Rounders, 3-4 Bowlers).

3. 🔮 Win Probability & Tactical Battle Dossier:
   - Actual historical Head-to-Head duel stats (runs, balls, wickets, strike rate, boundaries) between selected players.
   - Toss scenario modeling (Batting 1st vs Chasing).
   - Data-driven coaching directives.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import pandas as pd
from analytics.db import get_db_connection, DB_PATH

# Well-known international spinners for Pace vs Spin classification
KNOWN_SPINNERS = {
    # India
    "R Ashwin", "RA Jadeja", "YS Chahal", "Kuldeep Yadav", "A Mishra", "Harbhajan Singh",
    "A Kumble", "AR Patel", "Washington Sundar", "Varun Chakravarthy", "CV Varun",
    "Ravi Bishnoi", "PP Chawla", "RD Chahar", "S Gopal", "K Gowtham", "M Markande",
    # Australia
    "SK Warne", "NM Lyon", "A Zampa", "AC Agar", "GJ Maxwell", "T Murphy", "MJ Kuhnemann",
    "CL White", "XJ Doherty", "BE McGain", "GB Hogg", "S O'Keefe", "SNJ O'Keefe",
    # England
    "Adil Rashid", "AU Rashid", "MM Ali", "GP Swann", "MS Panesar", "Rehan Ahmed",
    "Shoaib Bashir", "TW Hartley", "LS Livingstone", "JE Root", "DA Briggs",
    # Pakistan
    "Saeed Ajmal", "Shahid Afridi", "Yasir Shah", "Shadab Khan", "Mohammad Nawaz",
    "Usama Mir", "Abrar Ahmed", "Imad Wasim", "Mushtaq Ahmed", "Saqlain Mushtaq",
    "Danish Kaneria", "Zulfiqar Babar", "Abdur Rehman",
    # Sri Lanka
    "M Muralitharan", "HMRKB Herath", "BAW Mendis", "PWH de Silva", "Wanindu Hasaranga",
    "M Theekshana", "DN Wellalage", "PVD Chameera", "PADLR Sandakan", "MDK Perera",
    # South Africa
    "KA Maharaj", "Imran Tahir", "Tabraiz Shamsi", "PL Harris", "RE van der Merwe",
    "BJ Fortuin", "G Linde", "SR Harmer", "N Boje",
    # New Zealand
    "DL Vettori", "MJ Santner", "IS Sodhi", "AY Patel", "GP Phillips", "C Mark Chapman",
    "JS Patel", "NT Broom", "BV Cairns",
    # West Indies
    "SP Narine", "Sunil Narine", "SJ Benn", "Devendra Bishoo", "Gudakesh Motie",
    "Akeal Hosein", "RA Reifer", "CH Gayle", "RR Sarwan", "NO Miller", "K Pierre",
    # Afghanistan & Bangladesh
    "Rashid Khan", "Mujeeb Ur Rahman", "Mohammad Nabi", "Noor Ahmad", "Qais Ahmad",
    "Shakib Al Hasan", "Mehidy Hasan Miraz", "Taijul Islam", "Nasum Ahmed", "Enamul Haque jnr",
    "Abdur Razzak", "Sohag Gazi"
}

# Known international wicketkeepers
KNOWN_WICKETKEEPERS = {
    "MS Dhoni", "Rishabh Pant", "KL Rahul", "SV Samson", "Ishan Kishan", "KD Karthik",
    "PA Patel", "WP Saha", "Jitesh Sharma", "Dhruv Jurel", "KS Bharat",
    "AC Gilchrist", "BJ Haddin", "TD Paine", "AT Carey", "MS Wade", "JP Inglis",
    "Jos Buttler", "JC Buttler", "JM Bairstow", "FO Bairstow", "SW Billings", "PD Collingwood",
    "MJ Prior", "GO Jones", "WK Hegg", "James Foster", "Sam Billings",
    "Q de Kock", "AB de Villiers", "MV Boucher", "H Klaasen", "Kyle Verreynne",
    "Mohammad Rizwan", "Sarfaraz Ahmed", "Kamran Akmal", "Umar Akmal", "Adnan Akmal",
    "KC Sangakkara", "RS Kaluwitharana", "Niroshan Dickwella", "Kusal Perera", "S Samarawickrama",
    "BB McCullum", "BJ Watling", "Tom Blundell", "Luke Ronchi", "Glenn Phillips", "Devon Conway",
    "D Ramdin", "RD Jacobs", "Nicholas Pooran", "Shai Hope", "Joshua Da Silva",
    "Mushfiqur Rahim", "Litton Das", "Nurul Hasan", "Anamul Haque",
    "Rahmanullah Gurbaz", "Ikram Alikhil", "Mohammad Shahzad"
}


def get_all_popular_teams() -> List[str]:
    """Returns primary international cricket teams."""
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


def get_all_popular_venues() -> List[str]:
    """Returns top venues hosted sorted by match count."""
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


def _get_venue_era_clause(era: str) -> str:
    """Returns date bounds for venue stats relevant to the chosen era and preceding window."""
    if era == "current":
        return "match_date >= '2020-01-01'"
    elif era == "2019_era":
        return "match_date BETWEEN '2015-01-01' AND '2019-12-31'"
    elif era == "2015_era":
        return "match_date BETWEEN '2011-01-01' AND '2015-12-31'"
    elif era == "2011_era":
        return "match_date BETWEEN '2007-01-01' AND '2011-12-31'"
    else:
        return "1=1"


# ══════════════════════════════════════════════════════════════════════════════
# 1. VENUE PITCH & CONDITIONS PROFILER (ERA-AWARE & FORMAT-STRICT)
# ══════════════════════════════════════════════════════════════════════════════

def get_venue_pitch_profile(venue_name: str, match_type: str = "T20I", era: str = "current") -> Dict[str, Any]:
    """
    Analyzes venue pitch behavior filtered to the selected era:
    - In Test: 4 Innings progression averages, Result vs Draw %, Pace vs Spin wickets.
    - In Limited Overs: 1st & 2nd Innings averages, RPO, Boundary %, Bat 1st vs Chase %.
    - Data proof backing all classifications without pre-meditated fluff.
    """
    con = get_db_connection()
    c_venue = venue_name.strip().replace("'", "''")
    c_fmt = match_type.strip().replace("'", "''")
    era_date = _get_venue_era_clause(era)

    # Check match sample in this era window
    sql_check = f"""
        SELECT COUNT(*)
        FROM dim_matches
        WHERE ground_name = '{c_venue}' AND match_type = '{c_fmt}' AND {era_date}
    """
    cnt_m = con.execute(sql_check).fetchone()[0]
    # If very few matches in that narrow window, relax to all-time so ground data is never blank
    if cnt_m < 2:
        era_date = "1=1"

    where_fact = [f"ground_name = '{c_venue}'", f"match_type = '{c_fmt}'", era_date]
    where_match = [f"ground_name = '{c_venue}'", f"match_type = '{c_fmt}'", era_date]

    is_test = (match_type == "Test")

    # Match outcomes (Toss & Bat 1st vs Chase)
    sql_outcomes = f"""
        SELECT 
            COUNT(*) as total_matches,
            SUM(CASE WHEN win_margin_runs > 0 THEN 1 ELSE 0 END) as bat_1st_wins,
            SUM(CASE WHEN win_margin_wkts > 0 THEN 1 ELSE 0 END) as chase_wins,
            SUM(CASE WHEN winner IS NULL OR winner = '' OR outcome_result = 'draw' THEN 1 ELSE 0 END) as draws
        FROM dim_matches
        WHERE {" AND ".join(where_match)}
    """
    row_m = con.execute(sql_outcomes).fetchone()
    total_m = row_m[0] if row_m else 0
    b1_wins = row_m[1] or 0 if row_m else 0
    ch_wins = row_m[2] or 0 if row_m else 0
    draws = row_m[3] or 0 if row_m else 0

    decided = b1_wins + ch_wins
    bat_1st_pct = round((b1_wins * 100.0) / decided, 1) if decided > 0 else 50.0
    chase_pct = round((ch_wins * 100.0) / decided, 1) if decided > 0 else 50.0
    decided_pct = round((decided * 100.0) / total_m, 1) if total_m > 0 else 80.0
    draw_pct = round((draws * 100.0) / total_m, 1) if total_m > 0 else 20.0

    # Deliveries statistics (Run rates, boundary %, dots)
    sql_deliveries = f"""
        SELECT 
            COUNT(*) as total_balls,
            SUM(total_runs) as total_runs,
            SUM(is_boundary::INT) as boundaries,
            SUM(is_dot::INT) as dots
        FROM fact_deliveries_master
        WHERE {" AND ".join(where_fact)}
    """
    row_d = con.execute(sql_deliveries).fetchone()
    balls = row_d[0] if row_d else 0
    runs = row_d[1] or 0 if row_d else 0
    bdry = row_d[2] or 0 if row_d else 0
    dots = row_d[3] or 0 if row_d else 0

    run_rate = round((runs * 6.0) / balls, 2) if balls > 0 else (3.10 if is_test else 7.50)
    bdry_pct = round((bdry * 100.0) / balls, 1) if balls > 0 else 12.0
    dot_pct = round((dots * 100.0) / balls, 1) if balls > 0 else 45.0

    # 4 Innings scores for Test vs 1st/2nd Innings for Limited Overs
    avg_1st = avg_2nd = avg_3rd = avg_4th = 0
    if is_test:
        sql_test_inn = f"""
            SELECT 
                inning,
                ROUND(AVG(inn_runs), 0) as avg_score,
                COUNT(DISTINCT match_id) as sample_size
            FROM (
                SELECT match_id, inning, SUM(total_runs) as inn_runs
                FROM fact_deliveries_master
                WHERE {" AND ".join(where_fact)} AND inning IN (1, 2, 3, 4)
                GROUP BY match_id, inning
            )
            GROUP BY inning
            ORDER BY inning
        """
        df_test_inn = con.execute(sql_test_inn).df()
        for _, r in df_test_inn.iterrows():
            inn = int(r["inning"])
            sc = int(r["avg_score"] or 0)
            if inn == 1: avg_1st = sc
            elif inn == 2: avg_2nd = sc
            elif inn == 3: avg_3rd = sc
            elif inn == 4: avg_4th = sc

        if avg_1st == 0: avg_1st = 320
        if avg_2nd == 0: avg_2nd = 290
        if avg_3rd == 0: avg_3rd = 240
        if avg_4th == 0: avg_4th = 180
        par_1st_inn = avg_1st
    else:
        sql_lo_inn = f"""
            SELECT 
                inning,
                ROUND(AVG(inn_runs), 0) as avg_score
            FROM (
                SELECT match_id, inning, SUM(total_runs) as inn_runs
                FROM fact_deliveries_master
                WHERE {" AND ".join(where_fact)} AND inning IN (1, 2)
                GROUP BY match_id, inning
                HAVING COUNT(*) >= 50
            )
            GROUP BY inning
        """
        df_lo_inn = con.execute(sql_lo_inn).df()
        for _, r in df_lo_inn.iterrows():
            inn = int(r["inning"])
            sc = int(r["avg_score"] or 0)
            if inn == 1: avg_1st = sc
            elif inn == 2: avg_2nd = sc

        if avg_1st == 0:
            avg_1st = 160 if match_type == "T20I" else 250
        if avg_2nd == 0:
            avg_2nd = 145 if match_type == "T20I" else 225
        par_1st_inn = avg_1st

    # Pace vs Spin Wickets and Averages
    sql_bowlers = f"""
        SELECT 
            bowler,
            COUNT(CASE WHEN is_wicket AND dismissal_kind IN ('bowled', 'caught', 'lbw', 'stumped', 'caught and bowled', 'hit wicket') THEN 1 END) as wickets,
            SUM(runs_batter + extra_wides + extra_noballs) as runs_conceded,
            SUM(CASE WHEN extra_wides = 0 AND extra_noballs = 0 THEN 1 ELSE 0 END) as legal_balls
        FROM fact_deliveries_master
        WHERE {" AND ".join(where_fact)}
        GROUP BY bowler
        HAVING COUNT(*) >= 20
    """
    df_bowlers = con.execute(sql_bowlers).df()
    con.close()

    spin_wkts = 0
    pace_wkts = 0
    spin_runs = 0
    pace_runs = 0
    spin_balls = 0
    pace_balls = 0

    if not df_bowlers.empty:
        for _, r in df_bowlers.iterrows():
            b_name = str(r["bowler"])
            w = int(r["wickets"] or 0)
            rc = int(r["runs_conceded"] or 0)
            lb = int(r["legal_balls"] or 0)
            if b_name in KNOWN_SPINNERS:
                spin_wkts += w
                spin_runs += rc
                spin_balls += lb
            else:
                pace_wkts += w
                pace_runs += rc
                pace_balls += lb

    total_classified = spin_wkts + pace_wkts
    if total_classified > 0:
        spin_wkt_pct = round((spin_wkts * 100.0) / total_classified, 1)
        pace_wkt_pct = round((pace_wkts * 100.0) / total_classified, 1)
    else:
        spin_wkt_pct = 32.0
        pace_wkt_pct = 68.0

    spin_avg = round(spin_runs / spin_wkts, 1) if spin_wkts > 0 else 32.0
    pace_avg = round(pace_runs / pace_wkts, 1) if pace_wkts > 0 else 28.0
    spin_econ = round((spin_runs * 6.0) / spin_balls, 2) if spin_balls > 0 else 7.20
    pace_econ = round((pace_runs * 6.0) / pace_balls, 2) if pace_balls > 0 else 7.80

    # Objective Classification with Data Proof
    if is_test:
        decay_ratio = round((avg_4th / avg_1st) * 100.0, 1) if avg_1st > 0 else 60.0
        if decay_ratio <= 65.0:
            bat_profile = "🏛️ Rapidly Deteriorating Surface"
            bat_proof = f"Inn 1: {avg_1st}r ➔ Inn 4: {avg_4th}r ({decay_ratio}% retention). Result rate: {decided_pct}%."
        else:
            bat_profile = "🏛️ Consistent Multi-Day Surface"
            bat_proof = f"Inn 1: {avg_1st}r ➔ Inn 4: {avg_4th}r. Draw rate: {draw_pct}%, Result rate: {decided_pct}%."
    else:
        if (match_type == "T20I" and run_rate >= 8.5) or (match_type == "ODI" and run_rate >= 5.8):
            bat_profile = "🏏 High-Scoring Batting Track"
            bat_proof = f"Par 1st Inn: {par_1st_inn} runs · {run_rate} RPO · {bdry_pct}% boundaries · {dot_pct}% dots."
        elif (match_type == "T20I" and run_rate <= 7.3) or (match_type == "ODI" and run_rate <= 4.9):
            bat_profile = "🔥 Restrictive Bowling Track"
            bat_proof = f"Par 1st Inn: {par_1st_inn} runs · {run_rate} RPO · {dot_pct}% dots · {bdry_pct}% boundaries."
        else:
            bat_profile = "⚖️ Balanced Scoring Track"
            bat_proof = f"Par 1st Inn: {par_1st_inn} runs · {run_rate} RPO · {bdry_pct}% boundaries · {dot_pct}% dots."

    # Bowling Proof
    if spin_wkt_pct >= 38.0:
        bowl_profile = "🌪️ Spin-Assisting Pitch"
        bowl_proof = f"Spin: {spin_wkts} wkts ({spin_wkt_pct}%, {spin_avg} avg) vs Pace: {pace_wkts} wkts ({pace_wkt_pct}%, {pace_avg} avg)."
        preferred_spinners = 3
        preferred_pacers = 2
    elif pace_wkt_pct >= 76.0:
        bowl_profile = "⚡ Fast & Bouncy Pace Pitch"
        bowl_proof = f"Pace: {pace_wkts} wkts ({pace_wkt_pct}%, {pace_avg} avg) vs Spin: {spin_wkts} wkts ({spin_wkt_pct}%, {spin_avg} avg)."
        preferred_spinners = 1
        preferred_pacers = 4
    else:
        bowl_profile = "⚖️ Balanced Bowling Surface"
        bowl_proof = f"Pace: {pace_wkts} wkts ({pace_wkt_pct}%, {pace_avg} avg) · Spin: {spin_wkts} wkts ({spin_wkt_pct}%, {spin_avg} avg)."
        preferred_spinners = 2
        preferred_pacers = 3

    # Toss decision
    if is_test:
        toss_rec = "🏏 Bat First (Pitch Deterioration)"
        toss_proof = f"1st Innings avg is {avg_1st} runs vs 4th Innings {avg_4th} runs ({decided_pct}% decided matches)."
    else:
        if bat_1st_pct >= 56.0:
            toss_rec = "🏏 Bat First (Defending Favored)"
            toss_proof = f"Defending teams win {bat_1st_pct}% vs Chasing {chase_pct}% ({b1_wins} vs {ch_wins} matches)."
        elif chase_pct >= 56.0:
            toss_rec = "🎯 Bowl First (Chasing Favored)"
            toss_proof = f"Chasing teams win {chase_pct}% vs Defending {bat_1st_pct}% ({ch_wins} vs {b1_wins} matches)."
        else:
            toss_rec = "⚖️ Balanced Toss Options"
            toss_proof = f"Even split: {bat_1st_pct}% Bat 1st wins vs {chase_pct}% Chase wins ({total_m} matches)."

    return {
        "venue_name": venue_name,
        "format": match_type,
        "era": era,
        "is_test": is_test,
        "total_matches": total_m,
        "bat_1st_win_pct": bat_1st_pct,
        "chase_win_pct": chase_pct,
        "decided_pct": decided_pct,
        "draw_pct": draw_pct,
        "par_1st_innings": par_1st_inn,
        "avg_1st_innings": avg_1st,
        "avg_2nd_innings": avg_2nd,
        "avg_3rd_innings": avg_3rd,
        "avg_4th_innings": avg_4th,
        "run_rate": run_rate,
        "boundary_pct": bdry_pct,
        "dot_ball_pct": dot_pct,
        "spin_wkts": spin_wkts,
        "pace_wkts": pace_wkts,
        "spin_wkt_pct": spin_wkt_pct,
        "pace_wkt_pct": pace_wkt_pct,
        "spin_avg": spin_avg,
        "pace_avg": pace_avg,
        "spin_econ": spin_econ,
        "pace_econ": pace_econ,
        "bat_profile": bat_profile,
        "bat_proof": bat_proof,
        "bowl_profile": bowl_profile,
        "bowl_proof": bowl_proof,
        "preferred_spinners": preferred_spinners,
        "preferred_pacers": preferred_pacers,
        "toss_rec": toss_rec,
        "toss_proof": toss_proof,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. SQUAD POOL & IDEAL PLAYING XI SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def _get_player_era_clause(era: str) -> str:
    """Returns SQL date bounds for player era filtering."""
    if era == "current":
        return "match_date >= '2023-01-01'"
    elif era == "2019_era":
        return "match_date BETWEEN '2017-01-01' AND '2019-12-31'"
    elif era == "2015_era":
        return "match_date BETWEEN '2013-01-01' AND '2015-12-31'"
    elif era == "2011_era":
        return "match_date BETWEEN '2009-01-01' AND '2011-12-31'"
    elif era == "all_time":
        return "1=1"
    else:
        return "match_date >= '2023-01-01'"


def get_team_player_pool(team_name: str, match_type: str = "T20I", era: str = "current") -> List[Dict[str, Any]]:
    """Extracts eligible players for a team within the chosen era (excluding retirees)."""
    con = get_db_connection()
    c_team = team_name.strip().replace("'", "''")
    date_clause = _get_player_era_clause(era)

    where_fmt = ""
    if match_type != "All":
        where_fmt = f"AND f.match_type = '{match_type.strip().replace('\'', '\'\'')}'"

    sql_bat = f"""
        SELECT 
            f.batter as player_name,
            COUNT(DISTINCT f.match_id) as era_matches,
            SUM(f.runs_batter) as runs,
            COUNT(f.ball) as balls,
            COUNT(CASE WHEN f.is_wicket THEN 1 END) as dismissals,
            SUM(f.is_four::INT) as fours,
            SUM(f.is_six::INT) as sixes
        FROM fact_deliveries_master f
        JOIN dim_matches m ON f.match_id = m.match_id
        WHERE f.batting_team = '{c_team}'
          {where_fmt}
          AND {date_clause.replace('match_date', 'f.match_date')}
          AND (m.tournament IS NULL OR m.tournament NOT ILIKE '%Women%')
        GROUP BY f.batter
        HAVING COUNT(DISTINCT f.match_id) >= 2
    """
    df_bat = con.execute(sql_bat).df()

    sql_bowl = f"""
        SELECT 
            f.bowler as player_name,
            COUNT(DISTINCT f.match_id) as era_matches,
            COUNT(CASE WHEN f.is_wicket AND f.dismissal_kind IN ('bowled', 'caught', 'lbw', 'stumped', 'caught and bowled', 'hit wicket') THEN 1 END) as wickets,
            SUM(runs_batter + extra_wides + extra_noballs) as runs_conceded,
            SUM(CASE WHEN extra_wides = 0 AND extra_noballs = 0 THEN 1 ELSE 0 END) as legal_balls
        FROM fact_deliveries_master f
        JOIN dim_matches m ON f.match_id = m.match_id
        WHERE f.bowling_team = '{c_team}'
          {where_fmt}
          AND {date_clause.replace('match_date', 'f.match_date')}
          AND (m.tournament IS NULL OR m.tournament NOT ILIKE '%Women%')
        GROUP BY f.bowler
        HAVING COUNT(DISTINCT f.match_id) >= 2
    """
    df_bowl = con.execute(sql_bowl).df()

    sql_dim = f"""
        SELECT player_name, primary_role, median_batting_position
        FROM dim_players
    """
    df_dim = con.execute(sql_dim).df()
    con.close()

    dim_dict = {r["player_name"]: (r["primary_role"], r["median_batting_position"]) for _, r in df_dim.iterrows()}

    players_dict = {}

    for _, r in df_bat.iterrows():
        p = str(r["player_name"])
        m = int(r["era_matches"])
        runs = int(r["runs"] or 0)
        balls = int(r["balls"] or 0)
        d = int(r["dismissals"] or 0)
        fours = int(r["fours"] or 0)
        sixes = int(r["sixes"] or 0)
        avg = round(runs / d, 1) if d > 0 else float(runs)
        sr = round((runs * 100.0) / balls, 1) if balls > 0 else 0.0

        players_dict[p] = {
            "player_name": p,
            "matches": m,
            "runs": runs,
            "balls": balls,
            "batting_avg": avg,
            "strike_rate": sr,
            "fours": fours,
            "sixes": sixes,
            "wickets": 0,
            "bowling_avg": 99.0,
            "economy": 12.0,
            "balls_bowled": 0,
        }

    for _, r in df_bowl.iterrows():
        p = str(r["player_name"])
        m = int(r["era_matches"])
        w = int(r["wickets"] or 0)
        rc = int(r["runs_conceded"] or 0)
        lb = int(r["legal_balls"] or 0)
        b_avg = round(rc / w, 1) if w > 0 else 99.0
        econ = round((rc * 6.0) / lb, 2) if lb > 0 else 12.0

        if p in players_dict:
            players_dict[p]["matches"] = max(players_dict[p]["matches"], m)
            players_dict[p]["wickets"] = w
            players_dict[p]["bowling_avg"] = b_avg
            players_dict[p]["economy"] = econ
            players_dict[p]["balls_bowled"] = lb
        else:
            players_dict[p] = {
                "player_name": p,
                "matches": m,
                "runs": 0,
                "balls": 0,
                "batting_avg": 0.0,
                "strike_rate": 0.0,
                "fours": 0,
                "sixes": 0,
                "wickets": w,
                "bowling_avg": b_avg,
                "economy": econ,
                "balls_bowled": lb,
            }

    player_pool = []
    for p, stats in players_dict.items():
        dim_info = dim_dict.get(p, ("Developing / Debutant", 6.0))
        dim_role = dim_info[0]
        med_pos = dim_info[1] if dim_info[1] else 6.0

        is_wk = p in KNOWN_WICKETKEEPERS
        is_spin = p in KNOWN_SPINNERS

        if is_wk:
            role_type = "WK-Batsman"
        elif dim_role == "All-Rounder" or (stats["runs"] >= 150 and stats["wickets"] >= 10):
            role_type = "All-Rounder"
        elif dim_role == "Bowler" or (stats["wickets"] >= 15 and stats["runs"] < 250):
            role_type = "Spinner" if is_spin else "Pacer"
        else:
            role_type = "Top-Order Batter" if med_pos <= 3.5 else "Middle-Order Batter"

        stats["role"] = role_type
        stats["is_spin"] = is_spin
        stats["is_wk"] = is_wk
        stats["median_pos"] = med_pos
        player_pool.append(stats)

    return player_pool


def select_ideal_xi(
    team_name: str,
    opponent_name: str,
    venue_name: str,
    match_type: str = "T20I",
    era: str = "current"
) -> List[Dict[str, Any]]:
    """Selects the condition-smart Playing XI."""
    pool = get_team_player_pool(team_name, match_type=match_type, era=era)
    if not pool:
        pool = get_team_player_pool(team_name, match_type=match_type, era="all_time")

    if not pool:
        return []

    pitch = get_venue_pitch_profile(venue_name, match_type=match_type, era=era)
    pref_spinners = pitch["preferred_spinners"]
    pref_pacers = pitch["preferred_pacers"]

    for p in pool:
        role = p["role"]
        score = 0.0
        runs = p["runs"]
        avg = p["batting_avg"]
        sr = p["strike_rate"]
        wkts = p["wickets"]
        econ = p["economy"]

        if "Batter" in role or role == "WK-Batsman":
            score += min(runs / 10.0, 150.0) + (avg * 2.5) + (sr * 0.5)
        elif role == "All-Rounder":
            score += min(runs / 15.0, 100.0) + (avg * 1.8) + (sr * 0.3) + (wkts * 8.0) + max(0, (10 - econ) * 8.0)
        else:
            score += (wkts * 12.0) + max(0, (30 - p["bowling_avg"]) * 2.0) + max(0, (10 - econ) * 10.0)

        if p["is_spin"]:
            score *= (1.0 + (pitch["spin_wkt_pct"] / 100.0) * 0.4)
        else:
            score *= (1.0 + (pitch["pace_wkt_pct"] / 100.0) * 0.2)

        p["selection_score"] = round(score, 1)

    wks = sorted([p for p in pool if p["is_wk"]], key=lambda x: x["selection_score"], reverse=True)
    top_bat = sorted([p for p in pool if p["role"] == "Top-Order Batter"], key=lambda x: x["selection_score"], reverse=True)
    mid_bat = sorted([p for p in pool if p["role"] == "Middle-Order Batter"], key=lambda x: x["selection_score"], reverse=True)
    all_rounders = sorted([p for p in pool if p["role"] == "All-Rounder"], key=lambda x: x["selection_score"], reverse=True)
    spinners = sorted([p for p in pool if p["role"] == "Spinner"], key=lambda x: x["selection_score"], reverse=True)
    pacers = sorted([p for p in pool if p["role"] == "Pacer"], key=lambda x: x["selection_score"], reverse=True)

    selected: List[Dict[str, Any]] = []

    # 1 WK
    if wks:
        selected.append(wks[0])
    elif top_bat or mid_bat:
        candidate = (top_bat + mid_bat)[0]
        candidate["role"] = "WK-Batsman"
        selected.append(candidate)

    def add_player(p_list, limit):
        added = 0
        for pl in p_list:
            if added >= limit:
                break
            if not any(s["player_name"] == pl["player_name"] for s in selected):
                selected.append(pl)
                added += 1

    add_player(top_bat, 3)
    add_player(mid_bat, 2)
    add_player(all_rounders, 2)
    add_player(spinners, pref_spinners)
    add_player(pacers, pref_pacers)

    remaining = sorted(
        [p for p in pool if not any(s["player_name"] == p["player_name"] for s in selected)],
        key=lambda x: x["selection_score"],
        reverse=True
    )
    for p in remaining:
        if len(selected) >= 11:
            break
        selected.append(p)

    selected = selected[:11]

    def batting_sort_key(p):
        role = p["role"]
        if role == "Top-Order Batter": return p.get("median_pos", 1.0)
        elif role == "WK-Batsman": return p.get("median_pos", 4.0)
        elif role == "Middle-Order Batter": return p.get("median_pos", 5.0)
        elif role == "All-Rounder": return 7.0
        elif role == "Spinner": return 9.0
        else: return 10.0

    selected = sorted(selected, key=batting_sort_key)

    captain_candidate = max(selected, key=lambda x: x["matches"])
    for i, p in enumerate(selected, 1):
        p["batting_order"] = i
        p["is_captain"] = (p["player_name"] == captain_candidate["player_name"])

    return selected


# ══════════════════════════════════════════════════════════════════════════════
# 3. REAL HEAD-TO-HEAD DUELS & MATCH SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def get_h2h_duel(batter: str, bowler: str) -> Dict[str, Any]:
    """Retrieves actual head-to-head encounter records between batter and bowler."""
    con = get_db_connection()
    c_bat = batter.strip().replace("'", "''")
    c_bowl = bowler.strip().replace("'", "''")

    sql = f"""
        SELECT 
            COUNT(ball) as balls,
            SUM(runs_batter) as runs,
            COUNT(CASE WHEN is_wicket AND dismissal_kind IN ('bowled', 'caught', 'lbw', 'stumped', 'caught and bowled', 'hit wicket') THEN 1 END) as dismissals,
            SUM(is_four::INT) as fours,
            SUM(is_six::INT) as sixes,
            SUM(is_dot::INT) as dots
        FROM fact_deliveries_master
        WHERE batter = '{c_bat}' AND bowler = '{c_bowl}'
    """
    row = con.execute(sql).fetchone()
    con.close()

    if not row or not row[0]:
        return {
            "batter": batter,
            "bowler": bowler,
            "balls": 0,
            "runs": 0,
            "dismissals": 0,
            "strike_rate": 0.0,
            "fours": 0,
            "sixes": 0,
            "dots": 0,
            "has_data": False,
            "summary": "0 balls faced in international cricket",
        }

    balls = int(row[0])
    runs = int(row[1] or 0)
    dismissals = int(row[2] or 0)
    fours = int(row[3] or 0)
    sixes = int(row[4] or 0)
    dots = int(row[5] or 0)
    sr = round((runs * 100.0) / balls, 1) if balls > 0 else 0.0

    return {
        "batter": batter,
        "bowler": bowler,
        "balls": balls,
        "runs": runs,
        "dismissals": dismissals,
        "strike_rate": sr,
        "fours": fours,
        "sixes": sixes,
        "dots": dots,
        "has_data": True,
        "summary": f"{runs} runs off {balls} balls · {dismissals} wickets · SR {sr} ({fours} fours, {sixes} sixes, {dots} dot balls)",
    }


def simulate_matchup(
    team1: str,
    team2: str,
    team1_xi: List[Dict[str, Any]],
    team2_xi: List[Dict[str, Any]],
    venue_name: str,
    match_type: str = "T20I",
    era: str = "current"
) -> Dict[str, Any]:
    """Computes win probabilities, projected scores, and factual player duels."""
    con = get_db_connection()
    c_t1 = team1.strip().replace("'", "''")
    c_t2 = team2.strip().replace("'", "''")
    c_fmt = match_type.strip().replace("'", "''")

    where_h2h = [
        f"((team_1 = '{c_t1}' AND team_2 = '{c_t2}') OR (team_1 = '{c_t2}' AND team_2 = '{c_t1}'))",
    ]
    if match_type != "All":
        where_h2h.append(f"match_type = '{c_fmt}'")

    sql_h2h = f"""
        SELECT 
            COUNT(*) as matches,
            SUM(CASE WHEN winner = '{c_t1}' THEN 1 ELSE 0 END) as t1_wins,
            SUM(CASE WHEN winner = '{c_t2}' THEN 1 ELSE 0 END) as t2_wins
        FROM dim_matches
        WHERE {" AND ".join(where_h2h)}
    """
    row_h = con.execute(sql_h2h).fetchone()
    con.close()

    h2h_matches = row_h[0] if row_h else 0
    t1_h2h_wins = row_h[1] or 0 if row_h else 0
    t2_h2h_wins = row_h[2] or 0 if row_h else 0

    if h2h_matches >= 3 and (t1_h2h_wins + t2_h2h_wins) > 0:
        base_p1 = (t1_h2h_wins * 100.0) / (t1_h2h_wins + t2_h2h_wins)
    else:
        base_p1 = 50.0

    pitch = get_venue_pitch_profile(venue_name, match_type=match_type, era=era)
    par_score = pitch["par_1st_innings"]

    t1_bat_score = sum(p["runs"] for p in team1_xi[:6]) / 6.0
    t2_bat_score = sum(p["runs"] for p in team2_xi[:6]) / 6.0

    t1_bowl_score = sum(p["wickets"] for p in team1_xi if p["role"] in ("Pacer", "Spinner", "All-Rounder"))
    t2_bowl_score = sum(p["wickets"] for p in team2_xi if p["role"] in ("Pacer", "Spinner", "All-Rounder"))

    xi_diff = (t1_bat_score - t2_bat_score) * 0.01 + (t1_bowl_score - t2_bowl_score) * 0.05
    xi_diff = max(min(xi_diff, 12.0), -12.0)

    p1_win = round(max(min(base_p1 * 0.70 + 50.0 * 0.15 + xi_diff, 85.0), 15.0), 1)
    p2_win = round(100.0 - p1_win, 1)

    bat_bias = pitch["bat_1st_win_pct"] - 50.0
    t1_bat_first_win = round(max(min(p1_win + (bat_bias * 0.6), 88.0), 12.0), 1)
    t1_chase_win = round(max(min(p1_win - (bat_bias * 0.6), 88.0), 12.0), 1)

    t2_bat_first_win = round(max(min(p2_win + (bat_bias * 0.6), 88.0), 12.0), 1)
    t2_chase_win = round(max(min(p2_win - (bat_bias * 0.6), 88.0), 12.0), 1)

    t1_proj_runs = int(par_score + (t1_bat_score - t2_bowl_score * 0.5) * 0.02)
    t2_proj_runs = int(par_score + (t2_bat_score - t1_bowl_score * 0.5) * 0.02)

    # Real Head-to-Head Tactical Battles (Factual Duel Numbers)
    tactical_battles = []
    batters_t1 = [p["player_name"] for p in team1_xi if "Batter" in p["role"] or p["is_wk"]][:3]
    bowlers_t2 = [p["player_name"] for p in team2_xi if p["role"] in ("Pacer", "Spinner")][:3]

    batters_t2 = [p["player_name"] for p in team2_xi if "Batter" in p["role"] or p["is_wk"]][:3]
    bowlers_t1 = [p["player_name"] for p in team1_xi if p["role"] in ("Pacer", "Spinner")][:3]

    # Find highest volume duels between the selected 11s
    for b in batters_t1:
        for bw in bowlers_t2:
            duel = get_h2h_duel(b, bw)
            if duel["balls"] >= 15:
                duel["batter_team"] = team1
                duel["bowler_team"] = team2
                tactical_battles.append(duel)
                break
        if len(tactical_battles) >= 2:
            break

    for b in batters_t2:
        for bw in bowlers_t1:
            duel = get_h2h_duel(b, bw)
            if duel["balls"] >= 15:
                duel["batter_team"] = team2
                duel["bowler_team"] = team1
                tactical_battles.append(duel)
                break
        if len(tactical_battles) >= 4:
            break

    # If fewer than 2 had >=15 balls, pair the opening batsmen with opening bowlers directly
    if len(tactical_battles) < 2:
        if batters_t1 and bowlers_t2:
            d1 = get_h2h_duel(batters_t1[0], bowlers_t2[0])
            d1["batter_team"] = team1
            d1["bowler_team"] = team2
            if not any(x["batter"] == d1["batter"] and x["bowler"] == d1["bowler"] for x in tactical_battles):
                tactical_battles.append(d1)
        if batters_t2 and bowlers_t1:
            d2 = get_h2h_duel(batters_t2[0], bowlers_t1[0])
            d2["batter_team"] = team2
            d2["bowler_team"] = team1
            if not any(x["batter"] == d2["batter"] and x["bowler"] == d2["bowler"] for x in tactical_battles):
                tactical_battles.append(d2)

    # Strategic Directives based strictly on data proof
    if pitch["is_test"]:
        directives_t1 = [
            f"1st Innings Par Benchmark: Construct a minimum total of {pitch['avg_1st_innings']} runs to establish a dominant lead.",
            f"Pitch Deterioration: 4th Innings target average drops to {pitch['avg_4th_innings']} runs ({pitch['decided_pct']}% result rate).",
            f"Bowling Directive: Capitalize on {pitch['bowl_proof']}.",
        ]
        directives_t2 = [
            f"1st Innings Benchmark: Match or exceed {pitch['avg_1st_innings']} runs to avoid batting last on a deteriorating pitch.",
            f"Bowling Execution: Fast bowlers and spinners should target the channel reflecting {pitch['bowl_proof']}.",
            f"Toss Priority: {pitch['toss_rec']}.",
        ]
    else:
        directives_t1 = [
            f"Target Total: Construct a score of {t1_proj_runs} runs on this {pitch['bat_profile']}.",
            f"Bowling Directive: Exploit {pitch['bowl_proof']}.",
            f"Toss Advantage: {pitch['toss_rec']} ({pitch['toss_proof']}).",
        ]
        directives_t2 = [
            f"Target Restriction: Restrict {team1} within {t2_proj_runs} runs.",
            f"Bowling Directive: Exploit {pitch['bowl_proof']}.",
            f"Toss Advantage: {pitch['toss_rec']} ({pitch['toss_proof']}).",
        ]

    return {
        "team1": team1,
        "team2": team2,
        "format": match_type,
        "venue": venue_name,
        "era": era,
        "h2h_matches": h2h_matches,
        "t1_h2h_wins": t1_h2h_wins,
        "t2_h2h_wins": t2_h2h_wins,
        "t1_win_pct": p1_win,
        "t2_win_pct": p2_win,
        "t1_bat_first_win": t1_bat_first_win,
        "t1_chase_win": t1_chase_win,
        "t2_bat_first_win": t2_bat_first_win,
        "t2_chase_win": t2_chase_win,
        "t1_proj_runs": t1_proj_runs,
        "t2_proj_runs": t2_proj_runs,
        "par_score": par_score,
        "pitch": pitch,
        "tactical_battles": tactical_battles,
        "directives_t1": directives_t1,
        "directives_t2": directives_t2,
    }
