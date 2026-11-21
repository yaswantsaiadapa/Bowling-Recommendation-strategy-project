"""
Cricsheet JSON Parser & Natural Key Generator.

Reads bulk Cricsheet zip archives directly in memory (zero disk unzipping overhead),
unrolls the nested match/inning/over/delivery tree into two clean relational tables
(dim_matches and fact_deliveries_base), and constructs deterministic Natural Keys
for seamless joining with telemetry and ground reference datasets.
"""

import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import duckdb

from data_pipeline.config import (
    FORMAT_MAP,
    PROCESSED_DIR,
    RAW_DIR,
    get_match_phase,
    logger,
)


def sanitize_token(text: str) -> str:
    """Sanitizes team/venue names for reliable natural key construction."""
    if not text:
        return "Unknown"
    # Replace whitespace and non-alphanumeric chars with an underscore
    clean = re.sub(r"[^\w]+", "_", text.strip())
    return clean.strip("_")


def parse_single_match(
    match_data: Dict[str, Any], match_file_id: str
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parses a single Cricsheet match JSON dictionary.

    Args:
        match_data: Parsed JSON content of a match.
        match_file_id: Numerical/string match ID from filename (e.g. '1478411').

    Returns:
        A tuple of (match_record_dict, list_of_delivery_dicts).
        Returns (None, []) if essential metadata is missing.
    """
    info = match_data.get("info", {})
    raw_teams = info.get("teams", [])
    if len(raw_teams) < 2:
        return None, []

    # 1. Standardize format and match date
    raw_format = info.get("match_type", "T20")
    format_type = FORMAT_MAP.get(raw_format, raw_format)
    dates = info.get("dates", [])
    match_date = str(dates[0]) if dates else "1970-01-01"

    # 2. Build canonical, order-independent Match Natural Key
    sorted_teams = sorted(raw_teams)
    team_a_clean = sanitize_token(sorted_teams[0])
    team_b_clean = sanitize_token(sorted_teams[1])
    match_key = f"{format_type}_{match_date}_{team_a_clean}_{team_b_clean}"

    # 3. Extract match outcomes
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")
    by_dict = outcome.get("by", {})
    win_margin_runs = by_dict.get("runs")
    win_margin_wkts = by_dict.get("wickets")
    outcome_result = outcome.get("result")  # 'tie', 'no result', etc.

    is_dls = False
    if "method" in outcome and "D/L" in str(outcome.get("method", "")).upper():
        is_dls = True

    toss = info.get("toss", {})
    toss_winner = toss.get("winner")
    toss_decision = toss.get("decision")

    venue = info.get("venue", "Unknown Venue")
    city = info.get("city")
    season = str(info.get("season", ""))
    event_info = info.get("event", {})
    tournament_name = event_info.get("name") if isinstance(event_info, dict) else str(event_info)

    player_of_match = info.get("player_of_match", [])
    pom_str = ", ".join(player_of_match) if isinstance(player_of_match, list) else str(player_of_match)

    # 4. Construct Match Record (dim_matches)
    match_record = {
        "match_id": str(match_file_id),
        "match_key": match_key,
        "match_type": format_type,
        "match_date": match_date,
        "season": season,
        "tournament": tournament_name,
        "team_1": sorted_teams[0],
        "team_2": sorted_teams[1],
        "venue": venue,
        "city": city,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "winner": winner,
        "win_margin_runs": win_margin_runs,
        "win_margin_wkts": win_margin_wkts,
        "outcome_result": outcome_result,
        "is_dls": is_dls,
        "player_of_match": pom_str if pom_str else None,
    }

    # 5. Extract Delivery Records (fact_deliveries_base)
    deliveries_list: List[Dict[str, Any]] = []
    innings = match_data.get("innings", [])

    for inning_idx, inning_data in enumerate(innings, start=1):
        batting_team = inning_data.get("team")
        # Bowling team is whichever of the two teams is not batting
        bowling_team = sorted_teams[1] if batting_team == sorted_teams[0] else sorted_teams[0]

        overs = inning_data.get("overs", [])
        for over_data in overs:
            over_num = int(over_data.get("over", 0))
            match_phase = get_match_phase(format_type, over_num)
            raw_deliveries = over_data.get("deliveries", [])

            for ball_idx, deliv in enumerate(raw_deliveries, start=1):
                batter = deliv.get("batter", "Unknown")
                bowler = deliv.get("bowler", "Unknown")
                non_striker = deliv.get("non_striker", "Unknown")

                # Scoring
                runs_info = deliv.get("runs", {})
                runs_batter = int(runs_info.get("batter", 0))
                runs_extras = int(runs_info.get("extras", 0))
                total_runs = int(runs_info.get("total", 0))

                extras_info = deliv.get("extras", {})
                extra_wides = int(extras_info.get("wides", 0))
                extra_noballs = int(extras_info.get("noballs", 0))
                extra_byes = int(extras_info.get("byes", 0))
                extra_legbyes = int(extras_info.get("legbyes", 0))
                extra_penalty = int(extras_info.get("penalty", 0))

                is_dot = total_runs == 0
                is_boundary = runs_batter >= 4
                is_four = runs_batter == 4
                is_six = runs_batter == 6

                # Wickets
                wickets = deliv.get("wickets", [])
                is_wicket = len(wickets) > 0
                dismissal_kind = None
                player_out = None
                fielders_str = None

                if is_wicket:
                    w0 = wickets[0]
                    dismissal_kind = w0.get("kind")
                    player_out = w0.get("player_out")
                    fielders_list = w0.get("fielders", [])
                    f_names = [f.get("name") for f in fielders_list if isinstance(f, dict) and "name" in f]
                    fielders_str = ", ".join(f_names) if f_names else None

                # Reviews
                review_info = deliv.get("review")
                has_drs = review_info is not None
                drs_by = review_info.get("by") if has_drs else None
                drs_decision = review_info.get("decision") if has_drs else None

                # Construct Deterministic Delivery Natural Key
                delivery_key = f"{match_key}_{inning_idx}_{over_num}_{ball_idx}"

                deliveries_list.append({
                    "delivery_key": delivery_key,
                    "match_id": str(match_file_id),
                    "match_key": match_key,
                    "match_type": format_type,
                    "match_date": match_date,
                    "venue": venue,
                    "inning": inning_idx,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "over": over_num,
                    "ball": ball_idx,
                    "match_phase": match_phase,
                    "batter": batter,
                    "bowler": bowler,
                    "non_striker": non_striker,
                    "runs_batter": runs_batter,
                    "runs_extras": runs_extras,
                    "total_runs": total_runs,
                    "extra_wides": extra_wides,
                    "extra_noballs": extra_noballs,
                    "extra_byes": extra_byes,
                    "extra_legbyes": extra_legbyes,
                    "extra_penalty": extra_penalty,
                    "is_dot": is_dot,
                    "is_boundary": is_boundary,
                    "is_four": is_four,
                    "is_six": is_six,
                    "is_wicket": is_wicket,
                    "dismissal_kind": dismissal_kind,
                    "player_out": player_out,
                    "fielders": fielders_str,
                    "has_drs": has_drs,
                    "drs_by": drs_by,
                    "drs_decision": drs_decision,
                })

    return match_record, deliveries_list


def parse_cricsheet_archive(
    zip_path: Path, max_matches: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Streams and parses JSON files directly from an in-memory zip archive.

    Args:
        zip_path: Path to the .zip archive on disk (e.g. data/raw/t20s_json.zip).
        max_matches: Optional limit on the number of matches to parse (ideal for quick tests).

    Returns:
        A tuple of (df_matches, df_deliveries).
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found at {zip_path}")

    matches_records: List[Dict[str, Any]] = []
    deliveries_records: List[Dict[str, Any]] = []

    logger.info(f"Opening archive in memory: {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        json_filenames = [name for name in zf.namelist() if name.endswith(".json")]
        total_available = len(json_filenames)

        if max_matches:
            json_filenames = json_filenames[:max_matches]
            logger.info(f"Sample mode enabled: parsing {len(json_filenames)} / {total_available} matches.")
        else:
            logger.info(f"Full mode enabled: parsing all {total_available} matches.")

        for idx, filename in enumerate(json_filenames, start=1):
            try:
                with zf.open(filename) as f:
                    match_data = json.load(f)

                # Match ID from filename (e.g. '1478411.json' -> '1478411')
                file_id = Path(filename).stem
                m_record, d_records = parse_single_match(match_data, file_id)

                if m_record:
                    matches_records.append(m_record)
                    deliveries_records.extend(d_records)

                # Progress indicator every 1,000 matches (or every 50 in sample mode)
                step = 50 if max_matches else 1000
                if idx % step == 0 or idx == len(json_filenames):
                    pct = (idx / len(json_filenames)) * 100
                    logger.info(
                        f"Parsed {idx:,} / {len(json_filenames):,} matches ({pct:.1f}%) | "
                        f"Deliveries collected: {len(deliveries_records):,}"
                    )

            except Exception as err:
                logger.warning(f"Skipping corrupt or unparseable match {filename}: {err}")
                continue

    logger.info(
        f"Parsing complete. Total Matches: {len(matches_records):,} | "
        f"Total Deliveries: {len(deliveries_records):,}"
    )

    df_matches = pd.DataFrame(matches_records)
    df_deliveries = pd.DataFrame(deliveries_records)

    return df_matches, df_deliveries


def save_parsed_tables(
    df_matches: pd.DataFrame, df_deliveries: pd.DataFrame, table_prefix: str = "t20"
) -> Tuple[Path, Path]:
    """
    Saves parsed DataFrames to optimized columnar Parquet format in data/processed/.

    Args:
        df_matches: DataFrame of match records.
        df_deliveries: DataFrame of delivery records.
        table_prefix: String prefix for filenames (default: 't20').

    Returns:
        Paths to the saved (matches_parquet, deliveries_parquet).
    """
    matches_path = PROCESSED_DIR / f"{table_prefix}_matches_raw.parquet"
    deliveries_path = PROCESSED_DIR / f"{table_prefix}_deliveries_raw.parquet"

    logger.info(f"Saving matches table to: {matches_path.name}")
    df_matches.to_parquet(matches_path, index=False, engine="pyarrow")

    logger.info(f"Saving deliveries table to: {deliveries_path.name}")
    df_deliveries.to_parquet(deliveries_path, index=False, engine="pyarrow")

    return matches_path, deliveries_path


# Self-test when running this script directly
if __name__ == "__main__":
    logger.info("Executing Subtask 3 self-test (parsing a sample of 100 matches)...")
    archive_file = RAW_DIR / "t20s_json.zip"

    # Fast test run on 100 matches to verify speed, schema, and keys
    df_m, df_d = parse_cricsheet_archive(archive_file, max_matches=100)

    # Save to data/processed/
    m_path, d_path = save_parsed_tables(df_m, df_d, table_prefix="sample_100")

    # Quick inspection using DuckDB
    con = duckdb.connect()
    summary = con.execute(f"""
        SELECT 
            COUNT(*) as total_deliveries,
            COUNT(DISTINCT match_key) as matches,
            COUNT(DISTINCT batter) as unique_batters,
            COUNT(DISTINCT bowler) as unique_bowlers,
            SUM(runs_batter) as total_runs,
            SUM(is_wicket::INT) as total_wickets
        FROM '{d_path.as_posix()}'
    """).df()

    logger.info(f"DuckDB Sample Summary:\n{summary.to_string(index=False)}")
    logger.info(f"Sample match keys:\n{df_m['match_key'].head(3).to_list()}")
    logger.info(f"Sample delivery keys:\n{df_d['delivery_key'].head(3).to_list()}")
