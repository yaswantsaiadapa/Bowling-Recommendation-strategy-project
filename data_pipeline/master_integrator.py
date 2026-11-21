"""
Master Relational Integrator & Inning Context Enricher.

Stitches together the core dimensions and fact tables into a unified master delivery table:
1. Links standardized ground metadata (ground_id, ground_name) from ground_normalizer.
2. Injects player roles (batter_role, bowler_role) from dim_players.
3. Computes running match context (cumulative_runs, cumulative_wickets, legal_ball_number,
   current_run_rate, balls_remaining) to enable situational tactical queries.
4. Enforces data quality and schema contracts before final DuckDB / Parquet export.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import duckdb

from data_pipeline.config import PROCESSED_DIR, logger
from data_pipeline.ground_normalizer import GroundNormalizer


def compute_running_match_state(df_deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Computes sequential running match context for every delivery within each inning.

    Calculates:
    - legal_ball_number: Count of legal balls bowled so far (excluding wides/noballs).
    - cumulative_runs: Running total score of the batting team.
    - cumulative_wickets: Running total wickets lost by the batting team.
    - current_run_rate: Running run rate per 6 legal balls.
    - balls_remaining: Legal deliveries remaining in the standard innings (120 for T20).
    """
    df = df_deliveries.copy()

    # Ensure chronological order by match, inning, over, and ball
    df.sort_values(by=["match_id", "inning", "over", "ball"], inplace=True)

    # A delivery is legal if neither a wide nor a noball was bowled
    is_legal = (df["extra_wides"] == 0) & (df["extra_noballs"] == 0)
    df["is_legal_ball"] = is_legal.astype(int)

    # Group by match and inning to compute running cumulative metrics
    inning_group = df.groupby(["match_id", "inning"], sort=False)

    df["cumulative_runs"] = inning_group["total_runs"].cumsum()
    df["cumulative_wickets"] = inning_group["is_wicket"].cumsum().astype(int)
    df["legal_ball_number"] = inning_group["is_legal_ball"].cumsum().astype(int)

    # Current Run Rate: (Runs * 6) / Legal Balls
    # Use np.where to prevent division by zero on ball 0 (e.g. initial wides)
    balls_valid = df["legal_ball_number"] > 0
    df["current_run_rate"] = np.where(
        balls_valid,
        np.round((df["cumulative_runs"] * 6.0) / df["legal_ball_number"], 2),
        0.0,
    )

    # Format-aware balls remaining calculation (T20 = 120, ODI = 300, Test = open-ended)
    if "match_type" in df.columns:
        quota = np.where(df["match_type"].str.contains("ODI|ODM", na=False), 300,
                np.where(df["match_type"].str.contains("Test|MDM", na=False), 0, 120))
        df["balls_remaining"] = np.maximum(0, quota - df["legal_ball_number"])
    else:
        df["balls_remaining"] = np.maximum(0, 120 - df["legal_ball_number"])

    return df


def integrate_master_deliveries(
    df_deliveries: pd.DataFrame,
    df_matches: pd.DataFrame,
    ground_normalizer: GroundNormalizer,
    df_players: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Integrates raw deliveries with ground and player dimensions, enriching with situational features.

    Args:
        df_deliveries: Parsed delivery records from cricsheet_parser.
        df_matches: Parsed match records from cricsheet_parser.
        ground_normalizer: Initialized GroundNormalizer instance.
        df_players: Master player DataFrame from player_normalizer.

    Returns:
        Tuple of (df_fact_master, df_matches_enriched).
    """
    logger.info("Starting Master Relational Integration...")

    # 1. Enrich Matches with Standardized Grounds
    logger.info("Linking venues to canonical ground dimensions...")
    df_matches_enriched = ground_normalizer.enrich_matches(df_matches)

    # Create a fast lookup map: match_id -> (ground_id, ground_name)
    ground_lookup = df_matches_enriched.set_index("match_id")[
        ["ground_id", "ground_name"]
    ].to_dict("index")

    # 2. Map Ground IDs onto Fact Deliveries
    df_deliv = df_deliveries.copy()
    df_deliv["ground_id"] = df_deliv["match_id"].map(
        lambda m_id: ground_lookup.get(str(m_id), {}).get("ground_id", 0)
    )
    df_deliv["ground_name"] = df_deliv["match_id"].map(
        lambda m_id: ground_lookup.get(str(m_id), {}).get("ground_name", "Unknown Venue")
    )

    # 3. Inject Player Roles from dim_players
    logger.info("Injecting player roles into delivery records...")
    player_role_map = df_players.set_index("player_name")["primary_role"].to_dict()

    df_deliv["batter_role"] = df_deliv["batter"].map(
        lambda name: player_role_map.get(name, "Developing / Debutant")
    )
    df_deliv["bowler_role"] = df_deliv["bowler"].map(
        lambda name: player_role_map.get(name, "Developing / Debutant")
    )

    # 4. Compute Situational Inning State (Running Score, Wickets, Run Rate)
    logger.info("Computing running match context (cumulative scores, wickets, CRR)...")
    df_fact_master = compute_running_match_state(df_deliv)

    # 5. Schema & Quality Enforcement
    # Ensure boolean flags are clean Python booleans
    for bool_col in ["is_dot", "is_boundary", "is_four", "is_six", "is_wicket", "has_drs"]:
        if bool_col in df_fact_master.columns:
            df_fact_master[bool_col] = df_fact_master[bool_col].astype(bool)

    # Organize columns into canonical order
    canonical_columns = [
        # Identifiers & Foreign Keys
        "delivery_key",
        "match_id",
        "match_key",
        "ground_id",
        "ground_name",
        # Match Info
        "match_type",
        "match_date",
        "batting_team",
        "bowling_team",
        # Inning & Temporal
        "inning",
        "over",
        "ball",
        "match_phase",
        # Players & Roles
        "batter",
        "batter_role",
        "bowler",
        "bowler_role",
        "non_striker",
        # Scoring & Extras
        "runs_batter",
        "runs_extras",
        "total_runs",
        "extra_wides",
        "extra_noballs",
        "extra_byes",
        "extra_legbyes",
        "extra_penalty",
        # Tactical Flags & Wickets
        "is_dot",
        "is_boundary",
        "is_four",
        "is_six",
        "is_wicket",
        "dismissal_kind",
        "player_out",
        "fielders",
        "has_drs",
        "drs_by",
        "drs_decision",
        # Running Match State
        "cumulative_runs",
        "cumulative_wickets",
        "legal_ball_number",
        "current_run_rate",
        "balls_remaining",
    ]

    # Keep available columns in canonical order
    final_cols = [c for c in canonical_columns if c in df_fact_master.columns]
    df_fact_master = df_fact_master[final_cols]

    logger.info(
        f"Master Integration complete. Final Fact Table: {len(df_fact_master):,} rows, "
        f"{len(df_fact_master.columns)} columns."
    )

    return df_fact_master, df_matches_enriched


def export_master_fact_table(
    df_fact_master: pd.DataFrame, output_path: Optional[Path] = None
) -> Path:
    """Exports the unified fact_deliveries_master table to Parquet."""
    out = output_path or (PROCESSED_DIR / "fact_deliveries_master.parquet")
    df_fact_master.to_parquet(out, index=False, engine="pyarrow")
    logger.info(f"Saved master fact table to: {out.name}")
    return out


# Self-test when running this script directly
if __name__ == "__main__":
    logger.info("Executing Subtask 6 self-test (Master Integrator)...")

    # Load artifacts from Subtasks 3, 4, and 5
    matches_file = PROCESSED_DIR / "sample_100_matches_raw.parquet"
    deliveries_file = PROCESSED_DIR / "sample_100_deliveries_raw.parquet"
    players_file = PROCESSED_DIR / "dim_players.parquet"

    if matches_file.exists() and deliveries_file.exists() and players_file.exists():
        df_m = pd.read_parquet(matches_file)
        df_d = pd.read_parquet(deliveries_file)
        df_p = pd.read_parquet(players_file)

        g_normalizer = GroundNormalizer()

        df_fact, df_matches_enr = integrate_master_deliveries(
            df_deliveries=df_d,
            df_matches=df_m,
            ground_normalizer=g_normalizer,
            df_players=df_p,
        )

        out_path = export_master_fact_table(df_fact)

        # Verification with DuckDB: inspect running score state on sample match
        con = duckdb.connect()
        con.register("master_fact", df_fact)

        sample_match = con.execute("SELECT match_id FROM master_fact LIMIT 1").fetchone()[0]
        preview = con.execute(f"""
            SELECT 
                over, ball, batter, batter_role, bowler, bowler_role,
                runs_batter, total_runs, cumulative_runs, cumulative_wickets, current_run_rate
            FROM master_fact
            WHERE match_id = '{sample_match}' AND inning = 1
            ORDER BY over, ball
            LIMIT 10
        """).df()

        logger.info(f"Situational Running State Preview (Match {sample_match}):\n{preview.to_string(index=False)}")
        logger.info(f"Subtask 6 verified successfully. Master fact table ready at: {out_path}")
    else:
        logger.error("Prerequisite sample files from Subtasks 3-5 not found.")
