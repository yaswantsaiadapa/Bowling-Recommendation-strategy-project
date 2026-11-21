"""
Master Pipeline Orchestrator & DuckDB Database Compiler.

Orchestrates the entire end-to-end data pipeline from raw Cricsheet data to a high-performance
analytical DuckDB database and optimized Parquet warehouse:
1. Ingestion: Streams or loads the latest Cricsheet archives (t20s, odis, tests) via downloader.py.
2. Parsing: Extracts matches and unrolls deliveries in-memory with Natural Keys (cricsheet_parser.py).
3. Ground Normalization: Canonicalizes stadium profiles and assigns unique ground_ids (ground_normalizer.py).
4. Player Dynamic Profiling: Compiles players and classifies hybrid roles (player_normalizer.py).
5. Master Integration: Injects dimensions and calculates running inning context (master_integrator.py).
6. Database Compilation: Builds the production DuckDB database (data/processed/cricket_master.duckdb).
7. Quality Assurance Suite: Runs automated sanity and integrity checks.
8. Cloud Deployment: Optionally pushes the unified dataset directly to MotherDuck.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import pandas as pd

from data_pipeline.config import (
    PROCESSED_DIR,
    RAW_DIR,
    logger,
)
from data_pipeline.downloader import download_cricsheet
from data_pipeline.cricsheet_parser import parse_cricsheet_archive
from data_pipeline.ground_normalizer import GroundNormalizer
from data_pipeline.player_normalizer import PlayerNormalizer
from data_pipeline.master_integrator import integrate_master_deliveries

# Default database and parquet locations
DEFAULT_DUCKDB_PATH = PROCESSED_DIR / "cricket_master.duckdb"


def run_pipeline(
    formats: Optional[List[str]] = None,
    sample_matches_per_format: Optional[int] = None,
    force_download: bool = False,
    duckdb_path: Path = DEFAULT_DUCKDB_PATH,
    export_parquet: bool = True,
    push_to_motherduck: bool = False,
) -> Dict[str, Any]:
    """
    Executes the end-to-end Cricket Analytics data compilation pipeline across one or multiple formats.

    Args:
        formats: List of formats to process (e.g. ['t20s', 'odis', 'tests']).
        sample_matches_per_format: Optional limit on matches to parse per format (None = full).
        force_download: Force re-download of raw zip from Cricsheet.
        duckdb_path: Destination path for the master DuckDB file.
        export_parquet: If True, writes companion Parquet files to data/processed/.
        push_to_motherduck: If True, pushes the compiled dataset to MotherDuck Cloud.

    Returns:
        Summary dict of pipeline execution statistics and QA results.
    """
    start_total_time = time.time()
    target_formats = formats or ["t20s"]

    logger.info("=" * 80)
    logger.info("CRICKET ANALYTICS MASTER PIPELINE ORCHESTRATOR")
    logger.info("=" * 80)
    logger.info(f"Target Formats   : {[f.upper() for f in target_formats]}")
    logger.info(f"Mode             : {'SAMPLE (' + str(sample_matches_per_format) + ' matches/fmt)' if sample_matches_per_format else 'FULL ARCHIVE'}")
    logger.info(f"DuckDB Target    : {duckdb_path}")
    logger.info(f"Export Parquet   : {export_parquet}")
    logger.info(f"Push MotherDuck  : {push_to_motherduck}")
    logger.info("-" * 80)

    # --------------------------------------------------------------------------
    # STAGE 1 & 2: INGESTION & PARSING (Multi-Format Loop)
    # --------------------------------------------------------------------------
    all_raw_matches: List[pd.DataFrame] = []
    all_raw_deliveries: List[pd.DataFrame] = []

    for fmt in target_formats:
        fmt_key = fmt.lower().strip()
        logger.info(f">>> Ingesting and Parsing Format: {fmt_key.upper()}...")
        zip_path = download_cricsheet(format_type=fmt_key, force=force_download)
        
        df_m, df_d = parse_cricsheet_archive(
            zip_path=zip_path,
            max_matches=sample_matches_per_format,
        )
        all_raw_matches.append(df_m)
        all_raw_deliveries.append(df_d)
        logger.info(f"    --> Collected {len(df_m):,} matches and {len(df_d):,} deliveries for {fmt_key.upper()}")

    # Concatenate all formats into unified raw tables
    df_matches_raw = pd.concat(all_raw_matches, ignore_index=True)
    df_deliveries_raw = pd.concat(all_raw_deliveries, ignore_index=True)

    # Ensure match_id uniqueness
    df_matches_raw.drop_duplicates(subset=["match_id"], inplace=True)

    logger.info("-" * 80)
    logger.info(
        f">>> MULTI-FORMAT COMBINED TOTAL: {len(df_matches_raw):,} matches | "
        f"{len(df_deliveries_raw):,} deliveries"
    )
    logger.info("-" * 80)

    # --------------------------------------------------------------------------
    # STAGE 3: GROUND NORMALIZATION (Canonical Venues & Aliases)
    # --------------------------------------------------------------------------
    stage3_start = time.time()
    logger.info(">>> STAGE 3: Normalizing Grounds & Resolving Stadium Aliases...")
    ground_normalizer = GroundNormalizer()
    df_matches_enriched = ground_normalizer.enrich_matches(df_matches_raw)
    df_grounds = pd.DataFrame(list(ground_normalizer.grounds.values()))
    if not df_grounds.empty:
        df_grounds.sort_values(by="ground_id", inplace=True)
    stage3_duration = time.time() - stage3_start
    logger.info(
        f">>> STAGE 3 COMPLETE: Resolved {len(df_grounds):,} distinct grounds ({stage3_duration:.2f}s)"
    )
    logger.info("-" * 80)

    # --------------------------------------------------------------------------
    # STAGE 4: PLAYER DYNAMIC PROFILING (Multi-Format Hybrid Classifier)
    # --------------------------------------------------------------------------
    stage4_start = time.time()
    logger.info(">>> STAGE 4: Profiling Players Across All Formats...")
    player_normalizer = PlayerNormalizer()
    player_normalizer.process_deliveries_dataframe(df_deliveries_raw)
    df_players = player_normalizer.build_dim_players_df()
    stage4_duration = time.time() - stage4_start
    logger.info(
        f">>> STAGE 4 COMPLETE: Profiled {len(df_players):,} unique players ({stage4_duration:.2f}s)"
    )
    logger.info("-" * 80)

    # --------------------------------------------------------------------------
    # STAGE 5: MASTER RELATIONAL INTEGRATION (Delivery Facts & Context)
    # --------------------------------------------------------------------------
    stage5_start = time.time()
    logger.info(">>> STAGE 5: Integrating Master Facts & Calculating Inning Running State...")
    df_fact_master, df_matches_final = integrate_master_deliveries(
        df_deliveries=df_deliveries_raw,
        df_matches=df_matches_enriched,
        ground_normalizer=ground_normalizer,
        df_players=df_players,
    )
    stage5_duration = time.time() - stage5_start
    logger.info(
        f">>> STAGE 5 COMPLETE: Assembled {len(df_fact_master):,} master delivery rows ({stage5_duration:.2f}s)"
    )
    logger.info("-" * 80)

    # --------------------------------------------------------------------------
    # STAGE 6: DUCKDB DATABASE COMPILATION
    # --------------------------------------------------------------------------
    stage6_start = time.time()
    logger.info(">>> STAGE 6: Compiling Master Tables into DuckDB Database...")
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clean previous local file to ensure clean schema compilation
    if duckdb_path.exists():
        try:
            duckdb_path.unlink()
        except Exception:
            pass

    con = duckdb.connect(str(duckdb_path))

    con.register("df_matches_view", df_matches_final)
    con.register("df_grounds_view", df_grounds)
    con.register("df_players_view", df_players)
    con.register("df_fact_view", df_fact_master)

    logger.info("Creating table: dim_matches...")
    con.execute("CREATE OR REPLACE TABLE dim_matches AS SELECT * FROM df_matches_view")

    logger.info("Creating table: dim_grounds...")
    con.execute("CREATE OR REPLACE TABLE dim_grounds AS SELECT * FROM df_grounds_view")

    logger.info("Creating table: dim_players...")
    con.execute("CREATE OR REPLACE TABLE dim_players AS SELECT * FROM df_players_view")

    logger.info("Creating table: fact_deliveries_master...")
    con.execute("CREATE OR REPLACE TABLE fact_deliveries_master AS SELECT * FROM df_fact_view")

    # Optional: Export companion Parquet files
    if export_parquet:
        logger.info("Exporting companion Parquet files to data/processed/...")
        df_matches_final.to_parquet(PROCESSED_DIR / "dim_matches.parquet", index=False, engine="pyarrow")
        df_grounds.to_parquet(PROCESSED_DIR / "dim_grounds.parquet", index=False, engine="pyarrow")
        df_players.to_parquet(PROCESSED_DIR / "dim_players.parquet", index=False, engine="pyarrow")
        df_fact_master.to_parquet(PROCESSED_DIR / "fact_deliveries_master.parquet", index=False, engine="pyarrow")

    stage6_duration = time.time() - stage6_start
    logger.info(f">>> STAGE 6 COMPLETE: DuckDB database built at {duckdb_path.name} ({stage6_duration:.2f}s)")
    logger.info("-" * 80)

    # --------------------------------------------------------------------------
    # STAGE 7: QUALITY ASSURANCE & INTEGRITY SUITE
    # --------------------------------------------------------------------------
    logger.info(">>> STAGE 7: Running Quality Assurance & Integrity Suite...")

    counts = {}
    for tbl in ["dim_matches", "dim_grounds", "dim_players", "fact_deliveries_master"]:
        count_res = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        counts[tbl] = count_res
        logger.info(f"  [QA Count] Table `{tbl}`: {count_res:,} rows")

    # Format Breakdown
    format_counts = con.execute("""
        SELECT match_type, COUNT(DISTINCT match_id) as matches, COUNT(*) as deliveries
        FROM fact_deliveries_master
        GROUP BY match_type
        ORDER BY deliveries DESC
    """).df()
    logger.info(f"  [QA Formats] Breakdown:\n{format_counts.to_string(index=False)}")

    con.close()

    # --------------------------------------------------------------------------
    # STAGE 8: MOTHERDUCK CLOUD PUSH (OPTIONAL)
    # --------------------------------------------------------------------------
    if push_to_motherduck:
        logger.info("-" * 80)
        logger.info(">>> STAGE 8: Pushing Multi-Format Production Tables to MotherDuck Cloud...")
        from analytics.db import upload_all
        upload_all(verify=True)
        logger.info(">>> STAGE 8 COMPLETE: MotherDuck cloud warehouse is fully updated!")

    total_duration = time.time() - start_total_time
    logger.info("=" * 80)
    logger.info(f"MASTER PIPELINE SUCCEEDED in {total_duration:.2f} seconds ({total_duration/60:.2f} mins)")
    logger.info("=" * 80)

    return {
        "status": "SUCCESS",
        "total_duration_sec": round(total_duration, 2),
        "table_counts": counts,
        "duckdb_file": str(duckdb_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Master Cricket Data Pipeline: Multi-Format (T20, ODI, Test) -> DuckDB & MotherDuck"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="all_formats",
        help="Format to process: 't20s', 'odis', 'tests', 'all_formats' (or comma-separated e.g. 't20s,odis,tests').",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional limit of matches per format for rapid verification. If omitted, parses full archives.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of archives from Cricsheet even if cached.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DEFAULT_DUCKDB_PATH),
        help=f"Target DuckDB database path (default: {DEFAULT_DUCKDB_PATH}).",
    )
    parser.add_argument(
        "--no-parquet",
        action="store_true",
        help="Skip writing Parquet files to disk.",
    )
    parser.add_argument(
        "--push-motherduck",
        action="store_true",
        help="Automatically push compiled tables to MotherDuck cloud warehouse after build.",
    )

    args = parser.parse_args()

    # Parse format argument
    raw_fmt = args.format.lower().strip()
    if raw_fmt in ("all", "all_formats", "combined"):
        selected_formats = ["t20s", "odis", "tests"]
    elif "," in raw_fmt:
        selected_formats = [f.strip() for f in raw_fmt.split(",") if f.strip()]
    else:
        selected_formats = [raw_fmt]

    run_pipeline(
        formats=selected_formats,
        sample_matches_per_format=args.sample,
        force_download=args.force_download,
        duckdb_path=Path(args.db_path),
        export_parquet=not args.no_parquet,
        push_to_motherduck=args.push_motherduck,
    )


if __name__ == "__main__":
    main()
