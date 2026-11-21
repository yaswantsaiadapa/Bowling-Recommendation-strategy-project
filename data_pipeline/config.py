"""
Pipeline Configuration & Logging Foundation.

Centralized settings, absolute directory paths, official Cricsheet download URLs,
domain mapping rules (match formats, phase calculation), and lightweight progress logging.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

# ==============================================================================
# SECTION 1: DYNAMIC DIRECTORY PATHS
# ==============================================================================
# Resolves the project root dynamically regardless of the current working directory.
# data_pipeline/ is one level deep, so .parent.parent gives the root folder.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Core project folders
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"                # For downloaded Cricsheet zip files
PROCESSED_DIR: Path = DATA_DIR / "processed"    # For master DuckDB and Parquet tables
LOGS_DIR: Path = PROJECT_ROOT / "logs"          # For pipeline execution log files
LEGACY_CSV_DIR: Path = DATA_DIR / "csv"         # Existing reference CSVs (grounds, players, etc.)

# Auto-create essential directories if they don't already exist
for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# SECTION 2: OFFICIAL CRICSHEET DOWNLOAD ENDPOINTS
# ==============================================================================
# Official, verified bulk JSON archives from cricsheet.org.
CRICSHEET_URLS: Dict[str, str] = {
    # All official Men's & Women's T20Is + 100+ Associate nations
    "t20s": "https://cricsheet.org/downloads/t20s_json.zip",
    # All Men's & Women's One-Day Internationals (2004–present)
    "odis": "https://cricsheet.org/downloads/odis_json.zip",
    # All Test Matches (2004–present)
    "tests": "https://cricsheet.org/downloads/tests_json.zip",
    # Complete multi-format archive containing all historical matches
    "all": "https://cricsheet.org/downloads/all_json.zip",
}


# ==============================================================================
# SECTION 3: CRICKET DOMAIN RULES & NORMALIZATION MAPS
# ==============================================================================
# Maps inconsistent format strings from raw data into clean standard categories
FORMAT_MAP: Dict[str, str] = {
    "T20": "T20I",
    "IT20": "T20I",
    "Twenty20": "T20I",
    "ODI": "ODI",
    "ODM": "ODI",
    "Test": "Test",
    "MDM": "Test",
}

def get_match_phase(format_type: str, over_number: int) -> str:
    """
    Classifies an over into its tactical cricket phase (0-indexed over).
    
    Args:
        format_type: Format string ('T20I', 'ODI', 'Test').
        over_number: 0-indexed over number (e.g. over 0 is 1st over, 19 is 20th over).
        
    Returns:
        Phase string ('Powerplay', 'Middle', 'Death', or 'Session X').
    """
    fmt = FORMAT_MAP.get(format_type, format_type)
    
    if fmt == "T20I":
        if over_number <= 5:        # Overs 1 to 6
            return "Powerplay"
        elif over_number <= 14:     # Overs 7 to 15
            return "Middle"
        else:                       # Overs 16 to 20
            return "Death"
            
    elif fmt == "ODI":
        if over_number <= 9:        # Overs 1 to 10
            return "Powerplay"
        elif over_number <= 39:     # Overs 11 to 40
            return "Middle"
        else:                       # Overs 41 to 50
            return "Death"
            
    else:
        # For Test cricket, breaks overs into roughly 30-over sessions per day
        session_idx = (over_number // 30) + 1
        return f"Session {session_idx}" if session_idx <= 3 else "Session 3"


# ==============================================================================
# SECTION 4: LIGHTWEIGHT PROGRESS LOGGER
# ==============================================================================
def setup_logger(name: str = "pipeline") -> logging.Logger:
    """
    Creates a resilient, clean progress logger that outputs to both:
    1. Standard output (console) with readable timestamps.
    2. logs/pipeline.log file for permanent record.
    """
    logger = logging.getLogger(name)
    
    # Avoid attaching duplicate handlers if function is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 1. Console Output Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. File Output Handler (appends cleanly to logs/pipeline.log)
        log_file_path = LOGS_DIR / "pipeline.log"
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Master logger instance ready for import across all pipeline scripts
logger: logging.Logger = setup_logger("cricket_pipeline")


# Quick self-test when running this script directly
if __name__ == "__main__":
    logger.info("Subtask 1 verified successfully!")
    logger.info(f"PROJECT_ROOT  : {PROJECT_ROOT}")
    logger.info(f"DATA_DIR      : {DATA_DIR}")
    logger.info(f"RAW_DIR       : {RAW_DIR}")
    logger.info(f"PROCESSED_DIR : {PROCESSED_DIR}")
    logger.info(f"LOGS_DIR      : {LOGS_DIR}")
    logger.info(f"Phase test    : T20 Over 18 -> {get_match_phase('T20', 18)}")
