"""
Venue & Ground Intelligence Normalizer.

Standardizes free-form venue strings from Cricsheet into canonical stadium profiles.
Leverages baseline metadata from ground.csv, town.csv, and country.csv, provides
alias and token matching, and automatically discovers and registers new international venues.
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import duckdb

from data_pipeline.config import LEGACY_CSV_DIR, PROCESSED_DIR, logger


# Well-known stadium nicknames and abbreviations
COMMON_ALIASES: Dict[str, str] = {
    "mcg": "melbourne cricket ground",
    "scg": "sydney cricket ground",
    "gabba": "brisbane cricket ground",
    "wankhede": "wankhede stadium",
    "eden gardens": "eden gardens kolkata",
    "chinnaswamy": "m chinnaswamy stadium",
    "chepauk": "m a chidambaram stadium",
    "motera": "narendra modi stadium",
    "ahmedabad": "narendra modi stadium",
    "lords": "lord s london",
    "oval": "kennington oval london",
    "dharamsala": "himachal pradesh cricket association stadium",
    "kotla": "arun jaitley stadium",
    "feroz shah kotla": "arun jaitley stadium",
    "rawalpindi": "rawalpindi cricket stadium",
    "gaddafi": "gaddafi stadium lahore",
}

STOPWORDS: Set[str] = {
    "cricket", "ground", "stadium", "oval", "sports", "complex", "international",
    "association", "the", "and", "at", "in"
}


def clean_venue_text(text: str) -> str:
    """Lowercases, normalizes possessives ('s -> s), removes punctuation, and standardizes spacing."""
    if not text or pd.isna(text):
        return ""
    # Normalize possessives (e.g., Lord's -> Lords)
    raw = re.sub(r"['’]s\b", "s", str(text).lower())
    clean = re.sub(r"[^\w\s]", " ", raw)
    tokens = clean.split()
    return " ".join(tokens)


def extract_keywords(text: str) -> Set[str]:
    """Extracts distinctive core keywords by stripping generic cricket stopwords."""
    tokens = clean_venue_text(text).split()
    return {t for t in tokens if t not in STOPWORDS and len(t) >= 3}


class GroundNormalizer:
    """
    Manages the canonical Ground Directory (dim_grounds).
    Resolves spelling variations to consistent ground IDs and auto-registers new venues.
    """

    def __init__(self, legacy_dir: Optional[Path] = None):
        self.legacy_dir = legacy_dir or LEGACY_CSV_DIR
        self.grounds: Dict[int, Dict[str, Any]] = {}
        self.alias_lookup: Dict[str, int] = {}
        # Start auto-discovered ground IDs at 10000 to avoid conflicting with baseline IDs (1-999)
        self.next_auto_id: int = 10000

        self._load_baseline_grounds()

    def _load_baseline_grounds(self) -> None:
        """Loads and merges ground.csv + town.csv + country.csv into the ground catalog."""
        ground_file = self.legacy_dir / "ground.csv"
        town_file = self.legacy_dir / "town.csv"
        country_file = self.legacy_dir / "country.csv"

        if not ground_file.exists():
            logger.warning(f"Baseline ground.csv not found at {ground_file}. Starting fresh catalog.")
            return

        try:
            df_g = pd.read_csv(ground_file)
            df_t = pd.read_csv(town_file) if town_file.exists() else pd.DataFrame()
            df_c = pd.read_csv(country_file) if country_file.exists() else pd.DataFrame()

            # Merge with town and country if available
            if not df_t.empty and "Town ID" in df_g.columns and "Town ID" in df_t.columns:
                df_merged = df_g.merge(df_t, on="Town ID", how="left")
            else:
                df_merged = df_g.copy()
                df_merged["Town Name"] = None

            if not df_c.empty and "Country ID" in df_merged.columns and "Country ID" in df_c.columns:
                df_merged = df_merged.merge(df_c, on="Country ID", how="left")
            else:
                df_merged["Country Name"] = None

            for _, row in df_merged.iterrows():
                try:
                    gid = int(row["Ground ID"])
                    name = str(row["Ground Name"]).strip()
                    slug = str(row.get("Ground Slug", "")).strip()
                    city = str(row.get("Town Name", "")).strip() if pd.notna(row.get("Town Name")) else None
                    country = str(row.get("Country Name", "")).strip() if pd.notna(row.get("Country Name")) else None

                    # Capacity can have commas like '30,000'
                    raw_cap = str(row.get("Capacity", "0")).replace(",", "").strip()
                    capacity = int(float(raw_cap)) if raw_cap.isdigit() else None

                    self.grounds[gid] = {
                        "ground_id": gid,
                        "ground_name": name,
                        "ground_slug": slug if slug else clean_venue_text(name).replace(" ", "-"),
                        "city": city,
                        "country": country,
                        "capacity": capacity,
                    }

                    # Index primary cleaned name
                    self.alias_lookup[clean_venue_text(name)] = gid

                except (ValueError, KeyError) as err:
                    continue

            logger.info(f"Loaded {len(self.grounds)} baseline stadiums from ground.csv.")

        except Exception as exc:
            logger.error(f"Error loading baseline grounds: {exc}")

    def resolve_ground(self, raw_venue: str, city: Optional[str] = None) -> Tuple[int, str]:
        """
        Resolves an arbitrary venue string into a canonical (ground_id, ground_name).
        If the venue is completely new, it automatically registers it.

        Args:
            raw_venue: Raw venue name from Cricsheet.
            city: Optional city string for geographic tie-breaking.

        Returns:
            Tuple of (canonical_ground_id, canonical_ground_name).
        """
        if not raw_venue or pd.isna(raw_venue):
            return 0, "Unknown Venue"

        raw_str = str(raw_venue).strip()
        cleaned = clean_venue_text(raw_str)

        # 1. Direct exact match in alias index
        if cleaned in self.alias_lookup:
            gid = self.alias_lookup[cleaned]
            return gid, self.grounds[gid]["ground_name"]

        # 2. Check well-known common aliases (e.g. 'mcg', 'lords')
        for alias_key, target in COMMON_ALIASES.items():
            if alias_key in cleaned:
                target_clean = clean_venue_text(target)
                if target_clean in self.alias_lookup:
                    gid = self.alias_lookup[target_clean]
                    return gid, self.grounds[gid]["ground_name"]

        # 3. Keyword / Token Overlap Match
        raw_keywords = extract_keywords(raw_str)
        if raw_keywords:
            best_match_id = None
            max_overlap = 0

            for gid, g_info in self.grounds.items():
                target_keywords = extract_keywords(g_info["ground_name"])
                common_tokens = raw_keywords.intersection(target_keywords)
                overlap = len(common_tokens)

                # Check city alignment for bonus confidence
                if city and g_info.get("city"):
                    if str(city).lower() in str(g_info["city"]).lower():
                        overlap += 1

                # Match if 2+ shared keywords OR 1 distinctive keyword with 5+ chars
                is_valid_match = (overlap >= 2) or (overlap == 1 and any(len(tok) >= 5 for tok in common_tokens))

                if overlap > max_overlap and is_valid_match:
                    max_overlap = overlap
                    best_match_id = gid

            if best_match_id is not None:
                # Cache this variant so subsequent lookups are instant O(1)
                self.alias_lookup[cleaned] = best_match_id
                return best_match_id, self.grounds[best_match_id]["ground_name"]

        # 4. Auto-Discovery for New Grounds (Associate stadiums, new venues)
        new_id = self.next_auto_id
        self.next_auto_id += 1

        clean_slug = clean_venue_text(raw_str).replace(" ", "-")
        self.grounds[new_id] = {
            "ground_id": new_id,
            "ground_name": raw_str,
            "ground_slug": clean_slug,
            "city": city if city and pd.notna(city) else None,
            "country": None,
            "capacity": None,
        }
        self.alias_lookup[cleaned] = new_id

        return new_id, raw_str

    def enrich_matches(self, df_matches: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches a matches DataFrame with standardized ground_id and canonical_ground_name.
        """
        df = df_matches.copy()
        ground_ids = []
        canonical_names = []

        for _, row in df.iterrows():
            venue = row.get("venue", "")
            city = row.get("city", None)
            gid, cname = self.resolve_ground(venue, city)
            ground_ids.append(gid)
            canonical_names.append(cname)

        df["ground_id"] = ground_ids
        df["ground_name"] = canonical_names
        return df

    def export_dim_grounds(self, output_path: Optional[Path] = None) -> Path:
        """Exports the master Ground Dimension to Parquet."""
        out = output_path or (PROCESSED_DIR / "dim_grounds.parquet")
        df_grounds = pd.DataFrame(list(self.grounds.values()))
        df_grounds.sort_values(by="ground_id", inplace=True)
        df_grounds.to_parquet(out, index=False, engine="pyarrow")
        logger.info(f"Exported {len(df_grounds)} stadiums to: {out.name}")
        return out


# Self-test when running this script directly
if __name__ == "__main__":
    logger.info("Executing Subtask 4 self-test (Ground Normalizer)...")
    normalizer = GroundNormalizer()

    # Test sample variations
    test_queries = [
        ("Lord's, London", "London"),
        ("Lord's", "London"),
        ("Lord's Cricket Ground", None),
        ("Melbourne Cricket Ground", "Melbourne"),
        ("MCG", "Melbourne"),
        ("Wankhede Stadium", "Mumbai"),
        ("Eden Gardens", "Kolkata"),
        ("Grand Prairie Stadium, Dallas", "Dallas"),  # New Associate ground test
    ]

    logger.info("Testing venue resolution against test cases:")
    for v_str, c_str in test_queries:
        gid, gname = normalizer.resolve_ground(v_str, c_str)
        logger.info(f"  Input: '{v_str}' -> Ground ID {gid}: '{gname}'")

    # Enrich our Subtask 3 sample matches
    sample_matches_file = PROCESSED_DIR / "sample_100_matches_raw.parquet"
    if sample_matches_file.exists():
        df_m = pd.read_parquet(sample_matches_file)
        df_enriched = normalizer.enrich_matches(df_m)
        logger.info(
            f"Successfully enriched {len(df_enriched)} sample matches with standardized ground_id."
        )

    # Export canonical dim_grounds table
    export_path = normalizer.export_dim_grounds()
    logger.info(f"Subtask 4 verification complete. Table ready at: {export_path}")
