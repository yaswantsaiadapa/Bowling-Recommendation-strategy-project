"""
Dynamic Player Directory & Hybrid Role Classifier.

Builds the master Player Dimension (dim_players) 100% dynamically from Cricsheet
data with zero reliance on legacy CSVs. Implements the tightened Hybrid Role Classification
framework:
- Roles: 'Batsman', 'Bowler', 'All-Rounder', and 'Developing / Debutant'.
- Clean Debutant Guard: matches_played <= 2 OR total balls < 40.
- Clean Bowler Lock: median_pos >= 7.5 AND overs_per_match >= 2.0.
- Tightened All-Rounder: median_pos <= 7.0 AND balls_per_innings >= 12.0 AND regular bowling.
- Native arrays (VARCHAR[]) for teams and formats.
"""

import sys
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import duckdb

from data_pipeline.config import PROCESSED_DIR, RAW_DIR, FORMAT_MAP, logger


def classify_hybrid_role(
    balls_faced: int,
    balls_bowled: int,
    matches_count: int,
    innings_batted: int,
    overs_per_match: float,
    bowling_part_rate: float,
    balls_per_innings: float,
    median_pos: float,
    pct_pos_1_3: float,
    pct_pos_4_6: float,
    pct_pos_7_11: float,
    is_bat_qual: bool,
    is_bowl_qual: bool,
) -> str:
    """
    Classifies a cricket player into one of three primary roles:
    'Batsman', 'Bowler', or 'All-Rounder' (with 'Developing / Debutant' for low-sample players).
    """
    # --------------------------------------------------------------------------
    # 1. CLEAN DEBUTANT / DEVELOPING GUARD
    # --------------------------------------------------------------------------
    total_balls = balls_faced + balls_bowled
    if matches_count <= 2 or total_balls < 40:
        return "Developing / Debutant"

    # --------------------------------------------------------------------------
    # 2. CLEAN BOWLER LOCK (Specialist Tailend Bowler)
    # --------------------------------------------------------------------------
    # Anyone who bats in positions 8-11 (median >= 7.5) and bowls regular overs (>= 2.0 ov/match)
    # is permanently locked as a Bowler. Occasional hitting does not make them an all-rounder.
    if median_pos >= 7.5 and overs_per_match >= 2.0 and bowling_part_rate >= 0.50:
        return "Bowler"

    # --------------------------------------------------------------------------
    # 3. TIGHTENED ALL-ROUNDER CHECK
    # --------------------------------------------------------------------------
    # Meaningful, regular contribution in BOTH departments:
    # - Bats strictly in positions 1 to 7 (median_pos <= 7.0)
    # - Faces >= 12.0 balls per batting innings (active presence at the crease)
    # - Bowls in at least 40% of matches, averaging >= 1.50 overs/match
    if (
        median_pos <= 7.0
        and balls_per_innings >= 12.0
        and overs_per_match >= 1.50
        and bowling_part_rate >= 0.40
    ):
        return "All-Rounder"

    # --------------------------------------------------------------------------
    # 4. PURE BATSMAN CHECK
    # --------------------------------------------------------------------------
    # Bats in top order (positions 1-6) with low or occasional bowling involvement
    if (
        (median_pos <= 6.5 or (pct_pos_1_3 + pct_pos_4_6) >= 0.65)
        and bowling_part_rate < 0.30
        and overs_per_match < 0.75
    ):
        return "Batsman"

    # --------------------------------------------------------------------------
    # 5. SECONDARY 250-BALL TIE-BREAKER & QUALIFICATION FALLBACK
    # --------------------------------------------------------------------------
    # Sustained career bowling pedigree (250+ balls) without batting qualification:
    if is_bowl_qual and not is_bat_qual:
        return "Bowler" if overs_per_match >= 1.4 else "All-Rounder"

    # Sustained career batting pedigree (250+ balls) without bowling qualification:
    if is_bat_qual and not is_bowl_qual:
        return "Batsman" if median_pos <= 7.0 and overs_per_match < 0.8 else "All-Rounder"

    # Sustained pedigree in BOTH disciplines:
    if is_bat_qual and is_bowl_qual:
        return "All-Rounder"

    # Default fallback based on comparative workload:
    if overs_per_match >= 1.50 or median_pos >= 7.5:
        return "Bowler"
    elif balls_per_innings >= 10.0 or median_pos <= 6.5:
        return "Batsman"
    else:
        return "All-Rounder"


class PlayerNormalizer:
    """
    Compiles and manages the canonical Player Dimension (dim_players).
    Extracts all players dynamically from Cricsheet, computes the hybrid features,
    and classifies into Batsman, Bowler, All-Rounder, and Developing / Debutant.
    """

    def __init__(self):
        self.players: Dict[str, Dict[str, Any]] = {}
        self.batting_positions: Dict[str, List[int]] = defaultdict(list)
        self.matches_bowled: Dict[str, Set[str]] = defaultdict(set)

    def _get_or_create_player(self, player_name: str) -> Dict[str, Any]:
        """Retrieves or initializes a player's profile record."""
        name = str(player_name).strip()
        if name not in self.players:
            slug_id = name.lower().replace(" ", "_")
            self.players[name] = {
                "player_id": slug_id,
                "player_name": name,
                "teams": set(),
                "formats": set(),
                "matches": set(),
                "balls_faced": 0,
                "runs_scored": 0,
                "balls_bowled": 0,
                "wickets_taken": 0,
            }
        return self.players[name]

    def process_deliveries_dataframe(self, df_deliveries: pd.DataFrame) -> None:
        """
        Extracts performance metrics, batting positions, and bowling participation
        from parsed delivery records.
        """
        if df_deliveries.empty:
            return

        logger.info(f"Computing hybrid player metrics across {len(df_deliveries):,} deliveries...")

        # 1. Track Batting Positions by Inning Order
        for (match_id, inn), group in df_deliveries.groupby(["match_id", "inning"], sort=False):
            seen_batters: List[str] = []
            for _, row in group.iterrows():
                b = str(row["batter"]).strip()
                ns = str(row.get("non_striker", "")).strip()
                if b and b not in seen_batters:
                    seen_batters.append(b)
                if ns and ns not in seen_batters:
                    seen_batters.append(ns)

            for pos, player in enumerate(seen_batters, start=1):
                self.batting_positions[player].append(pos)

        # 2. Track Matches Bowled In
        bowler_matches = df_deliveries.groupby(["bowler", "match_id"]).size().reset_index()
        for _, row in bowler_matches.iterrows():
            bowler_name = str(row["bowler"]).strip()
            self.matches_bowled[bowler_name].add(str(row["match_id"]))

        # 3. Batting Aggregations
        bat_stats = df_deliveries.groupby("batter").agg(
            balls_faced=("ball", "count"),
            runs_scored=("runs_batter", "sum"),
            teams=("batting_team", lambda s: set(s.dropna().unique())),
            formats=("match_type", lambda s: set(s.dropna().unique())),
            matches=("match_id", lambda s: set(s.dropna().unique())),
        ).reset_index()

        for _, row in bat_stats.iterrows():
            player = self._get_or_create_player(row["batter"])
            player["balls_faced"] += int(row["balls_faced"])
            player["runs_scored"] += int(row["runs_scored"])
            player["teams"].update(row["teams"])
            player["formats"].update(row["formats"])
            player["matches"].update(row["matches"])

        # 4. Bowling Aggregations
        bowl_stats = df_deliveries.groupby("bowler").agg(
            balls_bowled=("ball", "count"),
            wickets_taken=("is_wicket", "sum"),
            teams=("bowling_team", lambda s: set(s.dropna().unique())),
            formats=("match_type", lambda s: set(s.dropna().unique())),
            matches=("match_id", lambda s: set(s.dropna().unique())),
        ).reset_index()

        for _, row in bowl_stats.iterrows():
            player = self._get_or_create_player(row["bowler"])
            player["balls_bowled"] += int(row["balls_bowled"])
            player["wickets_taken"] += int(row["wickets_taken"])
            player["teams"].update(row["teams"])
            player["formats"].update(row["formats"])
            player["matches"].update(row["matches"])

    def build_dim_players_df(self) -> pd.DataFrame:
        """
        Computes all hybrid features and classifies roles into Batsman, Bowler,
        All-Rounder, or Developing / Debutant.
        """
        records: List[Dict[str, Any]] = []

        for name, data in self.players.items():
            matches_count = len(data["matches"])
            balls_faced = data["balls_faced"]
            balls_bowled = data["balls_bowled"]

            # Position calculations
            positions = self.batting_positions.get(name, [])
            innings_batted = len(positions)
            median_pos = float(statistics.median(positions)) if positions else 11.0

            if positions:
                pct_pos_1_3 = sum(1 for p in positions if 1 <= p <= 3) / len(positions)
                pct_pos_4_6 = sum(1 for p in positions if 4 <= p <= 6) / len(positions)
                pct_pos_7_11 = sum(1 for p in positions if p >= 7) / len(positions)
            else:
                pct_pos_1_3, pct_pos_4_6, pct_pos_7_11 = 0.0, 0.0, 1.0

            # Workload & Participation calculations
            overs_per_match = (balls_bowled / 6.0) / matches_count if matches_count > 0 else 0.0
            matches_bowled_count = len(self.matches_bowled.get(name, set()))
            bowling_part_rate = matches_bowled_count / matches_count if matches_count > 0 else 0.0

            balls_faced_per_match = balls_faced / matches_count if matches_count > 0 else 0.0
            balls_faced_per_innings = balls_faced / innings_batted if innings_batted > 0 else 0.0

            is_bat_qual = balls_faced >= 250
            is_bowl_qual = balls_bowled >= 250

            # Tightened Role Classification
            role = classify_hybrid_role(
                balls_faced=balls_faced,
                balls_bowled=balls_bowled,
                matches_count=matches_count,
                innings_batted=innings_batted,
                overs_per_match=overs_per_match,
                bowling_part_rate=bowling_part_rate,
                balls_per_innings=balls_faced_per_innings,
                median_pos=median_pos,
                pct_pos_1_3=pct_pos_1_3,
                pct_pos_4_6=pct_pos_4_6,
                pct_pos_7_11=pct_pos_7_11,
                is_bat_qual=is_bat_qual,
                is_bowl_qual=is_bowl_qual,
            )

            sorted_teams = sorted(list(data["teams"])) if data["teams"] else ["Independent / Unassigned"]
            sorted_formats = sorted(list(data["formats"])) if data["formats"] else ["T20I"]

            records.append({
                "player_id": data["player_id"],
                "player_name": name,
                "primary_role": role,
                "teams": sorted_teams,         # Native list -> VARCHAR[]
                "formats": sorted_formats,     # Native list -> VARCHAR[]
                "matches_played": matches_count,
                "balls_faced": balls_faced,
                "runs_scored": data["runs_scored"],
                "balls_bowled": balls_bowled,
                "wickets_taken": data["wickets_taken"],
                "overs_bowled_per_match": round(overs_per_match, 2),
                "bowling_participation_rate": round(bowling_part_rate, 2),
                "balls_faced_per_match": round(balls_faced_per_match, 1),
                "balls_faced_per_innings": round(balls_faced_per_innings, 1),
                "median_batting_position": round(median_pos, 1),
                "pct_pos_1_3": round(pct_pos_1_3, 2),
                "pct_pos_4_6": round(pct_pos_4_6, 2),
                "pct_pos_7_11": round(pct_pos_7_11, 2),
                "is_batting_qualified": is_bat_qual,
                "is_bowling_qualified": is_bowl_qual,
            })

        df = pd.DataFrame(records)
        df.sort_values(by="matches_played", ascending=False, inplace=True)
        return df

    def export_dim_players(self, output_path: Optional[Path] = None) -> Path:
        """Exports the master Player Dimension to Parquet with array columns."""
        out = output_path or (PROCESSED_DIR / "dim_players.parquet")
        df_players = self.build_dim_players_df()
        df_players.to_parquet(out, index=False, engine="pyarrow")
        logger.info(f"Exported {len(df_players):,} players to: {out.name}")
        return out


# Self-test when running this script directly
if __name__ == "__main__":
    logger.info("Executing Subtask 5 self-test (Tightened 3-Role Classifier)...")
    normalizer = PlayerNormalizer()

    sample_deliveries_file = PROCESSED_DIR / "sample_100_deliveries_raw.parquet"
    if sample_deliveries_file.exists():
        df_deliv = pd.read_parquet(sample_deliveries_file)
        normalizer.process_deliveries_dataframe(df_deliv)

    export_path = normalizer.export_dim_players()

    con = duckdb.connect()
    con.register("players_table", normalizer.build_dim_players_df())

    # 1. Check overall role distribution
    distribution = con.execute("""
        SELECT primary_role, COUNT(*) as count, 
               ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM players_table), 1) as pct
        FROM players_table
        GROUP BY primary_role
        ORDER BY count DESC
    """).df()
    logger.info(f"Tightened Role Distribution:\n{distribution.to_string(index=False)}")

    # 2. Check preview of top players
    preview = con.execute("""
        SELECT 
            player_name, 
            primary_role, 
            matches_played, 
            median_batting_position as med_pos, 
            overs_bowled_per_match as ov_pm, 
            bowling_participation_rate as bowl_part, 
            balls_faced_per_innings as b_pi
        FROM players_table
        WHERE matches_played >= 7
        ORDER BY matches_played DESC
        LIMIT 10
    """).df()
    logger.info(f"Player Preview:\n{preview.to_string(index=False)}")
