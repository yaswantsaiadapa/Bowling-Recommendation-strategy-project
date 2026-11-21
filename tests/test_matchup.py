import os, sys
from pathlib import Path
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
"""
Test suite for Matchup & Opponent Intelligence Engine.
Tests all 3 modes:
1. Batter vs Bowler (H2H)
2. Batter vs Team
3. Bowler vs Team
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import app

client = app.test_client()

# ── 1. Mode 1: Batter vs Bowler (V Kohli vs JM Anderson) ──────────────────────
res = client.get("/matchup?mode=h2h&batter=V+Kohli&bowler=JM+Anderson")
html = res.data.decode("utf-8")
assert res.status_code == 200, f"H2H route failed: {res.status_code}"
assert "Format-by-Format Rivalry Split" in html, "FAIL: Format split missing in H2H!"
assert "Scoring Anatomy off Bowler" in html, "FAIL: Shot distribution missing in H2H!"
assert "Dismissal History" in html, "FAIL: Dismissal history table missing in H2H!"
assert html.count('"bdata"') == 0, "FAIL: Plotly binary encoding bug present in H2H!"
print("MODE 1: H2H Batter vs Bowler    : PASS ✓  (All plots, dismissal table, 0 bdata)")

# ── 2. Mode 2: Batter vs Team (V Kohli vs Australia) ──────────────────────────
res = client.get("/matchup?mode=bat_team&batter=V+Kohli&team=Australia")
html = res.data.decode("utf-8")
assert res.status_code == 200, f"Batter vs Team route failed: {res.status_code}"
assert "Format Mastery against Australia" in html, "FAIL: Format mastery missing in Bat vs Team!"
assert "Australia Bowlers Faced Most Frequently" in html, "FAIL: Bowlers faced missing in Bat vs Team!"
assert html.count('"bdata"') == 0, "FAIL: Plotly binary encoding bug present in Bat vs Team!"
print("MODE 2: Batter vs Team          : PASS ✓  (Format mastery, Bowlers faced, 0 bdata)")

# ── 3. Mode 3: Bowler vs Team (JJ Bumrah vs Australia) ────────────────────────
res = client.get("/matchup?mode=bowl_team&bowler=JJ+Bumrah&team=Australia")
html = res.data.decode("utf-8")
assert res.status_code == 200, f"Bowler vs Team route failed: {res.status_code}"
assert "Format Mastery against Australia" in html, "FAIL: Format mastery missing in Bowl vs Team!"
assert "Favorite Victims from Australia" in html, "FAIL: Favorite victims missing in Bowl vs Team!"
assert "Dismissal Modes against Australia" in html, "FAIL: Dismissal modes missing in Bowl vs Team!"
assert html.count('"bdata"') == 0, "FAIL: Plotly binary encoding bug present in Bowl vs Team!"
print("MODE 3: Bowler vs Team          : PASS ✓  (Format mastery, Victims, Dismissals, 0 bdata)")

# ── 4. Empty State Fallback ───────────────────────────────────────────────────
res = client.get("/matchup?mode=h2h&batter=NonExistentPlayer123&bowler=NonExistentPlayer456")
html = res.data.decode("utf-8")
assert res.status_code == 200
assert "No confrontation found" in html, "FAIL: Empty confrontation message missing!"
print("EMPTY STATE FALLBACK            : PASS ✓  (Clean graceful empty message)")

print("\n=== ALL MATCHUP INTELLIGENCE TESTS PASSED SUCCESSFULLY ===")