import os, sys
from pathlib import Path
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
"""
Verification and regression test suite for Bowling Analytics.
Tests:
- Route response codes
- Format-specific conditional chart rendering
- Autocomplete API
- Zero Plotly binary encoding ("bdata")
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import app

client = app.test_client()

# ── 1. Test Format ───────────────────────────────────────────────────────────
res = client.get("/bowling?player=JJ+Bumrah&match_type=Test")
html = res.data.decode("utf-8")
assert res.status_code == 200, f"Test route failed: {res.status_code}"
assert "Test Innings Progression" in html, "FAIL: Test Innings Progression missing in Test!"
assert "Phase-Wise Impact Profile" not in html, "FAIL: Phase chart visible in Test!"
assert "Restricting vs Defending Target" not in html, "FAIL: Defending chart visible in Test!"
assert "Wicket Anatomy" in html, "FAIL: Dismissal anatomy missing in Test!"
print("TEST format       : PASS ✓  (Innings Progression shown; Phase + Defending hidden)")

# ── 2. All Formats ────────────────────────────────────────────────────────────
res = client.get("/bowling?player=JJ+Bumrah&match_type=All")
html = res.data.decode("utf-8")
assert res.status_code == 200, f"All formats failed: {res.status_code}"
assert "Phase-Wise Impact Profile" not in html, "FAIL: Phase chart visible in All Formats!"
assert "Restricting vs Defending Target" not in html, "FAIL: Defending chart visible in All Formats!"
assert "Opponent Team Mastery" in html, "FAIL: Opponent mastery missing in All Formats!"
assert "Wicket Anatomy" in html, "FAIL: Dismissal anatomy missing in All Formats!"
assert "Venue Mastery" in html, "FAIL: Venue mastery missing in All Formats!"
print("ALL FORMATS format : PASS ✓  (Phase + Defending hidden; Universal charts shown)")

# ── 3. ODI Format ─────────────────────────────────────────────────────────────
res = client.get("/bowling?player=JJ+Bumrah&match_type=ODI")
html = res.data.decode("utf-8")
assert res.status_code == 200, f"ODI failed: {res.status_code}"
assert "Phase-Wise Impact Profile" in html, "FAIL: Phase chart missing in ODI!"
assert "Restricting vs Defending Target" in html, "FAIL: Defending chart missing in ODI!"
print("ODI format        : PASS ✓  (Phase Impact + Defending Target both shown)")

# ── 4. T20I Format ────────────────────────────────────────────────────────────
res = client.get("/bowling?player=JJ+Bumrah&match_type=T20I")
html = res.data.decode("utf-8")
assert res.status_code == 200, f"T20I failed: {res.status_code}"
assert "Phase-Wise Impact Profile" in html, "FAIL: Phase chart missing in T20I!"
assert "Restricting vs Defending Target" in html, "FAIL: Defending chart missing in T20I!"
print("T20I format       : PASS ✓  (Phase Impact + Defending Target both shown)")

# ── 5. Search Autocomplete API ────────────────────────────────────────────────
res = client.get("/api/search_bowlers?q=Bumrah")
assert res.status_code == 200
data = res.get_json()
assert isinstance(data, list) and len(data) > 0, "FAIL: Autocomplete returned empty list!"
assert any("Bumrah" in name for name in data), f"FAIL: JJ Bumrah not in search results: {data}"
print(f"AUTOCOMPLETE API  : PASS ✓  (Found: {data})")

# ── 6. Binary Encoding Bug Check ─────────────────────────────────────────────
bdata_count = html.count('"bdata"')
assert bdata_count == 0, f"FAIL: {bdata_count} 'bdata' blocks found! Plotly binary encoding bug present."
print("PLOTLY ENCODING   : PASS ✓  (Zero 'bdata' blocks found, charts use pure native numbers)")

print("\n=== ALL BOWLING ANALYTICS TESTS PASSED SUCCESSFULLY ===")