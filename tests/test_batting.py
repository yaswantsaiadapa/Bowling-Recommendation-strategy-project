import os, sys
from pathlib import Path
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
"""Quick regression test for batting analytics chart visibility rules."""
import sys

from app import app

client = app.test_client()

# ── Test format ──────────────────────────────────────────────────────────────
res = client.get("/batting?player=V+Kohli&match_type=Test")
html = res.data.decode("utf-8")
assert res.status_code == 200, f"Status {res.status_code}"
assert "1st Batting (Setting Target)" not in html, "FAIL: Chase plot visible in Test!"
assert "Phase-Wise Attack Profile"    not in html, "FAIL: Phase plot visible in Test!"
assert "Test Innings Progression" in html, "FAIL: Innings heading missing in Test!"
print("TEST format  : PASS ✓  (Innings plot shown; Phase+Chase hidden)")

# ── All Formats ───────────────────────────────────────────────────────────────
res = client.get("/batting?player=V+Kohli&match_type=All")
html = res.data.decode("utf-8")
assert res.status_code == 200, f"Status {res.status_code}"
assert "1st Batting (Setting Target)" not in html, "FAIL: Chase plot visible in All Formats!"
assert "Phase-Wise Attack Profile"    not in html, "FAIL: Phase plot visible in All Formats!"
assert "Opponent Dominance"            in  html,    "FAIL: Team dominance missing in All Formats!"
print("ALL FORMATS  : PASS ✓  (Phase+Chase hidden; Opponent Dominance shown)")

# ── ODI ───────────────────────────────────────────────────────────────────────
res = client.get("/batting?player=V+Kohli&match_type=ODI")
html = res.data.decode("utf-8")
assert res.status_code == 200, f"Status {res.status_code}"
assert "Phase-Wise Attack Profile"       in html, "FAIL: Phase plot missing in ODI!"
assert "1st Batting (Setting Target)"    in html, "FAIL: Chase plot missing in ODI!"
print("ODI format   : PASS ✓  (Phase+Chase both shown)")

# ── T20I ──────────────────────────────────────────────────────────────────────
res = client.get("/batting?player=V+Kohli&match_type=T20I")
html = res.data.decode("utf-8")
assert res.status_code == 200, f"Status {res.status_code}"
assert "Phase-Wise Attack Profile"       in html, "FAIL: Phase plot missing in T20I!"
assert "1st Batting (Setting Target)"    in html, "FAIL: Chase plot missing in T20I!"
print("T20I format  : PASS ✓  (Phase+Chase both shown)")

print()
print("=== ALL ASSERTIONS PASSED ===")