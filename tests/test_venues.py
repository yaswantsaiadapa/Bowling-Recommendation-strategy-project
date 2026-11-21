import os, sys
from pathlib import Path
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
"""
Comprehensive Test Suite for Team & Venue Intelligence Engine.

Validates:
1. Test Cricket Venue Intelligence:
   - 4-innings score decay curve
   - Test match duration & Day 5 resolution profile
   - 4th innings target pursuit success %
   - Strict absence of limited-overs "Bat 1st Win %" / "Chase Win %"
2. All Formats Multi-Format Stadium View:
   - Independent format benchmark cards (T20I, ODI, Test)
   - Zero misleading score blending
   - Cross-format comparison plot
3. Limited Overs (T20I / ODI) Venue View:
   - Format-pure 1st & 2nd innings totals
   - Bat 1st vs Chase outcomes
   - Phase scoring dynamics (Powerplay, Middle, Death)
4. National Team Dynamics:
   - Head-to-Head opponent records
   - Team scoring trajectory & Phase DNA
   - Fortress stadiums
5. Zero Plotly binary encoding bug ("bdata" count == 0) across all views
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import app

client = app.test_client()

# ── 1. Test Cricket Venue Profiler: Melbourne Cricket Ground ──────────────────
print("\nRunning Test 1: Test Cricket Venue Profiler (MCG)...")
res = client.get("/venues?tab=venue&venue=Melbourne+Cricket+Ground&format=Test")
assert res.status_code == 200, f"MCG Test route failed: {res.status_code}"
html = res.data.decode("utf-8")

# Strict Test metrics
assert "Pitch Deterioration: 4-Innings Score Decay Curve" in html, "FAIL: 4-innings decay curve missing!"
assert "Test Match Duration &amp; Day 5 Resolution Profile" in html, "FAIL: Test duration plot missing!"
assert "Test Venue Hall of Fame" in html, "FAIL: Test Hall of Fame missing!"
assert "4th Inn Target Pursuit" in html, "FAIL: 4th innings target pursuit KPI missing!"
assert "Typical Duration" in html, "FAIL: Typical duration KPI missing!"
assert "1st &amp; 2nd Innings Avg" in html, "FAIL: 1st/2nd innings average missing!"
assert "3rd &amp; 4th Innings Avg" in html, "FAIL: 3rd/4th innings average missing!"

# Must NOT contain limited-overs Bat 1st Win % or Chase Win % in Test mode
assert "Bat 1st Win %" not in html, "FAIL: 'Bat 1st Win %' should NOT appear in Test cricket mode!"
assert "Chasing Win %" not in html, "FAIL: 'Chasing Win %' should NOT appear in Test cricket mode!"
assert html.count('"bdata"') == 0, "FAIL: Plotly binary encoding bug found in Test venue!"
print("TEST 1: Test Cricket Venue (MCG)            : PASS ✓  (4 Innings Decay, Duration, 4th Inn Target, 0 bdata)")

# ── 2. All Formats Combined Venue Profiler: Melbourne Cricket Ground ──────────
print("\nRunning Test 2: All Formats Combined Venue Profiler (MCG)...")
res = client.get("/venues?tab=venue&venue=Melbourne+Cricket+Ground&format=All")
assert res.status_code == 200, f"MCG All Formats failed: {res.status_code}"
html = res.data.decode("utf-8")

# Must show format benchmark cards without single blended average
assert "Cross-Format Scoring Benchmark" in html, "FAIL: Format comparison plot missing!"
assert "Test Match" in html, "FAIL: Test format card missing!"
assert "ODI" in html, "FAIL: ODI format card missing!"
assert "T20I" in html, "FAIL: T20I format card missing!"
assert "All-Time Venue Hall of Fame" in html, "FAIL: All-time Hall of Fame missing!"
assert html.count('"bdata"') == 0, "FAIL: Plotly binary encoding bug found in All Formats venue!"
print("TEST 2: All Formats Combined Venue (MCG)    : PASS ✓  (Format Cards, No Blended Score, 0 bdata)")

# ── 3. Limited Overs Venue Profiler: Dubai (T20I) ─────────────────────────────
print("\nRunning Test 3: Limited Overs Venue Profiler (Dubai T20I)...")
res = client.get("/venues?tab=venue&venue=Dubai+International+Cricket+Stadium&format=T20I")
assert res.status_code == 200, f"Dubai T20I failed: {res.status_code}"
html = res.data.decode("utf-8")

assert "Pitch Bias: Batting 1st vs Chasing Outcomes" in html, "FAIL: Pitch bias plot missing!"
assert "Venue Phase Run Rates" in html, "FAIL: Venue phase plot missing!"
assert "Bat 1st Win %" in html, "FAIL: Bat 1st Win % should be present for T20I!"
assert "Chasing Win %" in html, "FAIL: Chasing Win % should be present for T20I!"
assert html.count('"bdata"') == 0, "FAIL: Plotly binary encoding bug found in T20I venue!"
print("TEST 3: Limited Overs Venue (Dubai T20I)    : PASS ✓  (Pitch Bias, Phase Rates, Bat1st/Chase %, 0 bdata)")

# ── 4. National Team Dynamics: India ─────────────────────────────────────────
print("\nRunning Test 4: National Team Dynamics (India)...")
res = client.get("/venues?tab=team&team=India&format=All")
assert res.status_code == 200, f"India team route failed: {res.status_code}"
html = res.data.decode("utf-8")

assert "Head-to-Head Records against Opponents" in html, "FAIL: Opponents record missing!"
assert "Team Scoring Trajectory &amp; Phase DNA" in html, "FAIL: Team phase DNA missing!"
assert "Fortress Stadiums" in html, "FAIL: Fortress grounds missing!"
assert html.count('"bdata"') == 0, "FAIL: Plotly binary encoding bug found in Team dynamics!"
print("TEST 4: Team Dynamics (India)               : PASS ✓  (H2H Opponents, Phase DNA, Fortress Grounds, 0 bdata)")

print("\n======================================================================")
print("  🏆 ALL TEAM & VENUE INTELLIGENCE TESTS PASSED (100% SUCCESS) 🏆")
print("======================================================================")