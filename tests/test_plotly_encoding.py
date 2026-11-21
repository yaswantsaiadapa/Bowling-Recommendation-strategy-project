import os, sys
from pathlib import Path
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
"""Verify Plotly renders real numeric values, not bdata binary encoding."""
import sys

from app import app

client = app.test_client()

# Check ODI response for V Kohli - should have real SR numbers, not 'bdata'
res = client.get("/batting?player=V+Kohli&match_type=ODI")
html = res.data.decode("utf-8")

# bdata is Plotly's binary encoding - bad sign if present in chart sections
bdata_count = html.count('"bdata"')
print(f"'bdata' occurrences in ODI HTML: {bdata_count}")
if bdata_count == 0:
    print("PASS ✓  No binary encoding - Plotly values are plain numbers")
else:
    print(f"WARNING: {bdata_count} bdata blocks found - binary encoding still present")

# Check that real strike rate numbers appear (Kohli ODI SR ~93)
# Just verify there are actual float numbers in the chart data
import re
# Look for numbers like 93.45 in the response
floats_in_html = re.findall(r'\b\d{2,3}\.\d{1,2}\b', html)
sr_range = [f for f in floats_in_html if 30.0 <= float(f) <= 200.0]
print(f"Strike-rate-range floats found in HTML: {sr_range[:10]}")
if sr_range:
    print("PASS ✓  Real float values present in Plotly chart data")
else:
    print("WARNING: No float values in SR range found")