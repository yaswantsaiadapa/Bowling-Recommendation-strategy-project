import os, sys
from pathlib import Path
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
"""Find where the remaining bdata block lives."""
import sys
from app import app
client = app.test_client()
res = client.get("/batting?player=V+Kohli&match_type=ODI")
html = res.data.decode("utf-8")

idx = html.find('"bdata"')
if idx >= 0:
    # Print surrounding context
    snippet = html[max(0, idx-200):idx+200]
    print("Context around bdata:")
    print(snippet)