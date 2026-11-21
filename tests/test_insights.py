import os, sys
from pathlib import Path
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
"""
Comprehensive Automated Verification Test Suite for Global Cricket Insights & Historical Evolution.
Tests:
1. Status code 200 across default ('All Formats'), 'Test', 'ODI', and 'T20I'.
2. Format-pure metrics:
   - Test mode: Presence of 4 distinct innings lines (Innings 1 through 4), Draw rate decline, 4th innings targets.
   - Complete absence of limited-overs phase terms ('Powerplay', 'Death') in Test mode.
   - ODI/T20I mode: Presence of target chasing brackets.
3. Zero dew or day/night speculation.
4. Zero Plotly binary encoding ('bdata') bugs across all responses.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_insights_all_formats(client):
    """Test the default All Formats view."""
    res = client.get("/insights")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    # Assert title and KPI indicators
    assert "Global Cricket Insights &amp; Historical Evolution" in html or "Global Cricket Insights & Historical Evolution" in html
    assert "Matches Analyzed" in html
    assert "Boundary Run Share" in html
    assert "Format Comparison Matrix" in html

    # Assert zero binary encoding bug
    assert '"bdata":' not in html


def test_insights_test_format(client):
    """Test Test cricket format view: 4 innings lines, draw decline, zero powerplay."""
    res = client.get("/insights?format=Test")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    # Assert 4 innings lines
    assert "1st Innings" in html
    assert "2nd Innings" in html
    assert "3rd Innings" in html
    assert "4th Innings" in html

    # Assert Test Draw and Target pursuit analytics
    assert "Draw Rate Decline" in html or "Historical Test Match Draw Rate" in html
    assert "4th Innings Target Pursuits" in html

    # Assert Top Performers in Test
    assert "Top Batters" in html
    assert "Top Bowlers" in html

    # Assert absence of limited-overs concepts in Test
    assert "Powerplay" not in html
    assert "Death" not in html
    assert "dew" not in html.lower()
    assert "day/night" not in html.lower()

    # Zero bdata
    assert '"bdata":' not in html


def test_insights_t20i_format(client):
    """Test T20I format view: Target chasing brackets, scoring evolution."""
    res = client.get("/insights?format=T20I")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    # Assert T20I chasing brackets
    assert "T20I Target Chasing Success by Score Bracket" in html
    assert "&lt;140" in html or "<140" in html or "< 140" in html

    # Assert Top Performers in T20I
    assert "Top Batters" in html
    assert "Top Bowlers" in html

    # Zero bdata
    assert '"bdata":' not in html


def test_insights_odi_format(client):
    """Test ODI format view: Target chasing brackets, scoring evolution."""
    res = client.get("/insights?format=ODI")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    # Assert ODI chasing brackets
    assert "ODI Target Chasing Success by Score Bracket" in html
    assert "&lt;200" in html or "<200" in html or "< 200" in html

    # Assert Top Performers in ODI
    assert "Top Batters" in html
    assert "Top Bowlers" in html

    # Zero bdata
    assert '"bdata":' not in html


def test_insights_filter_team_and_venue(client):
    """Test filtered query with team and venue parameters."""
    res = client.get("/insights?format=ODI&team=India&venue=Wankhede+Stadium&year_min=2010&year_max=2024")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    assert "Matches Analyzed" in html
    assert '"bdata":' not in html


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])