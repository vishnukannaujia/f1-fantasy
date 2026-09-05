"""
Shared parsers for the structured (tabular) source files -- driver and constructor
prices. Used by both ingest.py (to build row-level chunks for the vector store) and
team_builder.py (to get exact prices for deterministic budget validation), so the
line-format regex lives in exactly one place.
"""

import re
from pathlib import Path

DRIVER_LINE_RE = re.compile(r"^(?P<driver>.+?)\s*\((?P<team>[^)]+)\):\s*\$(?P<price>[\d.]+)M$")
CONSTRUCTOR_LINE_RE = re.compile(r"^(?P<team>.+?):\s*\$(?P<price>[\d.]+)M$")


def parse_driver_prices(path) -> dict:
    """{"Lewis Hamilton": {"team": "Ferrari", "price": 25.1}, ...}"""
    prices = {}
    for line in Path(path).read_text().splitlines():
        m = DRIVER_LINE_RE.match(line.strip())
        if m:
            prices[m.group("driver")] = {"team": m.group("team"), "price": float(m.group("price"))}
    return prices


def parse_constructor_prices(path) -> dict:
    """{"Ferrari": 26.9, ...}"""
    prices = {}
    for line in Path(path).read_text().splitlines():
        m = CONSTRUCTOR_LINE_RE.match(line.strip())
        if m:
            prices[m.group("team")] = float(m.group("price"))
    return prices
