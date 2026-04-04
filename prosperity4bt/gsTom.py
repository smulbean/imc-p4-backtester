import itertools
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


# =========================================================
# CONFIG — MATCHED TO YOUR SETUP
# =========================================================

BASE_STRATEGY_FILE = "prosperity4bt/tomatoe_trader.py"
BACKTEST_CMD = ["prosperity4bt", "{strategy}", "0"]

RESULTS_JSON = "grid_search_results.json"
RESULTS_CSV = "grid_search_results.csv"

MAX_TESTS = 250


# =========================================================
# PARAM GRID — FOR YOUR CURRENT BIGGER-QUOTE STYLE
# These names must exist as class constants in TomatoesTrader
# =========================================================

PARAM_GRID = {
    "TAKE_EDGE": [1.0, 1.25, 1.5],
    "BASE_SIZE": [12, 14, 16, 18],
    "STRONG_SIZE": [24, 28, 32, 36],
    "SOFT_POS": [40, 45, 50],
    "HARD_POS": [75, 78],
    "INV_SKEW_PER_UNIT": [0.015, 0.018, 0.022],
    "MAX_SKEW": [4.0, 5.0, 6.0],
    "MAX_MAKE_EDGE": [0.75, 1.0, 1.25],
    "SPREAD_EDGE_MULT": [0.05, 0.08, 0.12],
}

INT_PARAMS = {"BASE_SIZE", "STRONG_SIZE", "SOFT_POS", "HARD_POS"}

PNL_WEIGHT = 1.0
SMOOTHNESS_WEIGHT = 0.15   # reward similar pnl across days
DOWNSIDE_WEIGHT = 0.20     # penalize weak worst day


# =========================================================
# DATA STRUCTURE
# =========================================================

@dataclass
class RunResult:
    params: Dict[str, float]
    day_profits: Dict[str, float]
    total_profit: float
    smoothness: float
    downside_penalty: float
    score: float


# =========================================================
# FILE HELPERS
# =========================================================

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# =========================================================
# PATCHING CLASS CONSTANTS
# =========================================================

def replace_class_constant(src: str, class_name: str, const_name: str, value) -> str:
    class_pattern = rf"(class\s+{re.escape(class_name)}\s*\(.*?\):\s*)(