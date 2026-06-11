from dataclasses import dataclass
from typing import Dict, List, Set
import pandas as pd


@dataclass
class AppContext:
    fixtures: pd.DataFrame
    fixtures_next: pd.DataFrame
    params_now: dict
    params_as_of: dict
    predictions: pd.DataFrame
    predictions_next: pd.DataFrame
    TEAMS: List[str]
    TEAMS_LOOKUP: Dict[str, str]
    null_guess_probability: float
    market_suppliers: Set[str]
