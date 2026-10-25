"""ScoringContract adapters. ContractV1 wraps the frozen v1 serving contract (contracts/v1)."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from decision.types import Assessment
from serving import contract_v1 as c1


class ContractV1:
    version = "v1"

    def __init__(self, contract_dir: str, booster=None, *, verify: bool = True):
        self.manifest = c1.load_manifest(contract_dir)
        self.booster = booster or c1.load_model(contract_dir)
        if verify:
            c1.verify_startup(contract_dir, self.booster)  # refuse to run a model that isn't the frozen one
        self.cutoff = float(self.manifest["cutoff"])

    def assess(self, token_id: str, bars: pd.DataFrame, as_of: datetime, *, launch_time: Any,
               launch_venue: str, observed_from_launch: bool) -> Assessment:
        row, why = c1.select_row(bars, as_of, launch_time=launch_time, launch_venue=launch_venue,
                                 observed_from_launch=observed_from_launch)
        if row is None:
            return Assessment(token_id, as_of, False, why)
        X = c1.to_model_frame([row], self.manifest["categories"])
        score = float(self.booster.predict(X)[0])
        return Assessment(token_id, as_of, True, why, score, float(row["price_at_t"]),
                          {k: row.get(k) for k in c1.FEATURE_COLUMNS})

    def resolve(self, bars, entry_time, entry_price, *, launch_venue, observed_until):
        return c1.resolve_outcome(bars, {"assessment_time": entry_time, "price_at_t": entry_price},
                                  launch_venue=launch_venue, observed_until=observed_until)

    def round_trip_cost(self, launch_venue: str) -> float:
        return c1.round_trip_cost_v1(launch_venue, True)
