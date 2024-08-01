from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class Condition:
    metric: str
    low_threshold: float
    high_threshold: float


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Parameters:
    start: Point
    end: Point
    variable_x: str
    variable_y: str
    budget_type: str
    budget_value: float


@dataclass
class TLMBoost:
    name: str
    recipient: str
    conditions: List[Condition]
    minting_func: Callable
    parameters: Parameters


def boost_cuttm_f_CUs_nonlinear(state: Any, network_macro: Any, parameters: Any) -> float:
    # Placeholder function definition
    pass


dao_boost = TLMBoost(
    name="DAO Boost",
    recipient="DAO",
    conditions=[Condition(metric="total_cus", low_threshold=0, high_threshold=2500 * 1e9)],
    minting_func=boost_cuttm_f_CUs_nonlinear,
    parameters=Parameters(
        start=Point(x=250 * 1e9, y=5e-9),
        end=Point(x=2500 * 1e9, y=0),
        variable_x="total_cus",
        variable_y="CUTTM",
        budget_type="annual_supply_growth",
        budget_value=0.5,
    ),
)

proposer_boost = TLMBoost(
    name="Proposer Boost",
    recipient="Validator",
    conditions=[Condition(metric="total_cus", low_threshold=0, high_threshold=2500 * 1e9)],
    minting_func=boost_cuttm_f_CUs_nonlinear,
    parameters=Parameters(
        start=Point(x=250 * 1e9, y=2.5e-9),
        end=Point(x=2500 * 1e9, y=0),
        variable_x="total_cus",
        variable_y="CUTTM",
        budget_type="annual_supply_growth",
        budget_value=0.25,
    ),
)

# Supplier Boost
supplier_boost = TLMBoost(
    name="Supplier Boost",
    recipient="Supplier",
    conditions=[Condition(metric="total_cus", low_threshold=0, high_threshold=2500 * 1e9)],
    minting_func=boost_cuttm_f_CUs_nonlinear,
    parameters=Parameters(
        start=Point(x=250 * 1e9, y=3.5e-8),
        end=Point(x=2500 * 1e9, y=0),
        variable_x="total_cus",
        variable_y="CUTTM",
        budget_type="annual_supply_growth",
        budget_value=3.5,
    ),
)

# Source Boost 1
source_boost_1 = TLMBoost(
    name="Sources Boost 1 - CU Based",
    recipient="Source",
    conditions=[Condition(metric="total_cus", low_threshold=0, high_threshold=2500 * 1e9)],
    minting_func=boost_cuttm_f_CUs_nonlinear,
    parameters=Parameters(
        start=Point(x=250 * 1e9, y=7.5e-9),
        end=Point(x=2500 * 1e9, y=0),
        variable_x="total_cus",
        variable_y="CUTTM",
        budget_type="annual_supply_growth",
        budget_value=0.75,
    ),
)

# Source Boost 2
source_boost_2 = TLMBoost(
    name="Sources Boost 2 - Shane's",
    recipient="Source",
    conditions=[Condition(metric="total_cus", low_threshold=0, high_threshold=1500 * 1e9)],
    minting_func=boost_prop_f_CUs_sources_custom,
    parameters=Parameters(
        start=Point(x=5 * 1e9, y=0.9 * 0.7),
        end=Point(x=1500 * 1e9, y=0.1 * 0.7),
        variable_x="total_cus",
        variable_y="prop. CUTTM",
        budget_type="POKT",
        budget_value=40e3,
    ),
)

# Assuming global_params_dict is defined
global_params_dict = {"boost_TLM": []}
global_params_dict["boost_TLM"].extend([proposer_boost, supplier_boost, source_boost_1, source_boost_2])

# Print the boosts to verify
for boost in global_params_dict["boost_TLM"]:
    print(boost)
