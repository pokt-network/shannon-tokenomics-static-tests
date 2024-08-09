from dataclasses import dataclass
from typing import Callable, List

import math_utils
import pandas as pd


@dataclass
class Condition:
    # TODO: Make ENUM for metrics
    metric: str  # total_cus, sum_emas_relays, etc...
    low_threshold: float
    high_threshold: float

# Report card: how well you did
# Punch card: how much you did -> EMA of # relays
# -> Low -> shitty job based on off chain (implcit QoS)
# -> high -> good job based on off chain (implicit QoS)


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Variables:
    x: str
    y: str


@dataclass
class Budget:
    type: str
    value: float


@dataclass
class Parameters:
    start: Point
    end: Point
    variables: Variables
    budget: Budget


@dataclass
class TLMBoost:
    name: str
    actor: str  # make this an enum too
    conditions: List[Condition]
    minting_func: Callable
    parameters: Parameters


def boost_cuttm_f_CUs_nonlinear(tlm_per_service_df: pd.DataFrame, network_macro: dict, params: dict) -> pd.Series:
    """
    This is a basic non-linear boost functions that modifies the CUTTM as a
    function of the CUs:
    CUTTM = f(CU)

    This is the result of separating the MINT-V2 mechanisms into stand-alone
    modules.

    Goal of this function: Trying to achieve a boost value for the CUTTM
    Boost should be near zero when in equilibrium.

    """

    # Assert that the TLM config is correct
    assert params.variables.x == "total_cus"
    assert params.variables.y == "CUTTM"

    # Calculate the non-linear parameters of this boost
    a_param, b_param = math_utils.get_non_linear_params(
        [params.start.x, params.start.y],
        [params.end.x, params.end.y],
    )
    # Calculate the parameter cap
    param_cap = None
    if params.budget.type == "annual_supply_growth":
        # Calculate annualized growth
        param_cap = [
            ((params.budget.value / 100.0) * network_macro["total_supply"]) / (network_macro["total_cus"] * 365.2)
        ]
    else:
        raise ValueError('Budget type "%s" not supported' % params.budget.type)
    # Calculate the parameter to use
    param_use = math_utils.calc_non_linear_param(
        [network_macro[params.variables.x]],
        a_param,
        b_param,
        params.end.x,
        bootstrap_start=params.start.x,
        max_param=param_cap,
    )

    # Calculate (maximum) total minted in each service
    # The same as the core TLM (or any TLM) this value will be potentially reduced
    return param_use * tlm_per_service_df["cu_per_node"] * tlm_per_service_df["active_nodes"]


# Instances of TLMBoost with the new Budget dataclass

dao_boost = TLMBoost(
    name="DAO Boost",
    actor="DAO",
    conditions=[Condition(metric="total_cus", low_threshold=0, high_threshold=2500 * 1e9)],
    minting_func=boost_cuttm_f_CUs_nonlinear,
    parameters=Parameters(
        start=Point(x=250 * 1e9, y=5e-9),
        end=Point(x=2500 * 1e9, y=0),
        variables=Variables(x="total_cus", y="CUTTM"),
        budget=Budget(
            type="annual_supply_growth",
            value=0.5,
        ),
    ),
)

proposer_boost = TLMBoost(
    name="Proposer Boost",
    actor="Validator",
    conditions=[Condition(metric="total_cus", low_threshold=0, high_threshold=2500 * 1e9)],
    minting_func=boost_cuttm_f_CUs_nonlinear,
    parameters=Parameters(
        start=Point(x=250 * 1e9, y=2.5e-9),
        end=Point(x=2500 * 1e9, y=0),
        variables=Variables(x="total_cus", y="CUTTM"),
        budget=Budget(
            type="annual_supply_growth",
            value=0.25,
        ),
    ),
)

supplier_boost = TLMBoost(
    name="Supplier Boost",
    actor="Supplier",
    conditions=[Condition(metric="total_cus", low_threshold=0, high_threshold=2500 * 1e9)],
    minting_func=boost_cuttm_f_CUs_nonlinear,
    parameters=Parameters(
        start=Point(x=250 * 1e9, y=3.5e-8),
        end=Point(x=2500 * 1e9, y=0),
        variables=Variables(x="total_cus", y="CUTTM"),
        budget=Budget(
            type="annual_supply_growth",
            value=3.5,
        ),
    ),
)

source_boost_1 = TLMBoost(
    name="Sources Boost 1 - CU Based",
    actor="Source",
    conditions=[Condition(metric="total_cus", low_threshold=0, high_threshold=2500 * 1e9)],
    minting_func=boost_cuttm_f_CUs_nonlinear,
    parameters=Parameters(
        start=Point(x=250 * 1e9, y=7.5e-9),
        end=Point(x=2500 * 1e9, y=0),
        variables=Variables(x="total_cus", y="CUTTM"),
        budget=Budget(
            type="annual_supply_growth",
            value=0.75,
        ),
    ),
)


# def boost_prop_f_CUs_sources_custom(tlm_per_service_df: pd.DataFrame, network_macro: dict, params: dict) -> pd.Series:
#     """
#     This boost is a proportional cuttm boost on top of sources boost.
#     It is intended to reflect the additional per-service boost that is applied
#     in the spreadsheet as the "sources boost" made by Shane.
#     """

#     assert params.variables.x == "total_cus"

#     # The modulation of the parameter is linear
#     a_param, b_param = math_utils.get_linear_params(
#         [params.start.x, params.start.y],
#         [params.end.x, params.end.y],
#     )
#     max_mint = -1
#     if params.budget.type == "annual_supply_growth":
#         # Calculate annualized growth
#         max_mint = ((params.budget.value / 100.0) * network_macro["total_supply"]) / (365.2)
#     elif params.budget.type == "POKT":
#         max_mint = params.budget.value
#     else:
#         raise ValueError('Budget type "%s" not supported' % params.budget.type)
#     param_use = math_utils.calc_linear_param(
#         [network_macro[params.variables.x]],
#         a_param,
#         b_param,
#         params.end.x,
#         bootstrap_start=params.start.x,
#     )

#     # Calculate (maximum) total minted in each service
#     per_service_max = (
#         param_use * network_macro["CUTTM"] * tlm_per_service_df["cu_per_node"] * tlm_per_service_df["active_nodes"]
#     )
#     # Apply budget
#     if max_mint > 0:
#         if max_mint < per_service_max.sum():
#             # Scale values
#             per_service_max *= max_mint / per_service_max.sum()
#     # Return amount to mint in each service by this boost
#     return per_service_max


# source_boost_2 = TLMBoost(
#     name="Sources Boost 2 - Shane's",
#     recipient="Source",
#     conditions=[Condition(metric="total_cus", low_threshold=0, high_threshold=1500 * 1e9)],
#     minting_func=boost_prop_f_CUs_sources_custom,
#     parameters=Parameters(
#         start=Point(x=5 * 1e9, y=0.9 * 0.7),
#         end=Point(x=1500 * 1e9, y=0.1 * 0.7),
#         variables=Variables(x="total_cus", y="CUTTM"),
#         budget=Budget(
#             type="POKT",
#             value=40e3,
#         ),
#     ),
# )
