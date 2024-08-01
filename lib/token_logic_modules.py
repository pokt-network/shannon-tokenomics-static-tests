from copy import deepcopy
from typing import Optional, Tuple

import math_utils
import numpy as np
import pandas as pd

# TODO: Add typing to all the helper functions


################################################################################
######################## Entropy Correction ####################################
################################################################################

# Helpers functions used to apply entropy in order to normalize the distribution of nodes over services.
# The main advantage, I think, is its simplicity for calculation (a single parameter)
# and the possibility of shaping the distribution of nodes per services (maybe we
# don't want it to be uniform)


def get_compute_units_by_node_distribution(
    compute_units_by_service: np.ndarray, nodes_by_service: np.ndarray, regularization_mask=Optional[np.ndarray]
):
    """
    Compute the distribution of compute units along the nodes in the network.

    Modifying this distribution with a custom mask will provide normalization for harder chains.
    If a chain is harder and we want to accept a higher number of compute units per node, then we just divide
    the corresponding bin by a factor, then the entropy will think it is under-provisioned and keep giving it bonus
    util we reach the expected value.
    """
    assert len(nodes_by_service) == len(compute_units_by_service)

    if regularization_mask != None:
        assert len(regularization_mask) == len(compute_units_by_service)
    else:
        regularization_mask = np.ones_like(nodes_by_service)

    r_by_c = compute_units_by_service / nodes_by_service
    r_by_c *= regularization_mask
    return r_by_c / np.sum(r_by_c)


def calculate_entropy_correction_values(dist_to_norm):
    """
    Calculates the entropy values for all services given the distance to
    uniform distribution per service.
    """
    entropy_bin = -(dist_to_norm * np.log2(dist_to_norm))
    num_bins = dist_to_norm.shape[0]
    max_entropy = np.sum(entropy_bin)
    return entropy_bin * num_bins / max_entropy, max_entropy


def limit_compute_by_node(compute_node_chain, global_compute_node_average, difficulty_factor=1.0):
    """
    We want to limit the rewards to under-provisioned chains and let the entropy factor be the only
    source of "boost". If we do not do this, the under-provisioned chains get too much rewards
    and they are the easier to game.

    TODO : just as in "get_compute_units_by_node_distribution", a mask can be used here to normalize.
    """
    return (
        compute_node_chain if global_compute_node_average > compute_node_chain else global_compute_node_average
    ) * difficulty_factor


def cap_cuttm_per_node(adjusted_CUTTM, base_CUTTM, cap_mult=5):
    return adjusted_CUTTM if adjusted_CUTTM < cap_mult * base_CUTTM else cap_mult * base_CUTTM


################################################################################
########## Global aggregation functions (pandas helpers) #######################
################################################################################


def network_state_calculation(
    chains_df: pd.DataFrame, network_macro: dict, apply_entropy: bool
) -> Tuple[pd.DataFrame, dict]:
    """
    Calculates parameters that provide a global view of the network.

    The results of these values are used by various functions & modules throughout.
    """

    # Global compute unit per node average
    global_compute_node_average = (chains_df["relays"] * chains_df["cu_per_relay"]).sum() / chains_df[
        "active_nodes"
    ].sum()

    # Total compute units used by the network
    network_macro["total_cus"] = (chains_df["relays"] * chains_df["cu_per_relay"]).sum()

    # Calculate computed units per node for this service
    chains_df["cu_per_node"] = chains_df["relays"] * chains_df["cu_per_relay"] / chains_df["active_nodes"]

    # --------------------------------------------------------------------------
    # Entropy Normalization on Nodes Per Service
    # --------------------------------------------------------------------------

    # Calculate Adjusted CUTTM on each node
    if apply_entropy:
        # Calculate entropy of the network
        compute_units_by_chain = chains_df["relays"].values * chains_df["cu_per_relay"].values
        nodes_by_chain = chains_df["active_nodes"].values
        relays_node_by_chain_norm = get_compute_units_by_node_distribution(compute_units_by_chain, nodes_by_chain)
        # Calculate the per-service entropy correction values
        entropy_correction, max_entropy = calculate_entropy_correction_values(relays_node_by_chain_norm)
        chains_df["entropy_correction"] = entropy_correction

        # Limit this value to the global average
        chains_df["cu_per_node_capped"] = chains_df["cu_per_node"].apply(
            lambda x: limit_compute_by_node(x, global_compute_node_average)
        )
        # Entropy correction normalization multiplier
        chains_df["normalization_correction"] = chains_df["entropy_correction"].apply(
            lambda x: cap_cuttm_per_node(x, 1)
        )
        # Average CU correction
        chains_df["normalization_correction"] *= chains_df["cu_per_node_capped"] / chains_df["cu_per_node"]
    else:
        chains_df["cu_per_node_capped"] = chains_df["cu_per_node"]
        chains_df["normalization_correction"] = 1.0

    return chains_df, network_macro


def core_TLM_budget(tlm_per_service_df: pd.DataFrame, network_macro: dict, params: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Calculates the total amount of POKT to be minted by the core TLM.

    The mint value calculated here is not the actual minted, its the amount that
    would be minted if there are no penalties or normalizations applied later.

    The burn calculated here is the actual value, since no normalization or
    correction is applied to burning.
    """

    # Calculate the Gateway Fee Per Compute Unit
    network_macro["GFPCU"] = params["core_TLM"]["cu_cost"] / network_macro["POKT_value"]
    # Calculate the Compute Unit To Token Multiplier, it uses the "supply" change parameter to achieve supply attrition or growth
    network_macro["CUTTM"] = network_macro["GFPCU"] * params["core_TLM"]["supply_change"]

    # This is the maximum amount of tokens to mint due to the core module.
    # (The entropy correction mechanism will not increase this amount)
    network_macro["core TLM mint budget"] = dict()
    network_macro["core TLM mint budget"]["total"] = network_macro["CUTTM"] * network_macro["total_cus"]
    # Assign per-actor
    for key in network_macro["mint_share"]:
        network_macro["core TLM mint budget"][key] = (
            network_macro["CUTTM"] * network_macro["total_cus"] * network_macro["mint_share"][key]
        )

    # This is the total to be burned
    network_macro["core TLM burn"] = network_macro["GFPCU"] * network_macro["total_cus"]

    # Calculate the same, but per-service
    tlm_per_service_df["core TLM budget"] = (
        network_macro["CUTTM"] * tlm_per_service_df["cu_per_node"] * tlm_per_service_df["active_nodes"]
    )
    tlm_per_service_df["core TLM burned"] = (
        network_macro["GFPCU"] * tlm_per_service_df["cu_per_node"] * tlm_per_service_df["active_nodes"]
    )
    # Assign per-actor
    for key in network_macro["mint_share"]:
        tlm_per_service_df["budget %s" % key] = network_macro["mint_share"][key] * tlm_per_service_df["core TLM budget"]

    return tlm_per_service_df, network_macro


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
    assert params["variables"]["x"] == "total_cus"
    assert params["variables"]["y"] == "CUTTM"

    # Calculate the non-linear parameters of this boost
    a_param, b_param = math_utils.get_non_linear_params(
        [params["start"]["x"], params["start"]["y"]],
        [params["end"]["x"], params["end"]["y"]],
    )
    # Calculate the parameter cap
    param_cap = None
    if params["budget"]["type"] == "annual_supply_growth":
        # Calculate annualized growth
        param_cap = [
            ((params["budget"]["value"] / 100.0) * network_macro["total_supply"]) / (network_macro["total_cus"] * 365.2)
        ]
    else:
        raise ValueError('Budget type "%s" not supported' % params["budget"]["type"])
    # Calculate the parameter to use
    param_use = math_utils.calc_non_linear_param(
        [network_macro[params["variables"]["x"]]],
        a_param,
        b_param,
        params["end"]["x"],
        bootstrap_start=params["start"]["x"],
        max_param=param_cap,
    )

    # Calculate (maximum) total minted in each service
    # The same as the core TLM (or any TLM) this value will be potentially reduced
    return param_use * tlm_per_service_df["cu_per_node"] * tlm_per_service_df["active_nodes"]


def boost_prop_f_CUs_sources_custom(tlm_per_service_df: pd.DataFrame, network_macro: dict, params: dict) -> pd.Series:
    """
    This boost is a proportional cuttm boost on top of sources boost.
    It is intended to reflect the additional per-service boost that is applied
    in the spreadsheet as the "sources boost" made by Shane.
    """

    assert params["variables"]["x"] == "total_cus"

    # The modulation of the parameter is linear
    a_param, b_param = math_utils.get_linear_params(
        [params["start"]["x"], params["start"]["y"]],
        [params["end"]["x"], params["end"]["y"]],
    )
    max_mint = -1
    if params["budget"]["type"] == "annual_supply_growth":
        # Calculate annualized growth
        max_mint = ((params["budget"]["value"] / 100.0) * network_macro["total_supply"]) / (365.2)
    elif params["budget"]["type"] == "POKT":
        max_mint = params["budget"]["value"]
    else:
        raise ValueError('Budget type "%s" not supported' % params["budget"]["type"])
    param_use = math_utils.calc_linear_param(
        [network_macro[params["variables"]["x"]]],
        a_param,
        b_param,
        params["end"]["x"],
        bootstrap_start=params["start"]["x"],
    )

    # Calculate (maximum) total minted in each service
    per_service_max = (
        param_use * network_macro["CUTTM"] * tlm_per_service_df["cu_per_node"] * tlm_per_service_df["active_nodes"]
    )
    # Apply budget
    if max_mint > 0:
        if max_mint < per_service_max.sum():
            # Scale values
            per_service_max *= max_mint / per_service_max.sum()
    # Return amount to mint in each service by this boost
    return per_service_max


def apply_global_limits_and_minimums(
    tlm_per_service_df: pd.DataFrame, network_macro: dict, params: dict
) -> Tuple[pd.DataFrame, dict]:
    """
    This function implement any minting limits or minimums that need to be
    applied after all TLMs were calculated.
    This function intends to enforce a cap on global supply growth and also to
    ensure minimum minting for each network actor.
    More things can be implemented here if they need to have access to the
    result of all TLMs minting.

    Order is important, first we apply minimum minting and then the global
    supply growth cap.
    """

    ############################################################################
    # Minimum Minting
    ############################################################################
    # For each actor check if minimum minting is OK or if we need to scale it
    for key in network_macro["mint_share"]:
        # Get data if exists
        min_mint = params["boundaries"]["min_mint"].get(key, None)
        if min_mint is not None:
            # Calculate the total budget here
            total_actor_budget = tlm_per_service_df["budget %s" % key].sum()

            if min_mint["type"] == "annual_supply_growth":
                min_budget = ((min_mint["value"] / 100.0) * network_macro["total_supply"]) / (365.2)
            elif min_mint["type"] == "USD":
                min_budget = min_mint["value"] / network_macro["POKT_value"]
            else:
                raise ValueError('Budget type "%s" not supported' % min_mint["type"])

            # Check against minimum
            if total_actor_budget < min_budget:
                # Calculate scaling
                scale = min_budget / total_actor_budget
                # Apply to column
                tlm_per_service_df["budget %s" % key] *= scale

    ############################################################################
    # Maximum Minting  / Supply Growth
    ############################################################################
    # Check for Global minting limits
    max_mint_params = params["boundaries"]["max_mint"].get("network", None)
    if max_mint_params is not None:

        if max_mint_params["type"] == "annual_supply_growth":
            max_budget = ((max_mint_params["value"] / 100.0) * network_macro["total_supply"]) / (365.2)
        else:
            raise ValueError('Budget type "%s" not supported' % max_mint_params["type"])

        # Get total minted, burn and growth
        total_burn = tlm_per_service_df["core TLM burned"].sum()
        total_mint = 0
        for key in network_macro["mint_share"]:
            total_mint += tlm_per_service_df["budget %s" % key].sum()
        supply_growth = total_mint - total_burn

        # Check against max
        if supply_growth > max_budget:

            # Scale
            scale = (max_budget + total_burn) / total_mint
            for key in network_macro["mint_share"]:
                tlm_per_service_df["budget %s" % key] *= scale

    return tlm_per_service_df, network_macro


def apply_penalties_and_normalizations(
    tlm_per_service_df: pd.DataFrame, network_macro: dict, params: dict
) -> Tuple[pd.DataFrame, dict]:
    """
    This function is the last step and defines the per-service minting.
    Here we take the calculated budgets for each service and apply the normalization
    correction factors.
    For example, the normalization correction factor is the entropy correction and
    the average CU per service cap.
    """

    # Calculate total minted in each service
    tlm_per_service_df["core TLM minted"] = (
        tlm_per_service_df["core TLM budget"] * tlm_per_service_df["normalization_correction"]
    )
    # Calculate minted for each service using the budgets
    for key in network_macro["mint_share"]:
        tlm_per_service_df["total TLM minted %s" % key] = (
            tlm_per_service_df["budget %s" % key] * tlm_per_service_df["normalization_correction"]
        )

    return tlm_per_service_df, network_macro


################################################################################
# ----- PANDAS PROCESS FUNCTION -----
################################################################################

# This function captures the full mechanics of the model implementation
# it is not really a part of the model, it is in fact an auxiliary function to
# enable model comparison and re-execution of the model for static tests.


def process(
    chains_df: pd.DataFrame,
    network_macro: dict,
    global_params_dict: dict,
) -> Tuple[pd.DataFrame, dict]:
    """
    Calculates all parameters and minting for a given chains_df and
    configuration using Token Logic Modules (TLMs) model.

    Parameters:
        chains_df          : A pandas DataFrame with all the network activity data.
        network_macro      : A dictionary containing global data of the network, like
                             total supply, POKT price, etc.
        global_params_dict : The parameters needed to run the TLM model.
    Returns:
        chains_df          : A pandas DataFrame with all the previous data plus some
                             additional data calculated by the model.
        global_params_dict : A dictionary with tokenomics data from the static
                             execution of this model.
    """

    # Get a hard copy of the original data
    chains_df = deepcopy(chains_df)
    network_macro = deepcopy(network_macro)

    # Empty result struct
    tlm_results = dict()
    tlm_results["chains"] = dict()
    tlm_results["total_mint"] = 0
    tlm_results["total_burn"] = 0
    tlm_results["total_mint_dao"] = 0
    tlm_results["total_mint_proposer"] = 0
    tlm_results["total_mint_supplier"] = 0
    tlm_results["total_mint_source"] = 0

    # Calculate base network state
    should_apply_entropy = global_params_dict.get("apply_entropy", True)
    chains_df, network_macro = network_state_calculation(chains_df, network_macro, should_apply_entropy)

    # Get core TLM mint budget and total burn
    chains_df, network_macro = core_TLM_budget(chains_df, network_macro, global_params_dict)

    # Create data struct for boosts
    network_macro["boost TLM mint budget"] = dict()
    network_macro["boost TLM mint budget"]["total"] = 0
    for key in network_macro["mint_share"]:
        network_macro["boost TLM mint budget"][key] = 0

    # Get budget from boost TLMs
    # This is the main feature of the TLM model, the simplicity for adding more
    # and arbitrary mechanics
    for boost_tlm in global_params_dict["boost_TLM"]:
        # Check conditions
        skip = False
        for condition in boost_tlm["conditions"]:
            if (network_macro[condition["metric"]] < condition["low_threshold"]) or (
                network_macro[condition["metric"]] > condition["high_threshold"]
            ):
                skip = True

        if not skip:
            # Get this boost budget
            this_budget = boost_tlm["minting_func"](chains_df, network_macro, boost_tlm["parameters"])
            assert this_budget.sum() > 0
            # Assign to actor
            chains_df["budget %s" % boost_tlm["recipient"]] += this_budget
            # Track globals
            network_macro["boost TLM mint budget"][boost_tlm["recipient"]] += this_budget
            network_macro["boost TLM mint budget"]["total"] += this_budget

    # Apply global limits and minimums to minting budget
    chains_df, network_macro = apply_global_limits_and_minimums(chains_df, network_macro, global_params_dict)

    # Apply penalties / normalizations
    chains_df, network_macro = apply_penalties_and_normalizations(chains_df, network_macro, global_params_dict)

    # Global compute unit per node average
    global_compute_node_average = (chains_df["relays"] * chains_df["cu_per_relay"]).sum() / chains_df[
        "active_nodes"
    ].sum()
    # Calculate total minted in each service
    chains_df["total minted chain"] = (
        chains_df["total TLM minted DAO"]
        + chains_df["total TLM minted Validator"]
        + chains_df["total TLM minted Supplier"]
        + chains_df["total TLM minted Source"]
    )

    for _, row in chains_df.iterrows():
        if row["relays"] <= 0:
            continue

        # Results entry for this service
        tlm_results["chains"][row["Chain"]] = dict()

        ##### Complete the results entry
        # How much we minted here due to work
        tlm_results["chains"][row["Chain"]]["mint_base"] = row["core TLM minted"]
        # How much we burnt here
        tlm_results["chains"][row["Chain"]]["burn_total"] = row["core TLM burned"]
        # How much extra we mint for sources
        tlm_results["chains"][row["Chain"]]["mint_boost_sources"] = (
            row["total TLM minted Source"] - network_macro["mint_share"]["Source"] * row["core TLM minted"]
        )
        # Total mint
        tlm_results["chains"][row["Chain"]]["mint_total"] = row["total minted chain"]
        # Calculate the minting per node in this service
        tlm_results["chains"][row["Chain"]]["mint_per_node"] = row["total minted chain"] / row["active_nodes"]
        # Calculate the imbalance
        tlm_results["chains"][row["Chain"]]["service_imbalance"] = row["cu_per_node"] / global_compute_node_average

        # Add to the global accumulators (all services)
        tlm_results["total_mint"] += tlm_results["chains"][row["Chain"]]["mint_total"]
        tlm_results["total_burn"] += tlm_results["chains"][row["Chain"]]["burn_total"]
        tlm_results["total_mint_dao"] += row["total TLM minted DAO"]
        tlm_results["total_mint_proposer"] += row["total TLM minted Validator"]
        tlm_results["total_mint_supplier"] += row["total TLM minted Supplier"]
        tlm_results["total_mint_source"] += row["total TLM minted Source"]

    return chains_df, tlm_results


################################################################################
########################## Configuration Structure #############################
################################################################################

# default_parameters_dict is a big structure that contains:
# 1. Global parameters
# 2. Definitions for the various TLMs
global_params_dict = dict()

############## Global Base Minting & Supply Growth ##############################

# Boundaries control two things:
# 1. The max amount of minting that can take place on the network (USD)
# 2. The min amount of minting that can take place on the network (USD)
#   - Enables not worrying about floating pocket prices
#   - If (2) is removed -> POKT/USD value is not necessary
#   - Potential post-mainnet candidate
#   - Will require changing the parameters of the boosts manually (i.e. like RTTM today)
global_params_dict["boundaries"] = dict()

# Supply Growth Global limit
global_params_dict["boundaries"]["max_mint"] = dict()
global_params_dict["boundaries"]["max_mint"]["network"] = dict()
global_params_dict["boundaries"]["max_mint"]["network"]["type"] = "annual_supply_growth"
global_params_dict["boundaries"]["max_mint"]["network"]["value"] = 5  # [%]

# Base Minting for DAO
global_params_dict["boundaries"]["min_mint"] = dict()
global_params_dict["boundaries"]["min_mint"]["DAO"] = dict()
global_params_dict["boundaries"]["min_mint"]["DAO"]["type"] = "USD"
global_params_dict["boundaries"]["min_mint"]["DAO"]["value"] = 2e3  # USD/day

# Base Minting for Block Proposer
global_params_dict["boundaries"]["min_mint"]["Validator"] = dict()
global_params_dict["boundaries"]["min_mint"]["Validator"]["type"] = "USD"
global_params_dict["boundaries"]["min_mint"]["Validator"]["value"] = 1e3  # USD/day

# Base Minting for Supplier
global_params_dict["boundaries"]["min_mint"]["Supplier"] = dict()
global_params_dict["boundaries"]["min_mint"]["Supplier"]["type"] = "USD"
global_params_dict["boundaries"]["min_mint"]["Supplier"]["value"] = 14e3  # USD/day

# Base Minting for Source owner
global_params_dict["boundaries"]["min_mint"]["Source"] = dict()
global_params_dict["boundaries"]["min_mint"]["Source"]["type"] = "USD"
global_params_dict["boundaries"]["min_mint"]["Source"]["value"] = 3e3  # USD/day

###################### Core TLM (Mint=Burn) ####################################

# core_TLM is the core module, encodes the end-game scenario of the network.
global_params_dict["core_TLM"] = dict()

# This sets the supply change.
#   < 1.0 means supply attrition.
#   > 1.0 means supply growth.
# This is not equal to the target supply change per year (i.e. inflation).
# The actual speed of growth or attrition is related to:
#   1. The amount of relays
#   2. The total supply, so, this is no equal
global_params_dict["core_TLM"]["supply_change"] = 1.0

# This value is the core of the model: the cost of a single compute unit.
# NB: The definition of the compute unit is a core component of the network economy and it will change over time.
global_params_dict["core_TLM"]["cu_cost"] = 0.0000000085  # USD/CU

############################# Boost TLMs #######################################

# These are a collection of modules that perform EXTRA minting. They are all defined by:
# - A recipient actor (DAO, Sources, Validators, Servicers, anything).
# - A condition logic, that dictates when the module is implemented.
# - A minting mechanism, that defines how the extra minting is calculated.
# - A budget, that sets the maximum spending for the module.
global_params_dict["boost_TLM"] = list()

### DAO Boost
aux_boost = dict()
aux_boost["name"] = "DAO Boost - CU Based"
aux_boost["recipient"] = "DAO"  # Who receives the minting
aux_boost["conditions"] = list()  # List of conditions to meet for TLM execution
aux_boost["conditions"].append(
    {
        "metric": "total_cus",  # Total CUs in the network
        "low_threshold": 0,
        "high_threshold": 2500 * 1e9,  # CU/day
    }
)
aux_boost["minting_func"] = (
    boost_cuttm_f_CUs_nonlinear  # A function that accepts as input the services state, the network macro state and the parameters below and return the amount to mint per service
)
aux_boost["parameters"] = {  # A structure containing all parameters needed by this module
    "start": {
        "x": 250 * 1e9,  # ComputeUnits/day
        "y": 5e-9,  # USD/ComputeUnits
    },
    "end": {
        "x": 2500 * 1e9,  # ComputeUnits/day
        "y": 0,  # USD/ComputeUnits
    },
    "variables": {
        "x": "total_cus",  # Control metric for this TLM
        "y": "CUTTM",  # Target parameter for this TLM
    },
    "budget": {  # Can be a fixed number of tokens [POKT] or a percentage of total supply (annualized) [annual_supply_growth]
        "type": "annual_supply_growth",
        "value": 0.5,
    },
}
# Append to boosts list
global_params_dict["boost_TLM"].append(aux_boost)

### Validator Boost
aux_boost = dict()
aux_boost["name"] = "Validator Boost - CU Based"
aux_boost["recipient"] = "Validator"
aux_boost["conditions"] = list()
aux_boost["conditions"].append(
    {"metric": "total_cus", "low_threshold": 0, "high_threshold": 2500 * 1e9}
)  # ComputeUnits/day
aux_boost["minting_func"] = boost_cuttm_f_CUs_nonlinear
aux_boost["parameters"] = {  #
    "start": {
        "x": 250 * 1e9,  # ComputeUnits/day
        "y": 2.5e-9,  # USD/ComputeUnits
    },
    "end": {
        "x": 2500 * 1e9,  # ComputeUnits/day
        "y": 0,  # USD/ComputeUnits
    },
    "variables": {
        "x": "total_cus",  # Control metric for this TLM
        "y": "CUTTM",  # Target parameter for this TLM
    },
    "budget": {"type": "annual_supply_growth", "value": 0.25},
}
global_params_dict["boost_TLM"].append(aux_boost)  # Append to boosts list


### Supplier Boost
aux_boost = dict()
aux_boost["name"] = "Supplier Boost - CU Based"
aux_boost["recipient"] = "Supplier"
aux_boost["metrics"] = ["total_cus"]
aux_boost["conditions"] = list()
aux_boost["conditions"].append({"metric": "total_cus", "low_threshold": 0, "high_threshold": 2500 * 1e9})
aux_boost["minting_func"] = boost_cuttm_f_CUs_nonlinear
aux_boost["parameters"] = {
    "start": {
        "x": 250 * 1e9,  # ComputeUnits/day
        "y": 3.5e-8,  # USD/ComputeUnits
    },
    "end": {
        "x": 2500 * 1e9,  # ComputeUnits/day
        "y": 0,  # USD/ComputeUnits
    },
    "variables": {
        "x": "total_cus",  # Control metric for this TLM
        "y": "CUTTM",  # Target parameter for this TLM
    },
    "budget": {"type": "annual_supply_growth", "value": 3.5},
}
global_params_dict["boost_TLM"].append(aux_boost)  # Append to boosts list


### Source Boost
aux_boost = dict()
aux_boost["name"] = "Sources Boost 1 - CU Based"
aux_boost["recipient"] = "Source"
aux_boost["conditions"] = list()
aux_boost["conditions"].append({"metric": "total_cus", "low_threshold": 0, "high_threshold": 2500 * 1e9})
aux_boost["parameters"] = {
    "start": {
        "x": 250 * 1e9,  # ComputeUnits/day
        "y": 7.5e-9,  # USD/ComputeUnits
    },
    "end": {
        "x": 2500 * 1e9,  # ComputeUnits/day
        "y": 0,  # USD/ComputeUnits
    },
    "variables": {
        "x": "total_cus",  # Control metric for this TLM
        "y": "CUTTM",  # Target parameter for this TLM
    },
    "budget": {"type": "annual_supply_growth", "value": 0.75},
}
aux_boost["minting_func"] = boost_cuttm_f_CUs_nonlinear
global_params_dict["boost_TLM"].append(aux_boost)  # Append to boosts list


### Source Boost 2 (Spreadsheet)
aux_boost = dict()
aux_boost["name"] = "Sources Boost 2 - Shane's"
aux_boost["recipient"] = "Source"
aux_boost["conditions"] = list()
aux_boost["conditions"].append({"metric": "total_cus", "low_threshold": 0, "high_threshold": 1500 * 1e9})
aux_boost["parameters"] = {
    "start": {
        "x": 5 * 1e9,  # ComputeUnits/day
        "y": 0.9 * 0.7,  # Proportion of CUTTF
    },
    "end": {
        "x": 1500 * 1e9,  # ComputeUnits/day
        "y": 0.1 * 0.7,  # Proportion of CUTTF
    },
    "variables": {
        "x": "total_cus",  # Control metric for this TLM
        "y": "prop. CUTTM",  # Target parameter for this TLM
    },
    "budget": {"type": "POKT", "value": 40e3},  # POKT per day
}
aux_boost["minting_func"] = boost_prop_f_CUs_sources_custom
global_params_dict["boost_TLM"].append(aux_boost)  # Append to boosts list
