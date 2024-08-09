from copy import deepcopy
from typing import Optional, Tuple

import boost
import math_utils
import numpy as np
import pandas as pd

################################################################################
########## Global aggregation functions (pandas helpers) #######################
################################################################################


def global_network_state_calculation(
    services_df: pd.DataFrame, network_macro: dict, params: dict
) -> Tuple[pd.DataFrame, dict]:
    """
    Calculates parameters that provide a global view of the network, updating both services_df and network_macro.

    The results of these values are used by various functions & modules throughout.

    This function can be thought of as a "Global ETL" that needs to be computed for all claims across
    all services in the session before we continue.
    """
    # Total compute units used by the network across all claims in a session
    network_macro["total_cus"] = (services_df["relays"] * services_df["cu_per_relay"]).sum()

    # Calculate computed units per node for this service; avg per node; used for normalization and exploitability
    services_df["cu_per_node"] = services_df["relays"] * services_df["cu_per_relay"] / services_df["active_nodes"]

    # Calculate the Gateway Fee Per Compute Unit
    # Assumption: POKT_value is a governance param until it's on service. Otherwise GFPCU is
    # The reason GFPCU != CUTTM is to enable deflation
    network_macro["GFPCU"] = params["core_TLM"]["cu_cost"] / network_macro["POKT_value"]

    # Calculate the Compute Unit To Token Multiplier.
    # It uses the "supply_change"parameter to achieve supply attrition or growth
    network_macro["CUTTM"] = network_macro["GFPCU"] * params["core_TLM"]["supply_change"]

    return services_df, network_macro


def core_TLM_mint(services_df: pd.DataFrame, network_macro: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Calculates the total amount of POKT to be minted by the core TLM.

    The MINT value calculated here IS NOT the actual minted, its the amount that
    would be minted if there are no penalties or normalizations applied later.

    The BURN calculated here is the actual value, since no normalization or
    correction is applied to burning.
    """

    # Implementation note: This can pretty much be done per claim.

    # Calculate the same, but per-service
    services_df["core_TLM_mint"] = network_macro["CUTTM"] * services_df["cu_per_node"] * services_df["active_nodes"]
    services_df["core_TLM_burnt"] = network_macro["GFPCU"] * services_df["cu_per_node"] * services_df["active_nodes"]

    # Assign per-actor
    for actor in network_macro["mint_share"]:
        services_df[f"mint_{actor}"] = network_macro["mint_share"][actor] * services_df["core_TLM_mint"]

    return services_df, network_macro


################################################################################
# ----- PANDAS PROCESS FUNCTION -----
################################################################################

# This function captures the full mechanics of the model implementation
# it is not really a part of the model, it is in fact an auxiliary function to
# enable model comparison and re-execution of the model for static tests.


def process(
    services_df: pd.DataFrame,
    network_macro: dict,
    global_params_dict: dict,
) -> Tuple[pd.DataFrame, dict]:
    """
    Calculates all parameters and minting for a given services_df and
    configuration using Token Logic Modules (TLMs) model.

    Parameters:
        services_df          : A pandas DataFrame with all the network activity data.
        network_macro      : A dictionary containing global data of the network, like total supply, POKT price, etc.
        global_params_dict : The parameters needed to run the TLM model.
    Returns:
        services_df          : A pandas DataFrame with all the previous data plus some additional data calculated by the model.
        global_params_dict : A dictionary with tokenomics data from the static execution of this model.
    """

    # Get a hard copy of the original data
    services_df = deepcopy(services_df)
    network_macro = deepcopy(network_macro)

    # Empty result struct
    tlm_results = dict()
    tlm_results["services"] = dict()
    tlm_results["total_mint"] = 0
    tlm_results["total_burn"] = 0
    tlm_results["total_mint_dao"] = 0
    tlm_results["total_mint_proposer"] = 0
    tlm_results["total_mint_supplier"] = 0
    tlm_results["total_mint_source"] = 0

    # Calculate global network state
    services_df, network_macro = global_network_state_calculation(services_df, network_macro, global_params_dict)

    # Get core_TLM_mint and total burn
    services_df, network_macro = core_TLM_mint(services_df, network_macro)

    # Execute all TLM boosts
    for boost_tlm in global_params_dict["boost_TLM"]:
        for condition in boost_tlm.conditions:
            if (network_macro[condition.metric] < condition.low_threshold) or (
                network_macro[condition.metric] > condition.high_threshold
            ):
                continue

        # Get this boost budget
        boost_mint = boost_tlm.minting_func(services_df, network_macro, boost_tlm.parameters)
        assert boost_mint.sum() > 0
        # Assign to actor
        services_df[f"mint_{boost_tlm.actor}"] += boost_mint

    return services_df, tlm_results


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
global_params_dict["boost_TLM"] = [boost.dao_boost, boost.proposer_boost, boost.supplier_boost, boost.source_boost_1]
