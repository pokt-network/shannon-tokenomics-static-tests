from copy import deepcopy
from typing import Tuple

import numpy as np
import pandas as pd

# This model is based on the final version of Shane's spreadsheet:
# https://docs.google.com/spreadsheets/d/1-_G8VCQ7WbytNmps_N6LhQJatPN3CvV6z2_e0bN9cC0/edit

################################################################################
# ----- GLOBAL ----- Things that are calculated for the whole network data
################################################################################


def get_CUs(compute_cost, relays):
    return compute_cost * relays


def get_relay_cost_usd(compute_cost, cu_cost):
    return compute_cost * cu_cost


def get_burned_POKT(compute_cost, relays, fee):
    return compute_cost * relays * fee


################################################################################
# ----- Suppliers Boost -----
################################################################################


def get_usd_target_mint(cu_cost, deflation_threshold, daily_CUs, daily_relays):
    """
    Calculates the amount of USD-equivalent to mint to reach equilibrium (no boost needed)
    """
    avg_cu_per_relay = daily_CUs / daily_relays
    return deflation_threshold * avg_cu_per_relay * cu_cost


def get_mint_per_day(max_mint_per_day, usd_target_mint, POKT_value, total_burned):
    """
    Calculate how much POKT we need to mint (globally):
    - if the total burned is greater than the POKT equivalent of USD to mint, return zero
    - if the total to mint in excess of the burn amount is greater than the hard cap of max_mint_per_day, return max_mint_per_day
    - else, return the total_pokt_left_to_mint
    """
    pokt_target_mint = usd_target_mint / POKT_value  # POKT
    total_pokt_left_to_mint = pokt_target_mint - total_burned  # POKT

    return np.min([max_mint_per_day, np.max([0, total_pokt_left_to_mint])])


def get_base_CUTTM(service_CUs, service_compute_cost, daily_CUs, mint_per_day):
    """
    Calculates the proportion of all that is going to be minted (in excess)
    that is assigned to this service, given the ratio of CUs that it processed
    compared to the rest of the services
    """
    # return ((service_CUs/daily_CUs)*mint_per_day) / (service_CUs/service_compute_cost) # Spreadsheet
    return mint_per_day * service_compute_cost / daily_CUs


def get_suppliers_supplier_boost(service_CUs, service_compute_cost, service_CUTTM, mint_share):
    """
    Computes how much of the total minted rewards (in excess) belong to the
    servicer.
    """
    share_suppliers_here = 1 - (mint_share["DAO"] + mint_share["Validator"])
    relays = service_CUs / service_compute_cost  # same as using ['relays']
    return service_CUTTM * relays * share_suppliers_here


def get_DAO_supplier_boost(service_CUs, service_compute_cost, service_CUTTM, mint_share):
    """
    Computes how much of the total minted rewards (in excess) belong to the
    DAO.
    """
    relays = service_CUs / service_compute_cost  # same as using ['relays']
    return service_CUTTM * relays * mint_share["DAO"]


def get_proposers_supplier_boost(service_CUs, service_compute_cost, service_CUTTM, mint_share):
    """
    Computes how much of the total minted rewards (in excess) belong to the
    Validator.
    """
    relays = service_CUs / service_compute_cost  # same as using ['relays']
    return service_CUTTM * relays * mint_share["Validator"]


################################################################################
# ----- Sources Boost -----
################################################################################
def get_prop_of_gfpr_supplier_boost(daily_relays, prop_of_CUF, floor_relays, cutoff_point, prop_ceiling):
    """
    *** I dont really understand this ***

    Calculates a proportion (percentage in the spreadsheet) used to reduce the source reward.
    Not really how to explain what it is but the function describes a piecewise linear function.
    """

    if daily_relays < floor_relays:
        return prop_of_CUF
    else:
        if daily_relays > cutoff_point:
            return prop_ceiling
        else:
            return prop_of_CUF - (prop_of_CUF - prop_ceiling) * (daily_relays - floor_relays) / (
                cutoff_point - floor_relays
            )


def get_mint_burn_reward_supplier(service_CUs, POKT_value, cu_cost, mint_share):
    """
    *** I dont really understand this ***

    Calculates the total POKT that is given to sources from the burning=minting mechanism
    """
    computed_usd = service_CUs * cu_cost  # total USD computed
    computed_usd = mint_share["Source"] * computed_usd  # Share of computed usd value
    return computed_usd / POKT_value  # Share of computed POKT value


# def get_boost_reward_supplier(data_df, POKT_value, cu_cost, mint_share):
#     computed_usd = data_df['Total CUs']*cu_cost # total USD computed
#     computed_usd = mint_share['Supplier']*computed_usd # Share of computed usd value
#     return computed_usd/POKT_value # Share of computed POKT value


def get_sources_prop_of_their_burn_from_CUF(
    relays, gfpr_rar_ceiling_relays, gfpr_rar_prop_ceiling, gfpr_rar_floor_relays, gfpr_rar_prop_floor
):
    """
    *** I dont really understand this ***

    Calculates using a piecewise linear function, how much from the total source reward comes from burning as opposed to boost.
    """
    if relays >= gfpr_rar_ceiling_relays:
        return gfpr_rar_prop_ceiling
    else:
        if relays <= gfpr_rar_floor_relays:
            return gfpr_rar_prop_floor
        else:
            # A linear equation
            return gfpr_rar_prop_ceiling + (gfpr_rar_prop_floor - gfpr_rar_prop_ceiling) * (
                relays - gfpr_rar_ceiling_relays
            ) / (gfpr_rar_floor_relays - gfpr_rar_ceiling_relays)


def get_uncapped_reward_sources_boost(
    service_CUs, prop_of_their_burn_from_CUF, cu_cost, prop_gfpr_suppliers, POKT_value
):
    """
    *** I don't really understand this ***

    Calculates how much POKT is going to be minted in excess to this supplier given the level of boost
    and the total income already received from burn/mint
    """
    total_usd = service_CUs * cu_cost  # total USD from CUs
    proportion_usd = total_usd * prop_of_their_burn_from_CUF
    proportion_usd = (
        proportion_usd * prop_gfpr_suppliers
    )  # Additional proportion? this further reduce the income of suppliers
    return proportion_usd / POKT_value


def get_reward_sources_boost(uncapped_boost, total_boost, gfpr_rar_max_supplier_boost):
    """
    Adds a cap to the total POKT that is going to be minted due to the suppliers boost.
    """
    if total_boost > gfpr_rar_max_supplier_boost:
        # Return the proportional, to keep boost fixed
        return (uncapped_boost / total_boost) * gfpr_rar_max_supplier_boost
    else:
        return uncapped_boost


################################################################################
# ----- PANDAS PROCESS FUNCTION -----
################################################################################
# This function captures the full mechanics of the model implementation
# it is not really a part of the model, it is in fact an auxiliary function to
# enable model comparison and re-execution of the model for static tests.


def process(data_df: pd.DataFrame, network_macro: dict, params: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Calculates all parameters and minting for a given chains DataFrame and
    configuration using Shane's model.

    Parameters:
        data_df : A pandas DataFrame with all the network activity data.
        network_macro : A dictionary containing global data of the network, like
                        total supply, POKT price, etc.
        params : The parameters needed to run Shane's model.
    Returns:
        data_df : A pandas DataFrame with all the previous data plus some
                  additional data calculated by the model.
        result_dict : A dictionary with tokenomics data from the static
                      execution of this model.
    """

    # Empty result struct
    result_dict = dict()
    result_dict["Chains"] = dict()
    result_dict["total_mint"] = 0
    result_dict["total_burn"] = 0
    # result_dict['total_mint_base'] = 0
    # result_dict['total_mint_others'] = 0
    result_dict["total_mint_dao"] = 0
    result_dict["total_mint_proposer"] = 0
    result_dict["total_mint_supplier"] = 0
    result_dict["total_mint_source"] = 0

    # Get a hard copy of the original data
    data_df = deepcopy(data_df)

    # Gateway fee per CU
    GFPCU = network_macro["cu_cost"] / network_macro["POKT_value"]

    # Calculate total CUs for each service
    data_df["Total CUs"] = get_CUs(data_df["cu_per_relay"], data_df["relays"])

    # Calculate gateway cost per relay for each service
    data_df["Gw Cost per Relay USD"] = get_relay_cost_usd(data_df["cu_per_relay"], network_macro["cu_cost"])

    # Total burnt for each service
    data_df["Total burn POKT"] = get_burned_POKT(data_df["cu_per_relay"], data_df["relays"], GFPCU)

    # total burned (and equally minted)
    total_burned = data_df["Total burn POKT"].sum()

    # total CUs processed
    daily_CUs = data_df["Total CUs"].sum()

    # Target mint in USD
    usd_target_mint = get_usd_target_mint(
        network_macro["cu_cost"], params["deflation_threshold"], daily_CUs, network_macro["daily_relays"]
    )

    # Calculate excess POKT to mint (excess from burn)
    mint_per_day = get_mint_per_day(
        params["max_mint_per_day"], usd_target_mint, network_macro["POKT_value"], total_burned
    )

    # Calculate the CUTTM for each service
    data_df["Base CUTTM"] = get_base_CUTTM(data_df["Total CUs"], data_df["cu_per_relay"], daily_CUs, mint_per_day)

    # Get the excess POKT minted to each actor
    data_df["Suppliers"] = get_suppliers_supplier_boost(
        data_df["Total CUs"], data_df["cu_per_relay"], data_df["Base CUTTM"], network_macro["mint_share"]
    )
    data_df["DAO"] = get_DAO_supplier_boost(
        data_df["Total CUs"], data_df["cu_per_relay"], data_df["Base CUTTM"], network_macro["mint_share"]
    )
    data_df["Validators"] = get_proposers_supplier_boost(
        data_df["Total CUs"], data_df["cu_per_relay"], data_df["Base CUTTM"], network_macro["mint_share"]
    )

    # Get some proportion of rewards, no really know how this works, but this is the main switch to turn on-off the boost as a function of relays.
    prop_gfpr_suppliers = get_prop_of_gfpr_supplier_boost(
        network_macro["daily_relays"],
        params["prop_of_CUF"],
        params["floor_relays"],
        params["cutoff_point"],
        params["prop_ceiling"],
    )
    # Get all sources rewards form burning/minting parity
    data_df["Mint Burn Reward Supplier"] = get_mint_burn_reward_supplier(
        data_df["Total CUs"],
        network_macro["POKT_value"],
        network_macro["cu_cost"],
        network_macro["mint_share"],
    )
    # Get proportion ofrewards from total taht comes from burning/minting
    data_df["Sources proportion of their burn from CUF"] = data_df["relays"].apply(
        lambda x: get_sources_prop_of_their_burn_from_CUF(
            x,
            params["gfpr_rar_ceiling_relays"],
            params["gfpr_rar_prop_ceiling"],
            params["gfpr_rar_floor_relays"],
            params["gfpr_rar_prop_floor"],
        )
    )
    # Calcualte the (uncapped) boost to mint for each Supplier
    data_df["uncapped_reward_sources_boost"] = get_uncapped_reward_sources_boost(
        data_df["Total CUs"],
        data_df["Sources proportion of their burn from CUF"],
        network_macro["cu_cost"],
        prop_gfpr_suppliers,
        network_macro["POKT_value"],
    )
    # Get the capped rewards
    data_df["reward_sources_boost"] = get_reward_sources_boost(
        data_df["uncapped_reward_sources_boost"],
        data_df["uncapped_reward_sources_boost"].sum(),
        params["gfpr_rar_max_supplier_boost"],
    )

    ##### Complete the results entry
    for idx, row in data_df.iterrows():

        if row["relays"] > 0:

            # Results entry for this service
            result_dict["Chains"][row["Chain"]] = dict()

            # How much we burnt here
            result_dict["Chains"][row["Chain"]]["burn_total"] = GFPCU * row["relays"] * row["cu_per_relay"]
            # How much we minted here due to work
            result_dict["Chains"][row["Chain"]]["mint_base"] = (
                row["Suppliers"] + row["DAO"] + row["Validators"] + result_dict["Chains"][row["Chain"]]["burn_total"]
            )
            # How much extra we mint for sources
            result_dict["Chains"][row["Chain"]]["mint_boost_sources"] = row["reward_sources_boost"]
            # Total mint
            result_dict["Chains"][row["Chain"]]["mint_total"] = (
                result_dict["Chains"][row["Chain"]]["mint_base"]
                + result_dict["Chains"][row["Chain"]]["mint_boost_sources"]
            )
            # Calculate the minting per node in this service
            result_dict["Chains"][row["Chain"]]["mint_per_node"] = (
                result_dict["Chains"][row["Chain"]]["mint_base"] / row["active_nodes"]
            )
            # Calculate the imbalance
            result_dict["Chains"][row["Chain"]]["service_imbalance"] = (row["Total CUs"] / row["active_nodes"]) / (
                daily_CUs / network_macro["supplier_nodes"]
            )

            # Add to the global accumulators (all services)
            result_dict["total_mint"] += result_dict["Chains"][row["Chain"]]["mint_total"]
            result_dict["total_burn"] += result_dict["Chains"][row["Chain"]]["burn_total"]
            # result_dict['total_mint_base'] += result_dict['services'][row['Chain']]['mint_base']
            # result_dict['total_mint_others'] += result_dict['services'][row['Chain']]['mint_boost_sources']
            result_dict["total_mint_dao"] += (
                network_macro["mint_share"]["DAO"] * result_dict["Chains"][row["Chain"]]["burn_total"] + row["DAO"]
            )
            result_dict["total_mint_proposer"] += (
                network_macro["mint_share"]["Validator"] * result_dict["Chains"][row["Chain"]]["burn_total"]
                + row["Validators"]
            )
            result_dict["total_mint_supplier"] += (
                network_macro["mint_share"]["Supplier"] * result_dict["Chains"][row["Chain"]]["burn_total"]
                + row["Suppliers"]
            )
            result_dict["total_mint_source"] += (
                network_macro["mint_share"]["Source"] * result_dict["Chains"][row["Chain"]]["burn_total"]
                + row["reward_sources_boost"]
            )

    return data_df, result_dict
