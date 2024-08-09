import marimo

__generated_with = "0.7.12"
app = marimo.App(width="full")


@app.cell
def __():
    import sys
    from copy import deepcopy

    import numpy as np
    import pandas as pd

    sys.path.append("../lib")
    import data_utils
    import token_logic_modules as tlm

    # defaults
    relay_multiplier = 1  # A multiplier to scale the # of relays per chain for experimentation
    base_CU_per_relay = 100  # CUs/Relay; the default if not service specific
    return (
        base_CU_per_relay,
        data_utils,
        deepcopy,
        np,
        pd,
        relay_multiplier,
        sys,
        tlm,
    )


@app.cell
def __(base_CU_per_relay, data_utils, relay_multiplier):
    # Download the "Performance" card from https://poktscan.com/explore?tab=chains
    CSV_FILE = "../data/chains_performance_2024-05-20_SAMPLE.csv"

    # Read chains data
    services_df = data_utils.get_services_df(
        CSV_FILE,
        relay_multiplier=relay_multiplier,
        spreadsheet_compact=True,
    )

    # Add compute cost column
    services_df = data_utils.add_compute_cost(services_df, sample_type="uniform", mean=base_CU_per_relay)
    services_df
    return CSV_FILE, services_df


@app.cell
def __(data_utils, tlm):
    data_utils.render_tree(tlm.global_params_dict)
    return


@app.cell
def __(base_CU_per_relay, data_utils, relay_multiplier, services_df):
    # Network macro - inputs to the token logic module procedures

    # These values captures *exogenous* network conditions
    network_macro = dict()
    network_macro["POKT_value"] = 0.50  # USD/POKT in off-chain marketplaces
    network_macro["POKT_stake_per_node"] = 60e3  # POKT required to stake a supplier; NOT NEEDED FOR MODELS
    network_macro["total_supply"] = 1679901301.43  # POKT tokens in circulation; NOT NEEDED FOR MODELS

    # These values are *global aggregated* views of *exogenous* network conditions
    # TODO: Replace daily_relays with session_relays -> total number of relays across all claims in a session.
    network_macro["daily_relays"] = services_df["relays"].sum()  # Total # of relays on the network
    network_macro["supplier_nodes"] = services_df["active_nodes"].sum()  # Total # of suppliers on the network

    # These values are additional parameters to scale the simulation
    network_macro["relay_multiplier"] = relay_multiplier  # Scales the # of relays per chain for experimentation

    # These values capture *global* on-chain parameters
    network_macro["base_CU_per_relay"] = base_CU_per_relay  # CUs/Relay; the default if not service specific
    network_macro["cu_cost"] = 0.0000000085  # USD/CU; an off-chain decision based on CU definition
    network_macro["mint_share"] = dict()
    network_macro["mint_share"]["DAO"] = 0.1
    network_macro["mint_share"]["Validator"] = 0.05
    network_macro["mint_share"]["Supplier"] = 0.7
    network_macro["mint_share"]["Source"] = 0.15
    assert sum(network_macro["mint_share"].values()) == 1, "The mint share must sum 1 (100%)"

    data_utils.render_tree(network_macro)
    return network_macro,


@app.cell
def __(deepcopy, network_macro, services_df, tlm):
    tlm_params = deepcopy(tlm.global_params_dict)
    tlm_params["core_TLM"]["cu_cost"] = network_macro["cu_cost"]
    tlm_params["apply_entropy"] = False
    tlm_per_service_df, tlm_results = tlm.process(services_df, network_macro, tlm_params)
    tlm_per_service_df
    return tlm_params, tlm_per_service_df, tlm_results


@app.cell
def __(data_utils, tlm_results):
    data_utils.render_tree(tlm_results)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
