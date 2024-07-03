import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Tuple

data_dict_defaults = {}
# Ecosystem Costs and Charges
data_dict_defaults["MaturityComputeUnitsCharge"]                 = 1.7e-6               # USD/ComputeUnits
data_dict_defaults["MaturityComputeUnitsCost"]                   = 1.5e-6               # USD/ComputeUnits
data_dict_defaults["SupplyGrowCap"]                       = 0.05                        # times

# Servicers
data_dict_defaults["MaxBootstrapServicerCostPerComputeUnits"]    = 5e-6                 # USD/ComputeUnits
data_dict_defaults["ServicersBootstrapUnwindStart"]       = 1.5                         # Billon ComputeUnits/day
data_dict_defaults["ServicersBootstrapEnd"]               = 10                          # Billon ComputeUnits/day
data_dict_defaults["MinUsdMint"] = 15e3 # USD/day

# Gateways
data_dict_defaults["MinBootstrapGatewayFeePerComputeUnits"]      = 0.00000085           # USD/ComputeUnits
data_dict_defaults["GatewaysBootstrapUnwindStart"]        = 2.5                         # Billon ComputeUnits/day
data_dict_defaults["GatewaysBootstrapEnd"]                = 20                          # Billon ComputeUnits/day


def calc_cuttm_cap(r, GFPCU, grow_target, current_supply):
    '''
    Calculates the maximum value that the CUTTM can take given the supply growth 
    limit that was selected.
    '''
    return (grow_target*current_supply)/(r*365.2*1e9) + GFPCU

    
################################################################################
# ----- Linear and Non-Linear Functions ----- 
################################################################################
# These are functions that implement linear or non-linear functions 
# They calculate parameters given equalities and also produce values given parameter
# They also implement caps on output values. Nothing fancy really.

def get_lin_params(p1,p2):
    '''
    Solves the system:
        p1[1] = a*p1[0]+b
        p2[1] = a*p2[0]+b
    '''
    
    a = (p1[1]-p2[1])/(p1[0]-p2[0])  
    b = p2[1] - a*p2[0]  

    return a,b

def calc_lin_param(daily_relays, a, b, bootstrap_end, bootstrap_start=0, max_param = [], min_param = []):
    '''
    Applies a piece-wise linear function given the parameters and the limits.
    '''
    daily_relays = np.clip(daily_relays,a_max=bootstrap_end,a_min=bootstrap_start)
    param = (daily_relays*a + b) 
    if len(max_param) > 0:
        param = np.clip(param,a_max=max_param,a_min=None)
    if len(min_param) > 0:
        param = np.clip(param,a_max=None,a_min=min_param)
    return param



def get_non_lin_params(p1,p2):
    '''
    Solves the system:
        p1[1] = a*p1[0]+b/p1[0] = a + b/p1[0]
        p2[1] = a*p2[0]+b/p2[0] = a + b/p2[0]

        p1[1] = a + b/p1[0]
        a = p1[1] - b/p1[0]
        p2[1] = p1[1] - b/p1[0] + b/p2[0]
        b = p1[0]*p2[0]*(p2[1]-p1[1]) / (p1[0]-p2[0])
    '''
    
    b = p1[0]*p2[0]*(p2[1]-p1[1]) / (p1[0]-p2[0])
    a = p1[1] - b/p1[0]

    return a,b

def calc_non_lin_param(daily_relays, a, b, bootstrap_end, bootstrap_start=0, max_param = [], min_param = []):
    daily_relays = np.clip(daily_relays,a_max=bootstrap_end,a_min=bootstrap_start)

    param = (daily_relays*a + b) / daily_relays
    
    if len(max_param) > 0:
        param = np.clip(param,a_max=max_param,a_min=None)
    if len(min_param) > 0:
        param = np.clip(param,a_max=None,a_min=min_param)
    return param


################################################################################
# ----- Entropy Correction ----- 
################################################################################
# This is a little more fancy, here we use the concept of entropy to normalize
# the distribution of nodes over services.
# The main advantage, I think, is its simplicity for calculation (a single parameter)
# and the possibility of shaping the distribution of nodes per services (maybe we 
# don't want it to be uniform)

def get_computed_units_by_node_distribution(computed_units_by_service, node_by_service, regularization_mask = None):
    '''
    Computes the distribution of CUs along the nodes in the network.

    Modifying this distribution with a custom mask will provide normalization for harder chains.
    If a chain is harder and we want to accept a higher number of compute units per node, then we just divide
    the corresponding bin by a factor, then the entropy will think it is under-provisioned and keep giving it bonus
    util we reach the expected value.
    '''
    assert len(node_by_service) == len(computed_units_by_service)
    if regularization_mask != None:
        assert len(regularization_mask) == len(computed_units_by_service)
    else:
        regularization_mask = np.ones_like(node_by_service)
        
    r_by_c = computed_units_by_service/node_by_service
    r_by_c *= regularization_mask
    return (r_by_c/np.sum(r_by_c))


def calculate_entropy_correction_values(dist_to_norm):
    '''
    Calculates the entropy values for all services given the distance to 
    uniform distribution per service.
    '''
    entropy_bin = -(dist_to_norm*np.log2(dist_to_norm))
    num_bins = dist_to_norm.shape[0]
    max_entropy = np.sum(entropy_bin) 
    return entropy_bin*num_bins/max_entropy, max_entropy


def limit_compute_by_node(compute_node_chain, global_compute_node_average, difficulty_factor=1.0):
    '''
    We want to limit the rewards to under-provisioned chains and let the entropy factor be the only
    source of "boost". If we do not do this, the under-provisioned chains get too much rewards
    and they are the easier to game.

    TODO : just as in "get_computed_units_by_node_distribution", a mask can be used here to normalize.
    '''
    return (compute_node_chain if global_compute_node_average > compute_node_chain else global_compute_node_average)*difficulty_factor

def cap_cuttm_per_node(adjusted_CUTTM, base_CUTTM, cap_mult = 5):
    return adjusted_CUTTM if adjusted_CUTTM < cap_mult*base_CUTTM else cap_mult*base_CUTTM

################################################################################
# ----- PANDAS PROCESS FUNCTION ----- 
################################################################################
# This function captures the full mechanics of the model implementation
# it is not really a part of the model, it is in fact an auxiliary function to
# enable model comparison and re-execution of the model for static tests.

def process(data_df: pd.DataFrame, network_macro: dict, params: dict) -> Tuple[pd.DataFrame, dict]:
    '''
    Calculates all parameters and minting for a given chains DataFrame and 
    configuration using MINT-V2 model.

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
    '''

    # Empty result struct
    result_dict = dict()
    result_dict['Chains'] = dict()
    result_dict['Total_Mint'] = 0
    result_dict['Total_Burn'] = 0
    result_dict['Total_Mint_base'] = 0
    result_dict['Total_Mint_others'] = 0
    

    # Get a hard copy of the original data
    data_df = deepcopy(data_df)

    # Total computed units
    computed_units = (data_df['Relays']*data_df['Compute Cost']).sum()

    # Get the cost in POKT for the given exchange rate fo each parameter
    target_pokt_compute_unit_charge = params['MaturityComputeUnitsCharge']/network_macro['POKT_value']
    target_pokt_compute_unit_cost = params['MaturityComputeUnitsCost']/network_macro['POKT_value']
    current_pokt_compute_unit_charge = params['MinBootstrapGatewayFeePerComputeUnits']/network_macro['POKT_value']
    CUTTM = params['MaxBootstrapServicerCostPerComputeUnits']/network_macro['POKT_value']
    current_pokt_compute_unit_cost = CUTTM
    current_pokt_compute_unit_cost_USD = current_pokt_compute_unit_cost*network_macro['POKT_value']

    # Calcualte linear functions

    # Servicer bootstrap
    a_serv, b_serv = get_non_lin_params([params['ServicersBootstrapUnwindStart'], current_pokt_compute_unit_cost],
                                    [params['ServicersBootstrapEnd'], target_pokt_compute_unit_cost])
    
    # Gateway bootstrap
    a_gate, b_gate = get_lin_params([params['GatewaysBootstrapUnwindStart'], current_pokt_compute_unit_charge],
                                    [params['GatewaysBootstrapEnd'], target_pokt_compute_unit_charge])
    

    # Get the GFPCU
    GFPCU_base = calc_lin_param([computed_units/1e9], 
                                  a_gate, 
                                  b_gate, 
                                  params['GatewaysBootstrapEnd'], 
                                  bootstrap_start=params['GatewaysBootstrapUnwindStart'],
                                  min_param=[current_pokt_compute_unit_charge])
    
    # Calculate the maximum allowed value for the CUTTM given the supply growth limit
    CUTTM_cap = calc_cuttm_cap(computed_units/1e9, GFPCU_base, params['SupplyGrowCap'], network_macro['total_supply'])

    # Calculate the CUTTM
    CUTTM_base=calc_non_lin_param([computed_units/1e9], 
                                  a_serv, 
                                  b_serv, 
                                  params['ServicersBootstrapEnd'], 
                                  bootstrap_start=params['ServicersBootstrapUnwindStart'],
                                  max_param=CUTTM_cap)
    
    # Check the minimum minting given the USD value
    to_mint_usd = CUTTM_base*computed_units*network_macro['POKT_value']
    if to_mint_usd < params["MinUsdMint"]:
        # If we are going to mint less than the USM minimum, compute the CUTTM for minting the USD minimum
        CUTTM_base = [params["MinUsdMint"]/(computed_units*network_macro['POKT_value'])]
        # Clip again, in case we go avobe the maximum CUTTM
        CUTTM_base = np.clip(CUTTM_base,a_max=CUTTM_cap,a_min=None)

    # (the module is created to allow matrix calculation)
    GFPCU_use = GFPCU_base[0]
    CUTTM_use = CUTTM_base[0]


    # --------------------------------------------------------------------------
    # Entropy Normalization on Nodes Per Service
    # --------------------------------------------------------------------------

    if params.get('Apply Entropy', True):
        # Calculate entropy of the network
        computed_units_by_chain = data_df['Relays'].values  *  data_df['Compute Cost'].values
        node_by_chain = data_df['Active Nodes'].values
        relays_node_by_chain_norm = get_computed_units_by_node_distribution(computed_units_by_chain, node_by_chain)
        # Calculate the per-service entropy correction values
        entropy_correction, max_entropy = calculate_entropy_correction_values(relays_node_by_chain_norm)
        data_df['Entropy Correction'] = entropy_correction
    # Global compute unit per node average
    global_compute_node_average = (data_df['Relays']  *  data_df['Compute Cost']).sum()/data_df['Active Nodes'].sum()

            
    # --------------------------------------------------------------------------
    # Calculate each service in the network
    # --------------------------------------------------------------------------

    # Calculate computed units per node for this service
    data_df['CU per Node'] = data_df['Relays']*data_df['Compute Cost']/data_df['Active Nodes']

    # Calculate Adjusted CUTTM on each node
    if params.get('Apply Entropy', True):
        # Limit this value to the global average
        data_df['CU per Node Capped'] = data_df['CU per Node'].apply(lambda x: limit_compute_by_node(x, global_compute_node_average))
        # Adjust the CUTTM using the entropy correction value for this service
        data_df['Adjusted CUTTM'] = CUTTM_use*data_df['Entropy Correction'] 
        # Set a hard cap for the CUTTM, it is not enforced normally but it is not a bad idea to have.
        data_df['Adjusted CUTTM'] = data_df['Adjusted CUTTM'].apply(lambda x: cap_cuttm_per_node(x, CUTTM_use))
    else:
        data_df['CU per Node Capped'] = data_df['CU per Node']
        data_df['Adjusted CUTTM'] = CUTTM_use

    # Calculate total minted in each service
    data_df['total minted chain'] = data_df['Adjusted CUTTM']*data_df['CU per Node Capped']*data_df['Active Nodes']
        
    for idx, row in data_df.iterrows():

        if row['Relays'] > 0:

            # Results entry for this service
            result_dict['Chains'][row['Chain']] = dict()

            ##### Complete the results entry
            # How much we minted here due to work
            result_dict['Chains'][row['Chain']]['mint_base'] = row['total minted chain']
            # How much we burnt here
            result_dict['Chains'][row['Chain']]['burn_total'] = GFPCU_use*row['Relays']*row['Compute Cost']
            # How much extra we mint for sources
            result_dict['Chains'][row['Chain']]['mint_boost_sources'] = 0 # TODO: I have not created this yet, not sure which are the requirements.
            # Total mint
            result_dict['Chains'][row['Chain']]['mint_total'] = row['total minted chain'] +result_dict['Chains'][row['Chain']]['mint_boost_sources']
            # Calculate the minting per node in this service
            result_dict['Chains'][row['Chain']]['mint_per_node'] = row['total minted chain']/row['Active Nodes']
            # Calculate the imbalance
            result_dict['Chains'][row['Chain']]['service_imbalance'] = row['CU per Node']/global_compute_node_average

            # Add to the global accumulators (all services)
            result_dict['Total_Mint'] += result_dict['Chains'][row['Chain']]['mint_total']
            result_dict['Total_Burn'] += result_dict['Chains'][row['Chain']]['burn_total']
            result_dict['Total_Mint_base'] += result_dict['Chains'][row['Chain']]['mint_base']
            result_dict['Total_Mint_others'] += result_dict['Chains'][row['Chain']]['mint_boost_sources']
            
    return data_df, result_dict

    
