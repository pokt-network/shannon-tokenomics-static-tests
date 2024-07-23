import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Tuple



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
# ----- GLOBAL FUNCTIONS (for Pandas) ----- 
################################################################################

def network_state_calculation(data_df: pd.DataFrame, network_macro: dict, apply_entropy: bool) -> Tuple[pd.DataFrame, dict]:
    '''
    Calculates parameters according the current state of the network.
    These parameters are then used by other functions or modules, they are not 
    specific for a single one.
    '''
    
    # Total computed units
    network_macro['total_cus'] = (data_df['Relays']*data_df['Compute Cost']).sum()

    # --------------------------------------------------------------------------
    # Entropy Normalization on Nodes Per Service
    # --------------------------------------------------------------------------

    if apply_entropy:
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
    # Calculate the normalization for each service in the network
    # --------------------------------------------------------------------------

    # Calculate computed units per node for this service
    data_df['CU per Node'] = data_df['Relays']*data_df['Compute Cost']/data_df['Active Nodes']

    # Calculate Adjusted CUTTM on each node
    if apply_entropy:
        # Limit this value to the global average
        data_df['CU per Node Capped'] = data_df['CU per Node'].apply(lambda x: limit_compute_by_node(x, global_compute_node_average))
        # Entropy correction normalization multiplier
        data_df['Normalization Correction'] = data_df['Entropy Correction'].apply(lambda x: cap_cuttm_per_node(x, 1))
        # Average CU correction
        data_df['Normalization Correction'] *= data_df['CU per Node Capped']/data_df['CU per Node']
    else:
        data_df['CU per Node Capped'] = data_df['CU per Node']
        data_df['Normalization Correction'] = 1.0

        
    return data_df, network_macro

    

def core_TLM_budget(data_df: pd.DataFrame, network_macro: dict, params: dict) -> Tuple[pd.DataFrame, dict]:
    '''
    Calculates the total amount of POKT to be minted by the core TLM.

    The mint value calculated here is not the actual minted, its the amount that 
    would be minted if there are no penalties or normalizations applied later.

    The burn calculated here is the actual value, since no normalization or
    correction is applied to burning.
    '''

    # Calculate the Gateway Fee Per Compute Unit
    network_macro['GFPCU'] = params['core_TLM']['cu_cost']/network_macro['POKT_value']
    # Calculate the Compute Unit To Token Multiplier, it uses the "supply" change parameter to achieve supply attrition or growth
    network_macro['CUTTM'] = network_macro['GFPCU']*params['core_TLM']['supply_change']

    # This is the maximum amount of tokens to mint due to the core module. 
    # (The entropy correction mechanism will not increase this amount)
    network_macro['core TLM mint budget'] = dict()
    network_macro['core TLM mint budget']['total'] = network_macro['CUTTM']*network_macro['total_cus']
    # Assign per-actor
    for key in network_macro['mint_share']:    
        network_macro['core TLM mint budget'][key] = network_macro['CUTTM']*network_macro['total_cus']*network_macro['mint_share'][key]

    # This is the total to be burned
    network_macro['core TLM burn'] = network_macro['GFPCU']*network_macro['total_cus']

    # Calculate the same, but per-service
    data_df['core TLM budget'] = network_macro['CUTTM']*data_df['CU per Node']*data_df['Active Nodes']
    data_df['core TLM burned'] = network_macro['GFPCU']*data_df['CU per Node']*data_df['Active Nodes']
    # Assign per-actor
    for key in network_macro['mint_share']:
        data_df['budget %s'%key] = network_macro['mint_share'][key]*data_df['core TLM budget']

    return data_df, network_macro



def boost_cuttm_f_CUs_nonlinear(data_df: pd.DataFrame, network_macro: dict, params: dict) -> pd.Series:
    '''
    This is a basic non-linear boost functions that modifies the CUTTM as a 
    function of the CUs:
    CUTTM = f(CU)

    This is the result of separating the MINT-V2 mechanisms into stand-alone
    modules.
    '''

    # Assert that the TLM config is correct
    assert params['variables']['x'] == 'total_cus'
    assert params['variables']['y'] == 'CUTTM'

    # Calculate the non-linear parameters of this boost
    a_param, b_param = get_non_lin_params([params['start']['x'], params['start']['y']],
                                    [params['end']['x'], params['end']['y']])
    # Calculate the parameter cap
    param_cap = None
    if params['budget']['type'] == 'annual_supply_growth':
        # Calculate annualized growth
        param_cap = [((params['budget']['value']/100.)*network_macro['total_supply'])/(network_macro['total_cus']*365.2)]
    else:
        raise ValueError("Budget type \"%s\" not supported"%params['budget']['type'])
    # Calculate the parameter to use
    param_use=calc_non_lin_param([network_macro[params['variables']['x']]], 
                                  a_param, 
                                  b_param, 
                                  params['end']['x'], 
                                  bootstrap_start=params['start']['x'],
                                  max_param=param_cap)
    
    # Calculate (maximum) total minted in each service
    # The same as the core TLM (or any TLM) this value will be potentially reduced
    return param_use*data_df['CU per Node']*data_df['Active Nodes']


def boost_prop_f_CUs_sources_custom(data_df: pd.DataFrame, network_macro: dict, params: dict) -> pd.Series:
    '''
    This boost is a proportional cuttm boost on top of sources boost.
    It is intended to reflect the additional per-service boost that is applied
    in the spreadsheet as the "sources boost" made by Shane.
    '''

    assert params['variables']['x'] == 'total_cus'

    # The modulation of the parameter is linear
    a_param, b_param = get_lin_params([params['start']['x'], params['start']['y']],
                                    [params['end']['x'], params['end']['y']])
    max_mint = -1
    if params['budget']['type'] == 'annual_supply_growth':
        # Calculate annualized growth
        max_mint = ((params['budget']['value']/100.)*network_macro['total_supply'])/(365.2)
    elif params['budget']['type'] == 'POKT':
        max_mint = params['budget']['value']
    else:
        raise ValueError("Budget type \"%s\" not supported"%params['budget']['type'])
    param_use=calc_lin_param([network_macro[params['variables']['x']]], 
                                  a_param, 
                                  b_param, 
                                  params['end']['x'], 
                                  bootstrap_start=params['start']['x']
                                  )
    
    # Calculate (maximum) total minted in each service
    per_service_max = param_use*network_macro['CUTTM']*data_df['CU per Node']*data_df['Active Nodes']
    # Apply budget
    if max_mint > 0:
        if max_mint < per_service_max.sum():
            # Scale values
            per_service_max *= (max_mint/per_service_max.sum())
    # Return amount to mint in each service by this boost
    return per_service_max
    






def apply_global_limits_and_minimums(data_df: pd.DataFrame, network_macro: dict, params: dict) -> Tuple[pd.DataFrame, dict]:
    '''
    This function implement any minting limits or minimums that need to be 
    applied after all TLMs were calculated. 
    This function intends to enforce a cap on global supply growth and also to
    ensure minimum minting for each network actor.
    More things can be implemented here if they need to have access to the 
    result of all TLMs minting.

    Order is important, first we apply minimum minting and then the global 
    supply growth cap.
    '''

    ############################################################################
    # Minimum Minting
    ############################################################################
    # For each actor check if minimum minting is OK or if we need to scale it
    for key in network_macro['mint_share']:
        # Get data if exists
        min_mint = params['global_boundaries']['min_mint'].get(key, None)
        if min_mint is not None:
            # Calculate the total budget here
            total_actor_budget = data_df['budget %s'%key].sum()

            if min_mint['type'] == 'annual_supply_growth':
                min_budget = ((min_mint['value']/100.)*network_macro['total_supply'])/(365.2)
            elif min_mint['type'] == 'USD':
                min_budget = min_mint['value']/network_macro['POKT_value']
            else:
                raise ValueError("Budget type \"%s\" not supported"%min_mint['type'])
                        
            # Check against minimum
            if total_actor_budget < min_budget:
                # Calculate scaling
                scale = min_budget/total_actor_budget
                # Apply to column
                data_df['budget %s'%key] *= scale

    ############################################################################
    # Maximum Minting  / Supply Growth
    ############################################################################
    # Check for Global minting limits
    max_mint_params = params['global_boundaries']['max_mint'].get("network", None)
    if max_mint_params is not None:

        if max_mint_params['type'] == 'annual_supply_growth':
            max_budget = ((max_mint_params['value']/100.)*network_macro['total_supply'])/(365.2)
        else:
            raise ValueError("Budget type \"%s\" not supported"%max_mint_params['type'])
        
        # Get total minted, burn and growth
        total_burn = data_df['core TLM burned'].sum()
        total_mint = 0
        for key in network_macro['mint_share']:
            total_mint += data_df['budget %s'%key].sum()
        supply_growth = total_mint - total_burn

        # Check against max
        if supply_growth > max_budget:
            
            # Scale
            scale = (max_budget+total_burn)/total_mint
            for key in network_macro['mint_share']:
                data_df['budget %s'%key] *= scale




    return data_df, network_macro

    
def apply_penalties_and_normalizations(data_df: pd.DataFrame, network_macro: dict, params: dict) -> Tuple[pd.DataFrame, dict]:
    '''
    This function is the last step and defines the per-service minting.
    Here we take the calculated budgets for each service and apply the normalization
    correction factors.
    For example, the normalization correction factor is the entropy correction and
    the average CU per service cap.
    '''


    # Calculate total minted in each service
    data_df['core TLM minted'] = data_df['core TLM budget']*data_df['Normalization Correction']
    # Calculate minted for each service using the budgets
    for key in network_macro['mint_share']:
        data_df['total TLM minted %s'%key] = data_df['budget %s'%key]*data_df['Normalization Correction']

    
    return data_df, network_macro






    

################################################################################
# ----- PANDAS PROCESS FUNCTION ----- 
################################################################################
# This function captures the full mechanics of the model implementation
# it is not really a part of the model, it is in fact an auxiliary function to
# enable model comparison and re-execution of the model for static tests.

def process(data_df: pd.DataFrame, network_macro: dict, params: dict) -> Tuple[pd.DataFrame, dict]:
    '''
    Calculates all parameters and minting for a given chains DataFrame and 
    configuration using Token Logic Modules (TLMs) model.

    Parameters:
        data_df : A pandas DataFrame with all the network activity data.
        network_macro : A dictionary containing global data of the network, like 
                        total supply, POKT price, etc.
        params : The parameters needed to run the TLM model.
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
    result_dict['Total_Mint_DAO'] = 0
    result_dict['Total_Mint_Validator'] = 0
    result_dict['Total_Mint_Supplier'] = 0
    result_dict['Total_Mint_Source'] = 0
    

    # Get a hard copy of the original data
    data_df = deepcopy(data_df)
    network_macro = deepcopy(network_macro)

    # Calculate base network state
    data_df, network_macro = network_state_calculation(data_df, network_macro, params.get('Apply Entropy', True))

    # Get core TLM mint budget and total burn
    data_df, network_macro = core_TLM_budget(data_df, network_macro, params)

    # Create data struct for boosts
    network_macro['boost TLM mint budget'] = dict()
    network_macro['boost TLM mint budget']['total'] = 0
    for key in network_macro['mint_share']:
        network_macro['boost TLM mint budget'][key] = 0  

    # Get budget from boost TLMs
    # This is the main feature of the TLM model, the simplicity for adding more
    # and arbitrary mechanics
    for boost_tlm in params['boost_TLM']:
        # Check conditions
        skip = False
        for condition in boost_tlm['conditions']:
            if (network_macro[condition['metric']] < condition['low_thrs']) or (network_macro[condition['metric']] > condition['high_thrs']):
                skip = True

        if not skip:
            # Get this boost budget
            this_budget = boost_tlm['minting_func'](data_df, network_macro, boost_tlm['parameters'])
            assert this_budget.sum() > 0
            # Assign to actor
            data_df['budget %s'%boost_tlm['recipient']] += this_budget
            # Track globals
            network_macro['boost TLM mint budget'][boost_tlm['recipient']] += this_budget
            network_macro['boost TLM mint budget']['total'] += this_budget
            

    # Apply global limits and minimums to minting budget
    data_df, network_macro = apply_global_limits_and_minimums(data_df, network_macro, params)

    # Apply penalties / normalizations
    data_df, network_macro = apply_penalties_and_normalizations(data_df, network_macro, params)


    # Global compute unit per node average
    global_compute_node_average = (data_df['Relays']  *  data_df['Compute Cost']).sum()/data_df['Active Nodes'].sum()
    # Calculate total minted in each service
    data_df['total minted chain'] =  data_df['total TLM minted DAO'] + \
                                    data_df['total TLM minted Validator'] + \
                                    data_df['total TLM minted Supplier'] + \
                                    data_df['total TLM minted Source'] 

    # print(data_df['total TLM minted DAO'].sum())
    # print(data_df['total TLM minted Validator'].sum())
    # print(data_df['total TLM minted Supplier'].sum())
    # print(data_df['total TLM minted Source'].sum())
    # print(data_df['core TLM minted'].sum())
    # print(data_df['core TLM burned'].sum())
                                    
    for idx, row in data_df.iterrows():

        if row['Relays'] > 0:

            # Results entry for this service
            result_dict['Chains'][row['Chain']] = dict()

            ##### Complete the results entry
            # How much we minted here due to work
            result_dict['Chains'][row['Chain']]['mint_base'] = row['core TLM minted']
            # How much we burnt here
            result_dict['Chains'][row['Chain']]['burn_total'] = row['core TLM burned']
            # How much extra we mint for sources
            result_dict['Chains'][row['Chain']]['mint_boost_sources'] = row['total TLM minted Source'] - network_macro["mint_share"]["Source"]*row['core TLM minted']
            # Total mint
            result_dict['Chains'][row['Chain']]['mint_total'] = row['total minted chain'] 
            # Calculate the minting per node in this service
            result_dict['Chains'][row['Chain']]['mint_per_node'] = row['total minted chain']/row['Active Nodes']
            # Calculate the imbalance
            result_dict['Chains'][row['Chain']]['service_imbalance'] = row['CU per Node']/global_compute_node_average

            # Add to the global accumulators (all services)
            result_dict['Total_Mint'] += result_dict['Chains'][row['Chain']]['mint_total']
            result_dict['Total_Burn'] += result_dict['Chains'][row['Chain']]['burn_total']
            result_dict['Total_Mint_DAO'] += row['total TLM minted DAO']
            result_dict['Total_Mint_Validator'] += row['total TLM minted Validator']
            result_dict['Total_Mint_Supplier'] += row['total TLM minted Supplier']
            result_dict['Total_Mint_Source'] += row['total TLM minted Source']
            
    return data_df, result_dict

    

################################################################################
# ----- Configuration Structure ----- 
################################################################################
# This is a big structure that not only contains the parameters but also 
# describes the different TLMs

default_parameters_dict = dict()

### Global Base Minting & Supply Growth
default_parameters_dict['global_boundaries'] = dict()
# Supply Growth Global limit
default_parameters_dict['global_boundaries']['max_mint'] = dict()
default_parameters_dict['global_boundaries']['max_mint']['network'] = dict()
default_parameters_dict['global_boundaries']['max_mint']['network']['type'] = 'annual_supply_growth'
default_parameters_dict['global_boundaries']['max_mint']['network']['value'] = 5                            # [%]
# Base Minting for all actors
default_parameters_dict['global_boundaries']['min_mint'] = dict()
default_parameters_dict['global_boundaries']['min_mint']['DAO'] = dict()
default_parameters_dict['global_boundaries']['min_mint']['DAO']['type'] = 'USD'
default_parameters_dict['global_boundaries']['min_mint']['DAO']['value'] = 2e3                              # USD/day
default_parameters_dict['global_boundaries']['min_mint']['Validator'] = dict()
default_parameters_dict['global_boundaries']['min_mint']['Validator']['type'] = 'USD'
default_parameters_dict['global_boundaries']['min_mint']['Validator']['value'] = 1e3                        # USD/day
default_parameters_dict['global_boundaries']['min_mint']['Supplier'] = dict()
default_parameters_dict['global_boundaries']['min_mint']['Supplier']['type'] = 'USD'
default_parameters_dict['global_boundaries']['min_mint']['Supplier']['value'] = 14e3                        # USD/day
default_parameters_dict['global_boundaries']['min_mint']['Source'] = dict()
default_parameters_dict['global_boundaries']['min_mint']['Source']['type'] = 'USD'
default_parameters_dict['global_boundaries']['min_mint']['Source']['value'] = 3e3                           # USD/day

### Core TLM
# This is the core mudule, encodes the end-game scenario of the network
default_parameters_dict['core_TLM'] = dict()
# This sets the supply change, a value less than 1.0 means supply attrition, a value higher than 1.0 means supply growth
# The actual speed of growth or attrition is related to the amount of relays and the total supply, so, this is no equal
# to the target supply change per year (commonly refferred as "inflation").
default_parameters_dict['core_TLM']['supply_change'] = 1.0 #0.99                                                # -
# This is the core of the model, the cost of a single compute unit.
# The definition of the compute unit is a core component of the network economy and it will change over time.
default_parameters_dict['core_TLM']['cu_cost'] = 0.0000000085                                               # USD/CU

### Boost TLMs
# These are a collection of modules that perform EXTRA minting. They are all defined by:
# - A recipient actor (DAO, Sources, Validators, Servicers, anything).
# - A condition logic, that dictates when the module is implemented.
# - A minting mechanism, that defines how the extra minting is calculated.
# - A budget, that sets the maximum spending for the module.
default_parameters_dict['boost_TLM'] = list()

### DAO Boost
aux_boost = dict()
aux_boost['name'] = "DAO Boost - CU Based"
aux_boost['recipient'] = 'DAO'                                                  # Who receives the minting
aux_boost['conditions'] = list()                                                # List of conditions to meet for TLM execution
aux_boost['conditions'].append(
    {
        'metric' : 'total_cus',                                                 # Total CUs in the network
        'low_thrs' : 0,
        'high_thrs' : 2500*1e9                                                  # CU/day
    }
)
aux_boost['minting_func'] = boost_cuttm_f_CUs_nonlinear                         # A function that accepts as input the services state, the network macro state and the parameters below and return the amount to mint per service
aux_boost['parameters'] = {                                                     # A structure containing all parameters needed by this module
    "start" : {
        "x" :  250*1e9,                                                         # ComputeUnits/day
        "y" : 5e-9,                                                             # USD/ComputeUnits
    },
    "end" : {
        "x" :  2500*1e9,                                                        # ComputeUnits/day
        "y" : 0,                                                                # USD/ComputeUnits
    },
    "variables" : {
        "x" : "total_cus",                                                      # Control metric for this TLM
        "y" : "CUTTM"                                                           # Target parameter for this TLM
    },
    'budget' : {                                                                # Can be a fixed number of tokens [POKT] or a percentage of total supply (annualized) [annual_supply_growth]
        'type' : 'annual_supply_growth',
        'value' : 0.5
    }
}
# Append to boosts list
default_parameters_dict['boost_TLM'].append(aux_boost)

### Validator Boost
aux_boost = dict()
aux_boost['name'] = "Validator Boost - CU Based"
aux_boost['recipient'] = 'Validator'
aux_boost['conditions'] = list()
aux_boost['conditions'].append(
    {
        'metric' : 'total_cus', 
        'low_thrs' : 0,
        'high_thrs' : 2500*1e9                                                  # ComputeUnits/day
    }
)
aux_boost['minting_func'] = boost_cuttm_f_CUs_nonlinear 
aux_boost['parameters'] = {         #
    "start" : {
        "x" :  250*1e9,                                                         # ComputeUnits/day
        "y" : 2.5e-9,                                                           # USD/ComputeUnits
    },
    "end" : {
        "x" :  2500*1e9,                                                        # ComputeUnits/day
        "y" : 0,                                                                # USD/ComputeUnits
    },
    "variables" : {
        "x" : "total_cus",                                                      # Control metric for this TLM
        "y" : "CUTTM"                                                           # Target parameter for this TLM
    },
    'budget' : {            
        'type' : 'annual_supply_growth',
        'value' : 0.25
    }
}
# Append to boosts list
default_parameters_dict['boost_TLM'].append(aux_boost)


### Supplier Boost
aux_boost = dict()
aux_boost['name'] = "Supplier Boost - CU Based"
aux_boost['recipient'] = 'Supplier'
aux_boost['metrics'] = [           
    'total_cus'     
    ]
aux_boost['conditions'] = list()
aux_boost['conditions'].append(
    {
        'metric' : 'total_cus', 
        'low_thrs' : 0,
        'high_thrs' : 2500*1e9      
    }
)
aux_boost['minting_func'] = boost_cuttm_f_CUs_nonlinear 
aux_boost['parameters'] = {       
    "start" : {
        "x" :  250*1e9,                                                         # ComputeUnits/day
        "y" : 3.5e-8,                                                           # USD/ComputeUnits
    },
    "end" : {
        "x" :  2500*1e9,                                                        # ComputeUnits/day
        "y" : 0,                                                                # USD/ComputeUnits
    },
    "variables" : {
        "x" : "total_cus",                                                      # Control metric for this TLM
        "y" : "CUTTM"                                                           # Target parameter for this TLM
    },
    'budget' : {            
        'type' : 'annual_supply_growth',
        'value' : 3.5
    }
}
# Append to boosts list
default_parameters_dict['boost_TLM'].append(aux_boost)


### Source Boost
aux_boost = dict()
aux_boost['name'] = "Sources Boost 1 - CU Based"
aux_boost['recipient'] = 'Source'
aux_boost['conditions'] = list()
aux_boost['conditions'].append(
    {
        'metric' : 'total_cus', 
        'low_thrs' : 0,
        'high_thrs' : 2500*1e9      
    }
)
aux_boost['parameters'] = {         
    "start" : {
        "x" :  250*1e9,                                                         # ComputeUnits/day
        "y" : 7.5e-9,                                                           # USD/ComputeUnits
    },
    "end" : {
        "x" :  2500*1e9,                                                        # ComputeUnits/day
        "y" : 0,                                                                # USD/ComputeUnits
    },
    "variables" : {
        "x" : "total_cus",                                                      # Control metric for this TLM
        "y" : "CUTTM"                                                           # Target parameter for this TLM
    },
    'budget' : {            
        'type' : 'annual_supply_growth',
        'value' : 0.75
    }
}
aux_boost['minting_func'] = boost_cuttm_f_CUs_nonlinear 
# Append to boosts list
default_parameters_dict['boost_TLM'].append(aux_boost)




### Source Boost 2 (Spreadsheet)
aux_boost = dict()
aux_boost['name'] = "Sources Boost 2 - Shane's"
aux_boost['recipient'] = 'Source'
aux_boost['conditions'] = list()
aux_boost['conditions'].append(
    {
        'metric' : 'total_cus', 
        'low_thrs' : 0,
        'high_thrs' : 1500*1e9      
    }
)
aux_boost['parameters'] = {         
    "start" : {
        "x" :  5*1e9,                                                           # ComputeUnits/day
        "y" : 0.9*0.7,                                                          # Proportion of CUTTF
    },
    "end" : {
        "x" :  1500*1e9,                                                        # ComputeUnits/day
        "y" : 0.1*0.7,                                                          # Proportion of CUTTF
    },
    "variables" : {
        "x" : "total_cus",                                                      # Control metric for this TLM
        "y" : "prop. CUTTM"                                                     # Target parameter for this TLM
    },
    'budget' : {            
        'type' : 'POKT',
        'value' : 40e3                                                          # POKT per day
    }
}
aux_boost['minting_func'] = boost_prop_f_CUs_sources_custom 
# Append to boosts list
default_parameters_dict['boost_TLM'].append(aux_boost)


