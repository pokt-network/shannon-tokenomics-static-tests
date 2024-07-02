import pandas as pd
import numpy as np

def add_compute_cost(data: pd.DataFrame, sample_type: str='uniform', mean: int=100, std: int=10) -> pd.DataFrame:
    '''
    Adds a column of "compute cost" to the chains data. 
    This is used to emulate the per-chain cost of the services that is not yet implemented.
    '''
    # Add compute cost data
    if sample_type=='uniform':
        data['Compute Cost'] = mean
    elif sample_type=='normal':
        data['Compute Cost'] = np.random.default_rng().normal(mean, std, len(data))
    else:
        raise ValueError(f'Unknown sample strategy: {sample_type}')
    
    return data



def get_chain_data(csv_path: str, relay_multiplier: float=1.0, spreadsheet_compat: bool = False) -> pd.DataFrame:
    '''
    Reads the CSV file containing the chains data.
    The data is expected to follow the format of the "Performance" card from https://poktscan.com/explore?tab=chains 
    '''
    # Read chain data
    chains_df = pd.read_csv(csv_path)
    def to_float(value):
        # Remove percentage sign and commas, then convert to float
        return float(value.replace('%', '').replace(',', '').replace('-', '0'))
    for column in ['Earn AVG', 'Network', 'Change', 'Relays', 'Active Nodes']:
        chains_df[column] = chains_df[column].apply(to_float)

    if spreadsheet_compat:
        # Remove all chains not in this list
        chains_df = chains_df[chains_df['Chain'].isin(chain_list)]

    # Scale the number of relays
    chains_df['Relays'] *= relay_multiplier

    # Filter no data
    chains_df = chains_df.loc[chains_df['Active Nodes']>0]
    chains_df = chains_df.loc[chains_df['Relays']>0]

    return chains_df

# List of chains used in Shane's spreadsheet. Useful for comparing / debugging.
chain_list = ['Polygon Mainnet (0009)',
'Ethereum (0021)',
'BSC Mainnet (0004)',
'Base Mainnet (0079)',
'Solana Custom (C006)',
'Arbitrum One (0066)',
'FUSE Mainnet (0005)',
'Polygon zkEVM (0074)',
'DFKchain Subnet (03DF)',
'Ethereum Archival (0022)',
'Fantom (0049)',
'Harmony Shard 0 (0040)',
'Optimism Mainnet (0053)',
'Polygon Archival (000B)',
'Sui Mainnet (0076)',
'Celo Mainnet (0065)',
'Scroll Mainnet (0082)',
'NEAR Mainnet (0052)',
'Optimism Archival (A053)',
'Klaytn Mainnet (0056)',
'Avalanche Mainnet (0003)',
'Arbitrum Sepolia Archival (A086)',
'Gnosis - xDai (0027)',
'Solana Mainnet (0006)',
'Base Testnet (0080)',
'Kava Mainnet (0071)',
'Metis Mainnet (0058)',
'Holesky Testnet (0081)',
'Boba Mainnet (0048)',
'Celestia Archival (A0CA)',
'IoTeX Mainnet (0044)',
'Amoy Testnet Archival (A085)',
'ETH Archival Trace (0028)',
'BSC Archival (0010)',
'Evmos Mainnet (0046)',
'Optimism Sepolia Archival (A087)',
'Osmosis Mainnet (0054)',
'Avalanche Archival (A003)',
'Gnosis - xDAI Archival (000C)',
'Sepolia Testnet (0077)',
'Moonbeam Mainnet (0050)',
'Kava Archival (0072)',
'Oasys Archival (0069)',
'Fraxtal Archival (A088)',
'OEC Mainnet (0047)',
'Moonriver Mainnet (0051)',
'Oasys Mainnet (0070)',
'Scroll Testnet (0075)',
'Sepolia Archival (0078)',
'Radix Mainnet (0083)']