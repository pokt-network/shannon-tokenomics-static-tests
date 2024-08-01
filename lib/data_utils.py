import numpy as np
import pandas as pd
from anytree import Node, RenderTree


def get_chains_df(csv_path: str, relay_multiplier: float = 1.0, spreadsheet_compact: bool = False) -> pd.DataFrame:
    """
    Reads the CSV file containing the chains data at csv_path to return chains_df.

    The data is expected to follow the format of the "Performance" card from https://poktscan.com/explore?tab=chains.
    """

    # Remove percentage sign and commas, then convert to float
    def to_float(value):
        return float(value.replace("%", "").replace(",", "").replace("-", "0"))

    # Read chain data
    chains_df = pd.read_csv(csv_path)
    chains_df.rename(columns={"Relays": "relays", "Active Nodes": "active_nodes"}, inplace=True)

    # Cleanup the data read in
    for column in ["Earn AVG", "Network", "Change", "relays", "active_nodes"]:
        chains_df[column] = chains_df[column].apply(to_float)

    # Remove all chains not in this list
    if spreadsheet_compact:
        chains_df = chains_df[chains_df["Chain"].isin(chain_list)]

    # Scale the number of relays for experimentation purposes
    chains_df["relays"] *= relay_multiplier

    # Filter rows where the data is empty or potentially incorrect
    print(chains_df.columns)
    chains_df = chains_df.loc[chains_df["active_nodes"] > 0]
    chains_df = chains_df.loc[chains_df["relays"] > 0]

    # Return chains_df
    return chains_df


def add_compute_cost(
    chains_df: pd.DataFrame,
    sample_type: str = "uniform",
    mean: int = 100,
    std: int = 10,
) -> pd.DataFrame:
    """
    Adds a "cu_per_relay" column to chains_df.

    This is used to emulate the per-chain cost of the services, which is not yet implemented.
    """
    if sample_type == "uniform":
        chains_df["cu_per_relay"] = mean
    elif sample_type == "normal":
        normal_dist = np.random.default_rng().normal(mean, std, len(chains_df))
        chains_df["cu_per_relay"] = normal_dist
    else:
        raise ValueError(f"Unknown sample strategy: {sample_type}")

    return chains_df


def render_tree(nested_dict: dict):
    """
    render_tree prints a nested dictionary as a tree for easier visualization.
    """

    def dict_to_tree(d, parent=None):
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict) or isinstance(v, list):
                    node = Node(k, parent=parent)
                    dict_to_tree(v, node)
                else:
                    Node(f"{k}: {v}", parent=parent)
        elif isinstance(d, list):
            for i, v in enumerate(d):
                if isinstance(v, dict) or isinstance(v, list):
                    node = Node(f"[{i}]", parent=parent)
                    dict_to_tree(v, node)
                else:
                    Node(f"[{i}]: {v}", parent=parent)
        else:
            Node(d, parent=parent)

    # Create the root node
    root = Node("root")

    # Convert the dictionary to a tree
    dict_to_tree(nested_dict, root)

    # Print the tree
    for pre, fill, node in RenderTree(root):
        print("%s%s" % (pre, node.name))


# List of chains used in Shane's spreadsheet. Useful for comparing / debugging.
chain_list = [
    "Polygon Mainnet (0009)",
    "Ethereum (0021)",
    "BSC Mainnet (0004)",
    "Base Mainnet (0079)",
    "Solana Custom (C006)",
    "Arbitrum One (0066)",
    "FUSE Mainnet (0005)",
    "Polygon zkEVM (0074)",
    "DFKchain Subnet (03DF)",
    "Ethereum Archival (0022)",
    "Fantom (0049)",
    "Harmony Shard 0 (0040)",
    "Optimism Mainnet (0053)",
    "Polygon Archival (000B)",
    "Sui Mainnet (0076)",
    "Celo Mainnet (0065)",
    "Scroll Mainnet (0082)",
    "NEAR Mainnet (0052)",
    "Optimism Archival (A053)",
    "Klaytn Mainnet (0056)",
    "Avalanche Mainnet (0003)",
    "Arbitrum Sepolia Archival (A086)",
    "Gnosis - xDai (0027)",
    "Solana Mainnet (0006)",
    "Base Testnet (0080)",
    "Kava Mainnet (0071)",
    "Metis Mainnet (0058)",
    "Holesky Testnet (0081)",
    "Boba Mainnet (0048)",
    "Celestia Archival (A0CA)",
    "IoTeX Mainnet (0044)",
    "Amoy Testnet Archival (A085)",
    "ETH Archival Trace (0028)",
    "BSC Archival (0010)",
    "Evmos Mainnet (0046)",
    "Optimism Sepolia Archival (A087)",
    "Osmosis Mainnet (0054)",
    "Avalanche Archival (A003)",
    "Gnosis - xDAI Archival (000C)",
    "Sepolia Testnet (0077)",
    "Moonbeam Mainnet (0050)",
    "Kava Archival (0072)",
    "Oasys Archival (0069)",
    "Fraxtal Archival (A088)",
    "OEC Mainnet (0047)",
    "Moonriver Mainnet (0051)",
    "Oasys Mainnet (0070)",
    "Scroll Testnet (0075)",
    "Sepolia Archival (0078)",
    "Radix Mainnet (0083)",
]
