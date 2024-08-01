# Shannon Tokenomics Static Testing

This repository contains preliminary tests for the Shannon tokenomics.

## Getting Started

### Environment Setup (One Time)

```bash
    make env_create
    $(make env_source)
    make pip_install
```

### Environment Usage (Every Time)

```bash
    $(make env_source)
    # If new dependencies were added
    make pip_freeze
```

## Source Data

The data used to run this models can be gathered from [POKTscan](https://poktscan.com/).

Currently only the `Chains` data is required, with can be obtained [here](https://poktscan.com/explore?tab=chains)
by downloading the `Performance` table. A sample of this data is provided in the `./data` folder.

## Experimental Notebooks

- **Tokenomics_Compare** : This notebook makes a simple comparison (a single scenario) of two proposed models.
- **Random_Scenarios** : This notebook tests many random network scenarios (services, relays and compute costs) and relays/nodes distributions over services on all the proposed models and then compares them head to head using several graphics.

## Olshansk TLM modules inspection

## Today

## How to avoid being gameable?

1. All the POKT to be minted in a session should be evenly distributed amongst all the nodes in that session.
   1. Why? To avoid gaming the system and pointing at your own node

2. The delay mechanism for updating RTTM for each service.
   1.

