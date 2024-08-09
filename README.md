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

## How to avoid being gamble?

1. Distribute rewards evenly.

   1. How? All the POKT to be minted in a session should be evenly distributed
      evenly amongst all the nodes in that session.
   2. Why? To avoid gaming the system and pointing at your own node
   3. Issues? Free loader nodes.
   4. How do we punish free loader nodes? Will require implicit QoS.

2. Delaying CUTTM (discuss more later)
   1. How? The delay mechanism for updating RTTM for each service.
   2. Why?
