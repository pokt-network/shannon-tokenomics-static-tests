# Shannon Tokenomics Static Testing

This repository contains preliminary tests for the Shannon tokenomics.

The data used to run this models can be gathered from [POKTscan](https://poktscan.com/). 
Currently only the `Chains` data is required, with can be obtained [here](https://poktscan.com/explore?tab=chains) by downloading the `Performance` table. A sample of this data is provided in the `./data` folder.

### Experiment Notebooks 
- **Tokenomics_Compare** : This notebook makes a simple comparison (a single scenario) of two proposed models.
- **Random_Scenarios** : This notebook tests many random network scenarios (services, relays and compute costs) and relays/nodes distributions over services on all the proposed models and then compares them head to head using several graphics.
