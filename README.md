# EGAT: Edge-Featured Graph Attention Networks for Reaction and Molecular Property Prediction

EGAT is a repository that uses Edge-Featured Graph Attention Networks for reaction and molecular property prediction. This is initially described in the paper published on ArXiv (https://chemrxiv.org/engage/chemrxiv/article-details/65410dc248dad23120c6e954) that is under review by the Journal of Physical Chemistry A. More details will come on that soon. Please cite these papers if EGAT is helpful to your research.

**Applications:**
 - Reaction Prediction: ChemArXiv [https://chemrxiv.org/engage/chemrxiv/article-details/65410dc248dad23120c6e954]
 - Mass Spectra Peak Prediction: under construction
 - Molecular Property Prediction: under construction
 - Heat of Formation Prediction: under construction
 - Reaction Classification: under construction

**Documentation:** 

#### 1. Creating a Config File:

The first step in creating EGAT is to make a configuration file. You can do this in two ways: 1) by modifying the current config files available in the config file folder, or using a CLI based solution which can be done with Config.py. To see what kinds of arguements are there, please use the command below:

python Config.py --help

#### 2. Generating Graphs:

To generate the graphs, please use the command below:

python Generate.py --config 

#### 3. Training:

To train the model, please use the command below:

python Train.py --config 

#### 4. Model Prediction:

To predict the model using another model, please use the command below:

python Predict.py --config 

**Tutorial:** Sides. 

**License:** 

# Hardware Reqirements 












