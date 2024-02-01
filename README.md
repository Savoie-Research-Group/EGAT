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

```
python Config.py --help
```

#### 2. Generating Graphs:

To generate the molecular graphs necessary for model training, please use the command below:

```
python Generate.py --config [config_file]
```

To generate molecular graphs instead of reaction based graphs for property prediction, you must set the --molecular flag when running python Config.py (or you can change it in the .yaml file itself)

#### 3. Training:

To train the model, please use the command below:

```
python Train.py --config [config_file]
```

#### 4. Model Prediction:

To predict the model using another model, please use the command below:

```
python Predict.py --config [config_file]
```

#### 5. Obtaining Model Embeddings:

To obtain the fingerprints for a set of reactions, please set the **--Embed** flag to either **1** or **2** when writing Config.py. Setting **--Embed** to **2** means that you will only get embeddings returned to you. To train and obtain Embeddings, please use the training command below:

```
python Train.py --config [config_file]
```

To obtain embeddings based on a model, please use: 

```
python Predict.py --config [config_file]
```

#### 5. Obtaining Attention Maps for Chemical Interpretation:

To obtain the fingerprints for a set of reactions, please set the **--AttentionMaps** flag when writing Config.py. To train and obtain Embeddings, please use the training command below:

```
python Train.py --config [config_file]
```

To obtain embeddings based on a model, please use: 

```
python Predict.py --config [config_file]
```

**Tutorial:** Slides and Video coming soon. 

**License:** 
MIT License

# Hardware Reqirements 
- NVIDIA GPU
- Python 3.8

# How to Download EGAT to your Home Computer/Cluster

Please set up the conda environment with the given packages in the environment.yml file. Then clone the repository to your home computer.


## Authors/Contributors

- Sai Mahit Vaddadi (svaddadi@purdue.edu)
- Qiyuan Zhao (zhaoqy@umich.edu)
- Brett Savoie (bsavoie@purdue.edu)















