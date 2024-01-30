import pandas as pd 
import umap
import os,sys,argparse
import joblib
import importlib
import shutil
import hydra
import omegaconf
import pandas as pd

def UMAP(args):
    omegaconf.OmegaConf.set_struct(args, False)
    if args.UMAP_model is None:
        reducer = umap.UMAP(n_neighbors=args.n_neighbors,min_dist=args.min_dist,n_components=args.n_components,metric=args.metric)
        embeddings = pd.read_csv(args.umap_input,index_col=0).values
        UMAPembeds = reducer.fit_transform(embeddings)
        UMAPcsv = pd.DataFrame({'X':UMAPembeds[:,0],'Y':UMAPembeds[:,1]})
        UMAPcsv.to_csv(args.umap_outfile)
        joblib.dump(reducer, args.umap_model_filename)
    else:
        reducer = joblib.load(args.UMAP_model)
        embeddings = pd.read_csv(args.umap_input,index_col=0).values
        UMAPembeds = reducer.fit_transform(embeddings)
        UMAPcsv = pd.DataFrame({'X':UMAPembeds[:,0],'Y':UMAPembeds[:,1]})
        UMAPcsv.to_csv(args.umap_outfile)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hydra-based script with a config file argument")
    parser.add_argument("--config", type=str, default="config/train.yaml", help="Path to the config file")

    args = parser.parse_args()

    # Load the specified config file
    config = omegaconf.OmegaConf.load(args.config)
    omegaconf.OmegaConf.set_struct(args, False)
    
    # Determine the config_name based on the name of the loaded config file
    file_name = os.path.basename(args.config)
    config_name, _ = os.path.splitext(file_name)

    # Set the config_name for the Hydra function
    hydra.utils.set_config_name(config_name)

    # Run the Hydra function with the merged configuration
    UMAP(config)