import argparse
import yaml
import os,sys
    
    

def main(argv):
        parser = argparse.ArgumentParser(description='Create a YAML configuration file for EGAT to use. Also runs the EGAT with all steps in one go.')
        parser.add_argument('-output', default = 'EGAT.yaml', help='Output YAML file name')

        
        parser.add_argument('--weightsandbiases', action='store_true', help='Check if need to live monitor things with weights and biases.')
        parser.add_argument('--wandbproject', type=str, default=None,help='Name of project to save for live monitoring.')
        parser.add_argument('--wandbrun', type=str, default=None,help='Name of run to save for live monitoring.')
        
        parser.add_argument('--startpoint', type=str, default=None,help='Kind of Starting Point to use for Training.')
        parser.add_argument('--base_model', type=str, default=None,help='Trained model used for Transfer Learning and Prediction')
        parser.add_argument('--ablation_EGAT_model', type=str, default=None,help='Trained model used for Ablations runs. This only takes the EGAT weights.')
        parser.add_argument('--ablation_NN_model', type=str, default=None,help='Trained model used for Ablations runs. This only takes the EGAT weights.')
        parser.add_argument('--gpu', type=int, default=1,help='Number of gpus involved.')

        parser.add_argument('--hidden_dim', type=int, default=128,help='Number of nodes in each FFNN and EGAT layer.')
        parser.add_argument('--num_heads', type=int, default=4,help='Number of Attention Heads Being Used.')

        parser.add_argument('--input', type=str, default=None,help='Input CSV file with the smiles and targets along with any additional features')
        parser.add_argument('--data_path', type=str, default=None,help='Path where the DGL graph data is stored')
        parser.add_argument('--save_path', type=str, default=None,help='Path where the model results will be saved for training.')
        parser.add_argument('--exclude', type=str, default = 'exclude.txt',help='Location of the Exclude file.')
        parser.add_argument('--class_choice', type=str, default=None,help='Types of reactions to look at.') 
        parser.add_argument('--npoints', type=str, default=33,help='Max Number of Nodes to look at in each graph.')  
        parser.add_argument('--split', type=str, default='all',help='Part of the Split used for Prediction.')  
        parser.add_argument('--randomize', action='store_true', help='Shuffle all of the Data being loaded so that the data is truly randomized. Turning this off means that the data is loaded along each reaction type.') 
        parser.add_argument('--fold_train', dest= 'fold',type=int,default=None, help='Fold to look at for training and prediction.') 
        parser.add_argument('--foldtype', type=str,default=None, help='Fold type to look at') 
        parser.add_argument('--fold_split', dest='folds',type=int,default=1, help='# of Folds to Split the Data As.') 
        parser.add_argument('--random_state', dest='state',type=int,default=1, help='Random State for doing all of the operations for reproducibility.') 
        parser.add_argument('--ResidBias', action='store_true', help='Types of reactions to look at.')
        parser.add_argument('--size', type=int, default=None, help='Look at the first N samples.') 
        parser.add_argument('--batch_size', type=int, default=50, help='Batch size of data.') 
        parser.add_argument('--shuffle', action='store_true', help='Shuffles the data at every iteration.') 
        parser.add_argument('--datastorage', type=str, default=None, help='How to store data. Default is a folder full of .json files.') 
        
        parser.add_argument('--useOld', action='store_true', help='Use the node and edge vector sizes initially done by Qiyuan.')
        parser.add_argument('--useFullHyb', action='store_true', help='Use the full RDKit Hybridization suite.')
        
        
        parser.add_argument('--optimizer', type=str, default = 'Adam', help='Kind of Optimizer.')
        parser.add_argument('--learning_rate', type=float, default = 1e-05, help='Initial Learning Rate.')
        parser.add_argument('--betas', type=float, default = [.9,.999], help='Initial Learning Rate Betas.',nargs=2)
        parser.add_argument('--lr_min', type=float, default = 2e-05, help='Minimum Learning Rate.')
        parser.add_argument('--lr_decay', type=float, default = .6, help='Learning Rate Decay Rate.')
        parser.add_argument('--exp_decay', type=float, default = 5, help='Learning Rate Decay Rate for exponential scheduler.')
        
        parser.add_argument('--weight_decay', type=float, default = 1e-04, help='Learning Rate Decay Rate for weights.')
        parser.add_argument('--epsilon', type=float, default = 1e-08, help='Learning Rate Decay Rate factor.')
        parser.add_argument('--momentum', type=float, default = .9, help='Learning Rate Decay Rate momentum.')
        parser.add_argument('--step_size', type=int, default = 20, help='Learning Rate Decay Rate step size.')
        parser.add_argument('--momentum_orig', type=int, default = .1, help='Learning Rate Decay Rate original momentum.')

        parser.add_argument('--target', type=str, default = 'DE',help=' Target column or set of columns to predict',nargs='+') 
        parser.add_argument('--tweights', type=str, default = [.50,.50],help='Weights of the Target column or set of columns to predict',nargs='+') 
        parser.add_argument('--additionals', type=str, default=None,help='Column or set of columns to use for predictions.',nargs='+') 
        parser.add_argument('--hasaddons', action='store_true', help='Check if RDKit Features are needed.')
        parser.add_argument('--hasnormedaddons', action='store_true', help='Check if NormalizedRDKit Features are needed. hasaddons also has to be flagged too.')
        parser.add_argument('--drop_last', action='store_true', help='Drop the last batch when loading Data.')
        parser.add_argument('--split_type', type=str, default='random',help='Type of Split to Split the data with.')
        
        parser.add_argument('--model', type=str, help='Directory of the model being used. Only use if you have made a modified version of EGAT not currently available.')
        parser.add_argument('--model_type', type=str, default='direct',help='Type of model being used')
        parser.add_argument('--ablation_EGAT_model_type', type=str, default='direct',help='Type of model being used for EGAT ablation.')
        parser.add_argument('--ablation_NN_model_type', type=str, default='direct',help='Type of model being used for NN ablation.')
        parser.add_argument('--destination', type=str, help='Where to save the Prediction Results.')
        parser.add_argument('--test_only', action='store_true', help='Only look at the validation set for evaulation in Training.') 
        parser.add_argument('--test_split', type=float, help='Percent of Data in the Training Set.') 
        parser.add_argument('--Classification', action='store_true', help='Run the Classification steps.')

        parser.add_argument('--AtomPropPrediction', action='store_true',help='Run Atom Property Prediction')
        parser.add_argument('--AtomPropAddition', type=str,default =None,help='Add Atom Property column to Prediction')
        parser.add_argument('--BondPropPrediction', action='store_true',help='Run Bond Property Prediction')
        parser.add_argument('--BondPropAddition', type=str,default =None,help='Add Bond Property column to Prediction')
        
        parser.add_argument('--3D', action='store_true',help='Use the 3D property suite instead of 2D.')
        parser.add_argument('--GraphPropSteps', type=str,default = 'EGAT' , help='Type of layers to use to predict Atom/Bond Properties.')

        parser.add_argument('--EGAT_layers', type=int,default=4,help='# of layers for EGAT')
        parser.add_argument('--EGT_layers', type=int,default=0,help='# of layers for EGT')
        parser.add_argument('--EGAT_EGT_Style', type=str,default='Block',help='How the EGAT and EGT interact.')

        parser.add_argument('--NN_layers', type=int,default=3,help='# of layers for NN')
        parser.add_argument('--NN_dim', type=int,default=100,help='# of nodes for NN')
        parser.add_argument('--NN_cascade', action='store_true',help='Check if you want your nodes to shrink in size by half after each layer for NN')
        
        parser.add_argument('--Aggregate', action='store_true', help='Check if we want to aggregate R and P features after EGAT instead of getting the difference.')
        parser.add_argument('--AddOnAgg', action='store_true', help='Check if we want to aggregate R and P RDKit features after EGAT instead of getting the difference.')
        parser.add_argument('--Norm', action='store_true', help='Normalize the Additional Columns.') 
        parser.add_argument('--molecular', action='store_true', help='only use the moleuclar version of EGAT.') 
        parser.add_argument('--Resid', type=str, default = None,help='Type of Residual architecture we want to use.')
        parser.add_argument('--SA', action='store_true',help='Check if we can use Self-Attention.')
        parser.add_argument('--GRU', action='store_true',help='Add a GRU after each step')


        
        
        parser.add_argument('--AttentionMaps', action='store_true',help='Check if need to obtain the learned Attention Mapping.')
        parser.add_argument('--Embed', action='store_true', help='Check if need to obtain the learned fingerprint.')
        parser.add_argument('--UMAP', action='store_true', help='Run UMAP on the embeddings.')
        
        parser.add_argument('--smiles', type=str, default= 'smiles',help='Column with the smiles strings.')
        
        parser.add_argument('--folders', type=str, default='Molecularity',help='Way to make the folders.')

        parser.add_argument('--atom_map', action='store_true', help='Tells EGAT if we need to make our own atom-mapping.') 
        parser.add_argument('--method_mapping', type=str, help='Way to Atom-Map Reactions. Do not use for running molecular prediction.')

        parser.add_argument('--getradical', type=str, default = None,help='Method by which we can grab radical electron counts. (RDKit or YARP)')
        parser.add_argument('--getspiro', action='store_true',help='Method by which we can grab counts of the Spiro Atom. (RDKit or YARP)')
        parser.add_argument('--getbridgehead', action='store_true',help='Method by which we can grab Bridgehead Atom. (RDKit or YARP)')
        parser.add_argument('--gethbinfo', action='store_true',help='Method by which we can grab Hydrogen Bonding Info. (RDKit or YARP)')
        parser.add_argument('--geteneg', action='store_true',help='Method by which we can grab electronegativity. (RDKit or YARP)')
        
        parser.add_argument('--onlyH', action='store_true', help='Check if we only need to look at neighboring H.')         
        parser.add_argument('--removeelementinfo', action='store_true', help='Does not look at element info. This is useful for sensitivity analysis of variables.') 
        parser.add_argument('--removereactiveinfo', action='store_true', help='Does not look at distance from reacting atoms. This is useful for sensitivity analysis of variables.') 
        parser.add_argument('--removeringinfo', action='store_true', help='Does not look at distance from reacting atoms. This is useful for sensitivity analysis of variables.') 
        parser.add_argument('--removeformalchargeinfo', action='store_true', help='Does not look at distance from reacting atoms. This is useful for sensitivity analysis of variables.') 
        parser.add_argument('--removearomaticity', action='store_true', help='Does not look at distance from reacting atoms. This is useful for sensitivity analysis of variables.') 
        parser.add_argument('--removechiralinfo', action='store_true', help='Does not look at distance from reacting atoms. This is useful for sensitivity analysis of variables.') 
        parser.add_argument('--removehybridinfo', action='store_true', help='Does not look at distance from reacting atoms. This is useful for sensitivity analysis of variables.') 
        parser.add_argument('--removeneighborcount', action='store_true', help='Does not look at distance from reacting atoms. This is useful for sensitivity analysis of variables.') 
        
        parser.add_argument('--getbondrot', action='store_true', help='Check if we need to get the location of rotatable bonds.')
        parser.add_argument('--getbondpolarity', action='store_true', help='Check if we need to get the polarity bonds.')
        
        parser.add_argument('--removebondorderinfo', action='store_true', help='Does not look at bond order info. This is useful for sensitivity analysis of variables.') 
        parser.add_argument('--removebondtypeinfo', action='store_true', help='Does not look at bond type info. This is useful for sensitivity analysis of variables.') 
        parser.add_argument('--removeconjinfo', action='store_true', help='Does not look at bond conjugation. This is useful for sensitivity analysis of variables.') 
        parser.add_argument('--removestereoinfo', action='store_true', help='Does not look at bond stereochemistry. This is useful for sensitivity analysis of variables.') 

        parser.add_argument('--epoch', type=int, default = 30, help='Epochs with changing learning rate.')
        parser.add_argument('--epoch_const', type=int, default = 10, help='Epochs with constant learning rate.')
        parser.add_argument('--warmup', type=int, default = 2, help='Epochs with a linearly rising learning rate.')
        parser.add_argument('--patience', type=int, default = 2, help='Max no. of consecutive epochs useful for.')
        parser.add_argument('--scheduler', type=str, default = 'cos', help='Learning Rate Decay Rate.')

        parser.add_argument('--train_loss', type=str, default='MAE', help='Loss used for Training.')
        parser.add_argument('--pred_loss', type=str, default = 'MAE',help='Loss used in model prediction',nargs='+')

        parser.add_argument("--UMAP_model",type=str, default=None, help="Input CSV file of reactions")
        
        parser.add_argument("--umap_input",type=str, default=None, help="Input CSV file of reactions")
        parser.add_argument("--umap_outfile", type=str, default=None, help="Output file")
        parser.add_argument("--n_neighbors", type=int, default=32, help="Number of neighbors to look at")
        parser.add_argument("--dist", type=float, default=.1, help="Overlap")
        parser.add_argument("--n_components", type=int, default=1, help="Output Dimensions")
        parser.add_argument("--metric", type=str, default='euclidian', help="Method to find distance")

        
        parser.add_argument('--num_workers', type=int, default=1, help='Number of Parallel processors used.')
        args = parser.parse_args()

        # Convert the Namespace object to a dictionary
        args_dict = vars(args)
        #print(args_dict)
        # Specify the output YAML file name
        #output_yaml_file = f'config/{args.output}'
        output_yaml_file = args.output
        #args_dict = {key: value for key, value in args_dict.items() if key.startswith('--')}
        with open(output_yaml_file, 'w') as yaml_file:
                yaml.dump(args_dict, yaml_file, default_flow_style=False)

        with open(output_yaml_file, 'r') as yaml_file:
                yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        
        # Write the dictionary to a YAML file
        if args.model_type == 'direct':
                if args.molecular:
                        if args.hasaddons:
                                yaml_data['defaults'] = {'model': 'EGAT_3MLP_1OUT_MOLEC_RDKIT'}
                        else:
                                yaml_data['defaults'] = {'model': 'EGAT_3MLP_1OUT_MOLEC'}
                else:
                        if args.hasaddons:
                                yaml_data['defaults'] = {'model': 'EGAT_3MLP_1OUT_RDKIT'}
                        else:
                                yaml_data['defaults'] = {'model': 'EGAT_3MLP_1OUT'}
        elif args.model_type in ['multi','BEP','multitask']:
                if args.molecular:
                        if args.hasaddons:
                                yaml_data['defaults'] = {'model': 'EGAT_3MLP_2OUT_MOLEC_RDKIT'}
                        else:
                                yaml_data['defaults'] = {'model': 'EGAT_3MLP_2OUT_MOLEC'}
                else:
                        if args.hasaddons:
                                yaml_data['defaults'] = {'model': 'EGAT_3MLP_2OUT_RDKIT'}
                        else:
                                yaml_data['defaults'] = {'model': 'EGAT_3MLP_2OUT'}
        elif args.model_type == 'Hr':
                if args.molecular:
                        if args.hasaddons:
                                yaml_data['defaults'] = {'model': 'EGAT_3MLP_Hr_MOLEC_RDKIT'}
                        else:
                                yaml_data['defaults'] = {'model': 'EGAT_3MLP_Hr_MOLEC'}
                else:
                        if args.hasaddons:
                                yaml_data['defaults'] = {'model': 'EGAT_3MLP_Hr_RDKIT'}
                        else:
                                yaml_data['defaults'] = {'model': 'EGAT_3MLP_Hr'}
        
        if args.save_path is not None:
                if not os.path.isdir(args.save_path): os.mkdir(args.save_path)
                if not os.path.join(args.save_path,yaml_data['defaults']['model']): os.path.join(args.save_path,yaml_data['defaults']['model'])
                yaml_data['hydra'] = {'run':{'dir':os.path.join(args.save_path,yaml_data['defaults']['model'])},'sweep':{'dir':args.save_path,'subdir':yaml_data['defaults']['model']}}
        else:
                yaml_data['hydra'] = {'run':{'dir':f'log/{args.model_type}/{args.model}'},'sweep':{'dir':f'log/{args.model_type}','subdir':args.model}}
                
        # Write the updated YAML data back to the file
        with open(output_yaml_file, 'w') as yaml_file:
                yaml.dump(yaml_data, yaml_file, default_flow_style=False)

        
if __name__ == '__main__':
    main(sys.argv[1:])
