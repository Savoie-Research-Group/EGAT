import argparse
import yaml
import os,sys
    
    

def main(argv):
        parser = argparse.ArgumentParser(description='Create a YAML configuration file for EGAT to use. Also runs the EGAT with all steps in one go.')
        parser.add_argument('-output', default = 'EGAT.yaml', help='Output YAML file name')

        parser.add_argument('-purpose', type = str, default = 'Train', help='What the goal of the run is.')

        parser.add_argument('--weightsandbiases', action='store_true', help='Check if need to live monitor things with weights and biases.')
        parser.add_argument('--wandbproject', type=str, default=None,help='Name of project to save for live monitoring.')
        parser.add_argument('--wandbrun', type=str, default=None,help='Name of run to save for live monitoring.')
        
        parser.add_argument('--startpoint', type=str, default=None,help='Kind of Starting Point to use for Training.')
        parser.add_argument('--base_model', type=str, default=None,help='Trained model used for Transfer Learning and Prediction')
        parser.add_argument('--gpu', type=int, default=1,help='Number of gpus involved.')

        parser.add_argument('--input', type=str, default=None,help='Path where the data is stored')
        parser.add_argument('--data_path', type=str, default=None,help='Path where the data is stored')
        parser.add_argument('--save_path', type=str, default=None,help='Path where the data is stored')
        parser.add_argument('--exclude', type=str, default = 'exclude.txt',help='Location of the Exclude file.')
        parser.add_argument('--class_choice', type=str, default=None,help='Types of reactions to look at.') 
        parser.add_argument('--npoints', type=str, default=33,help='Loss used for Training.')  
        parser.add_argument('--split', type=str, default=None,help='Loss used for Training.')  
        parser.add_argument('--randomize', action='store_true', help='Types of reactions to look at.') 
        parser.add_argument('--fold', type=int,default=None, help='Fold to look at.') 
        parser.add_argument('--foldtype', type=str,default=None, help='Fold type to look at') 
        parser.add_argument('--size', type=int, default=None, help='Look at the first N samples.') 
        parser.add_argument('--batch_size', type=int, default=50, help='Batch size of data.') 
        parser.add_argument('--shuffle', action='store_true', help='shuffles the data at every iteration.') 
        parser.add_argument('--datastorage', type=str, default=None, help='How to store data. Default is a folder full of .json files.') 
        


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
        parser.add_argument('--hasnormedaddons', action='store_true', help='Check if NormalizedRDKit Features are needed.')
        parser.add_argument('--drop_list', action='store_true', help='Check if NormalizedRDKit Features are needed.')
        parser.add_argument('--split_type', type=str, default='random',help='Check if NormalizedRDKit Features are needed.')
        
        
        parser.add_argument('--model', type=str, help='Directory of the model being used')
        parser.add_argument('--model_type', type=str, default='direct',help='Type of model being used')
        parser.add_argument('--destination', type=str, help='Where to save the Prediction Results.')
        parser.add_argument('--test_only', action='store_true', help='Only look at the validation set for evaulation in Training.') 
        parser.add_argument('--test_split', type=float, help='Only look at the validation set for evaulation in Training.') 
        
        parser.add_argument('--EGAT_layers', type=int,default=4,help='Check if need to obtain the learned fingerprint.')
        parser.add_argument('--Aggregate', action='store_true', help='Check if we want to aggregate R and P features after EGAT instead of getting the difference.')
        parser.add_argument('--AddOnAgg', action='store_true', help='Check if we want to aggregate R and P RDKit features after EGAT instead of getting the difference.')
        parser.add_argument('--Norm', action='store_true', help='Add Normalization to Columns.') 
        parser.add_argument('--molecular', action='store_true', help='Types of reactions to look at.') 
        parser.add_argument('--Residual', type=str, default = None,help='Check if need to obtain the learned fingerprint.')
        parser.add_argument('--SA', action='store_true',help='Check if need to obtain the learned fingerprint.')
        
        parser.add_argument('--AttnMaps', action='store_true',help='Check if need to obtain the learned fingerprint.')
        parser.add_argument('--Embed', action='store_true', help='Check if need to obtain the learned fingerprint.')
        parser.add_argument('--UMAP', action='store_true', help='Run UMAP on the embeddings.')
        
        parser.add_argument('--smiles', type=str, default= 'smiles',help='Column with the smiles strings.')
        
        parser.add_argument('--folders', type=str, default='Molecularity',help='Way to make the folders.')

        parser.add_argument('--atom_map', action='store_true', help='Tells EGAT if we need to make our own atom-mapping.') 
        parser.add_argument('--method_mapping', type=str, help='Way to Atom-Map Reactions. Do not use for running molecular prediction.')

        parser.add_argument('--getradical', type=str, default = None,help='Method by which we can grab radical electron counts. (RDKit or YARP)')
        parser.add_argument('--getspiro', action='store_true',help='Method by which we can grab radical electron counts. (RDKit or YARP)')
        parser.add_argument('--getbridgehead', action='store_true',help='Method by which we can grab radical electron counts. (RDKit or YARP)')
        parser.add_argument('--gethbinfo', action='store_true',help='Method by which we can grab radical electron counts. (RDKit or YARP)')
        parser.add_argument('--geteneg', action='store_true',help='Method by which we can grab radical electron counts. (RDKit or YARP)')
        
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
        parser.add_argument('--getbondpolarity', action='store_true', help='Check if we need to get the location of rotatable bonds.')
        
        parser.add_argument('--removebondorderinfo', action='store_true', help='Does not look at element info. This is useful for sensitivity analysis of variables.') 
        parser.add_argument('--removebondtypeinfo', action='store_true', help='Does not look at distance from reacting atoms. This is useful for sensitivity analysis of variables.') 
        parser.add_argument('--removeconjinfo', action='store_true', help='Does not look at distance from reacting atoms. This is useful for sensitivity analysis of variables.') 
        parser.add_argument('--removestereoinfo', action='store_true', help='Does not look at distance from reacting atoms. This is useful for sensitivity analysis of variables.') 
        
        
        

        parser.add_argument('--epoch', type=int, default = 30, help='Epochs with changing learning rate.')
        parser.add_argument('--epoch_const', type=int, default = 10, help='Epochs with constant learning rate.')
        parser.add_argument('--warmup', type=int, default = 2, help='Learning Rate Decay Rate.')
        parser.add_argument('--patience', type=int, default = 2, help='Learning Rate Decay Rate.')
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

        
        parser.add_argument('--num_workers', type=int, default=1, help='Loss used in model prediction')
        parser.add_argument('--nodes',dest='nodes',default=1,help = 'Nodes to run on')
        parser.add_argument('--days',dest='days',default='00',help = 'Time length (Day)')
        parser.add_argument('--hours',dest='hours',default='23',help = 'Time length (Hr.)')
        parser.add_argument('--minutes',dest='minutes',default='50',help = 'Time length (Min.)')
        parser.add_argument('--cpus',dest='cpus',default=10,help = 'CPUs to run on')
        parser.add_argument('--user',dest='user',default=None,help = 'username')
        parser.add_argument('--partition',dest='partition',default='standby',help = 'username')    
    
        args = parser.parse_args()

        # Convert the Namespace object to a dictionary
        args_dict = vars(args)
        #print(args_dict)
        # Specify the output YAML file name
        output_yaml_file = f'config/{args.output}'
        
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
