import argparse
from operator import truediv
import yaml
import os 

def load_yaml_config(file_path):
    with open(file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def add_arguments_from_yaml(parser, config):
    for key, value in config.items():
        parser.add_argument(f'--{key}', default=value, type=int if isinstance(value, int) else float if isinstance(value, float) else str,help=f'{key} parameter (default: {value})')

def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def modify_and_save_yaml(args, output_yaml):
    for arg, new_value in args.modify_arguments.items():
        if isinstance(new_value, str):
            if new_value.isdigit():
                new_value = int(new_value)
            elif is_number(new_value):
                new_value = float(new_value)
            elif new_value == 'true':
                new_value = True
            elif new_value == 'false':
                new_value = False
        setattr(args,arg, new_value)


    # Save the modified arguments to a new YAML file
    modified_config = {arg: getattr(args, arg) for arg in vars(args)}
    with open(output_yaml, 'w') as output_file:
        yaml.dump(modified_config, output_file, default_flow_style=False)

    with open(output_yaml, 'r') as yaml_file:
        yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    
    # Write the dictionary to a YAML file
    if yaml_data['model_type'] == 'direct':
            if yaml_data['molecular'] == True:
                    if yaml_data['hasaddons']:
                            yaml_data['defaults'] = {'model': 'EGAT_3MLP_1OUT_MOLEC_RDKIT'}
                    else:
                            yaml_data['defaults'] = {'model': 'EGAT_3MLP_1OUT_MOLEC'}
            else:
                    if yaml_data['hasaddons']:
                            yaml_data['defaults'] = {'model': 'EGAT_3MLP_1OUT_RDKIT'}
                    else:
                            yaml_data['defaults'] = {'model': 'EGAT_3MLP_1OUT'}
    elif yaml_data['model_type'] in ['multi','BEP','multitask']:
            if yaml_data['molecular'] == True:
                    if yaml_data['hasaddons']:
                            yaml_data['defaults'] = {'model': 'EGAT_3MLP_2OUT_MOLEC_RDKIT'}
                    else:
                            yaml_data['defaults'] = {'model': 'EGAT_3MLP_2OUT_MOLEC'}
            else:
                    if yaml_data['hasaddons']:
                            yaml_data['defaults'] = {'model': 'EGAT_3MLP_2OUT_RDKIT'}
                    else:
                            yaml_data['defaults'] = {'model': 'EGAT_3MLP_2OUT'}
    elif yaml_data['model_type'] == 'Hr':
            if yaml_data['molecular'] == True:
                    if yaml_data['hasaddons']:
                            yaml_data['defaults'] = {'model': 'EGAT_3MLP_Hr_MOLEC_RDKIT'}
                    else:
                            yaml_data['defaults'] = {'model': 'EGAT_3MLP_Hr_MOLEC'}
            else:
                    if yaml_data['hasaddons']:
                            yaml_data['defaults'] = {'model': 'EGAT_3MLP_Hr_RDKIT'}
                    else:
                            yaml_data['defaults'] = {'model': 'EGAT_3MLP_Hr'}
    
    yaml_data['hydra'] = {'run':{'dir':os.path.join(yaml_data['save_path'],yaml_data['defaults']['model'])},'sweep':{'dir':yaml_data['save_path'],'subdir':yaml_data['defaults']['model']}}

    with open(output_yaml, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser(description='Load arguments from YAML file and add them as argparse arguments')
    parser.add_argument('--yaml_config', required=True, help='Path to the YAML configuration file')
    parser.add_argument('--output_yaml', required=True, help='Path to save the modified YAML configuration file')
    parser.add_argument('--modify_arguments', nargs='+', metavar='ARG=VALUE', help='Specify arguments and their new values for modification')

    args = parser.parse_args()

    yaml_config = load_yaml_config(args.yaml_config)
    add_arguments_from_yaml(parser, yaml_config)

    # Parse arguments
    args = parser.parse_args()

    # Check if modify_arguments is provided and not empty
    if args.modify_arguments:
        modify_args_dict = dict(arg.split('=') for arg in args.modify_arguments)
        args.modify_arguments = modify_args_dict
        # Modify arguments and save to a new YAML file
        modify_and_save_yaml(args, args.output_yaml)

if __name__ == "__main__":
    main()
