import sys,os,argparse,subprocess,shutil,time,glob,fnmatch
import pickle,json,csv
import numpy as np

# Load modules
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2])+'/utilities')
from xtb_functions import xtb_energy,xtb_geo_opt
from taffi_functions import *
from job_submit import *
from utility import *

sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2])+'/model_reaction')
from id_reaction_type import id_reaction_types

def main(argv):

    parser = argparse.ArgumentParser(description='Driver script for analyzing YARP outputs with Gaussian engine.')

    #optional arguments                                             
    parser.add_argument('-w', dest='working_folder', help = 'Specify the working (output) folder of a YARP task')

    parser.add_argument('-o', dest='outputname', default='reaction_db_feature.csv', help = 'Controls the output file name of the features')

    parser.add_argument('-c', dest='check_duplicate', help = 'a txt file to check duplicate reactions, always keep the lowest energy TS')

    ### parse energy dictionary
    args=parser.parse_args()    

    ### specify folder names
    record       = args.working_folder + '/reactions.txt'
    input_folder = args.working_folder + '/input_files_conf'
    IRC_folder   = args.working_folder + '/low-IRC-result'
    TS_folder    = args.working_folder + '/TS-folder'

    ### loop over each reaction 
    with open(record,'r') as f:
        for lc,lines in enumerate(f):
            if lc == 0: continue
            fields = lines.split()
            if len(fields) != 5: continue
            Rindex = args.working_folder.split('/')[-2]+'-{}'.format(fields[0])

            if fields[-1] == 'intended':
                E,RG,PG,Radj_mat,Padj_mat = parse_input("{}/{}.xyz".format(input_folder,fields[0]),return_adj=True)
                
            else:
                E,_,_ = parse_input("{}/{}.xyz".format(input_folder,fields[0]),return_adj=False)
                E,RG,PG,Radj_mat,Padj_mat = parse_IRC_xyz("{}/{}-IRC.xyz".format(IRC_folder,fields[0]),len(E))
                
            # determine reaction type and reaction features
            results = return_rection_features(E,RG,PG,Radj_mat,Padj_mat)
            if results:

                reaction_features,reaction_type = results[0],results[1]
                
                with open(args.outputname, 'a') as csvfile: 
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([Rindex]+reaction_features)

                # check duplicate
                '''
                finish_flag,imag_flag,SPE,zero_E,H_298,F_298,_ = read_Gaussian_output('{}/{}-TS.out'.format(TS_folder,fields[0]))
                if reaction_type in total_reaction.keys():
                    total_reaction[reaction_type] = {"Rindex":Rindex,"TS_Ene":}
                '''
            else:
                print("Have trouble idfentifying the reaction type of {}, might be a b4f4 reaction...".format(Rindex))

    return

def parse_IRC_xyz(IRC_xyz,Natoms):
    with open(IRC_xyz,'r') as g: lines = g.readlines()

    count = 0
    write_reactant= []
    write_product = []
    Energy_list   = []
    N_image = int(len(lines)/(Natoms+2))

    for lc,line in enumerate(lines):
        fields = line.split()
        if len(fields)==1 and fields[0] == str(Natoms): count += 1
        if len(fields) == 3 and 'energy' in fields[2]: Energy_list += [float(fields[2].split(':')[-1])]
        if 'Coordinates' in fields and 'E' in fields: Energy_list += [float(fields[-1])]
        if count == 1: write_reactant+= [line]
        if count == N_image: write_product += [line]

        Energy_list = np.array(Energy_list)

    # write the reactant and product
    with open(IRC_xyz.replace('-IRC.xyz','-start.xyz'),'w') as g:
        for line in write_reactant: g.write(line)

    # parse IRC start point xyz file
    NE,NG1  = xyz_parse(IRC_xyz.replace('-IRC.xyz','-start.xyz'))
    N_adj_1 = Table_generator(NE,NG1)

    # generate end point of IRC
    with open(IRC_xyz.replace('-IRC.xyz','-product.xyz'),'w') as g:
        for line in write_product: g.write(line)
            
    # parse IRC start point xyz file
    NE,NG2  = xyz_parse(IRC_xyz.replace('-IRC.xyz','-product.xyz'))
    N_adj_2 = Table_generator(NE,NG2)

    os.system("rm {} {}".format(IRC_xyz.replace('-IRC.xyz','-start.xyz'),IRC_xyz.replace('-IRC.xyz','-product.xyz')))

    return NE,NG1,NG2,N_adj_1,N_adj_2

# Function to predict the intended score
def return_rection_features(E,RG,PG,Radj_mat=None,Padj_mat=None):
    
    # obtain adj_mat if not given  
    if Radj_mat is None: Radj_mat = Table_generator(E,RG)
    if Padj_mat is None: Padj_mat = Table_generator(E,PG)

    # obtain reaction BE matrix
    lone,bond,core,Rbond_mat,fc = find_lewis(E,Radj_mat,return_pref=False,return_FC=True)
    lone,bond,core,Pbond_mat,fc = find_lewis(E,Padj_mat,return_pref=False,return_FC=True)
    BE_change   = Pbond_mat[0] - Rbond_mat[0]

    # determine bond changes (break and form)
    bond_break  = []
    bond_form   = []
    normal_bonds= []
    for i in range(len(E)):
        for j in range(i+1,len(E)):
            if BE_change[i][j] == -1:
                bond_break += [(i,j)]
                
            if BE_change[i][j] == 1:
                bond_form += [(i,j)]                

            if BE_change[i][j] == 0 and Radj_mat[i][j] == 1:
                normal_bonds += [(i,j)]
                
    # determine "reactive atom list"
    involve = [set(sorted(list(sum(bond_break, ())))),set(sorted(list(sum(bond_form, ()))))]
    if involve[0]==involve[1]:
        involve_atoms = involve[0]
    else:
        involve_atoms = set(list(involve[0])+list(involve[1]))
    
    # determine number of heavy atoms and number of each atom
    N_atom_list   = [len([e for e in E if e.lower() == i ]) for i in ['c','n','o','h']]
    N_Total_heavy = sum(N_atom_list[:-1])
    
    # determine bond change type
    bond_dict = {('c','c'):0, ('c','n'):0, ('c','o'):0, ('c','h'):0, ('n','n'):0, ('n','o'):0, ('n','h'):0, ('o','o'):0, ('o','h'):0, ('h','h'):0}
    for bond in bond_break+bond_break:
        bond_type = (E[bond[0]].lower(),E[bond[1]].lower())
        if bond_type in bond_dict.keys():
            bond_dict[bond_type] += 1
        else:
            try:
                bond_dict[(bond_type[1],bond_type[0])] += 1
            except:
                return False

    bond_changes = [N for _,N in bond_dict.items()]

    # determine number of each atom in the nearest, second-nearest and third-nearest lists
    # operate on the reactant side (same of operate on the product side)
    Radj_mat_break = deepcopy(Radj_mat)
    for bond in bond_break: Radj_mat_break[bond[0]][bond[1]] = Radj_mat_break[bond[1]][bond[0]] = 0
        
    Rgs     = graph_seps(Radj_mat)
    nearest = involve_atoms
    s_nearest = sorted(sum([ tuple([ count_i for count_i,i in enumerate(Rgs[j]) if i == 1]) for j in nearest],()))
    t_nearest = sorted(sum([ tuple([ count_i for count_i,i in enumerate(Rgs[j]) if i == 2]) for j in nearest],()))
    
    N_1_atom_list   = [len([E[ind] for ind in list(nearest) if E[ind].lower() == i ]) for i in ['c','n','o','h']]
    N_2_atom_list   = [len([E[ind] for ind in list(s_nearest) if E[ind].lower() == i ]) for i in ['c','n','o','h']]
    N_3_atom_list   = [len([E[ind] for ind in list(t_nearest) if E[ind].lower() == i ]) for i in ['c','n','o','h']]

    # determine the bond distance
    try:
        reaction_type,seq,bond_dis = id_reaction_types(E,[Radj_mat,Padj_mat],bond_changes=[bond_break,bond_form],gens=1,algorithm="matrix",return_bond_dis=True)
        if seq == [0,1]: N_bond_change = [len(bond_break),len(bond_form)]
        else: N_bond_change = [len(bond_form),len(bond_break)]

    except:
        return False

    # sum up into reaction feature
    reaction_features = [N_Total_heavy] + N_atom_list + N_bond_change + bond_changes + N_1_atom_list + N_2_atom_list + N_3_atom_list + bond_dis

    return [reaction_features,reaction_type]

if __name__ == "__main__":
    main(sys.argv[1:])

