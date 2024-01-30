from RDkitHelpers import *
from rdkit import Chem
smi = Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles('CCO')))



# Hydrogen Bonding #

# Generate a random 3D confirmation of the total product and reactant.
# Get the molecules. 
# find the location of the acceptor, donor, and H involved in the network
acceptors_mol1,donors_mol1,H_mol1,_,_ = find_hbond_acceptors_donors(smi)
acceptors_mol2,donors_mol2,H_mol2,_,_ = find_hbond_acceptors_donors(smi)


# Get the maximum number of hydrogen bonds one can have
max_Hbonds = {'F':3,'O':2,'N':1,'Cl':3,'Br':3,'I':3,'H':1}

# Connect the acceptors to the Hydrogen if possible. Keep count of the number of H_bonds each one has.
# 1. Iterate through the acceptors in the first molecule.
# 2. Find a Hydrogen in the other molecule. Check if it's a Hydrogen that can be in an H bond. Add a Hydrogen bond between the Acceptor and Hydrogen. 
#    That Hydrogen cannot be used again. The acceptor can be used until the number of H bonds it can do is at it's max. 
# 3. For the hydrogen in the other molecule, the atom next to it cannot bond to the hydrogen in the first molecule. 




# Hydrophobic Contacts #
# Generate a random 3D confirmation of the total product and reactant.

#Find the location of the hydrophobe


# Hydrophobic Contacts #
# Generate a random 3D confirmation of the total product and reactant.

#Find the location of the hydrophobe














# Pi-Pi stacking # 
pi_pi = {'Parallel':[1,0,0,0],'Perpendicular':[0,1,0,0],'DNE':[0,0,1,0],'NA':[0,0,0,1]}
# Generate a set of random 3D confirmations. 

# Get the lowest n configurations.

from oddt.toolkits.rdk import Molecule

rdkit_mol = Chem.MolFromSmiles('CCC')

# Get the lowest n configurations.
oddt_mol = Molecule(rdkit_mol)




