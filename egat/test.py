# Read the content of the file
with open('/depot/bsavoie/data/Mahit-TS-Energy-Project/GitHub/EGAT/electroneg-pauling.txt', 'r') as file:
    lines = file.readlines()

# Initialize an empty dictionary to store atomic numbers and electronegativity values
electronegativity_dict = {}

# Iterate through the lines and extract relevant information
for line in lines:
   
    parts = [part.strip() for part in line.split('#') if part.strip()]
    if len(parts) == 2:
        if parts[0] == '-':
            electronegativity = 0
        else:
            electronegativity = float(parts[0])
        
        element = parts[1].split(' ')[1]
        electronegativity_dict[element] = electronegativity


        




        

# Display the dictionary
print(electronegativity_dict)
