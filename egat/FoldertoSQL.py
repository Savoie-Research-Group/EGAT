import sqlite3
import json
import os
import argparse
from tqdm import tqdm
def TurnJSONfolderintoSQL(folder):
    # Connect to SQLite database
    conn = sqlite3.connect(os.path.join(folder,f"{folder.split('/')[-1]}.db"))
    cursor = conn.cursor()

    # Base folder path containing subdirectories with JSON files
    base_folder_path = folder

    # Iterate through subdirectories
    for subdirectory in os.listdir(base_folder_path):
        subdirectory_path = os.path.join(base_folder_path, subdirectory)

        # Check if the item is a subdirectory
        if os.path.isdir(subdirectory_path):
            # Iterate through JSON files in the subdirectory
            for _,filename in enumerate(tqdm(os.listdir(subdirectory_path), total=len(os.listdir(subdirectory_path)), smoothing=0.9)):
            #for filename in os.listdir(subdirectory_path):
                if filename.endswith('.json'):
                    with open(os.path.join(subdirectory_path, filename), 'r') as file:
                        data = json.load(file)
                        data['rxntype'] = subdirectory
                        # Assuming each JSON object corresponds to a table row
                        table_name = folder.split('/')[-1] + '_' + subdirectory + '_' + filename.split('.')[0]
                        # Ensure the table does not exist before creating it
                        cursor.execute(f'DROP TABLE IF EXISTS {table_name}')
                        cursor.execute(f'CREATE TABLE {table_name} (id INTEGER PRIMARY KEY AUTOINCREMENT,key TEXT, value TEXT)')

                        # Iterate through JSON items and insert key-value pairs
                        key_value_pairs = [(key,str(data[key])) for key in list(data.keys())]

                        cursor.executemany(f'INSERT INTO {table_name} (key, value) VALUES (?, ?)',
                                           key_value_pairs)
                            
    conn.commit()
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hydra-based script with a config file argument")
    parser.add_argument("--folder", type=str, default="Graphs", help="Path to the config file")

    args = parser.parse_args()

    TurnJSONfolderintoSQL(args.folder)