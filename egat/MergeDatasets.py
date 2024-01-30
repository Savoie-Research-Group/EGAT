import sqlite3
import os,sys,argparse

def merge_databases(main_db, other_dbs):
    # Connect to the main database
    conn_main = sqlite3.connect(main_db)

    # Attach other databases
    for db_file in other_dbs:
        db_name = os.path.splitext(os.path.basename(db_file))[0]
        conn_main.execute(f"ATTACH DATABASE '{db_file}' AS {db_name};")

    # Get the list of tables from one of the attached databases
    cursor = conn_main.execute(f"SELECT name FROM {db_name}.sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    # Copy tables from attached databases to the main database
    for table in tables:
        for db_name in other_dbs:
            db_name = os.path.splitext(os.path.basename(db_name))[0]
            query = f"INSERT INTO main.{table} SELECT * FROM {db_name}.{table};"
            conn_main.execute(query)

    # Commit changes and close connections
    conn_main.commit()
    conn_main.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hydra-based script with a config file argument")
    parser.add_argument("--name", type=str, default="Graphs/Graphs.db", help="Path to the new dataset")
    parser.add_argument("--db", type=str, default="Graphs/Graphs.db", help="Path to the new dataset",nargs= '+')
    args = parser.parse_args()
    merge_databases(args.name,args.db)