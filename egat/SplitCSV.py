import pandas as pd
from sklearn.model_selection import train_test_split

def create_split_columns(input, train_size=0.8, random_state=42):
    # Step 1: Create a column with unique identifiers for each row
    if input[-3:] == 'csv':
        df = pd.read_csv(input,index_col = 0)
    elif input[-3:] == 'tsv':
        df = pd.read_csv(input,sep='\t')
    # Step 2: Use train_test_split to split the data into training, testing, and validation sets
    train_ids, test_ids = train_test_split(df.index, test_size=1-train_size, random_state=random_state)
    test_ids, val_ids = train_test_split(test_ids, test_size=.5, random_state=random_state)
    # Step 3: Create new columns for training, testing, and validation sets
    df['split'] = 'none'
    df.loc[train_ids, 'split'] = 'train'
    df.loc[test_ids, 'split'] = 'test'
    df.loc[val_ids, 'split'] = 'val'
    return df

