import pandas as pd
from sklearn.model_selection import train_test_split
import sys

def stratified_split_and_save(data_path, test_size=0.2, random_state=42, verbose=True,
                              train_output="train_data.csv", test_output="test_data.csv",
                              labels_output="labels.csv"):
    """
    Loads data from the given CSV, performs a stratified 80:20 train–test split based on the 
    'inhospital_mortality' column, checks consistency of categorical variables and prevalence, 
    saves the splits to CSV files, and also saves the test split labels to a separate file.

    Parameters:
      - data_path (str): Path to the CSV file.
      - test_size (float): Proportion of data to use as test set (default is 0.2).
      - random_state (int): Random state for reproducibility.
      - verbose (bool): If True, prints detailed information.
      - train_output (str): Filename for saving the train split.
      - test_output (str): Filename for saving the test split.
      - labels_output (str): Filename for saving the test labels.
      
    Returns:
      - train_df (DataFrame): Training split.
      - test_df (DataFrame): Testing split.
      - info (dict): Dictionary containing split statistics.
    """
    # Load full data.
    data = pd.read_csv(data_path)
    if 'inhospital_mortality' not in data.columns:
        raise ValueError("Column 'inhospital_mortality' not found in the dataset.")
    
    # Stratified split based on the outcome column.
    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data['inhospital_mortality']
    )
    
    # Print shapes.
    if verbose:
        print("Train shape:", train_df.shape)
        print("Test shape:", test_df.shape)
    
    # Compute prevalence.
    train_prev = train_df['inhospital_mortality'].mean()
    test_prev = test_df['inhospital_mortality'].mean()
    if verbose:
        print(f"Prevalence in train set: {train_prev:.4f}")
        print(f"Prevalence in test set: {test_prev:.4f}")
    
    # Check consistency of categorical variables.
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object' 
                        or pd.api.types.is_categorical_dtype(data[col])]
    category_warnings = {}
    for col in categorical_cols:
        train_cats = set(train_df[col].dropna().unique())
        test_cats = set(test_df[col].dropna().unique())
        if train_cats != test_cats:
            category_warnings[col] = {
                "train_categories": list(train_cats),
                "test_categories": list(test_cats)
            }
    if verbose:
        if category_warnings:
            print("Warning: Some categorical columns have inconsistent categories across splits:")
            for col, details in category_warnings.items():
                print(f"  Column '{col}':")
                print("    Train categories:", details["train_categories"])
                print("    Test  categories:", details["test_categories"])
        else:
            print("All categorical columns are consistent across splits.")
    
    info = {
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "train_prevalence": train_prev,
        "test_prevalence": test_prev,
        "category_warnings": category_warnings
    }
    
    # Save train and test splits.
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    if verbose:
        print(f"Train split saved to: {train_output}")
        print(f"Test split saved to: {test_output}")
    
    # Extract labels (and patient IDs) from the test split.
    # We assume that the columns 'studyid_adm' and 'inhospital_mortality' are present.
    if 'studyid_adm' in test_df.columns and 'inhospital_mortality' in test_df.columns:
        labels_df = test_df[['studyid_adm', 'inhospital_mortality']]
    else:
        # Fallback: if these columns are missing, just use the outcome.
        labels_df = test_df[['inhospital_mortality']]
    
    labels_df.to_csv(labels_output, index=False)
    if verbose:
        print(f"Labels from test split saved to: {labels_output}")
    
    return train_df, test_df, info

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python split_data.py <data.csv>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    train_df, test_df, info = stratified_split_and_save(data_path, verbose=True)
