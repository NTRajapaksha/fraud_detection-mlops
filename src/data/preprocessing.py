import pandas as pd
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess(input_path: str, output_dir: str):
    logging.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Simple validation
    if 'Class' not in df.columns:
        raise ValueError("Dataset missing 'Class' target column")

    # Split features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    # Stratified split to maintain fraud ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save as Parquet (faster, preserves schema)
    logging.info(f"Saving processed data to {output_dir}")
    X_train.to_parquet(f"{output_dir}/X_train.parquet")
    X_test.to_parquet(f"{output_dir}/X_test.parquet")
    pd.DataFrame(y_train).to_parquet(f"{output_dir}/y_train.parquet")
    pd.DataFrame(y_test).to_parquet(f"{output_dir}/y_test.parquet")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    load_and_preprocess("data/raw/creditcard.csv", "data/processed")