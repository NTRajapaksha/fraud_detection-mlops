import pandas as pd
import numpy as np

def generate_drifted_data():
    # 1. Load the original structure (to get column names)
    ref_data = pd.read_csv("data/reference_data.csv")
    columns = ref_data.columns
    
    # 2. Generate "Normal" Data (similar to training)
    # create 100 rows of random normal data
    normal_data = pd.DataFrame(
        np.random.normal(0, 1, size=(100, len(columns))), 
        columns=columns
    )
    
    # 3. Generate "Drifted" Data (significantly different)
    # create 100 rows where values are multiplied by 5 (Major Drift!)
    drifted_data = pd.DataFrame(
        np.random.normal(2, 5, size=(100, len(columns))), 
        columns=columns
    )
    
    # 4. Combine and Save
    # This simulates a week's worth of data where things went wrong
    current_data = pd.concat([normal_data, drifted_data])
    current_data.to_csv("data/current_data.csv", index=False)
    print("✅ Generated 'current_data.csv' with synthetic drift.")

if __name__ == "__main__":
    generate_drifted_data()