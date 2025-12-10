import pandas as pd
import argparse
import os
import glob

def process_single_csv(filepath):
    """
    Reads a CSV file, adds a 'normalized_row_index' column, and saves the result
    by overwriting the original file.

    The normalized row index is calculated as the row's zero-based index divided
    by the total number of rows minus one. This results in a value from 0.0 to 1.0.
    
    Args:
        filepath (str): The path to the CSV file to be processed.
    """
    try:
        # Read the input CSV file into a pandas DataFrame
        df = pd.read_csv(filepath)
        
        # Check if the DataFrame is empty
        if df.empty:
            print(f"Warning: '{os.path.basename(filepath)}' is empty. Skipping.")
            return

        num_rows = len(df)
        if num_rows <= 1:
            # Handle edge case with 0 or 1 row to avoid division by zero
            df['normalized_row_index'] = 0.0
        else:
            # Create a new column with the normalized row index
            df['normalized_row_index'] = df.index / (num_rows - 1)
        
        # Save the new DataFrame, overwriting the original file
        df.to_csv(filepath, index=False)
        
        print(f"Successfully processed and updated '{os.path.basename(filepath)}'.")
    
    except pd.errors.EmptyDataError:
        print(f"Warning: '{os.path.basename(filepath)}' is empty. Skipping.")
    except Exception as e:
        print(f"An error occurred while processing '{os.path.basename(filepath)}': {e}")

def process_directory_inplace(target_dir):
    """
    Processes all CSV files in a given directory by adding a normalized row index
    and overwriting the original files.
    
    Args:
        target_dir (str): The path to the directory containing CSV files.
    """
    # Check if the target directory exists
    if not os.path.isdir(target_dir):
        print(f"Error: The directory '{target_dir}' was not found.")
        return
    
    # Find all CSV files in the target directory
    csv_files = glob.glob(os.path.join(target_dir, '*.csv'))
    
    if not csv_files:
        print(f"Warning: No CSV files found in the directory '{target_dir}'.")
        return

    print(f"Found {len(csv_files)} CSV files to process.")
    print("\n--- WARNING: This will overwrite your original files! ---")
    
    # Process each CSV file individually
    for filepath in csv_files:
        process_single_csv(filepath)
    
    print("\nAll files processed.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Add a normalized row index to all CSV files in a directory, overwriting them.")
    parser.add_argument('target_dir', type=str, help="Path to the directory containing CSV files.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the function with the provided directory
    process_directory_inplace(args.target_dir)
