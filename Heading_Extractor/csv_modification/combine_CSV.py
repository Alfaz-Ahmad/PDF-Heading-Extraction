import pandas as pd
import argparse
import os
import glob

def combine_csvs(input_dir, output_filepath):
    """
    Combines all CSV files in a directory into a single CSV file.
    
    A unique integer identifier is added to each row to prevent data loss and
    maintain context from the original file.

    Args:
        input_dir (str): The path to the directory containing the CSV files.
        output_filepath (str): The path where the combined CSV file will be saved.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: The input directory '{input_dir}' was not found.")
        return

    # Find all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    if not csv_files:
        print(f"Warning: No CSV files found in the directory '{input_dir}'.")
        return

    print(f"Found {len(csv_files)} CSV files to combine.")
    
    # List to hold the dataframes from each CSV file
    all_dataframes = []
    
    # Initialize a counter for the unique file ID
    file_id_counter = 1
    
    for filepath in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(filepath)
            
            # Check if the dataframe is empty to prevent future warnings and errors
            if df.empty:
                print(f"Warning: '{os.path.basename(filepath)}' is empty. Skipping.")
                continue

            # Add a new column with the unique integer identifier
            df['document_id'] = file_id_counter
            
            # Append the dataframe to our list
            all_dataframes.append(df)
            print(f"Processed '{os.path.basename(filepath)}' and assigned 'document_id' {file_id_counter}.")
            
            # Increment the counter for the next file
            file_id_counter += 1
            
        except pd.errors.EmptyDataError:
            print(f"Warning: '{os.path.basename(filepath)}' is empty. Skipping.")
        except Exception as e:
            print(f"An error occurred while processing '{os.path.basename(filepath)}': {e}")

    # Concatenate all dataframes into a single dataframe
    if all_dataframes:
        # Concatenating only non-empty dataframes to avoid the FutureWarning
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Ensure the output directory exists before writing the file
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        # Save the final combined dataframe to the output file
        combined_df.to_csv(output_filepath, index=False)
        
        print(f"\nSuccessfully combined all CSV files into '{output_filepath}'.")
    else:
        print("\nNo data to combine. The output file was not created.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Combine all CSV files in a directory into a single file with a unique integer ID.")
    parser.add_argument('-i', '--input_dir', type=str, required=True, help="Path to the directory containing CSV files.")
    parser.add_argument('-o', '--output_filepath', type=str, required=True, help="Path where the combined CSV file will be saved.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the function with the provided arguments
    combine_csvs(args.input_dir, args.output_filepath)

