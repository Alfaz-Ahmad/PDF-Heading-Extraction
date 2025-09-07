import pandas as pd


def rank_column_values(input_csv_path, output_csv_path, column_to_rank):
    """
    Reads a CSV, ranks the unique values in a specified column,
    and saves the result to a new CSV.

    The ranking is done in descending order (largest value gets rank 1).

    Args:
        input_csv_path (str): The path to the input CSV file.
        output_csv_path (str): The path where the output CSV will be saved.
        column_to_rank (str): The name of the column to rank.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(input_csv_path)

        # --- Step 1: Check if the specified column exists ---
        if column_to_rank not in df.columns:
            print(f"Error: Column '{column_to_rank}' not found in the CSV file.")
            print(f"Available columns are: {list(df.columns)}")
            return

        # --- Step 2: Find unique values and sort them ---
        # Get all unique values from the specified column
        unique_values = df[column_to_rank].unique()

        # Sort the unique values in descending order (from largest to smallest)
        sorted_unique_values = sorted(unique_values, reverse=True)

        # --- Step 3: Create a rank mapping ---
        # Create a dictionary that maps each unique value to its rank.
        # The rank is its position (index + 1) in the sorted list.
        # e.g., {0.8879: 1, 0.5356: 2, ...}
        rank_mapping = {value: rank + 1 for rank, value in enumerate(sorted_unique_values)}

        # --- Step 4: Add the new rank column to the DataFrame ---
        # The .map() function applies the rank_mapping to each value in the column
        new_rank_column_name = f'{column_to_rank}_rank'
        df[new_rank_column_name] = df[column_to_rank].map(rank_mapping)

        # --- Step 5: Save the result to a new CSV file ---
        # index=False prevents pandas from writing the DataFrame index as a column
        df.to_csv(output_csv_path, index=False)

        print(f"Success! The file has been processed.")
        print(f"The new CSV with ranks is saved as: '{output_csv_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found.")
        print("Please make sure the file exists and the path is correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Change this to the actual path of your input file
    input_file = 'your_data.csv'

    # You can change the output file name if you wish
    output_file = 'data_with_ranks.csv'

    # The column you want to find unique values in and rank
    column_name = 'norm_font_size'

    # --- Run the function ---
    rank_column_values(input_file, output_file, column_name)