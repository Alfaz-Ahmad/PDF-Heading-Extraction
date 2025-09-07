import os
import subprocess
import sys
import pandas as pd


def rank_column_values_in_place(csv_path, column_to_rank):
    """
    Reads a CSV, ranks the unique values in a specified column,
    and adds the rank as a new column to the *same* CSV file.
    The ranking is done in descending order (largest value gets rank 1).

    Args:
        csv_path (str): The path to the CSV file to modify.
        column_to_rank (str): The name of the column to rank.
    """
    try:
        df = pd.read_csv(csv_path)

        if column_to_rank not in df.columns:
            print(f"Error: Column '{column_to_rank}' not found in the CSV file '{csv_path}'.")
            print(f"Available columns are: {list(df.columns)}")
            return

        unique_values = df[column_to_rank].unique()
        sorted_unique_values = sorted(unique_values, reverse=True)
        rank_mapping = {value: rank + 1 for rank, value in enumerate(sorted_unique_values)}

        new_rank_column_name = f'{column_to_rank}_rank'

        # Get all columns except the last one
        cols = list(df.columns)

        # Determine the position for the new column (second to last)
        insert_position = len(cols) - 1  # This will insert it before the last column

        # Insert the new column at the desired position
        df.insert(loc=insert_position, column=new_rank_column_name, value=df[column_to_rank].map(rank_mapping))

        df.to_csv(csv_path, index=False)
        print(f"Successfully added '{new_rank_column_name}' to '{csv_path}'.")

    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred while ranking: {e}")


def process_pdfs_in_folder(input_pdf_folder, output_csv_folder):
    """
    Processes all PDF files in a given folder:
    1. Extracts features to a CSV using feature-extractor.py.
    2. Adds font size rank to the second last column of the *same* CSV file.

    Args:
        input_pdf_folder (str): The path to the folder containing PDF files.
        output_csv_folder (str): The path where the processed CSV files will be stored.
    """
    if not os.path.isdir(input_pdf_folder):
        print(f"Error: The input folder '{input_pdf_folder}' does not exist.")
        sys.exit(1)

    os.makedirs(output_csv_folder, exist_ok=True)
    print(f"Output CSVs will be saved in: {output_csv_folder}\n")

    pdf_files = [f for f in os.listdir(input_pdf_folder) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in '{input_pdf_folder}'.")
        return

    for idx, pdf_file in enumerate(pdf_files, start=1):
        pdf_path = os.path.join(input_pdf_folder, pdf_file)

        output_csv_path = os.path.join(output_csv_folder, f"output{idx}.csv")

        print(f"--- Processing '{pdf_file}' ---")

        # Step 1: Run feature-extractor.py
        try:
            print(f"Running feature-extractor for '{pdf_file}'...")
            subprocess.run(
                [sys.executable, "./Parser/PyMuPDF/pdf_to_csv_code_essentials/feature-extractor.py", pdf_path, output_csv_path],
                check=True
            )
            print(f"Features extracted to '{output_csv_path}'")
        except subprocess.CalledProcessError as e:
            print(f"Error running feature-extractor for '{pdf_file}': {e}")
            continue
        except FileNotFoundError:
            print("Error: 'feature-extractor.py' not found. Make sure it's in the same directory as this script.")
            sys.exit(1)

        # Step 2: Add font size rank to the same CSV file
        try:
            print(f"Adding font size rank to '{output_csv_path}'...")
            rank_column_values_in_place(output_csv_path, "norm_font_size")
        except Exception as e:
            print(f"Error adding font size rank to '{output_csv_path}': {e}")
            continue

        print(f"--- Finished processing '{pdf_file}' ---\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python orchestrator_script.py <path_to_input_pdf_folder> <path_to_output_csv_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    process_pdfs_in_folder(input_folder, output_folder)