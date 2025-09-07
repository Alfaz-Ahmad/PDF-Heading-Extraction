#
# Inference Script for Fine-Tuned DeBERTa-v3 Heading Detector
#
# This script loads a pre-trained text classification model and uses it to
# predict whether text chunks in new CSV files are headings.
#
# This version is designed for prediction on new data and processes
# ALL CSV files found in a specified input folder. It saves the
# results for each processed file to a corresponding file in an output folder.
#
# How to run:
# 1. Ensure you have the required libraries installed:
#    pip install transformers pandas torch scikit-learn sentencepiece
# 2. Run the script from your terminal, providing the paths to your model,
#    the new data folder, and an output path for the results:
#    python test-berttiny9.py --model_path ./b --input_folder ./input_data --output_folder ./output_predictions
#
#    - --model_path should be the directory where the fine-tuned model was saved.
#    - --input_folder must be the path to a folder containing the CSV files you want to predict on.
#    - --output_folder is the path where the new CSV files with predictions will be saved.
#      The script will create this folder if it doesn't exist.
#

import pandas as pd
import argparse
import os
import torch
from transformers import pipeline
import warnings
import glob

# Suppress specific pandas warnings that are not relevant here
warnings.filterwarnings("ignore", 'This pattern is deprecated.*')
warnings.filterwarnings("ignore", 'The sentencepiece tokenizer does not have a max length specified.*')


# --- 1. Configuration ---

# These names are used for clear output. They should correspond to the
# model's output labels (e.g., 'LABEL_0', 'LABEL_1').
LABEL_NAMES = ["NOT_HEADING", "HEADING"]


# --- 2. Data Loading and Pre-processing ---

def load_and_prepare_data_for_prediction(file_path):
    """
    Loads data from a single CSV and prepares it for prediction by creating the
    'combined_input' feature. This function does not expect a 'class' column.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The specified input file was not found: {file_path}")

    print(f"Loading and preparing data from {file_path}...")
    df = pd.read_csv(file_path, encoding_errors='ignore')

    # --- Column Mapping ---
    # Maps the expected column names in your CSV to the names the script uses.
    column_map = {
        'text': 'text',
        'font': 'font',
        'norm_font_size': 'normalised_font_size'
        # 'class' column is not required for prediction.
    }

    # Check if all required source columns exist in the DataFrame
    for source_col in column_map.keys():
        if source_col not in df.columns:
            raise ValueError(f"Missing required column '{source_col}' in the input CSV file: {file_path}")

    # Rename columns based on the map
    df = df.rename(columns=column_map)

    # --- Feature Engineering ---
    # Combine text and style features into the single input string the model expects.
    # This format must exactly match the one used during training.
    df['combined_input'] = (
        "text: " + df['text'].astype(str).str.strip() +
        " | font: " + df['font'].astype(str).str.strip() +
        " | size: " + df['normalised_font_size'].astype(str)
    )

    print("Data preparation complete.")
    return df


# --- 3. Main Inference Logic ---

def main():
    """Main function to run the prediction process on a folder of CSV files and save results."""

    parser = argparse.ArgumentParser(description="Run prediction with a fine-tuned DeBERTa-v3 model on a folder of CSV files.")
    parser.add_argument(
        "--m",
        type=str,
        required=True,
        help="Path to the directory containing the fine-tuned model and tokenizer."
    )
    parser.add_argument(
        "--i",
        type=str,
        required=True,
        help="Path to the folder containing the CSV files for prediction."
    )
    parser.add_argument(
        "--o",
        type=str,
        required=True,
        help="Path where the new CSV files with predictions will be saved."
    )
    args = parser.parse_args()

    # --- Load Model ---
    if os.path.isdir(args.m):
        model_path = args.m  # local folder
    else:
        model_path = args.m

    print(f"\n--- Loading model from {args.m} ---")
    # Use the pipeline for easy text classification inference
    classifier = pipeline("text-classification", model=args.m, tokenizer=args.m, device=0 if torch.cuda.is_available() else -1)
    print("Model loaded successfully.")

    # --- Get list of files to process ---
    if not os.path.isdir(args.i):
        raise FileNotFoundError(f"Input folder not found at: {args.i}")

    csv_files = glob.glob(os.path.join(args.i, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in the specified input folder: {args.i}")
        return

    # Create output folder if it doesn't exist
    os.makedirs(args.o, exist_ok=True)
    print(f"Processing {len(csv_files)} files...")

    # --- Process each file ---
    for input_file_path in csv_files:
        try:
            # --- Load and Prepare Data ---
            inference_df = load_and_prepare_data_for_prediction(input_file_path)
            texts_to_classify = inference_df['combined_input'].tolist()
            
            if not texts_to_classify:
                print(f"Skipping empty file: {input_file_path}")
                continue

            # --- Run Inference ---
            print(f"\n--- Running prediction on {len(texts_to_classify)} text chunks in {os.path.basename(input_file_path)} ---")
            
            # This fixes the tensor size mismatch error. The DeBERTa-v3 model has a max
            # input size of 512 tokens. While `truncation=True` is set, some models
            # require the `max_length` to be explicitly provided.
            predictions = classifier(texts_to_classify, padding=True, truncation=True, max_length=512)
            
            print("Prediction complete.")

            # --- Process Results and Prepare for CSV ---
            if not predictions:
                print("No predictions were generated. Skipping save.")
                continue

            # Extract labels and scores from the predictions
            predicted_labels = []
            confidence_scores = []
            
            for result in predictions:
                # Determine the predicted class name
                predicted_label_name = ''
                if result['label'] == 'HEADING':
                    predicted_label_name = 'HEADING'
                elif result['label'] == 'NOT_HEADING':
                    predicted_label_name = 'NOT_HEADING'
                else:  # Fallback for models that output generic labels like 'LABEL_0', 'LABEL_1'
                    try:
                        predicted_label_id = int(result['label'].split('_')[-1])
                        predicted_label_name = LABEL_NAMES[predicted_label_id]
                    except (ValueError, IndexError):
                        predicted_label_name = result['label']

                predicted_labels.append(predicted_label_name)
                confidence_scores.append(result['score'])

            # Add the predictions as new columns to the DataFrame
            inference_df['predicted_class'] = predicted_labels
            inference_df['confidence_score'] = confidence_scores

            # --- Print a summary of the results ---
            print("\n--- Prediction Results Summary ---")
            print(f"Total rows predicted: {len(inference_df)}")
            print(inference_df[['text', 'predicted_class', 'confidence_score']].head()) # Show a few rows

            # --- Save the DataFrame to a new CSV file ---
            output_file_name = os.path.basename(input_file_path)
            output_file_path = os.path.join(args.o, output_file_name)
            
            print(f"Saving results to {output_file_path}...")
            # We drop the 'combined_input' column to clean up the final output.
            output_df = inference_df.drop(columns=['combined_input'])
            output_df.to_csv(output_file_path, index=False)
            print(f"Results successfully saved to {output_file_path}")
            
        except Exception as e:
            print(f"An error occurred while processing {input_file_path}: {e}")
            continue

    print("\n\nScript finished.")


if __name__ == "__main__":
    main()