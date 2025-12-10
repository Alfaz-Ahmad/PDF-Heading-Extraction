#
# Inference Script for Fine-Tuned ELECTRA-small Heading Detector
#
# This script loads a pre-trained text classification model (local or from Hugging Face Hub)
# and uses it to predict whether text chunks in new CSV files are headings.
#
# Example usage:
#   python inference.py --m itsGoodman/electra_small_heading_classifier --i ./input_csvs --o ./predictions
#   python inference.py --m ./local_model_folder --i ./input_csvs --o ./predictions
#

import pandas as pd
import argparse
import os
import torch
import glob
from transformers import pipeline
import warnings

# Suppress pandas warnings
warnings.filterwarnings("ignore", 'This pattern is deprecated.*')
warnings.filterwarnings("ignore", 'A value is trying to be set on a copy of a DataFrame from a chain walking method.*')

LABEL_NAMES = ["NOT_HEADING", "HEADING"]

def load_and_prepare_data_for_prediction(file_path):
    """Loads and prepares CSV data for prediction."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path, encoding_errors='ignore', on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    column_map = {
        'text': 'text',
        'font': 'font',
        'norm_font_size': 'norm_font_size',
        'page_number': 'page_number',
        'is_bold': 'is_bold',
        'is_italic': 'is_italic',
        'is_all_caps': 'is_all_caps',
        'is_centered': 'is_centered',
        'contains_number': 'contains_number',
        'contains_fullstop': 'contains_fullstop',
        'word_count': 'word_count',
        'norm_space_above': 'norm_space_above',
        'norm_space_below': 'norm_space_below',
        'norm_space_left': 'norm_space_left',
        'norm_space_right': 'norm_space_right',
        'norm_r': 'norm_r',
        'norm_g': 'norm_g',
        'norm_b': 'norm_b',
        'norm_x0': 'norm_x0',
        'norm_y0': 'norm_y0',
        'norm_x1': 'norm_x1',
        'norm_y1': 'norm_y1',
    }

    for col in column_map.keys():
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {file_path}. Ensure input CSV matches training format.")

    df_mapped = df.rename(columns=column_map)

    # Create combined_input string
    df_mapped['combined_input'] = "text: " + df_mapped['text'].fillna('').astype(str)
    for col in column_map.values():
        if col != 'text':
            df_mapped['combined_input'] += f" | {col.replace('_', ' ')}: " + df_mapped[col].fillna('').astype(str)

    return df_mapped


def main():
    parser = argparse.ArgumentParser(description="Run prediction with a fine-tuned ELECTRA-small model.")
    parser.add_argument(
        "--m", type=str, required=True,
        help="Model path or Hugging Face repo ID (e.g. 'itsgoodman/finetuned-electra-small')."
    )
    parser.add_argument(
        "--i", type=str, required=True,
        help="Input folder containing CSV files."
    )
    parser.add_argument(
        "--o", type=str, required=True,
        help="Output folder to save predictions."
    )
    args = parser.parse_args()

    os.makedirs(args.o, exist_ok=True)
    all_csv_files = glob.glob(os.path.join(args.i, "*.csv"))

    if not all_csv_files:
        print("No CSV files found. Exiting.")
        return

    print(f"Found {len(all_csv_files)} CSV file(s).")

    # --- Load Model (local path or repo ID) ---
    print(f"\n--- Loading model: {args.m} ---")
    try:
        classifier = pipeline(
            "text-classification",
            model=args.m,
            tokenizer=args.m,
            device=0 if torch.cuda.is_available() else -1
        )
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model {args.m}: {e}")
        return

    # --- Process each file ---
    for file_path in all_csv_files:
        print(f"\n--- Processing file: {file_path} ---")
        inference_df = load_and_prepare_data_for_prediction(file_path)
        if inference_df is None or inference_df.empty:
            print(f"Skipping invalid/empty file: {file_path}")
            continue

        texts_to_classify = inference_df['combined_input'].tolist()
        if not texts_to_classify:
            print("No text to classify.")
            continue

        try:
            predictions = classifier(texts_to_classify, padding=True, truncation=True, max_length=512)
            print("Prediction complete.")
        except Exception as e:
            print(f"Error during prediction: {e}")
            continue

        inference_df['predicted_class'] = [p['label'] for p in predictions]
        inference_df['prediction_score'] = [p['score'] for p in predictions]

        output_csv_path = os.path.join(args.o, os.path.splitext(os.path.basename(file_path))[0] + ".csv")
        inference_df.to_csv(output_csv_path, index=False)
        print(f"✅ Saved predictions: {output_csv_path}")

    print("\n🎉 All files processed successfully.")


if __name__ == "__main__":
    main()
