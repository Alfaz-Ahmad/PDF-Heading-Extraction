import pandas as pd
import argparse
import os
from transformers import pipeline
import warnings

# Suppress specific pandas warnings that are not relevant here
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# --- 1. Configuration ---

# These names are used for clear output. They correspond to the
# model's output labels (e.g., 'LABEL_0', 'LABEL_1').
LABEL_MAPPING = {
    'LABEL_0': 'NOT_HEADING',
    'LABEL_1': 'HEADING'
}


# --- 2. Data Loading and Pre-processing ---

def load_and_prepare_data_for_prediction(file_path):
    """
    Loads data from a single CSV and prepares it for prediction.
    It expects a column named 'text'.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The specified input file was not found: {file_path}")

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, encoding_errors='ignore')

    # Ensure the required 'text' column exists
    if 'text' not in df.columns:
        raise ValueError(f"Missing required column 'text' in the input CSV file: {file_path}")

    return df


# --- 3. Main Inference Logic ---

def main():
    """Main function to run the prediction process on a folder of CSV files."""

    parser = argparse.ArgumentParser(description="Run sentiment analysis with a pre-trained DistilBERT model on multiple CSV files in a folder.")
    # New argument to specify the local model directory
    parser.add_argument(
        "--m",
        type=str,
        required=True,
        help="Path to the local directory containing the downloaded model and tokenizer."
    )
    # Changed argument from --input_csv to --input_dir for folder processing
    parser.add_argument(
        "--i",
        type=str,
        required=True,
        help="Path to the folder containing the CSV files for prediction. Each CSV must contain a 'text' column."
    )
    # Changed argument from --output_csv to --output_dir for folder processing
    parser.add_argument(
        "--o",
        type=str,
        required=True,
        help="Path where the new CSV files with predictions will be saved."
    )
    args = parser.parse_args()

    # --- Load Model ---
    print(f"\n--- Loading model from local directory: {args.m} ---")
    if os.path.isdir(args.m):
        model_path = args.m  # local folder
    else:
        model_path = args.m

    try:
        # We need to specify truncation=True here to handle text that is longer
        # than the model's maximum input size (which is 512 for this model).
        # This will prevent the runtime error you were seeing.
        classifier = pipeline(
            "sentiment-analysis",
            model=args.m,
            tokenizer=args.m,
            truncation=True,
            padding=True
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Create output directory if it doesn't exist ---
    if not os.path.exists(args.o):
        os.makedirs(args.o)

    # --- Process each CSV file in the input directory ---
    csv_files = [f for f in os.listdir(args.i) if f.endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in the input directory: {args.i}")
        return

    for csv_file in csv_files:
        input_path = os.path.join(args.i, csv_file)
        output_path = os.path.join(args.o, csv_file)

        try:
            inference_df = load_and_prepare_data_for_prediction(input_path)
            texts_to_classify = inference_df['text'].tolist()

            print(f"\n--- Running prediction on {len(texts_to_classify)} text chunks in {csv_file}... ---")
            predictions = classifier(texts_to_classify)
            print("Prediction complete.")

            if not predictions:
                print(f"No predictions were generated for {csv_file}. Skipping save.")
                continue

            # --- Process Results and Prepare for CSV ---
            predicted_labels = []
            confidence_scores = []

            for result in predictions:
                predicted_sentiment = LABEL_MAPPING.get(result['label'], result['label'])
                predicted_labels.append(predicted_sentiment)
                confidence_scores.append(result['score'])

            # Add the predictions as new columns to the DataFrame
            inference_df['predicted_class'] = predicted_labels
            inference_df['confidence_score'] = confidence_scores

            # --- Print a summary of the results ---
            print(f"Saving results for {csv_file} to {output_path}...")
            inference_df.to_csv(output_path, index=False)
            print(f"Results successfully saved.")

        except (FileNotFoundError, ValueError) as e:
            print(f"Error processing {csv_file}: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred while processing {csv_file}: {e}")
            continue

    print("\nAll files processed. Script finished.")


if __name__ == "__main__":
    main()
