import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier  # Changed from xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, \
    recall_score
import os
import argparse
import joblib

# This dictionary maps the original column names to the standardized names.
column_map = {
    'text': 'text',
    'font': 'font',
    'normalised_font_size_rank': 'norm_font_size_rank',
    'page_number': 'page_number',
    'is_bold': 'is_bold',
    'is_italic': 'is_italic',
    'is_underline': 'is_underline',
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
    'class': 'class'
}

# The user-specified target variable is 'class'.
TARGET_COLUMN = 'class'

# The features for the model are selected from the provided list.
feature_columns = [
    'is_bold', 'is_italic', 'is_underline', 'is_all_caps', 'is_centered',
    'contains_number', 'contains_fullstop', 'word_count',
    'norm_font_size_rank', 'norm_space_above', 'norm_space_below',
    'norm_space_left', 'norm_space_right',
    'norm_r', 'norm_g', 'norm_b', 'norm_x0', 'norm_y0', 'norm_x1', 'norm_y1'
]


def train_and_evaluate_model(df, output_path):
    """
    Trains a Random Forest model, evaluates it, and saves the results.
    """
    # --- 2. DATA PREPARATION ---
    # Separate features (X) and target variable (y)
    # Perform a document-aware train-test split to prevent data leakage.
    document_ids = df['document_id'].unique()
    train_docs, test_docs = train_test_split(document_ids, test_size=0.2, random_state=42)

    # Filter the DataFrame to create training and testing sets based on the document IDs
    train_df = df[df['document_id'].isin(train_docs)].reset_index(drop=True)
    test_df = df[df['document_id'].isin(test_docs)].reset_index(drop=True)

    # Extract features and target variable from the document-split DataFrames
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    print(f"\nTraining set size: {X_train.shape[0]} samples from {len(train_docs)} documents")
    print(f"Testing set size: {X_test.shape[0]} samples from {len(test_docs)} documents")

    # Label encode the target variable
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    print("Labels encoded successfully.")

    # Save the label encoder
    label_encoder_path = os.path.join(os.getcwd(), 'label_encoder.joblib')
    joblib.dump(le, label_encoder_path)
    print(f"Label encoder saved to: {label_encoder_path}")

    # Scale the numerical features using a MinMaxScaler
    scaler = MinMaxScaler()
    numerical_features = [col for col in feature_columns if 'norm_' in col or 'word_count' in col]
    X_train_scaled_numerical = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled_numerical = scaler.transform(X_test[numerical_features])

    # Combine the scaled numerical features with the unscaled boolean features
    boolean_features = [col for col in feature_columns if 'is_' in col or 'contains_' in col]
    X_train_scaled = np.hstack([X_train_scaled_numerical, X_train[boolean_features].values])
    X_test_scaled = np.hstack([X_test_scaled_numerical, X_test[boolean_features].values])
    print("Features scaled successfully.")

    # --- 3. MODEL TRAINING (RANDOM FOREST) ---
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    print("\nStarting model training...")
    model.fit(X_train_scaled, y_train_encoded)
    print("Model training complete.")

    # Save the trained model and scaler
    model_path = os.path.join(os.getcwd(), 'random_forest_model.joblib')  # Renamed file
    scaler_path = os.path.join(os.getcwd(), 'minmax_scaler.joblib')
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

    # --- 4. MODEL EVALUATION ---
    print("\n--- Model Evaluation ---")
    y_pred_encoded = model.predict(X_test_scaled)
    y_pred = le.inverse_transform(y_pred_encoded)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Save metrics to a text file
    metrics_path = os.path.join(output_path, 'model_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Model Evaluation Metrics (Random Forest)\n")
        f.write("----------------------------------------\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score (weighted): {f1:.4f}\n")
        f.write(f"Precision (weighted): {precision:.4f}\n")
        f.write(f"Recall (weighted): {recall:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        np.savetxt(f, cm, fmt='%d')

    print(f"Metrics saved to: {metrics_path}")

    # Save confusion matrix as an image
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix (Random Forest)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_path = os.path.join(output_path, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix image saved to: {cm_path}")
    plt.close()

    return model, scaler, numerical_features, boolean_features, le


def main():
    print("Script execution started.")
    parser = argparse.ArgumentParser(description="Train a Random Forest model for document element classification.")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Path to the input dataset file (e.g., 'your_dataset.csv').")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Path to the directory where results (metrics, plots) will be saved.")
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Load data from the specified input file
    print(f"Loading data from {args.input}...")
    try:
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Error: Input file not found at {args.input}")

        df = pd.read_csv(args.input)

        # Clean the data by dropping rows with missing 'class' values
        df.dropna(subset=[TARGET_COLUMN], inplace=True)
        # Convert the 'class' column to a string type to avoid mixed-type errors
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str)

        # Filter out rows with the 'h' class
        df = df[df['class'] != 'h'].reset_index(drop=True)

        # Check for required columns again after cleaning
        required_cols = feature_columns + [TARGET_COLUMN, 'document_id']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Error: The input CSV is missing required columns: {missing_cols}")

    except (FileNotFoundError, ValueError) as e:
        print(e)
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading the CSV file: {e}")
        return

    # Train and evaluate the model
    model, scaler, numerical_features, boolean_features, le = train_and_evaluate_model(df, args.output)

    # --- Example of making a prediction using the loaded model ---
    print("\n--- Example Prediction with Loaded Model ---")

    # Create a new, unseen data point
    new_data_point = {
        'is_bold': 1, 'is_italic': 0, 'is_underline': 0, 'is_all_caps': 1,
        'is_centered': 1, 'contains_number': 0, 'contains_fullstop': 0,
        'word_count': 3, 'norm_font_size_rank': 0.98, 'norm_space_above': 0.85,
        'norm_space_below': 0.1, 'norm_space_left': 0.05, 'norm_space_right': 0.05,
        'norm_r': 0.1, 'norm_g': 0.1, 'norm_b': 0.1, 'norm_x0': 0.2,
        'norm_y0': 0.2, 'norm_x1': 0.8, 'norm_y1': 0.3
    }

    # Load the saved model and scaler
    try:
        loaded_model = joblib.load('random_forest_model.joblib') # Renamed file
        loaded_scaler = joblib.load('minmax_scaler.joblib')
        loaded_le = joblib.load('label_encoder.joblib')

        # Create a DataFrame and scale the new data point
        new_df = pd.DataFrame([new_data_point])
        new_data_scaled_numerical = loaded_scaler.transform(new_df[numerical_features])
        new_data_scaled_boolean = new_df[boolean_features].values
        new_data_scaled = np.hstack([new_data_scaled_numerical, new_data_scaled_boolean])

        # Predict the class for the new data point
        predicted_class_encoded = loaded_model.predict(new_data_scaled)
        predicted_class = loaded_le.inverse_transform(predicted_class_encoded)
        predicted_proba = loaded_model.predict_proba(new_data_scaled)

        print(f"The new data point is predicted to be class: {predicted_class[0]}")
        print("Prediction probabilities per class:")
        for i, prob in enumerate(predicted_proba[0]):
            print(f"Class '{loaded_le.classes_[i]}': {prob:.4f}")

    except FileNotFoundError:
        print(
            "Error: Could not load the saved model, scaler, or label encoder. They must be in the same directory as the script.")
        return


if __name__ == "__main__":
    main()
