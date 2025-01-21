import pandas as pd
import json
import joblib
import os
import logging
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from ..check_structure import check_existing_file, check_existing_folder
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FOLDER = BASE_DIR / "data" / "processed"
MODEL_FOLDER = BASE_DIR / "models"
METRICS_FOLDER = BASE_DIR / "metrics"
OUTPUT_FOLDER = METRICS_FOLDER

# Load the trained model from a joblib file
def load_trained_model():
    """
    Load the trained model from the saved file.
    """
    model_path = MODEL_FOLDER / "trained_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

# Load test data
def load_test_data():
    """
    Load test dataset (X_test and y_test).
    """
    X_test = pd.read_csv(os.path.join(INPUT_FOLDER, "X_test_scaled.csv"))
    y_test = pd.read_csv(os.path.join(INPUT_FOLDER, "y_test.csv"))
    y_test = y_test.values.ravel()  
    return X_test, y_test

# Evaluate the model
def evaluate_model(model, X_test, y_test, output_folderpath):
    """
    Evaluate the model using test data and calculate various metrics.
    """
    # Predict with the trained model
    y_pred = model.predict(X_test)

    # Calculate MSE, R^2, and optionally more metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Store metrics in a dictionary
    metrics = {
        "mse": mse,
        "r2": r2,
        "mae": mae,
    }
    # Create folder if necessary 
    if check_existing_folder(output_folderpath) :
        os.makedirs(output_folderpath)

    #--Saving the best params in .pkl file
    for file, filename in zip([metrics], ['scores']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.json')
        if check_existing_file(output_filepath):
            joblib.dump(metrics, output_filepath)
    return metrics

# Main function
def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info("Loading the trained model...")
    model = load_trained_model()

    logger.info("Loading the test data...")
    X_test, y_test = load_test_data()

    logger.info("Evaluating the model...")
    metrics = evaluate_model(model, X_test, y_test, OUTPUT_FOLDER)

    logger.info(f"Model evaluation metrics: {metrics}")
    
if __name__ == '__main__':
    main()
