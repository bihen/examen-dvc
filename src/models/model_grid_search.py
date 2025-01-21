import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from check_structure import check_existing_file, check_existing_folder
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FOLDER = BASE_DIR / "data" / "processed"
OUTPUT_FOLDER = BASE_DIR / "data" / "processed"

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    
    logger = logging.getLogger(__name__)
    logger.info('normalizing feature data')
    
    input_filepath_test = os.path.join(INPUT_FOLDER, "X_test_scaled.csv")
    input_filepath_train = os.path.join(INPUT_FOLDER, "X_train_scaled.csv")
    output_folderpath = OUTPUT_FOLDER
    
    # Call the main data processing function with the provided file paths
    process_data(input_filepath_test, input_filepath_train, output_folderpath)

def process_data(input_filepath_test, input_filepath_train, output_folderpath):
 
    #--Importing dataset
    X_train = pd.read_csv(input_filepath_train, sep=",")
    X_test = pd.read_csv(input_filepath_test, sep=",")
    
    elasticnet = ElasticNet()
    
    param_grid = {
    'alpha': np.logspace(-5, 1, 7),        # Regularization strength
    'l1_ratio': np.linspace(0, 1, 11)      # Ratio between Lasso (L1) and Ridge (L2)
    }
    # Create folder if necessary 
    if check_existing_folder(output_folderpath) :
        os.makedirs(output_folderpath)

    #--Saving the dataframes to their respective output file paths
    for file, filename in zip([scaled_train, scaled_test], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]


    main()