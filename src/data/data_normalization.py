# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from ..check_structure import check_existing_file, check_existing_folder
import os

#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=False), required=0)
#@click.argument('output_filepath', type=click.Path(exists=False), required=0)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FOLDER = BASE_DIR / "data" / "processed"
OUTPUT_FOLDER = BASE_DIR / "data" / "processed"

def main():
    """ Normalizes the X_test, X_train datasets
    """
    
    logger = logging.getLogger(__name__)
    logger.info('normalizing feature data')
    
    input_filepath_test = os.path.join(INPUT_FOLDER, "X_test.csv")
    input_filepath_train = os.path.join(INPUT_FOLDER, "X_train.csv")
    output_folderpath = OUTPUT_FOLDER
    
    # Call the main data processing function with the provided file paths
    normalize_data(input_filepath_test, input_filepath_train, output_folderpath)

def normalize_data(input_filepath_test, input_filepath_train, output_folderpath):
 
    #--Importing dataset
    X_train = pd.read_csv(input_filepath_train, sep=",")
    X_test = pd.read_csv(input_filepath_test, sep=",")
    
    features = ["ave_flot_air_flow","ave_flot_level","iron_feed","starch_flow","amina_flow","ore_pulp_flow","ore_pulp_pH","ore_pulp_density"]
    
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(X_train)
    scaled_test = scaler.transform(X_test)
    
    scaled_train = pd.DataFrame(scaled_train, columns = features)
    scaled_test = pd.DataFrame(scaled_test, columns = features)

    # Create folder if necessary 
    if check_existing_folder(output_folderpath) :
        os.makedirs(output_folderpath)

    #--Saving the dataframes to their respective output file paths
    for file, filename in zip([scaled_train, scaled_test], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            print(file.head())
            file.to_csv(output_filepath, index=False, header=True, encoding='utf-8')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]


    main()
