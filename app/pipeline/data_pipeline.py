#Import libraries
import drms
from drms import DrmsQueryError
from astropy.time import Time
import pandas as pd
from datetime import timedelta
import csv
import json
from os.path import exists
import numpy as np
import joblib
import torch
import yaml
import os
import logging
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from app.config.config import load_config

def setup_logger(log_file):
    # Create a logger object
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a file handler to write log messages to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler to print logs to the console (optional)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Adjust log level for console

    # Define a log format (with timestamp, log level, and message)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def download_data(logger, data_download_file):
    #Download data to csv, searching iteratively for the most recent available day's data.
    
    c = drms.Client()
    c.series(r'hmi\.sharp_')

    fields = ["T_REC", "HARPNUM", "NOAA_NUM", "NOAA_ARS", "NOAA_AR", "QUALITY", "TOTUSJH", "TOTUSJZ", "SAVNCPP", "USFLUX", "ABSNJZH", "TOTPOT","SIZE_ACR", 
        "NACR", "MEANPOT", "SIZE", "MEANJZH", "SHRGT45", "MEANSHR","MEANJZD", "MEANALP", "MEANGBT", "MEANGBL", "MEANGAM", "MEANGBZ", "MEANGBH", "NPIX"
        ]

    query_string = ",".join(fields)
    time_diff = timedelta(minutes=1440)
    first_write = True
    end = Time.now()
    start = end - time_diff
    
    if os.path.exists(data_download_file):
        os.remove(data_download_file)
        logger.info(f"File {data_download_file} has been deleted.")
    else:
        logger.info(f"File {data_download_file} does not exist.")
    while True:
        
        logger.info(f"start = {start}")
        logger.info(f"end = {end}")
        logger.info("Downloading data")

        t_2_str = end.strftime("%Y.%m.%d_%H:%M:%S_TAI") 
        t_1_str = start.strftime("%Y.%m.%d_%H:%M:%S_TAI") 

        try:
            extract = c.query(f'hmi.sharp_720s[1-13459][{t_1_str}-{t_2_str}]', key=query_string)

            if not extract.empty:
                extract.to_csv(data_download_file, mode='a', index=False, header=first_write)
                logger.info(f"Wrote {len(extract)} rows for {start} - {end}")
                if first_write:
                    first_write = False
                    start = start - time_diff
                    end = end - time_diff
                    continue
                else:
                    break
            else:
                logger.warning(f"No records available for {start} - {end}")
                start = start - time_diff
                end = end - time_diff

        except (DrmsQueryError, TimeoutError) as e:
            logger.error(f"JSOC query failed for {start}-{end}: {e}")
            start = start - time_diff
            end = end - time_diff

        except Exception as ex:
            logger.error(f"Fatal download error: {ex}", exc_info=True)
            raise


def replace_median_features(logger, data):
    
    try:
        median_features = ['SIZE_ACR', 'SIZE', 'NPIX', 'NACR']

        for feature in median_features:
            medians = data.groupby('HARPNUM')[feature].transform('median')
            data[feature] = data[feature].fillna(medians)

        return data
    
    except Exception as ex:
            logger.error(f"Failed to replace nulls with medians: {ex}", exc_info=True)
            raise 


def linear_interpolation(logger, data):

    try:
        linear_interpolation_features = ['TOTUSJH','TOTUSJZ', 'SAVNCPP', 'USFLUX', 'ABSNJZH', 'TOTPOT', 'MEANPOT', 'MEANJZH',
                                        'SHRGT45', 'MEANSHR', 'MEANJZD', 'MEANALP', 'MEANGBT', 'MEANGBL', 'MEANGAM', 'MEANGBZ', 'MEANGBH']

        LI_numeric_data = data.copy()
        LI_numeric_data[linear_interpolation_features] = data[linear_interpolation_features].apply(
            pd.to_numeric, errors='coerce'
        )

        LI_numeric_data_sorted = LI_numeric_data.sort_values(['HARPNUM', 'T_REC']).copy()

        for col in linear_interpolation_features:
            LI_numeric_data_sorted[col] = LI_numeric_data_sorted.groupby('HARPNUM')[col].transform(lambda g: g.interpolate(method='linear', limit_direction = 'both'))

        return LI_numeric_data_sorted
    
    except Exception as ex:
            logger.error(f"Failed to replace nulls with linear interpolation: {ex}", exc_info=True)
            raise


def prepare_sequence_dictionary(logger, data):

    try:
        harp_dict = {}
        grouped = data.groupby('HARPNUM')

        for harpnum, group in grouped:
            harp_dict[harpnum] = group

        sequence_length = 30  # 6 hours of 12-minute cadence
        cadence_upper = pd.Timedelta(minutes=13)
        cadence_lower = pd.Timedelta(minutes=11)
        sequence_dict = {}

        for harp_ID, sample in harp_dict.items():
            
            valid_sequences = []
            sample = sample.sort_values('T_REC').reset_index(drop=True)
            start_idx = 0

            while start_idx < (len(sample) - sequence_length + 1):
                    seq = sample.iloc[start_idx : start_idx + sequence_length]
                    time_deltas = seq['T_REC'].diff().dropna()

                    if all(time_deltas < cadence_upper) and all(time_deltas > cadence_lower):
                        valid_sequences.append(seq.reset_index(drop=True))
                        start_idx = start_idx + sequence_length
                    else:
                        start_idx += 1

            if len(valid_sequences) > 0:
                sequence_dict[harp_ID] = valid_sequences

        return sequence_dict

    except Exception as ex:
            logger.error(f"Failed to build sequence dictionary: {ex}", exc_info=True)
            raise
        
def extract_recent_sequences(logger, sequence_dict):

    try:
        sequence_list = []
        # Assuming `sequence_dict` is already defined
        for harpnum, sequences in sequence_dict.items():
            for index, sequence in enumerate(sequences):
                # Add the row to the list (instead of concatenating repeatedly)
                sequence_list.append({
                    "Harpnum": harpnum,
                    "T_REC": sequence["T_REC"].iloc[29],
                    "seq_number": index
                })

        # Create a DataFrame from the list of rows
        sequence_end_time = pd.DataFrame(sequence_list, columns=["Harpnum", "T_REC", "seq_number"])
        # Ensure 'T_REC' is in datetime format
        sequence_end_time['T_REC'] = pd.to_datetime(sequence_end_time['T_REC'])
        # Find the maximum T_REC (end time)
        max_end_time = sequence_end_time['T_REC'].max()
        # Filter the DataFrame to keep only the rows where T_REC is equal to the max end time
        max_end_time_records = sequence_end_time[sequence_end_time['T_REC'] == max_end_time]

        recent_sequence_dict = {}
        # Assuming `max_end_time_records` is the DataFrame and `harpnum` is the key you're filtering by
        for harpnum, sequences in sequence_dict.items():
            # Check if the harpnum exists in max_end_time_records to avoid errors
            if harpnum in max_end_time_records['Harpnum'].values:
                seq_index = max_end_time_records[max_end_time_records['Harpnum'] == harpnum]['seq_number'].iloc[0]
                recent_sequence_dict[harpnum] = sequences[seq_index]
        
        return recent_sequence_dict, max_end_time
    
    except Exception as ex:
            logger.error(f"Failed to extract recent sequences: {ex}", exc_info=True)
            raise

def transform_to_tensor(logger, recent_sequence_dict, data_pipeline_file, scaler_file):
    
    try:
        X_list = []
        for harpnum, sequence in recent_sequence_dict.items():
            #sequence_array = sequence.select_dtypes(include='number').to_numpy()
            sequence_array = sequence.drop(columns=['T_REC', 'HARPNUM', 'NOAA_NUM', 'NOAA_AR', 'QUALITY']).to_numpy()
            X_list.append(sequence_array)
    
        X_real_time_scaled = []
        # Load the pre-fitted scaler (from training data)
        scaler = joblib.load(scaler_file)
        # Assuming `X_list` contains multiple sequences (each sequence is a 2D array of shape (30, n_features))
        for seq in X_list:
            # Transform the sequence using the pre-fitted scaler (no fitting, just transforming)
            seq_scaled = scaler.transform(seq)
            # Append the scaled sequence to the list
            X_real_time_scaled.append(seq_scaled)

        # Convert the scaled sequences list to a numpy array if needed
        X_real_time_scaled = np.array(X_real_time_scaled)
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_real_time_scaled, dtype=torch.float32)
        torch.save({'X_tensor': X_tensor,}, data_pipeline_file)
    
    except Exception as ex:
            logger.error(f"Failed to transform data to tensor: {ex}", exc_info=True)
            raise
        

def prepare_data(logger, data_download_file, scaler_file, data_pipeline_file):
    
    data = pd.read_csv(data_download_file)
    data = data.drop_duplicates()
    data.drop(columns='NOAA_ARS', inplace=True)
    data.replace(['MISSING', 'NaN'], np.nan, inplace=True)
    data['T_REC'] = pd.to_datetime(data['T_REC'].str.replace('_TAI', ''), format='%Y.%m.%d_%H:%M:%S')

    data = replace_median_features(data)
    data = linear_interpolation(data)
    sequence_dict = prepare_sequence_dictionary(data)
    recent_sequence_dict, forecast_time = extract_recent_sequences(sequence_dict)
    harpnums = list(recent_sequence_dict.keys())
    transform_to_tensor(recent_sequence_dict, data_pipeline_file, scaler_file) 

    return forecast_time, harpnums


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, bidirectional=True, dropout_p=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout_p, bidirectional=bidirectional,
                            batch_first=True)

        self.dropout = nn.Dropout(dropout_p)
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, 1)  # Regression output

    def forward(self, x):
        out, _ = self.lstm(x)
        last_time_step_out = out[:, -1, :]
        dropped = self.dropout(last_time_step_out)  # MC Dropout always applied
        out = self.fc(dropped)
        return out.squeeze()


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()  # Force dropout ON


def mc_predict(model, x_batch, n_passes=50):
    try:
        model.eval()
        enable_mc_dropout(model)

        preds = []
        for _ in range(n_passes):
            with torch.no_grad():
                preds.append(model(x_batch).cpu().numpy())

        preds = np.stack(preds, axis=0)  # shape: (n_passes, batch_size)
        mean = preds.mean(axis=0)        # shape: (batch_size,)
        std = preds.std(axis=0)          # shape: (batch_size,)
        return mean, std
    
    except Exception as ex:
            logger.error(f"Failed to generate monte carlo prediction distributon: {ex}", exc_info=True)
            raise


def initialise_model(logger, model_weights_file):
    
    try:
        input_size = 21  # <-- replace with your real input size
        model = LSTMModel(input_size=input_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_weights_file, map_location=device))
        model.to(device)
        model.eval()
        logger.info("Model initialised")

        return model, device
    
    except Exception as ex:
            logger.error(f"Failed to initialise model: {ex}", exc_info=True)
            raise


def evaluate_predictions(logger, model, device, data_pipeline_file):
    
    try:
        logger.info("Loading data")
        data = torch.load(data_pipeline_file)
        X_tensor = data['X_tensor']
        logger.info("Preparing data sequence")
        real_time_dataset = SequenceDataset(X_tensor)

        # Create the DataLoader for real-time evaluation
        logger.info("Building data loader")
        real_time_loader = DataLoader(real_time_dataset, batch_size=32, shuffle=False)

        all_means = []
        all_stds = []
        
        logger.info("Enabling Monte Carlo dropout")
        enable_mc_dropout(model)
        
        logger.info("Executing predictions")
        with torch.no_grad():
            for X_batch in real_time_loader:
                X_batch = X_batch.to(device)
                batch_mean, batch_std = mc_predict(model, X_batch, n_passes=50)
                all_means.extend(batch_mean)
                all_stds.extend(batch_std)

        all_means = np.array(all_means)
        all_stds = np.array(all_stds)
        mean_linear = all_means
        std_linear = ((all_means + all_stds)) - mean_linear  # approximate upper range

        # for i in range(len(mean_linear)):
        #     print(f"Sample {i}: Flare = {mean_linear[i]:.2f}, ± {std_linear[i]:.2f}")

        return mean_linear, std_linear
    
    except Exception as ex:
            logger.error(f"Failed to evaluate predictions: {ex}", exc_info=True)
            raise


def publish_forecast(logger, forecast_file, mean_linear, std_linear, forecast_time, harpnums):
    
    try:
        forecast_time = str(forecast_time)
        data = {"forecast_time": forecast_time, "measurements": []}

        for i in range(len(harpnums)):
            measurement = {
                "harpnum": harpnums[i],
                "intensity": round(float(mean_linear[i]), 2),
                "error": round(float(std_linear[i]), 2)
            }
            data["measurements"].append(measurement)

        with open(forecast_file, 'w') as file:
            json.dump(data, file, indent=2)

    except Exception as ex:
            logger.error(f"Failed to publish forecast: {ex}", exc_info=True)
            raise



if __name__ == "__main__":

    config = load_config()

    # Accessing file paths
    pipeline_log_file = config['paths']['pipeline_log_file']
    data_download_file = config['paths']['data_download_file']
    scaler_file = config['paths']['scaler_file']
    data_pipeline_file = config['paths']['data_pipeline_file']
    model_weights_file = config['paths']['model_weights_file']
    forecast_file = config['paths']['forecast_file']

    logger = setup_logger(pipeline_log_file)
    logger.info("Starting download...")
    download_data(logger, data_download_file)
    logger.info("Preparing data...")
    forecast_time, harpnums = prepare_data(data_download_file, scaler_file, data_pipeline_file)
    logger.info("Success. Data prepared for model input.")
    logger.info("Initialising model...")
    model, device = initialise_model(model_weights_file)
    logger.info("Evaluating forecast...")
    mean_linear, std_linear = evaluate_predictions(model, device, data_pipeline_file)
    publish_forecast(forecast_file, mean_linear, std_linear, forecast_time, harpnums)
