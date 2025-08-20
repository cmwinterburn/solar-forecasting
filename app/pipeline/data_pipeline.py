import drms
from drms import DrmsQueryError
from astropy.time import Time
import pandas as pd
from datetime import timedelta
import json
from os.path import exists
import numpy as np
import joblib
import torch
import os
import logging
import torch.nn as nn
from torch.utils.data import DataLoader
from app.config.config import load_config
from app.database.io import insert_forecast
import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger(log_file, max_bytes=5_000_000, backup_count=3):
    """Return a logger object to record pipeline execution for debugging."""
    
    # Define a job specific logger.
    logger = logging.getLogger('solar_forecast')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False   

    # Prevent duplicate handlers if logger is already set up.
    if logger.handlers:
        return logger

    # Define a filehandler to rotate the log after exceeding the set size threshold.
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(logging.DEBUG)

    # Output logs to console at INFO level and above.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Standardise log format.
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger object.
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def download_data(logger, data_download_file):
    """Download SHARP data from the JSOC API to csv, searching iteratively for the most recent available day's data."""

    logger.info("Starting download.")
    # Define API client and data series for extraction.
    c = drms.Client()
    c.series(r'hmi\.sharp_')
    
    # Define data features to be extracted.
    fields = ["T_REC", "HARPNUM", "NOAA_NUM", "NOAA_ARS", "NOAA_AR", "QUALITY", "TOTUSJH", "TOTUSJZ", "SAVNCPP", "USFLUX", "ABSNJZH", "TOTPOT","SIZE_ACR", 
        "NACR", "MEANPOT", "SIZE", "MEANJZH", "SHRGT45", "MEANSHR","MEANJZD", "MEANALP", "MEANGBT", "MEANGBL", "MEANGAM", "MEANGBZ", "MEANGBH", "NPIX"
        ]
    
    # Build query to JSOC API.
    query_string = ",".join(fields)
    time_diff = timedelta(minutes=1440)
    first_write = True
    end = Time.now()
    start = end - time_diff
    days_of_data = 1
    i = 0
    
    # Remove any existing data dump before proceeding with data extraction.
    if os.path.exists(data_download_file):
        os.remove(data_download_file)
        logger.info(f"File {data_download_file} has been deleted.")
    else:
        logger.info(f"File {data_download_file} does not exist.")

    # Loop back through days to extract the most recent available JSOC SHARP data.
    while True:
        
        logger.info(f"start = {start}")
        logger.info(f"end = {end}")
        logger.info("Downloading data")
        
        t_2_str = end.strftime("%Y.%m.%d_%H:%M:%S_TAI") 
        t_1_str = start.strftime("%Y.%m.%d_%H:%M:%S_TAI") 

        try:
            extract = c.query(f'hmi.sharp_720s[1-13459][{t_1_str}-{t_2_str}]', key=query_string)
            
            # Extract defined count of days of data to ensure sufficient sequences available for prediction.
            if not extract.empty:
                extract.to_csv(data_download_file, mode='a', index=False, header=first_write)
                logger.info(f"Wrote {len(extract)} rows for {start} - {end}")
                if first_write:
                    first_write = False
                    start = start - time_diff
                    end = end - time_diff
                    i += 1
                    continue
                elif i < days_of_data:
                    i += 1
                    start = start - time_diff
                    end = end - time_diff
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
    """Replace null values with feature medians for selected SHARP features. Return transformed dataframe."""

    try:
        median_features = ['SIZE_ACR', 'SIZE', 'NPIX', 'NACR']
    
        # Group by HARPNUM to fill with medians per active region.
        for feature in median_features:
            medians = data.groupby('HARPNUM')[feature].transform('median')
            data[feature] = data[feature].fillna(medians)

        return data
    
    except Exception as ex:
            logger.error(f"Failed to replace nulls with medians: {ex}", exc_info=True)
            raise 


def linear_interpolation(logger, data):
    """Replace null values with linear interpolated values for selected time dependent SHARP features. Return transformed dataframe."""

    try:
        linear_interpolation_features = ['TOTUSJH','TOTUSJZ', 'SAVNCPP', 'USFLUX', 'ABSNJZH', 'TOTPOT', 'MEANPOT', 'MEANJZH',
                                        'SHRGT45', 'MEANSHR', 'MEANJZD', 'MEANALP', 'MEANGBT', 'MEANGBL', 'MEANGAM', 'MEANGBZ', 'MEANGBH']
        
        # Convert features to numeric datatype.
        LI_numeric_data = data.copy()
        LI_numeric_data[linear_interpolation_features] = data[linear_interpolation_features].apply(
            pd.to_numeric, errors='coerce'
        )
    
        # Sort features by Harpnum, timestamp.
        LI_numeric_data_sorted = LI_numeric_data.sort_values(['HARPNUM', 'T_REC']).copy()
        
        # Calculate linear interpolated value from adjacent records and replace.
        for col in linear_interpolation_features:
            LI_numeric_data_sorted[col] = LI_numeric_data_sorted.groupby('HARPNUM')[col].transform(lambda g: g.interpolate(method='linear', limit_direction = 'both'))

        return LI_numeric_data_sorted
    
    except Exception as ex:
            logger.error(f"Failed to replace nulls with linear interpolation: {ex}", exc_info=True)
            raise


def prepare_sequence_dictionary(logger, data):
    """Return a dictionary of lists of SHARP data sequences mapped to each distinct HARMPNUM (solar active region)."""
    
    try:
        # Initialise dictionary and dataframe per HARPNUM.
        harp_dict = {}
        grouped = data.groupby('HARPNUM')
        
        # Create keys for each HARPNUM.
        for harpnum, group in grouped:
                harp_dict[harpnum] = group
        
        # Sequence length corresponding to 6 hours of 12-minute cadence
        sequence_length = 30

        # Define short 2 minute buffer to ensure no valid sequences are overlooked.
        cadence_upper = pd.Timedelta(minutes=13)
        cadence_lower = pd.Timedelta(minutes=11)
        sequence_dict = {}

        # Search through each sample to extract valid sequences with no gaps.
        for harp_ID, sample in harp_dict.items():
            
            valid_sequences = []
            sample = sample.sort_values('T_REC').reset_index(drop=True)
            i = 0
            
            while i <= (len(sample) - sequence_length):
                    # Extract 30 consecutive records and respective timedeltas.
                    seq = sample.iloc[i : i + sequence_length]
                    time_deltas = seq['T_REC'].diff().dropna()
                    
                    # Affirm that all timedeltas are within bounds (records are consecutive) and record seuqence.
                    # Then skip ahead by one sequence length to ensure no overlapping sequences.
                    if all(time_deltas < cadence_upper) and all(time_deltas > cadence_lower):
                        valid_sequences.append(seq.reset_index(drop=True))
                        i = i + sequence_length
                    # Otherwise skip ahead by one record and reassess.
                    else:
                        i += 1
            
            # Update the dictionary with the valid sequences from the sample.
            if len(valid_sequences) > 0:
                sequence_dict[harp_ID] = valid_sequences

        return sequence_dict

    except Exception as ex:
            logger.error(f"Failed to build sequence dictionary: {ex}", exc_info=True)
            raise


def extract_recent_sequences(logger, sequence_dict):
    """Return only valid sequences with the most recent available start time, and timestamp, to provide the most up to date forecast."""

    try:
        sequence_list = []

        # For each sequence for each HARPNUM, create and format a dataframe containing only ID's and timestamp.
        for harpnum, sequences in sequence_dict.items():
            for index, sequence in enumerate(sequences):
                sequence_list.append({
                    "Harpnum": harpnum,
                    "T_REC": sequence["T_REC"].iloc[29],
                    "seq_number": index
                })
        sequence_end_time = pd.DataFrame(sequence_list, columns=["Harpnum", "T_REC", "seq_number"])
        sequence_end_time['T_REC'] = pd.to_datetime(sequence_end_time['T_REC'])

        # Define the most recent end time from all sequences as max_end_time.
        max_end_time = sequence_end_time['T_REC'].max()

        # Filter the DataFrame to retain only the records where timestamp is equal to the max end time.
        max_end_time_records = sequence_end_time[sequence_end_time['T_REC'] == max_end_time]

        # Build a dictionary mapping HARPNUMs to their valid sequences ending at the max end time.
        recent_sequence_dict = {}
        for harpnum, sequences in sequence_dict.items():
            if harpnum in max_end_time_records['Harpnum'].values:
                seq_index = max_end_time_records[max_end_time_records['Harpnum'] == harpnum]['seq_number'].iloc[0]
                recent_sequence_dict[harpnum] = sequences[seq_index]
        
        return recent_sequence_dict, max_end_time
    
    except Exception as ex:
            logger.error(f"Failed to extract recent sequences: {ex}", exc_info=True)
            raise


def transform_to_tensor(logger, recent_sequence_dict, data_pipeline_file, scaler_file):
    """Transform prepared recent data sequences to Pytorch tensor for ML model prediction."""

    try:
        X_list = []
        for harpnum, sequence in recent_sequence_dict.items():
            # Drop all unrequired features from the sequence dataset for each Harpnum and append to a list.
            sequence_array = sequence.drop(columns=['T_REC', 'HARPNUM', 'NOAA_NUM', 'NOAA_AR', 'QUALITY']).to_numpy()
            X_list.append(sequence_array)
        
        # Use the scaler from the training set to normalise the data in line with the model expectations.
        X_real_time_scaled = []
        scaler = joblib.load(scaler_file)  
        for seq in X_list:
            seq_scaled = scaler.transform(seq)
            X_real_time_scaled.append(seq_scaled)

        # Convert the scaled sequences list to a Pytorch Tensor via a numpy array, and save to file.
        X_real_time_scaled = np.array(X_real_time_scaled)
        X_tensor = torch.tensor(X_real_time_scaled, dtype=torch.float32)
        torch.save({'X_tensor': X_tensor,}, data_pipeline_file)
    
    except Exception as ex:
            logger.error(f"Failed to transform data to tensor: {ex}", exc_info=True)
            raise
        

def prepare_data(logger, data_download_file, scaler_file, data_pipeline_file):
    """Co-ordinate data preparation from raw data dump to ML model ready tensor file. Return forecast time and list of active regions."""
    
    # Download and clean the raw data.
    logger.info("Preparing data.")
    data = pd.read_csv(data_download_file)
    data = data.drop_duplicates()
    data.drop(columns='NOAA_ARS', inplace=True)
    data.replace(['MISSING', 'NaN'], np.nan, inplace=True)
    data['T_REC'] = pd.to_datetime(data['T_REC'].str.replace('_TAI', ''), format='%Y.%m.%d_%H:%M:%S')
    
    # Apply data preparation functions and save resulting tensor to file.
    data = replace_median_features(logger, data)
    data = linear_interpolation(logger, data)
    sequence_dict = prepare_sequence_dictionary(logger, data)
    recent_sequence_dict, forecast_time = extract_recent_sequences(logger, sequence_dict)
    harpnums = list(recent_sequence_dict.keys())
    transform_to_tensor(logger, recent_sequence_dict, data_pipeline_file, scaler_file) 
    logger.info("Success. Data prepared for model input.")
    return forecast_time, harpnums


class LSTMModel(nn.Module):
    """Define the LSTM architecture from training for prediction evaluation"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, bidirectional=True, dropout_p=0.3):
        """Initialise an LSTM model with chosen hyperparameters."""

        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout_p, bidirectional=bidirectional,
                            batch_first=True)
        
        # Apply a dropout layer for regularisation to reduce overfitting.
        self.dropout = nn.Dropout(dropout_p)
        # Adjust hidden size depending on directionality of the model.
        direction_factor = 2 if bidirectional else 1
        # Apply a fully connected layer to linearise output into regression prediction
        self.fc = nn.Linear(hidden_size * direction_factor, 1)

    def forward(self, x):
        """Forward pass data through the LSTM, returning regression prediction output."""

        # Output hidden state for each sequence time step.
        out, _ = self.lstm(x)
        # Extract final hidden state.
        last_time_step_out = out[:, -1, :]
        # Apply random dropout to final hidden state.
        dropped = self.dropout(last_time_step_out)
        # Pass through fully connected layer and reduce dimension to return regression prediction.
        out = self.fc(dropped)
        return out.squeeze()


class SequenceDataset(torch.utils.data.Dataset):
    """Create a dataset object for sequences to be prepare data for model evaluation."""

    def __init__(self, X):
        """Store data inside dataset object."""
        self.X = X

    def __len__(self):
        """Return length of dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Fetch sequence by index."""
        return self.X[idx]


def enable_mc_dropout(model):
    """Force dropout on during model evaluation phase to allow monte carlo dropout Baysean uncertainty estimation."""
    
    # Activate training mode for only nn.Dropout module.
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_predict(logger, model, x_batch, n_passes=50):
    """Implement monte carlo dropout by evaluating prediction with random dropout repeatedly, 
    returning the mean and standard deviation of the resulting prediction distribution."""
    
    try:
        # Enable evaluation mode and force dropout on.
        model.eval()
        enable_mc_dropout(model)
        
        # Evaluate prediction on the same data repeatedly for n_passes.
        preds = []
        for _ in range(n_passes):
            with torch.no_grad():
                preds.append(model(x_batch).cpu().numpy())
        
        # Calculate mean and standard deviation of resulting predictions.
        preds = np.stack(preds, axis=0)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        
        return mean, std
    
    except Exception as ex:
            logger.error(f"Failed to generate monte carlo prediction distributon: {ex}", exc_info=True)
            raise


def initialise_model(logger, model_weights_file):
    """Initialise the LSTM using pretrained weights, returning the model and the device type (CPU / GPU)."""

    logger.info("Initialising model.")
    try:
        model = LSTMModel(input_size=21)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_weights_file, map_location=device))
        model.to(device)
        model.eval()
        logger.info("Model initialised")

        return model, device
    
    except Exception as ex:
            logger.error(f"Failed to initialise model: {ex}", exc_info=True)
            raise


def classify_predictions(intensity, intensity_classifier):
    """Return the flare class for a given log10 flare intensity."""

    with open(intensity_classifier, "r") as f:
        thresholds = json.load(f)

    # Sort the intensity thresholds by magnitude.
    thresholds.sort(key=lambda x: x["min_intensity"])
    
    # Assign default as minimum class, and iteratively test if intensity exceeds each threshold.
    flare_class = thresholds[0]["class"]
    for entry in thresholds:
        if intensity >= entry["min_intensity"]:
            flare_class = entry["class"]
        else:
            break
    
    return flare_class


def evaluate_predictions(logger, model, device, data_pipeline_file, intensity_classifier):
    """Return intensity, error, and class predictions for each HARPNUM in the latest forecast window"""

    logger.info("Evaluating forecast.")
    try:
        # Load and prepare the dataset.
        logger.info("Loading data")
        data = torch.load(data_pipeline_file)
        X_tensor = data['X_tensor']
        logger.info("Preparing data sequence")
        real_time_dataset = SequenceDataset(X_tensor)

        # Create a DataLoader for real-time evaluation.
        logger.info("Building data loader")
        real_time_loader = DataLoader(real_time_dataset, batch_size=32, shuffle=False)
        
        all_means = []
        all_stds = []
        all_classes = []
        
        # Force MC droput on.
        logger.info("Enabling Monte Carlo dropout")
        enable_mc_dropout(model)
        
        logger.info("Executing predictions")
        # Pass each sequence through the model, recording mean and standard deviation predictions.
        with torch.no_grad():
            for X_batch in real_time_loader:
                X_batch = X_batch.to(device)
                batch_means, batch_stds = mc_predict(logger, model, X_batch, n_passes=50)
                all_means.extend(np.atleast_1d(batch_means).astype(float).ravel().tolist())
                all_stds.extend(np.atleast_1d(batch_stds).astype(float).ravel().tolist())
        
        # Classify each intensity prediction.
        for mean in all_means:
            all_classes.append(classify_predictions(float(mean), intensity_classifier))
        
        all_means = np.array(all_means, dtype=float)
        all_stds = np.array(all_stds, dtype=float)

        return all_means, all_stds, all_classes
    
    except Exception as ex:
            logger.error(f"Failed to evaluate predictions: {ex}", exc_info=True)
            raise


def publish_forecast(logger, forecast_file, means, stds, classes, forecast_time, harpnums):
    """Write forecast predictions to JSON for consumption by Flask app."""
    
    logger.info("Publishing predictions to file.")
    try:
        forecast_time = str(forecast_time)
        data = {"forecast_time": forecast_time, "measurements": []}

        # Write each HARPNUM prediction to JSON.
        for i in range(len(harpnums)):
            measurement = {
                "harpnum": harpnums[i],
                "intensity": round(float(means[i]), 2),
                "error": round(float(stds[i]), 2),
                "class": classes[i]
            }
            data["measurements"].append(measurement)

        with open(forecast_file, 'w') as file:
            json.dump(data, file, indent=2)
        logger.info("Forecast file published successfully.")

    except Exception as ex:
            logger.error(f"Failed to publish forecast: {ex}", exc_info=True)
            raise


def write_predictions_to_db(logger, forecast_file, database_file):
    """Insert each Harpnum prediction into the forecast table in SQLite DB."""

    logger.info("Writing predictions to DB.")
    try:
        with open(forecast_file, "r") as f:
            data = json.load(f) 

        forecast_time = data["forecast_time"]
    
        for measurement in data["measurements"]:
            row_id = insert_forecast(
                database_file,
                forecast_time=forecast_time,
                harpnum=str(measurement["harpnum"]),
                flare_class=str(measurement["class"]),
                intensity=float(measurement["intensity"]),
                std_error=float(measurement["error"]),
            )
            logger.info(f"Inserted record to DB: ID: {row_id}, harpnum: {measurement['harpnum']}")

    except Exception as ex:
        logger.error(f"Failed to write prediction to DB: {ex}", exc_info=True)
        raise


def main():
    """Execute the full forecast pipeline and serve to Flask app and DB."""

    config = load_config()

    # Access file paths from config.yaml
    pipeline_log_file = config['paths']['pipeline_log_file']
    data_download_file = config['paths']['data_download_file']
    scaler_file = config['paths']['scaler_file']
    data_pipeline_file = config['paths']['data_pipeline_file']
    model_weights_file = config['paths']['model_weights_file']
    forecast_file = config['paths']['forecast_file']
    intensity_classifier = config['paths']["intensity_classifier"]
    database_file = config['paths']['database_file']

    logger = setup_logger(pipeline_log_file)
    download_data(logger, data_download_file)
    forecast_time, harpnums = prepare_data(logger, data_download_file, scaler_file, data_pipeline_file)
    model, device = initialise_model(logger, model_weights_file)
    means, stds, classes = evaluate_predictions(logger, model, device, data_pipeline_file, intensity_classifier)
    publish_forecast(logger, forecast_file, means, stds, classes, forecast_time, harpnums)
    write_predictions_to_db(logger, forecast_file, database_file)


if __name__ == "__main__":
    main()
