import os
import subprocess
import h5py
import pandas as pd
from scipy.io import arff
import arff


def extract_audio_features(opensmile_path, config_path, audio_file, LLD_path):
    # Command to run openSMILE
    command = [
        os.path.join(opensmile_path, "bin", "SMILExtract"),
        "-C", config_path,
        "-I", audio_file,
        "-lldcsvoutput", LLD_path,
        "-timestampcsvlld", "0",  # Disable timestamps in the LLD output
        "-headercsvlld", "0"
    ]
    # Run the command
    subprocess.run(command, check=True)


def convert_arff_to_dataframe(arff_path):
    # Load ARFF file using liac-arff
    with open(arff_path, 'r') as f:
        data = arff.load(f)

    # Create a DataFrame from the data
    columns = [attr[0] for attr in data['attributes']][1:-1]
    df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
    df = df.iloc[:, 1:-1]  # Use slicing to exclude the first and last columns

    # Convert all columns to numeric if possible, ignoring errors to keep strings intact
    df = df.apply(pd.to_numeric, errors='coerce')
    print(df.head())
    return df


def process_audio_directory(opensmile_path, audio_directory, output_directory, hdf5_file_path):
    config_file = f"{opensmile_path}/config/compare16/ComParE_2016.conf"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with h5py.File(hdf5_file_path, 'w') as hdf:
        audio_group = hdf.create_group('audio')

        for filename in os.listdir(audio_directory):
            if filename.endswith(".wav"):
                audio_path = os.path.join(audio_directory, filename)
                LLD_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_LLDs.csv")

                # Extract features and convert to ARFF
                extract_audio_features(opensmile_path, config_file, audio_path, LLD_path)

                df = pd.read_csv(LLD_path, header=None, delimiter=';')
                print(df.head())
                df = df.iloc[:, 1:]  # Remove the first column
                print(df.head())
                audio_group.create_dataset(os.path.splitext(filename)[0], data=df.to_numpy())

                # Remove temporary ARFF file
                #os.remove(LLD_path)


# Example usage
opensmile_path = '/models/opensmile'
audio_directory = '/data/all_utterances_touse'
output_directory = '/output dir'
hdf5_file_path = '/h5 file to save output features'
process_audio_directory(opensmile_path, audio_directory, output_directory, hdf5_file_path)
