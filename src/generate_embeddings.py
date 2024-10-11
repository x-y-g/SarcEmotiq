import os
import argparse
import h5py
import subprocess
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import torchaudio
from sklearn.preprocessing import StandardScaler
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Text Embedding (BERT)
def extract_text_embeddings(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()

    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**encoded_input)
    return output.last_hidden_state.cpu().numpy()

# Audio Feature Extraction (OpenSMILE)
def extract_audio_features(opensmile_path, config_path, audio_file, LLD_path):
    command = [
        os.path.join(opensmile_path, "bin", "SMILExtract"),
        "-C", config_path,
        "-I", audio_file,
        "-lldcsvoutput", LLD_path,
        "-timestampcsvlld", "0",  # Disable timestamps in the LLD output
        "-headercsvlld", "0"
    ]
    subprocess.run(command, check=True)
    df = pd.read_csv(LLD_path, header=None, delimiter=';').iloc[:, 1:]  # Remove timestamp
    return df.to_numpy()

# Emotion Embedding Extraction (Wav2Vec2)
def extract_emotion_embeddings(audio_file):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", output_hidden_states=True).to(device)
    waveform, _ = torchaudio.load(audio_file)
    waveform = waveform.mean(dim=0, keepdim=True)  # Handle multichannel
    inputs = feature_extractor(waveform, return_tensors="pt", sampling_rate=16000, padding=True).to(device)
    with torch.no_grad():
        outputs = model(inputs.input_values)
    return outputs.hidden_states[-1].cpu().numpy()

# Sentiment Embedding Extraction (RoBERTa)
def extract_sentiment_embeddings(text):
    tokenizer = BertTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
    model = BertModel.from_pretrained("siebert/sentiment-roberta-large-english").to(device)
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**encoded_input)
    return outputs.last_hidden_state.cpu().numpy()

# Normalize embeddings
def normalize_embeddings(embeddings):
    scaler = StandardScaler()
    return scaler.fit_transform(embeddings)

# Main function to process directory and save embeddings
def process_directory(audio_directory, text_csv, opensmile_path, output_directory):
    # Correctly define the OpenSMILE config file path
    config_file = f"{opensmile_path}/config/compare16/ComParE_2016.conf"

    df = pd.read_csv(text_csv)

    # Create separate HDF5 files for text, audio, sentiment, and emotion
    text_hdf5 = h5py.File('data/text_embeddings.h5', 'w')
    audio_hdf5 = h5py.File('data/audio_embeddings.h5', 'w')
    sentiment_hdf5 = h5py.File('data/sentiment_embeddings.h5', 'w')
    emotion_hdf5 = h5py.File('data/emotion_embeddings.h5', 'w')

    text_group = text_hdf5.create_group('text')
    audio_group = audio_hdf5.create_group('audio')
    sentiment_group = sentiment_hdf5.create_group('sentiment')
    emotion_group = emotion_hdf5.create_group('emotion')

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for _, row in df.iterrows():
        audio_name = row['KEY']
        text = row['SENTENCE']
        audio_file = os.path.join(audio_directory, f"{audio_name}.wav")
        LLD_path = os.path.join(output_directory, f"{audio_name}_LLDs.csv")

        # Extract embeddings
        text_embeddings = normalize_embeddings(extract_text_embeddings(text))
        audio_features = normalize_embeddings(extract_audio_features(opensmile_path, config_file, audio_file, LLD_path))
        sentiment_embeddings = normalize_embeddings(extract_sentiment_embeddings(text))
        emotion_embeddings = normalize_embeddings(extract_emotion_embeddings(audio_file))

        # Save to separate HDF5 files
        text_group.create_dataset(audio_name, data=text_embeddings)
        audio_group.create_dataset(audio_name, data=audio_features)
        sentiment_group.create_dataset(audio_name, data=sentiment_embeddings)
        emotion_group.create_dataset(audio_name, data=emotion_embeddings)

        # Remove temporary LLD CSV file
        os.remove(LLD_path)

    # Close all HDF5 files
    text_hdf5.close()
    audio_hdf5.close()
    sentiment_hdf5.close()
    emotion_hdf5.close()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate embeddings for audio, text, sentiment, and emotion.")
    parser.add_argument('--audio_directory', type=str, required=True, help="Path to the audio files directory.")
    parser.add_argument('--text_csv', type=str, required=True, help="Path to the text CSV file containing 'KEY' and 'SENTENCE'.")
    parser.add_argument('--opensmile_path', type=str, required=True, help="Path to OpenSMILE directory.")
    parser.add_argument('--output_directory', type=str, required=True, help="Directory to store temporary audio features.")

    args = parser.parse_args()

    # Process embeddings
    process_directory(
        audio_directory=args.audio_directory,
        text_csv=args.text_csv,
        opensmile_path=args.opensmile_path,
        output_directory=args.output_directory
    )
