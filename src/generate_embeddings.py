import os
import argparse
import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import BertTokenizer, BertModel, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor, \
    AutoConfig, AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
import torchaudio
import subprocess
import warnings
from transformers import logging
logging.set_verbosity_error()  # Suppress most warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Text Embedding (BERT)
def get_bert_sequential_embeddings(text):
    print(f"Extracting text embeddings for: {text[:30]}...")  # User feedback
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoded_input = tokenizer(text, return_tensors='pt',
                              truncation=True,
                              return_attention_mask=True,
                              max_length=512,
                              padding=True,
                              add_special_tokens=True)
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
    embeddings = output.last_hidden_state.squeeze(0)
    return embeddings.cpu().numpy()


# Audio Feature Extraction (OpenSMILE)
def extract_audio_features(config_path, audio_file_path, LLD_path):
    print(f"Extracting audio features from: {audio_file_path}")  # User feedback
    opensmile_path = 'opensmile'
    command = [
        os.path.join(opensmile_path, "bin", "SMILExtract"),
        "-C", config_path,
        "-I", audio_file_path,
        "-lldcsvoutput", LLD_path,
        "-timestampcsvlld", "0",
        "-headercsvlld", "0"
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Audio feature extraction completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error running SMILExtract: {e.stderr}")
        raise

    df = pd.read_csv(LLD_path, header=None, delimiter=';')
    df = df.iloc[:, 1:]  # Remove the first column
    return df.to_numpy()


# Emotion Embedding Extraction (Wav2Vec2)
def extract_emotion_embeddings(audio_file):
    print(f"Extracting emotion embeddings from: {audio_file}")  # User feedback
    model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
    model = AutoModelForAudioClassification.from_pretrained(model_name, config=config)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

    waveform, sample_rate = torchaudio.load(audio_file)

    if waveform.shape[0] > 1:  # Handle multiple channels -- average across channels
        waveform = torch.mean(waveform, dim=0)
    else:  # single channel, ensure it's the correct shape [1, N]
        waveform = waveform.squeeze()
    inputs = feature_extractor(waveform, return_tensors="pt", sampling_rate=16000, padding=True)
    with torch.no_grad():
        outputs = model(inputs.input_values.float())
        embeddings = outputs.hidden_states[-1].squeeze(0)
    return embeddings.cpu().numpy()


# Sentiment Embedding Extraction (RoBERTa)
def get_senti_embeddings(text):
    print(f"Extracting sentiment embeddings for: {text[:30]}...")  # User feedback
    tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
    model = AutoModel.from_pretrained("siebert/sentiment-roberta-large-english")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    encoded_input = tokenizer(text, return_tensors='pt',
                              truncation=True,
                              return_attention_mask=True,
                              max_length=512,
                              padding=True,
                              add_special_tokens=True)

    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    embeddings = outputs.last_hidden_state.squeeze(0)
    return embeddings.cpu().numpy()


# Projection for audio embeddings
class NonLinearAudioProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NonLinearAudioProjection, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


# Main function to process directory and save embeddings
def process_directory(audio_directory, text_csv, output_directory):
    config_file = "opensmile/config/compare16/ComParE_2016.conf"
    df = pd.read_csv(text_csv)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    audio_projection = NonLinearAudioProjection(130, 768).to(device)

    # Create HDF5 files for all embeddings
    embedding_file_path = '../data/embeddings.h5'
    with h5py.File(embedding_file_path, 'w') as h5f:
        text_group = h5f.create_group('text')
        label_group = h5f.create_group('label')
        audio_group = h5f.create_group('audio')
        sentiment_group = h5f.create_group('sentiment')
        emotion_group = h5f.create_group('emotion')

        for i, row in enumerate(df.iterrows()):
            audio_name = row[1]['KEY']
            text = row[1]['SENTENCE']
            sarcasm_label = int(row[1]['Sarcasm'])
            audio_file = os.path.join(audio_directory, f"{audio_name}.wav")
            LLD_path = os.path.join(output_directory, f"{audio_name}_LLDs.csv")

            print(f"Processing {i + 1}/{len(df)}: {audio_name}")  # User feedback

            # Extract embeddings
            text_embeddings = get_bert_sequential_embeddings(text)
            audio_features = extract_audio_features(config_file, audio_file, LLD_path)

            audio_embeddings = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).to(device)
            projected_audio = audio_projection(audio_embeddings).detach().cpu().numpy().squeeze(0)

            sentiment_embeddings = get_senti_embeddings(text)
            emotion_embeddings = extract_emotion_embeddings(audio_file)

            # Store embeddings in HDF5
            text_group.create_dataset(audio_name, data=text_embeddings)
            label_group.create_dataset(audio_name, data=np.array([sarcasm_label]))
            audio_group.create_dataset(audio_name, data=projected_audio)
            sentiment_group.create_dataset(audio_name, data=sentiment_embeddings)
            emotion_group.create_dataset(audio_name, data=emotion_embeddings)

            os.remove(LLD_path)

        print(f"All embeddings have been processed and saved to {embedding_file_path}")  # Final feedback


def check_h5_contents(file_path):
    with h5py.File(file_path, 'r') as f:
        def print_group(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")

        f.visititems(print_group)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for text, audio, sentiment, and emotion.")
    parser.add_argument('--audio_directory', type=str, required=True, help="Directory containing audio files.")
    parser.add_argument('--text_csv', type=str, required=True, help="Path to CSV containing text and labels.")
    parser.add_argument('--output_directory', type=str, required=True,
                        help="Directory to save temporary audio features.")

    args = parser.parse_args()

    process_directory(args.audio_directory, args.text_csv, args.output_directory)
