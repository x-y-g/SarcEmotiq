import torchaudio
import whisper
import os
import subprocess
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor, AutoConfig, AutoTokenizer, AutoModel


def transcribe_audio_whisper(audio_file_path):
    whisper_model = whisper.load_model("small")
    result = whisper_model.transcribe(audio_file_path)
    print('transcribed text:', result['text'])
    return result['text']


def extract_audio_features(opensmile_path, config_path, audio_file_path, LLD_path):
    command = [
        os.path.join(opensmile_path, "bin", "SMILExtract"),
        "-C", config_path,
        "-I", audio_file_path,
        "-lldcsvoutput", LLD_path,
        "-timestampcsvlld", "0",
        "-headercsvlld", "0"
    ]
    print("Running command:", " ".join(command))

    try:
        result = subprocess.run(["echo", "Hello, World!"], check=True, capture_output=True, text=True)
        print("Diagnostic ls Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running diagnostic ls command: {e.stderr}")

    try:
        # Run the command and capture output
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("SMILExtract Output:", result.stdout)
        print("SMILExtract Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running SMILExtract: {e.stderr}")
        raise

        # Ensure the output file exists before reading
    if not os.path.exists(LLD_path):
        raise FileNotFoundError(f"Output file not found: {LLD_path}")

    df = pd.read_csv(LLD_path, header=None, delimiter=';')
    df = df.iloc[:, 1:]  # Remove the first column
    return df.to_numpy()


def get_bert_sequential_embeddings(text):
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
    #print('text features:', embeddings)
    #print(embeddings.shape)
    return embeddings.cpu().numpy()


def get_emotion(audio_file_path):
    model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
    model = AutoModelForAudioClassification.from_pretrained(model_name, config=config)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

    waveform, sample_rate = torchaudio.load(audio_file_path)

    if waveform.shape[0] > 1:  # Handle multiple channels -- average across channels
        waveform = torch.mean(waveform, dim=0)
    else:  # single channel, ensure it's the correct shape [1, N]
        waveform = waveform.squeeze()
    inputs = feature_extractor(waveform, return_tensors="pt", sampling_rate=16000, padding=True)
    with torch.no_grad():  # no need for gradients
        outputs = model(inputs.input_values.float())
        embeddings = outputs.hidden_states[-1].squeeze(0) # the output features from the final layer of the model for each time step
    embeddings = embeddings
    return embeddings.cpu().numpy()


def get_sentiment(text):
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

    with torch.no_grad():  # Ensures no gradients are calculated to save memory and computations
        outputs = model(input_ids, attention_mask=attention_mask)
    embeddings = outputs.last_hidden_state.squeeze(0)
    embeddings = embeddings
    return embeddings.cpu().numpy()

def normalize_embeddings(embeddings):
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)
    return normalized_embeddings



