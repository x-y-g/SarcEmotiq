import os
import h5py
import pandas as pd
import torch
import torchaudio
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor, AutoConfig
import numpy as np

model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
model = AutoModelForAudioClassification.from_pretrained(model_name, config=config)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")


def load_audio(file_path, sampling_rate=16000):
    waveform, sample_rate = torchaudio.load(file_path)
    # Resample if necessary
    if sample_rate != sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sampling_rate)
        waveform = resampler(waveform)

    return waveform


def extract_emo_embeddings(audio_path):
    waveform = load_audio(audio_path)
    if waveform.shape[0] > 1:  # Handle multiple channels -- average across channels
        waveform = waveform.mean(dim=0, keepdim=True)  # keep the tensor structure consistent
    else:  # single channel, ensure it's the correct shape [1, N]
        waveform = waveform.squeeze()  # Remove channel dimension if it's singleton
    print("Processed waveform shape:", waveform.shape)

    inputs = feature_extractor(waveform, return_tensors="pt", sampling_rate=16000, padding=True)

    with torch.no_grad():  # no need for gradients
        outputs = model(inputs.input_values.float())
        embeddings = outputs.hidden_states[-1]  # the output features from the final layer of the model for each time step
    embeddings = embeddings
    print(embeddings.shape)
    return embeddings.cpu().numpy()


def extract_emo(audio_path):
    waveform = load_audio(audio_path)
    if waveform.shape[0] > 1:  # Handle multiple channels -- average across channels
        waveform = waveform.mean(dim=0, keepdim=True)  # keep the tensor structure consistent
    else:  # single channel, ensure it's the correct shape [1, N]
        waveform = waveform.squeeze()  # Remove channel dimension if it's singleton
    print("Processed waveform shape:", waveform.shape)

    inputs = feature_extractor(waveform, return_tensors="pt", sampling_rate=16000, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_label = model.config.id2label[predicted_ids.item()]

    return predicted_label

def process_directory(audio_directory, output_hdf5):
    with h5py.File(output_hdf5, 'w') as hdf:
        group = hdf.create_group('emotion')

        for filename in os.listdir(audio_directory):
            if filename.endswith('.wav'):
                audio_name = filename[:-4]
                audio_path = os.path.join(audio_directory, filename)
                #embeddings = extract_emo_embeddings(audio_path)
                emotion = extract_emo(audio_path)
                print(emotion)

                #group.create_dataset(audio_name, data=embeddings.squeeze(0))
                group.create_dataset(audio_name, data=emotion)


audio_directory = '/data/all_utterances_touse'
output_hdf5 = 'output h5 file to save emotion embeddings'
process_directory(audio_directory, output_hdf5)
