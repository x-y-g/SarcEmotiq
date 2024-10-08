import os

import numpy as np
import pandas as pd
import torch

from transformers import BertTokenizer, BertModel
import h5py

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_bert_sequential_embeddings(text):
    model.eval()  # Put model in evaluation mode
    # Check if CUDA is available and set device accordingly
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
    embeddings = output.last_hidden_state
    embeddings = embeddings
    print(embeddings.shape)
    return embeddings.cpu().numpy()


def get_bert_global_embeddings(text):
    model.eval()  # Put model in evaluation mode

    # Check if CUDA is available and set device accordingly
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
    embeddings = output.last_hidden_state
    embeddings = embeddings.mean(dim=1)  # [batch size, sequence length, feature dimension] --> reduces each sequence of embeddings down to a single embedding vector per item in the batch.
    #embeddings = embeddings.squeeze(0)
    embeddings = embeddings
    print(embeddings.shape)
    return embeddings.cpu().numpy()


def process_csv_and_save_features(csv_file_path, output_file_path, label_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    data = df[['KEY', 'SENTENCE', 'Sarcasm']]
    # print(data.head())
    audio_files = os.listdir('/home/xiyuan/PycharmProjects/ASA2024/data/all_utterances_touse')
    modified_audio_names = [file[:-4] for file in audio_files if file.endswith('.wav')]
    audio_df = pd.DataFrame(modified_audio_names, columns=['KEY'])
    selected_data = pd.merge(data, audio_df, on='KEY')

    # Print the filtered data
    print(len(selected_data))
    print(selected_data.head())

    # Prepare the h5py file
    with h5py.File(output_file_path, 'w') as h5f:
        text_group = h5f.create_group('text')

        with h5py.File(label_file_path, 'w') as label_h5f:
            label_group = label_h5f.create_group('label')

            for index, row in selected_data.iterrows():
                text = row['SENTENCE']
                key = row['KEY']
                sarcasm_label = int(row['Sarcasm'])

                # Extract text features using BERT
                text_features = get_bert_sequential_embeddings(text)

                # Store the features in the h5py file, indexed by the audio name
                text_group.create_dataset(key, data=text_features.squeeze(0))
                label_group.create_dataset(key, data=np.array([sarcasm_label]))


csv_file_path = '/data/mustard++_onlyU.csv'
output_file_path = 'output h5 file to save text embeddings'
label_file_path = '/data/label.h5'  # save labels
process_csv_and_save_features(csv_file_path, output_file_path,label_file_path)
