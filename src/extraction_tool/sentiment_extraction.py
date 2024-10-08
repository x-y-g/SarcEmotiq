import os
import h5py
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
model = AutoModel.from_pretrained("siebert/sentiment-roberta-large-english")

def get_senti_embeddings(text):

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
    embeddings = outputs.last_hidden_state
    embeddings = embeddings
    print(embeddings.shape)
    return embeddings.cpu().numpy()

def process_csv_and_save_features(csv_file_path, output_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    data = df[['KEY', 'SENTENCE', 'Sarcasm']]
    # print(data.head())
    audio_files = os.listdir('/data/all_utterances_touse')
    modified_audio_names = [file[:-4] for file in audio_files if file.endswith('.wav')]
    audio_df = pd.DataFrame(modified_audio_names, columns=['KEY'])
    selected_data = pd.merge(data, audio_df, on='KEY')

    # Print the filtered data
    print(len(selected_data))
    print(selected_data.head())

    # Prepare the h5py file
    with h5py.File(output_file_path, 'w') as h5f:
        group = h5f.create_group('sentiment')

        for index, row in selected_data.iterrows():
            text = row['SENTENCE']
            key = row['KEY']

            # Extract text features using BERT
            text_features = get_senti_embeddings(text)

            # Store the features in the h5py file, indexed by the audio name
            group.create_dataset(key, data=text_features.squeeze(0))

csv_file_path = '/data/mustard++_onlyU.csv'
output_file_path = 'output h5 file to save sentiment embeddings'
process_csv_and_save_features(csv_file_path, output_file_path)

