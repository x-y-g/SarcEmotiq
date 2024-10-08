import numpy as np
import torch
import h5py
from sklearn.preprocessing import StandardScaler


def normalize_embeddings(embeddings):
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)
    return normalized_embeddings


def process_file(file_path, group_name, output_path):
    normalized_embeddings = {}
    with h5py.File(file_path, 'r') as file:
        embeddings_group = file[group_name]

        all_embeddings = np.vstack([embeddings_group[name][()] for name in embeddings_group])
        print(all_embeddings.shape)
        scaler = StandardScaler()
        scaler.fit(all_embeddings)

        with h5py.File(output_path, 'w') as output_file:
            output_group = output_file.create_group(group_name)

            # Process each embedding, normalize and save
            for name in embeddings_group:
                normalized_data = scaler.transform(embeddings_group[name][()])
                print('normalized size', normalized_data.shape)
                if normalized_data.shape[0] == 1:
                    normalized_data = np.squeeze(normalized_data, axis=0)
                output_group.create_dataset(name, data=normalized_data)
                print(name, 'normalized size', normalized_data.shape)



modalities_info = {
    #'/extracted opensmile features': ('audio', 'normalized audio features.h5'),
    #'/extracted BERT features': ('text', 'normalized text embeddings.h5'),
    #'/extracted sentiment features': ('sentiment', 'normalized sentiment embeddings.h5'),
    #'/extracted emotion features': ('emotion', 'normalized emotion embeddings.h5')
}

# Process each modality
for file_path, (group_name, output_path) in modalities_info.items():
    process_file(file_path, group_name, output_path)


