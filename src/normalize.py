import argparse
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler


def normalize_embeddings(embeddings):
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)
    return normalized_embeddings


def process_file(file_path, group_name, output_path):
    print(f"Processing {group_name} from {file_path}...")  # Feedback to the user
    with h5py.File(file_path, 'r') as file:
        embeddings_group = file[group_name]
        all_embeddings = np.vstack([embeddings_group[name][()] for name in embeddings_group])
        scaler = StandardScaler()
        scaler.fit(all_embeddings)

        with h5py.File(output_path, 'w') as output_file:
            output_group = output_file.create_group(group_name)
            for i, name in enumerate(embeddings_group):
                normalized_data = scaler.transform(embeddings_group[name][()])
                if normalized_data.shape[0] == 1:
                    normalized_data = np.squeeze(normalized_data, axis=0)
                output_group.create_dataset(name, data=normalized_data)
                if (i + 1) % 10 == 0:  # Print feedback every 10 processed items
                    print(f"Processed {i + 1}/{len(embeddings_group)} embeddings from {group_name}")

    print(f"Finished processing {group_name}. Output saved to {output_path}")


def process_labels(file_path, output_path):
    print(f"Processing labels from {file_path}...")  # Feedback to the user
    with h5py.File(file_path, 'r') as file:
        labels_group = file['label']

        with h5py.File(output_path, 'w') as output_file:
            output_file.create_group('label')

            for key in labels_group:
                label_data = labels_group[key][()]
                output_file['label'].create_dataset(key, data=label_data)
                print(f"Copied label for {key}: {label_data}")

    print(f"Finished processing labels. Output saved to {output_path}")


def check_h5_file(file_path, group_name=None):
    with h5py.File(file_path, 'r') as f:
        if group_name:
            group = f[group_name]
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset):  # Check if it's a dataset
                    print(f"Dataset: {group_name}/{key}, Shape: {item.shape}, Dtype: {item.dtype}")
                else:
                    print(f"Group: {group_name}/{key}")
        else:
            # List all groups in the file
            for group in f.keys():
                print(f"Group: {group}")
                check_h5_file(file_path, group)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Normalize embeddings in HDF5 file.")
    parser.add_argument('--embeddings', type=str, required=True, help="Path to the input embeddings file")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save the normalized embeddings")

    args = parser.parse_args()

    modalities_info = {
        'audio': 'normalized_audio.h5',
        'text': 'normalized_text.h5',
        'sentiment': 'normalized_sentiment.h5',
        'emotion': 'normalized_emotion.h5'
    }

    for group_name, output_file_name in modalities_info.items():
        output_path = f"{args.output_dir}/{output_file_name}"
        process_file(args.embeddings, group_name, output_path)

    process_labels(args.embeddings, f"{args.output_dir}/label.h5")
