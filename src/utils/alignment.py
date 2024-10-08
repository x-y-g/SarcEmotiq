import h5py
import torch
import torch.nn as nn


class NonLinearAudioProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NonLinearAudioProjection, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.relu = nn.ReLU()
        #self.layer_norm = nn.LayerNorm(output_dim)  # add normalization

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        #x = self.layer_norm(x)
        return x


def transform_embeddings(input_file_path, output_file_path, audio_dim, text_dim):
    audio_projection = NonLinearAudioProjection(audio_dim, text_dim)
    # Open the HDF5 file
    with h5py.File(input_file_path, 'r') as input_file:

        with h5py.File(output_file_path, 'w') as output_file:
            input_group = input_file['audio']
            output_group = output_file.create_group('audio')

            for dataset_name in input_group:
                audio_data = input_group[dataset_name][()]
                audio_embeddings = torch.tensor(audio_data, dtype=torch.float32)
                if audio_embeddings.dim() == 2:
                    audio_embeddings = audio_embeddings.unsqueeze(0)  # add batch_size dim

                transformed_audio_embeddings = audio_projection(audio_embeddings)

                transformed_audio_embeddings = transformed_audio_embeddings.detach()
                # Save the transformed embeddings to the new file
                output_group.create_dataset(dataset_name, data=transformed_audio_embeddings.numpy().squeeze(0))
                print("Shape of transformed audio embeddings:", transformed_audio_embeddings.numpy().squeeze(0).shape)


def create_transformed_audio_embeddings():
    input_file_path = '/data/input_features/normalized_audio_features_lld.h5' # normalized audio features
    output_file_path = '/data/input_features/normalized_transformed_audio_features_lld.h5'
    audio_dim = 130
    text_dim = 768

    # Process all audio embeddings and save the transformed data
    transform_embeddings(input_file_path, output_file_path, audio_dim, text_dim)


if __name__ == '__main__':
    create_transformed_audio_embeddings()
