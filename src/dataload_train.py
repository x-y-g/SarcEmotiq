import h5py
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence


class SarcasmDataset(Dataset):
    def __init__(self, data, audio_names):
        super().__init__()
        self.data = data
        self.audio_names = audio_names

    def __len__(self):
        return len(self.audio_names)

    def __getitem__(self, idx):
        audio_name = self.audio_names[idx]
        return ((self.data['text'][audio_name],
                self.data['audio'][audio_name],
                self.data['sentiment'][audio_name],
                self.data['emotion'][audio_name],
                self.data['label'][audio_name].squeeze()),
                audio_name)


def load_data(text_file, audio_file, sentiment_file, emotion_file, label_file):
    # Open the files
    data={}
    with h5py.File(text_file, 'r') as text_data, \
            h5py.File(audio_file, 'r') as audio_data, \
            h5py.File(sentiment_file, 'r') as sentiment_data, \
            h5py.File(emotion_file, 'r') as emotion_data, \
            h5py.File(label_file, 'r') as label_data:
        # Assume 'label' contains the keys for indexing all other datasets
        audio_names = list(label_data['label'].keys())

        data['text'] = {name: torch.tensor(text_data['text'][name][:], dtype=torch.float) for name in audio_names}
        data['audio'] = {name: torch.tensor(audio_data['audio'][name][:], dtype=torch.float) for name in
                          audio_names}
        data['sentiment'] = {name: torch.tensor(sentiment_data['sentiment'][name][:], dtype=torch.float) for name in
                              audio_names}
        data['emotion'] = {name: torch.tensor(emotion_data['emotion'][name][:], dtype=torch.float) for name in
                            audio_names}
        data['label'] = {name: torch.tensor(label_data['label'][name][:], dtype=torch.long) for name in audio_names}
    return data, audio_names


def collate_fn(batch):
    #texts, audios, sentiments, emotions, labels = zip(*batch)
    data, audio_names = zip(*batch)
    texts, audios, sentiments, emotions, labels = zip(*data)
    # Pad sequences and create masks
    texts, text_masks = pad_and_create_mask(texts)

    audios, audio_masks = pad_and_create_mask(audios)
    sentiments, sentiment_masks = pad_and_create_mask(sentiments)
    emotions, emotion_masks = pad_and_create_mask(emotions)
    # Assuming labels are already appropriately shaped
    labels = torch.stack(labels)

    return texts, audios, sentiments, emotions, labels, text_masks, audio_masks, sentiment_masks, emotion_masks


def pad_and_create_mask(sequences, padding_value=0):
    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value)
    masks = (padded_sequences != padding_value).any(dim=2)  # keep only (batch, max_seq_length)
    #masks = padded_sequences != padding_value
    return padded_sequences, masks


def create_dataloader(texts, audios, sentiments, emotions, labels, batch_size):
    data, audio_names = load_data(texts, audios, sentiments, emotions, labels)
    dataset = SarcasmDataset(data, audio_names)
    total_size = len(dataset)
    train_ratio = 0.7
    valid_ratio = 0.15
    test_ratio = 0.15

    train_dataset, temp_dataset = train_test_split(dataset, test_size=1 - train_ratio, random_state=2024)
    valid_dataset, test_dataset = train_test_split(temp_dataset, test_size=test_ratio / (valid_ratio + test_ratio),
                                                   random_state=2024)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader
