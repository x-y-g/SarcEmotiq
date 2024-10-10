import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from attention import SarcasmDetectionModel
from dataload_train import create_dataloader


class EarlyStopping:
    def __init__(self, patience, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience, model_path):
    early_stopping = EarlyStopping(patience=patience)
    epoch_details = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for texts, audios, sentiments, emotions, labels, text_masks, audio_masks, sentiment_masks, emotion_masks in train_loader:
            optimizer.zero_grad()
            outputs = model(texts, audios, sentiments, emotions, text_masks, audio_masks, sentiment_masks,
                            emotion_masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            for texts, audios, sentiments, emotions, labels, text_masks, audio_masks, sentiment_masks, emotion_masks in val_loader:
                outputs = model(texts, audios, sentiments, emotions, text_masks, audio_masks, sentiment_masks,
                                emotion_masks)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        epoch_details.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        })

        early_stopping(avg_val_loss)

        if early_stopping.best_loss == avg_val_loss:
            torch.save(model.state_dict(), model_path)
            print("Saved new best model")

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    return epoch_details


def test_model(model, test_loader, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for texts, audios, sentiments, emotions, labels, text_masks, audio_masks, sentiment_masks, emotion_masks in test_loader:
            outputs = model(texts, audios, sentiments, emotions, text_masks, audio_masks, sentiment_masks,
                            emotion_masks)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f'Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    return {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }


def main():
    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Train Sarcasm Recognition Model")
    parser.add_argument('--data', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=10, help="Batch size for training")
    parser.add_argument('--model_path', type=str, default='sarcasm_detection_model.pth', help="Path to save the trained model")
    parser.add_argument('--patience', type=int, default=5, help="Patience for early stopping")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate for training")  # Added learning rate
    args = parser.parse_args()

    # Load data
    text_file = f'{args.data}/embeddings/normalized_text_embeddings.h5'
    audio_file = f'{args.data}/embeddings/normalized_transformed_audio_features_lld.h5'
    sentiment_file = f'{args.data}/embeddings/normalized_sentiment_embeddings.h5'
    emotion_file = f'{args.data}/embeddings/normalized_emotion_embeddings.h5'
    label_file = f'{args.data}/label.h5'

    train_loader, val_loader, test_loader = create_dataloader(text_file, audio_file, sentiment_file, emotion_file, label_file, batch_size=args.batch_size)

    # Model initialization
    model = SarcasmDetectionModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # Use learning rate from args

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, args.epochs, args.patience, args.model_path)

    # Test the model
    test_model(model, test_loader, args.model_path)

if __name__ == '__main__':
    main()
