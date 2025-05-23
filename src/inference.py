import argparse
import torch
from torch import nn
from dataload import transcribe_audio_whisper, extract_audio_features, get_bert_sequential_embeddings, get_emotion, \
    get_sentiment, normalize_embeddings
from attention import load_model


class AudioDimensionRegulator(nn.Module):
    def __init__(self, input_dim=130, output_dim=768):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class SarcasmRecognize():
    def __init__(self, model_path):
        self.opensmile_path = '/opensmile'
        self.config_path = f"{self.opensmile_path}/config/compare16/ComParE_2016.conf"
        self.LLD_path = "temp_lld.csv"
        self.model_path = model_path  # Use the model path passed during initialization

    def predict_sarcasm(self, audio_file_path):
        print("🔄 Transcribing audio and extracting features...")

        # Transcribe audio
        text = transcribe_audio_whisper(audio_file_path)
        audio_features = extract_audio_features(self.opensmile_path, self.config_path, audio_file_path, self.LLD_path)
        text_features = get_bert_sequential_embeddings(text)
        sentiment_features = get_sentiment(text)
        emotion_features = get_emotion(audio_file_path)

        # Normalize features
        audio_features = normalize_embeddings(audio_features)
        text_features = normalize_embeddings(text_features)
        sentiment_features = normalize_embeddings(sentiment_features)
        emotion_features = normalize_embeddings(emotion_features)

        # Load model
        print("🔄 Loading pre-trained model...")
        model = load_model(self.model_path)

        # Convert features to tensor
        audio_features = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0)
        text_features = torch.tensor(text_features, dtype=torch.float32).unsqueeze(0)
        sentiment_features = torch.tensor(sentiment_features, dtype=torch.float32).unsqueeze(0)
        emotion_features = torch.tensor(emotion_features, dtype=torch.float32).unsqueeze(0)

        # Regulate audio dimension
        regulator = AudioDimensionRegulator()
        audio_features = regulator(audio_features)

        # Create masks
        audio_mask = torch.ones(audio_features.size()[:-1], dtype=torch.bool)
        text_mask = torch.ones(text_features.size()[:-1], dtype=torch.bool)
        sentiment_mask = torch.ones(sentiment_features.size()[:-1], dtype=torch.bool)
        emotion_mask = torch.ones(emotion_features.size()[:-1], dtype=torch.bool)

        # Predict
        print("🔄 Making predictions...")
        with torch.no_grad():
            output = model(text_features, audio_features, sentiment_features, emotion_features, text_mask, audio_mask,
                           sentiment_mask, emotion_mask)
            _, predicted = torch.max(output, 1)

        return text, "Sarcastic 😏" if predicted.item() == 1 else "Not Sarcastic!"


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Sarcasm Recognition using a pre-trained model")
    parser.add_argument('--input', type=str, required=True, help="Path to the input audio file (.wav)")
    parser.add_argument('--model', type=str, required=True, help="Path to the pre-trained model (.pth)")
    args = parser.parse_args()

    # Initialize recognizer with the model path
    Recognizer = SarcasmRecognize(model_path=args.model)

    # Predict sarcasm
    print("🚀 Starting Sarcasm Recognition...")
    text, result = Recognizer.predict_sarcasm(args.input)
    print("\n===============================")
    print(f"📝 Transcribed Text: {text}")
    print(f"🎯 Sarcasm Prediction: {result}")
    print("===============================")


if __name__ == '__main__':
    main()
