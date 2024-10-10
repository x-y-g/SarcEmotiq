import torch
import torch.nn as nn


class ContrastiveAttention(nn.Module):
    # anchor on emotion
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm_sentiments = nn.LayerNorm(embed_dim)
        self.norm_emotions = nn.LayerNorm(embed_dim)

        self.query_transform = nn.Linear(embed_dim, embed_dim)
        self.key_transform = nn.Linear(embed_dim, embed_dim)
        self.value_transform = nn.Linear(embed_dim, embed_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
        self.norm_post_attention = nn.LayerNorm(embed_dim)

    def forward(self, sentiments, emotions, sentiments_masks):
        sentiments = self.norm_sentiments(sentiments)
        emotions = self.norm_emotions(emotions)

        query = self.query_transform(emotions)
        key = self.key_transform(sentiments)
        values = self.value_transform(sentiments)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        scores = scores.masked_fill(sentiments_masks.unsqueeze(1) == 0, -1e9)

        attention_weights = self.softmax(scores)

        # Assuming opponent function is 1 - ac
        contrastive_weights = 1 - attention_weights
        contrastive_weights = self.softmax(contrastive_weights)
        contrastive_weights = self.dropout(contrastive_weights) # add dropout to attention weights

        contrast_vector = torch.bmm(contrastive_weights, values)
        contrast_vector = self.norm_post_attention(contrast_vector)

        return contrast_vector  # 1024


class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm_text = nn.LayerNorm(embed_dim)
        self.norm_audio = nn.LayerNorm(embed_dim)

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
        self.norm_post = nn.LayerNorm(embed_dim)

    def forward(self, text, audio, audio_masks):
        text = self.norm_text(text)
        audio = self.norm_audio(audio)

        query = self.query(text)
        key = self.key(audio)
        v = self.value(audio)

        #scores = self.torch.bmm(q, k)  # matrix multiplication between queries and keys.
        scores = torch.matmul(query, key.transpose(-2, -1)) / (audio.size(-1) ** 0.5)
        scores = scores.masked_fill(audio_masks.unsqueeze(1) == 0, -1e9)

        attention_weights = self.softmax(scores)
        attention_weights = self.dropout(attention_weights)  # add dropout to atten weights

        cross_vector = torch.bmm(attention_weights, v)  # matrix multiplication between attention weights and values
        cross_vector = self.norm_post(cross_vector)

        return cross_vector  #768


class SarcasmDetectionModel(nn.Module):
    def __init__(self, text_dim=768, audio_dim=768, sentiment_dim=1024, emotion_dim = 1024, num_classes=2):
        super().__init__()
        self.contrastive_attention = ContrastiveAttention(sentiment_dim)
        self.cross_attention = CrossAttention(text_dim)
        self.mlp = nn.Sequential(
            nn.Linear(text_dim + audio_dim + sentiment_dim + emotion_dim + emotion_dim + text_dim, 256),  # First MLP layer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),  # Second MLP layer
            )

    @staticmethod
    def masked_average_pooling(x, mask):
        mask = mask.unsqueeze(-1).expand_as(x)
        sum_x = torch.sum(x * mask, dim=1)
        count_x = torch.sum(mask, dim=1)

        return sum_x / count_x

    def forward(self, texts, audios, sentiments, emotions, text_masks, audio_masks, sentiment_masks, emotion_masks):
        # Extract Contrastive and Cross vectors
        contrastive_vector = self.contrastive_attention(sentiments, emotions, sentiment_masks)
        fusion_vector = self.cross_attention(texts, audios, audio_masks)

        pooled_text = self.masked_average_pooling(texts, text_masks)
        pooled_audio = self.masked_average_pooling(audios, audio_masks)
        pooled_sentiment = self.masked_average_pooling(sentiments, sentiment_masks)
        pooled_emotion = self.masked_average_pooling(emotions, emotion_masks)

        pooled_contrastive = self.masked_average_pooling(contrastive_vector, emotion_masks)
        pooled_fusion = self.masked_average_pooling(fusion_vector, text_masks)

        combined_features = torch.cat([pooled_text, pooled_audio, pooled_sentiment, pooled_emotion, pooled_contrastive, pooled_fusion], dim=1)

        sarcasm_output = self.mlp(combined_features)

        return sarcasm_output


def load_model(model_path):
    model = SarcasmDetectionModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
