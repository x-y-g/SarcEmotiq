# 🎙️😜 SarcEmotiq
SarcEmotiq is a deep learning-based tool for recognizing sarcasm in English audio. It uses pre-trained models trained on open-sourced datasets [MUStARD++](https://github.com/cfiltnlp/MUStARD_Plus_Plus) but also allows users to retrain the model with their own data.

## Model Architecture
SarcEmotiq integrates multiple modalities, acoustic + textual + emotional + sentiment cues, into a unified attention-based fusion model. Below is a summary of the system.

### Modalities Used:

| Modality  | Feature source                                                       |
|-----------|----------------------------------------------------------------------|
| Audio     | openSMILE (ComParE_2016)                                             |
| Text      | BERT-base-uncased                                                    |
| Emotion   | wav2vec2-large-xlsr → Speech emotion classifier                      |
| Sentiment | RoBERTa (sentiment-roberta-large-english) Text sentiment classifier  |

### Fusion Mechanisms:
1. **Contrastive attention**  
   Aligns emotions (as query) with sentiments (as key-value) to emphasize conflicting affective states -> indicative of sarcasm.

2. **Cross attention**  
   Aligns textual content (as query) with audio features (as key-value) to capture prosody-semantic mismatches (e.g., flat tone with exaggerated words).

3. **Masked average pooling**  
   Reduces all embeddings over time to handle variable-length sequences.

4. **Multimodal concatenation + MLP**  
   Pooled outputs (text, audio, sentiment, emotion, cross, contrastive) are concatenated and passed through an MLP for final classification.

➡️ More information about the model please visit our [published paper] (https://doi.org/10.1121/2.0001918)


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SarcEmotiq.git
   cd SarcEmotiq
   ```
2. Set up the environment:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the pretrained model (Place it in the `models/` folder)
   ```bash
   wget <https://drive.google.com/file/d/1xqyofUELCl2oBlA6151Vvd-f8pIG2biF/view?usp=drive_link> -O models/model.pth
   ```
4. Run inference
   ```bash
   python src/inference.py --input path/to/data --model path/to/model
   ```
   ⚠️ Note: The audio file should be in .wav format, ranging from 1s to 20s. No need to include the contextual sentence. Check the example under /samples/.


## Input Requirements
The input audio and associated text should meet the following criteria:

### Audio
- **Format**: WAV (.wav)
- **Channels**: Mono preferred (if stereo, the script will average channels)
- **Sample Rate**: 16 kHz is recommended (resampling is not automatic)
- **Duration**: 1 to 20 seconds
- **Bit Depth**: 16-bit PCM is ideal
- **Environment**: Clean audio (avoid overlapping speech or heavy background noise)
- **Recommended tool to enhance speech**: [Adobe Podcast](https://podcast.adobe.com/en)

### Transcription
- For **inference**, the system will automatically transcribe audio using [Whisper](https://github.com/openai/whisper).


## Retrain the model
You can retrain the model with your own dataset (audio + text).
The preprocessing and training are organized as separate steps to maximize flexibility and control for research purpose.

### Step 1: Prepare the input dataset

You need:
- A folder of `.wav` audio files
- A transcript CSV file

**CSV format**:
```csv
KEY,SENTENCE
audio1,This is a test sentence.
audio2,Another example of sarcasm.
```
The text_csv file is expected to contain at least two columns:
- KEY: This should be a unique identifier for each audio file, corresponding to the file name (without the .wav extension).
- SENTENCE: This column contains the transcriptions (or textual representations) of the audio files. Each row should correspond to the text spoken in the associated audio file.

    
Example directory:
```bash
/dataset/
├── audio/
│   ├── audio1.wav
│   └── audio2.wav
└── transcript.csv
```

### Step 2: Generate feature embeddings 
   ```bash
   python generate_embeddings.py \
    --audio_directory /path/to/audios \ 
    --text_csv /path/to/text.csv \  #Path to the CSV file containing the audio file keys and text, sample: "/data/mustard++_onlyU.csv"
    --output_directory /path/to/store/tmp_audio_features  #Directory where the temporary extracted audio feature files (LLDs) will be stored and removed after processing.
   ```

### Step 3: Normalize feature embeddings
Once the embeddings (audio, text, sentiment, and emotion) are extracted, you can normalize them using the following command:
   ```
   python normalize_embeddings.py --embeddings path/to/embeddings.h5 --output_dir path/to/output_directory
   ```
   This will generate four normalized files:
    - normalized_audio.h5
    - normalized_text.h5
    - normalized_sentiment.h5
    - normalized_emotion.h5
   
   Additionally, a separate label.h5 file will be generated containing the unmodified labels.

### Step 4: Train the model
When the normalized embeddings are generated, you can use the following command to train the model.
The default path for normalized embedding files are under data/.
   ```bash
   python train.py --data path/to/data --epochs 20 --batch_size 32 --model_path ./models/model.pth --patience 5 --lr 0.001
   ```
   
   -- data: Path to the folder containing preprocessed embeddings.
   -- epochs: Number of epochs for training (default is 20).
   -- batch_size: Number of samples per batch (default is 32).
   -- model_path: Path to save the trained model. Make sure to provide a full file name like ./models/model.pth.
   -- patience: Number of epochs to wait for improvement before early stopping.
   -- lr: Learning rate for the optimizer (default is 0.001).

   (You can adjust epochs, batch_size, and lr to your needs.)

   The script will output training progress, including the loss on the training and validation sets. The best model will be saved at the specified model path.

   The training process includes early stopping based on validation loss. If the model doesn't improve for patience number of epochs, training will stop early.


## Limitations
While SarcEmotiq shows commendable performance on benchmark data (74% F1-score) for binary sarcasm detection, users should be aware of its current limitations:

- **Domain generalization**: Trained primarily on MUStARD++ (scripted, American-accented data). Accuracy may drop for spontaneous speech or unfamiliar accents.
- **Transcript dependency**: Model performance degrades with inaccurate transcriptions. Whisper is robust but may misinterpret noisy speech.
- **Emotion/sentiment models**: Emotion and sentiment embeddings come from general-purpose models (not sarcasm-specific), which may add noise in some cases.
- **Real-time limitations**: This is a batch inference system; not optimized for real-time streaming or large-scale deployment without adaptation.
- **Context-free inference**: Each audio clip is processed independently. No dialog or speaker context is used.

(Future work may include context-aware transformers for broader deployment.)


## License
This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

You are free to:

- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:

- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the authors endorse you or your use.
- **NonCommercial** — You may not use the material for **commercial purposes**.

📄 Full license text: [https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/)


### Citation: 
Xiyuan Gao, Shekhar Nayak, Matt Coler; Improving sarcasm detection from speech and text through attention-based fusion exploiting the interplay of emotions and sentiments. Proc. Mtgs. Acoust. 13 May 2024; 54 (1): 060002. https://doi.org/10.1121/2.0001918

