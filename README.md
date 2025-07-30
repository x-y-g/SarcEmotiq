# 🎙️😜 SarcEmotiq
SarcEmotiq is a deep learning-based tool for recognizing sarcasm in English audio. It uses pre-trained models trained on specific datasets (MUStARD++) but also allows users to retrain the model with their own data.

➡️ More information about the model please visit our publication: https://pubs.aip.org/asa/poma/article/54/1/060002/3305267/Improving-sarcasm-detection-from-speech-and-text

## 🚀 Installation
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


## 🔄 Retrain the model
You can retrain the model with your own dataset. Here's how:

1. First, you need to generate embeddings compatible with the model from your dataset:
   ```bash
   python generate_embeddings.py \
    --audio_directory /path/to/audios \ 
    --text_csv /path/to/text.csv \  #Path to the CSV file containing the audio file keys and text, sample: "/data/mustard++_onlyU.csv"
    --output_directory /path/to/store/tmp_audio_features  #Directory where the temporary extracted audio feature files (LLDs) will be stored and removed after processing.
   ```
   📋 Text CSV format:

   ```
   KEY,SENTENCE
   audio1,"This is a test sentence."
   audio2,"Another example of audio transcription."
   ```
   The text_csv file is expected to contain at least two columns:

   KEY: This should be a unique identifier for each audio file, corresponding to the file name (without the .wav extension).

   SENTENCE: This column contains the transcriptions (or textual representations) of the audio files. Each row should correspond to the text spoken in the associated audio file.


2. Once the embeddings (audio, text, sentiment, and emotion) are extracted, you can normalize them using the following command:
   ```
   python normalize_embeddings.py --embeddings path/to/embeddings.h5 --output_dir path/to/output_directory
   ```
   This will generate four normalized files:

    - normalized_audio.h5
    - normalized_text.h5
    - normalized_sentiment.h5
    - normalized_emotion.h5
   
   Additionally, a separate label.h5 file will be generated containing the unmodified labels.


3. When the normalized embeddings are generated, you can use the following command to train the model.
The default path for normalized embedding files are under data/.
   ```bash
   python train.py --data path/to/data --epochs 20 --batch_size 32 --model_path ./models/model.pth --patience 5 --lr 0.001
   ```
   
   - data: Path to the folder containing preprocessed embeddings.
   - epochs: Number of epochs for training (default is 20).
   - batch_size: Number of samples per batch (default is 32).
   - model_path: Path to save the trained model. Make sure to provide a full file name like ./models/model.pth.
   - patience: Number of epochs to wait for improvement before early stopping.
   - lr: Learning rate for the optimizer (default is 0.001).

   🖇️ You can adjust epochs, batch_size, and lr to your needs.

   🖇️ The script will output training progress, including the loss on the training and validation sets. The best model will be saved at the specified model path.

   🖇️ The training process includes early stopping based on validation loss. If the model doesn't improve for patience number of epochs, training will stop early.

## 📜 License
Licensed under the Apache License, Version 2.0 (the "License");
You may not use this file except in compliance with the License.
You may obtain a copy of the License at:

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

However, **commercial use of this software, including any associated patents, is prohibited** without the express written consent of the copyright holder. For non-commercial, academic, and research purposes, the software is free to use, modify, and distribute, provided that all modifications and attributions are clearly noted.

### 🛡️ Patent Reservation:
The copyright holder explicitly reserves the right to apply for patent protection on any inventions or innovations disclosed in this software. The release of this software under this license does not constitute a waiver of any patent rights, including the right to file for patent protection in the future.

### ⚙️ Attribution:
When using, redistributing, or modifying this software for non-commercial purposes, you must provide appropriate credit, indicating the source as follows:

"This software is based on work developed by Gao et al. (2024) and licensed under the Apache 2.0. For more information, visit: https://github.com/x-y-g/SarcEmotiq."

### 📖 Citation: 
Xiyuan Gao, Shekhar Nayak, Matt Coler; Improving sarcasm detection from speech and text through attention-based fusion exploiting the interplay of emotions and sentiments. Proc. Mtgs. Acoust. 13 May 2024; 54 (1): 060002. https://doi.org/10.1121/2.0001918

