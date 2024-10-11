# SarcEmotiq
SarcEmotiq is a deep learning-based tool for recognizing sarcasm in English audio. It uses pre-trained models trained on specific datasets (MUStARD++) but also allows users to retrain the model with their own data.
More information about the model please visit: https://pubs.aip.org/asa/poma/article/54/1/060002/3305267/Improving-sarcasm-detection-from-speech-and-text

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SarcEmotiq.git
   ```
2. Set up the environment:
   ```bash
   pip install -r requirements.txt
   ```
3. You can choose to use the pre-trained model or retrain the model on your dataset. Details are shown as follows:

## Pre-trained model
Download the pre-trained model from [https://drive.google.com/file/d/1xqyofUELCl2oBlA6151Vvd-f8pIG2biF/view?usp=drive_link]. Place it in the `models/` folder.

### Inference with Pre-trained Model
To use the sarcasm recognition model on a new audio file (.wav):
   ```bash
   python src/inference.py --input path/to/data --model path/to/model
   ```
The audio file should be in .wav format, ranging from 1s to 20s. No need to include the contextual sentence.
An example can be found under "/samples/."

## For retraining
You can retrain the model with your own dataset.

1. First, you need to generate embeddings compatible with the model from your dataset:
   ```bash
   python generate_embeddings.py \
    --audio_directory /path/to/audios \ 
    --text_csv /path/to/text.csv \  #Path to the CSV file containing the audio file keys and text, sample: "/data/mustard++_onlyU.csv"
    --opensmile_path /path/to/opensmile \  #path to OpenSMILE installation directory, default: compare16 
    --output_directory /path/to/store/tmp_audio_features  #Directory where the temporary extracted audio feature files (LLDs) will be stored and removed after processing.
   ```
   The text_csv file is expected to contain at least two columns:

   KEY: This should be a unique identifier for each audio file, corresponding to the file name (without the .wav extension).

   SENTENCE: This column contains the transcriptions (or textual representations) of the audio files. Each row should correspond to the text spoken in the associated audio file.
   ```
   KEY,SENTENCE
   audio1,"This is a test sentence."
   audio2,"Another example of audio transcription."
   ```

2. When the embeddings are generated, you can use the following command to train the model.
The default path for embedding files are under data/.
   ```bash
   python src/train.py --data path/to/your/data --epochs 20 --batch_size 32 --model_path path/to/save/model.pth --lr 0.001
   ```
   You can set up your own epochs, batch_size and lr.

## License
Licensed under the Apache License, Version 2.0 (the "License");
You may not use this file except in compliance with the License.
You may obtain a copy of the License at:

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

However, **commercial use of this software, including any associated patents, is prohibited** without the express written consent of the copyright holder. For non-commercial, academic, and research purposes, the software is free to use, modify, and distribute, provided that all modifications and attributions are clearly noted.

### Patent Reservation:
The copyright holder explicitly reserves the right to apply for patent protection on any inventions or innovations disclosed in this software. The release of this software under this license does not constitute a waiver of any patent rights, including the right to file for patent protection in the future.

### Attribution:
When using, redistributing, or modifying this software for non-commercial purposes, you must provide appropriate credit, indicating the source as follows:

"This software is based on work developed by Gao et al. (2024) and licensed under the Apache 2.0. For more information, visit: https://github.com/x-y-g/SarcEmotiq."

### Citation: 
Xiyuan Gao, Shekhar Nayak, Matt Coler; Improving sarcasm detection from speech and text through attention-based fusion exploiting the interplay of emotions and sentiments. Proc. Mtgs. Acoust. 13 May 2024; 54 (1): 060002. https://doi.org/10.1121/2.0001918

