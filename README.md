# SarcEmotiq
SarcEmotiq is a deep learning-based tool for recognizing sarcasm in English audio. It uses pre-trained models trained on specific datasets (MUStARD++) but also allows users to retrain the model with their own data.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SarcEmotiq.git
   ```
pip install -r requirements.txt

## Pre-trained model
Download the pre-trained model from [https://drive.google.com/file/d/1xqyofUELCl2oBlA6151Vvd-f8pIG2biF/view?usp=drive_link]. Place it in the `models/` folder.

## Usage
### Inference with Pre-trained Model
To use the sarcasm recognition model on a new audio file (.wav):
   ```bash
   python src/inference.py --input path/to/data --model path/to/model
   ```

For retraining:
You can retrain the model on your own dataset.

First, you need to generate embeddings compatible with the model from your dataset:
   ```bash
python generate_embeddings.py \
    --audio_directory /path/to/audio \
    --text_csv /path/to/text.csv \
    --opensmile_path /path/to/opensmile \
    --config_file /path/to/opensmile/config/compare16/ComParE_2016.conf
   ```

When the embeddings are generated, you can use the following command to train the model.
The default path for embedding files are under data/.
   ```bash
   python src/train.py --data path/to/your/data --epochs 20 --batch_size 32 --model_path path/to/save/model.pth --lr 0.001
   ```
## Example use

## Repository Structure
SarcEmotiq/
├── src/                    # Source code for the project
│   ├── train.py            # Training script
│   ├── inference.py        # Inference script
├── data/                   # (Optional) Sample data
├── models/                 # Folder for pre-trained models
├── notebooks/              # Jupyter notebooks for examples
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── LICENSE                 # License file

## License
This project is licensed under the MIT License.



