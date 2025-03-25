# Unit-Based Vocoder Training

This repository contains the workflow for training a unit-based vocoder. The pipeline involves processing data, extracting acoustic features, learning K-means clusters, quantizing the data, training the vocoder, and doing inference.


## Steps

### 1. Data Preparation

- Original Dataset by J2Team: [Link](https://www.facebook.com/groups/j2team.community/permalink/1010834009248719/)

It contains 25-hour speech by a single female speaker.

You can denoise it with this notebook from NTT123 [Link](https://github.com/NTT123/vietTTS/blob/master/notebooks/denoise_infore_dataset.ipynb)

### 2. Acoustic Model & Feature Extraction
Pretrained acoustic representation model [Link](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h)

Notebook name: 2_feature_extraction.ipynb

### 3. K-Means Clustering
Learn K-means clustering from extracted acoustic representations.

Notebook name: 3_kmeans_training.ipynb

### 4. Quantization
Quantize the extracted features using the trained K-means model.

Notebook name: 4_quantization.ipynb

### 5. Vocoder Training
Train a vocoder model using the quantized data.

Notebook name: 5_vocoder_training.ipynb

### 6. Inference
Generate audio using the trained vocoder.

Notebook name: 6_inference.ipynb

## Acknowledgments
- The workflow largely follows the guide by fairseq toolkit: [Speech to Unit Model (speech2unit)
](https://github.com/facebookresearch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit)
- J2Team for the high quality dataset
- The Wav2Vec2 Vietnamese Pretrained Model by author Thai Binh Nguyen [Link repo
](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h)
