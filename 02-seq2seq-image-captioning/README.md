# Seq2Seq Image Captioning (Encoder–Decoder LSTM)

This repository contains a concise implementation of an image captioning pipeline using a sequence-to-sequence (encoder–decoder) LSTM architecture. The goal is to map an image to a variable-length caption by first extracting visual features with a pretrained CNN (ResNet-50) and then generating text with an LSTM decoder.

## Introduction
We use the Flickr30k dataset (~30K images, ~158K captions, five captions per image). The worked notebook and runnable code are available on Kaggle and GitHub:
- Dataset: https://www.kaggle.com/datasets/adityajn105/flickr30k
- Kaggle notebook: https://www.kaggle.com/code/zeeshankhalid666/image-captioning-model-using-encoder-decoder
- Blog post: https://zeshankhalid.com/blog/seq2seq-image_captioning/

Images are preprocessed into 2048-D features by removing the final classification layer of a pretrained ResNet-50. Captions are cleaned, tokenized, and converted to indices; uncommon words map to `<unk>`. We fix a maximum caption length of 25 and use `<start>`, `<end>`, and `<pad>` tokens.

## Architecture 

Encoder:
- Pretrained **ResNet-50** (final classifier removed) -> 2048-D feature vector per image.
- A linear layer maps the 2048-D vector into the decoder's hidden space.

Decoder:
- Custom **LSTM-based decoder** that receives the encoder vector as an initial hidden state and generates words one token at a time.
- During training the decoder is teacher-forced with ground-truth tokens; during inference it autoregressively feeds predictions back as inputs.

## Training & Pipeline notes
Precomputed visual features to speed up training, and build the vocabulary using a minimum frequency threshold (`min_freq = 5`), so rare words are mapped to `<unk>`. Used token padding to a fixed maximum caption length and mask pad tokens during loss computation. Typical hyperparameters for this project include an LSTM hidden size of approximately 512 and a max caption length of 25. Training was performed in batches with validation, and  BLEU score, token-level F1, and perplexity was added track model progress and guide tuning.

## Discussion (why results are modest)
The stored model is a basic, prototype implementation rather than a state-of-the-art captioner. Key reasons the scores are low:
- Fixed-size context bottleneck: compressing all visual information into a single vector loses spatial detail, so the decoder often misses fine-grained cues.
- LSTM limitations: vanilla LSTMs struggle to align long-range visual-language correspondences without explicit mechanisms for focusing on image regions.
- Limited capacity/tuning: fewer epochs, simple preprocessing, and lack of attention or richer visual features restrict performance.

Practical improvements that can improve results may include adding attention (spatial or transformer attention), using grid-level CNN features or object detectors, increasing training time, and tuning model/hyperparameters.

Below is a representative sample from the experiments (see notebook for more examples and quantitative scores)

![Sample result 1](./Seq2Seq-Image_Captioning/img2.png)
