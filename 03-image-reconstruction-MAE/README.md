# Self-Supervised Image Reconstruction with Masked Autoencoder (MAE)

This repo implements Masked Autoencoders (MAE) for self-supervised learning on Tiny ImageNet and demonstrates how an encoder learns robust visual representations by reconstructing masked image patches.

## Introduction
MAE trains a Vision Transformer encoder to reconstruct missing patches from heavily masked images. By keeping only a small portion of patches (typically 25%) and predicting the remaining 75%, the model is forced to learn high-level semantics rather than trivial pixel interpolation.

Project resources:
- Notebook & code: [Self-Supervised Learning using Masked Autoencoders on Kaggle](https://www.kaggle.com/code/zeeshankhalid666/self-supervised-learning-using-masked-autoencoders)
- Dataset used: [Tiny ImageNet on Kaggle](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)
- Full blog: [here](https://zeshankhalid.com/blog/image-reconstruction-mae/)

## Architecture
Input images are patchified into a sequence of tokens. A high fraction of these tokens (mask_ratio ≈ 0.75) is randomly masked so the model must predict missing content from limited visible context. A Vision Transformer (ViT) encoder processes only the visible tokens for computational efficiency. A lightweight decoder (a small ViT) reconstructs pixel values for the masked tokens by combining the visible-token latents with learned mask tokens. The training objective is mean squared error (MSE) computed only on the masked patches.

## Training & practical notes
- Mask ratio: 75% (common MAE setting).
- Precompute patches or features to speed experiments when needed.
- Optimizer: AdamW; scale learning rate with batch size (linear scaling rule).
- Typical setup: train for many epochs; MAEs often require long schedules (hundreds of epochs) to learn high-frequency details.

## Why results may not be good
- Short training: MAEs rely on long self-supervised schedules. short runs mainly learn low-frequency/color statistics.
- Poor hyperparameter tuning: unstable optimization reduces final quality.
- Low-resolution sources: upscaling small images (e.g., 64->224) blurs high-frequency details and handicaps reconstruction.
- Limited model capacity: a too-small encoder/decoder cannot capture fine texture.

## Suggestions to improve
To improve results:
- train for many more epochs (hundreds) and use larger batches when possible.
- Perform experiments with hyperparameters
- Use higher-quality or higher-resolution images and avoid aggressive upscaling that destroys high-frequency detail. 
- Increase decoder capacity or add perceptual and/or adversarial refinement stages to obtain sharper reconstructions. 
- Prefer using the trained encoder for downstream fine-tuning (classification or detection) instead of focusing solely on pixel-level reconstruction.
