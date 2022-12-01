
# Lifelong Variational Autoencoder via Online Adversarial Expansion Strategy

>📋 This is the implementation of Learning Dynamic Latent Spaces for Lifelong Generative Modelling

>📋 Accepted by AAAI 2023

# Title : Learning Dynamic Latent Spaces for Lifelong Generative Modelling

# Paper link : 


# Abstract
Task Free Continual Learning (TFCL) aims to capture novel concepts from non-stationary data streams without forgetting previously learned knowledge. Mixture models, which add new components when certain conditions are met, have shown promising results in TFCL tasks. However, such approaches do not make use of the knowledge already accumulated for positive knowledge transfer. In this paper, we develop a new model, namely the Online Recursive Variational Autoencoder (ORVAE). ORVAE utilizes the prior knowledge by recursively incorporating the information flow learned by previously trained components with those learned by newly created components into its latent space. To fully explore this information when learning novel samples, we introduce a new attention mechanism to regularize the structural latent space in which the most important information are reused while the information that interferes with novel samples is inactivated. The proposed attention mechanism can maximize the benefit from the forward transfer for learning novel samples without causing forgetting on previously learnt knowledge. We perform several experiments which show that ORVAE achieves state-of-the-art results under TFCL.

# Environment

1. Tensorflow 2.1
2. Python 3.6

# Training and evaluation

>📋 Python xxx.py, the model will be automatically trained and then report the results after the training.

>📋 Different parameter settings of OCM would lead different results and we also provide different settings used in our experiments.

# BibTex
>📋 If you use our code, please cite our paper as:


