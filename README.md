# Self-Contrastive Learning based Semi-Supervised Radio Modulation Classification
This repo is our semi-supervised model, namely SemiAMC, for radio modulation classification. The architecture of SemiAMC is shown in the following figure. SemiAMC aims at training a classifier to accurately recognize the modulation type for any given radio signal. As a semi-supervised framework, SemiAMC is trained with both labeled and unlabeled data. The illustrated workflow is as follows.

The first step is called self-supervised contrastive pretraining, where we train an encoder to map the original radio measurements into low-dimensional representations. This is done in a self-supervised manner, with unlabeled data only. The supervision here comes from optimizing the contrastive
loss function, that maximizes the agreement between the representations of differently augmented views for the same data sample. Here, we apply [simclr](https://github.com/google-research/simclr) as our self-supervised contrastive learning framework.

In step two, we freeze the encoder learned during the self-supervised contrastive pre-training step, and map the labeled input radio signals to their corresponding representations in the low-dimensional space. The classifier can be trained based on these representations and their corresponding labels. Here a relatively simple classifier (e.g., linear model) usually work well, because the latent representation has already extracted the intrinsic information from the signal input. In this way, a small number of labeled data samples is enough to train the classifier.

Please refer https://ieeexplore.ieee.org/document/9652914 for mre details.

![Screen Shot 2022-03-23 at 8 55 52 PM](https://user-images.githubusercontent.com/9064192/159826484-8c7bfec4-9b44-49c1-993c-8afaee4d0517.png)

## Environment
Python 3.7.10  
Tensorflow 2.4.0  

Install the required packages by running  
`pip install -r requirements.txt`

## Dataset
We use **RML2016.10a** as our dataset. Please go to https://www.deepsig.ai/datasets, and download RML2016.10a. Then put **RML2016.10a_dict.pkl** to 'data/' folder.

## Run the code
We show how to run our code in **main.ipynb**.
