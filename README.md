# ICLHIF


An incremental bearing fault-diagnosis technique based on contrastive learning.

File description:

oneD: Stores the Case Western Reserve University (CWRU) benchmark bearing dataset.
models: Stores the weights (checkpoints) for each step in contrastive learning.
picture: Stores feature-visualization plots after each round of feature extraction.
Features: Stores the features after CL dimensionality reduction, and the subsequent anomaly detection and novel-class detection results.

data_util: Utilities for processing the CWRU data.
model_CL: Network design for the feature extractor.
train: Training the network.
experience: Validating/evaluating the network.
envs: Environment setup and required libraries/packages.