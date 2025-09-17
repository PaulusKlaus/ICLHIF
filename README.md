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


Fault tipe identidication 

Ball_007_1: Ball defect (0.007 inch)
Ball_014_1: Ball defect (0.014 inch)
Ball_021_1: Ball defect (0.021 inch)
IR_007_1: Inner race fault (0.007 inch)
IR_014_1: Inner race fault (0.014 inch)
IR_021_1: Inner race fault (0.021 inch)
Normal_1: Normal
OR_007_6_1: Outer race fault (0.007 inch, data collected from 6 O'clock position)
OR_014_6_1: Outer race fault (0.014 inch, 6 O'clock)
OR_021_6_1: Outer race fault (0.021 inch, 6 O'clock)

