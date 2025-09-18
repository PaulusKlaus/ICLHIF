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

Index/Label 
0 :Normal_1: Normal
1: Ball_007_1: Ball defect (0.007 inch)
2: Ball_014_1: Ball defect (0.014 inch)
3: Ball_021_1: Ball defect (0.021 inch)
4: IR_007_1: Inner race fault (0.007 inch)
5: IR_014_1: Inner race fault (0.014 inch)
6: IR_021_1: Inner race fault (0.021 inch)
7: OR_007_6_1: Outer race fault (0.007 inch, data collected from 6 O'clock position)
8: OR_014_6_1: Outer race fault (0.014 inch, 6 O'clock)
9: OR_021_6_1: Outer race fault (0.021 inch, 6 O'clock)