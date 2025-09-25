import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F 

from torchvision.models import resnet50 

import torch_geometric.nn as pyg_nn

# Questions:
"""
How do we know how many convolutional layers do we need?
what is stride, and kernel_size , pading 


kernels - convolutional layers extract featurs from their inputs with the helh of the filters called kernels 

stride - the movement of the kernels over the current data is given by strides, which tells the kernes how many rows and columns to move before doing a new calculation 

batch size - number of training samples processed by neural network before updating its internal parameters

featurs - 1024 (input) -> 16 (output)

 .detach() is the gradientstop?
"""



# Augmentations for contrastive learning -> FFT
# represents what features the model will learn 

class SimSiam(nn.Module):
    # Why SimSiam?
    """
    SimSiam is an important follow up from the Siamese Network.
    It removes the need for negative samples or momentum encoders, which were central in methodes like SimCLR and BYOL. 
    This simplicity makes SimSiam highly computationaly efficient while retaining exlellent preformance.
    """
    def __init__(self, backbone, projector, device, stride =2, in_channel = 1, out_channel = 16):
        super(SimSiam, self).__init__()
        self.in_channel  = in_channel
        self.output_channel = out_channel
        self.predictor_hidden_size = 1024 # ???

        self.backbone = backbone
        self.projector = projector 
        
        self.projector =nn.Sequential( # also called backbone
            # First convolutional layer
           nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            # Second convolutional layer
           nn.Conv1d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            # Third convolutional layer 
           nn.Conv1d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            # Forurth Convolutional layer 
           nn.Conv1d(256, 512, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            # Fifth conv layer 
           nn.Conv1d(512, 1024, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(1024)
            )
        # There is also another way 
        # Encoder head 
        # Fully connected layers for output 
        self.lin1 = nn.Linear(1024*6, 512) # the size depends on the backbone after flattening
        # Projection head 
        self.mlp  = pyg_nn.MLP(in_channels=1024*2, hidden_channels=[512], out_channels=1024*2, 
                              num_layers=3, dropout=0.0, batch_norm=True, act="relu")
        # Alternative 2 
        self.predictor_mlp = nn.Sequential(nn.Linear(self.output_size, self.predictor_hidden_size) , # the size depends on the backbone after flattening 
                                           nn.BatchNorm1d(self.predictor_hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.predictor_hidden_size, self.output_size)).to(device)
        

        # Shortcut connection for matching dimensions if necessary
        self.shortcut = nn.Sequential()
        if stride != 2:
            self.shortcut = nn.Sequential(
                nn.Conv1d(),
                nn.BatchNorm1d()
            )


                                        


    def forward(self,s_i, a_i):
        # Online (student network)
        x_i = self.projector(self.backbone(s_i))  # (N, 1024, L)
        p1 = self.predictor_mlp(x_i)

        # Target branch (with stop-gradient)
        # Target (teacher network)
        f_i = self.projector(self.backbone(a_i))

        return p1, f_i
    

# Augmentation 
"""
Invariance: The network should produce similar embeddings regardless of the augmentation
Diversity: The learned representation should remain distinguishable across different images
Avoiding Collapse: Preventing the network from producing identical outputs for all inputs. 
"""

# Encoder 
"""
Encode the augmented views into feature representations to learn a rich,
 high-level representation of the input data. 
"""

# Projection 
"""
Project Features to an Embedding Space to transform extracted features 
into a lower-dimensional embedding space where comparisons (e.g., similarity calculations) 
are more meaningful.

Multi-layer perceptron (MLP) for projection in SimSiam 
this module projects the backbone's output to a feature space for contrastive learning
"""

# Prediction
"""
To learn a mapping from the projected embeddings to a predicted space. T
he predictor helps the model align representations without collapsing into trivial solutions 
(e.g., predicting constant vectors).

Prediction_MLp is applied to the output of one branch 
"""
# loss function 
def simsiam_loss(p1,f_i, ver):
    """
    Consine similarity between prediction_MLP and stopgradient ( projection vector) branch is computed and maximized 
    p1 (torch.Tensor): prediction vector 
    f_i ( torch.Tensor): projection vector 
    """
    # Detach z to stop gradient flow
    if ver == " original":
        f_i = f_i.detach()

        # Normalize vectors 
        p1 = F.normalize(p1, dim=1)
        f_i = F.normalize(f_i, dim=1)

        # Original formulation: negative dot product 
        return -(p1 * f_i).sum(dim =1).mean()
    elif ver == "simple":
        return -F.cosine_similarity(p1, f_i.detach(), dim=-1).mean()
    else: 
        raise Exception 

def loss_forward(z1, z2, p1, p2):
        """
        Compute the SimSiam loss for two pairs of projections and predictions.
 
        Args:
            z1 (torch.Tensor): Projection vector from the first augmented view.
            z2 (torch.Tensor): Projection vector from the second augmented view.
            p1 (torch.Tensor): Prediction vector corresponding to z1.
            p2 (torch.Tensor): Prediction vector corresponding to z2.
 
        Returns:
            torch.Tensor: Averaged SimSiam loss.
        """
        # Compute the loss for each pair (p1, z2) and (p2, z1)
        loss1 = simsiam_loss(p1, z2)
        loss2 = simsiam_loss(p2, z1)
 
        # Average the two losses
        return 0.5 * loss1 + 0.5 * loss2



# Loss function 
# https://towardsdatascience.com/a-practical-guide-to-contrastive-learning-26e912c0362f/
# https://arxiv.org/pdf/2011.10566
cos = nn.CosineSimilarity(dim=1)
def negative_cosine_similarity(pred, proj):
   return -cos(pred, proj.detach()).mean()

# .detach() is the gradientstop?
