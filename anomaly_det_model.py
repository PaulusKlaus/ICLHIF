import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F 

from torchvision.models import resnet50 

import torch_geometric.nn as pyg_nn

class SimSiam(nn.Module):
    """
    SimSiam is an important follow up from the Siamese Network.
    It removes the need for negative samples or momentum encoders, which were central in methodes like SimCLR and BYOL. 
    This simplicity makes SimSiam highly computationaly efficient while retaining exlellent preformance.
    """
    def __init__(self, input_size = 1024, out_size= 16):
        super(SimSiam, self).__init__()
        self.input_size = input_size # input of the time domain and frequency domain should be the same -> one dimentional and with 1024 features each 
        self.output_size = out_size 
        self.predictor_hidden_size = 1024 # ???

        self.backbone = nn.Sequential(
            # this should be ResNet-1001 (1D CNN)
            # nn.Conv1d
            # in article it says that this consists of:
            """
            one-dimentional convolutional layer 
            activation layer (ReLU)
            fully connected layer 
            """
            # but in the article there is also reference to the source which shows this 
            # TODO: can implement different backbone function, even as differente classes 
            # such as : resnet-1001, resnet-18, ....

        )
        
        self.projector =nn.Sequential(
            # Here should be the implementation of the mlp (multi layer perceptron) with two layers 
            # nn.Linear().to(device), but i dont yet understand what does to(device) mean 
            )# or using 
            # pyg_nn.MLP
            # im not sure how many layers there are either 

        self.prediction = nn.Sequential(
            # two layers perceptron 
            # the output of this presictor should be the same at the outpur of the projector+gradient stop
        )

                                        


    def forward(self,s_i, a_i):
        # Online (student network)
        x_i = self.projector(self.backbone(s_i))  # (N, 1024, L)
        p1 = self.predictor_mlp(x_i)

        # Target branch (with stop-gradient)
        # Target (teacher network)
        f_i = self.projector(self.backbone(a_i))

        return p1, f_i
    


class loss:

    def __init__(self, projection_time, projection_freq, prediction_time, prediction_freq):
        self.loss = self.simsiam_loss(prediction_time,projection_freq)
        self.loss_forward = 0.5 * self.simsiam_loss(prediction_time, projection_freq) + 0.5 * self.simsiam_loss(prediction_freq, projection_time)

    # loss function 
    def simsiam_loss(self, p1,f_i, ver= "simple"):
        """
        Consine similarity between prediction_MLP and stopgradient ( projection vector) branch is computed and maximized 
        p1 (torch.Tensor): prediction vector 
        f_i ( torch.Tensor): projection vector 
        """
        # Detach z to stop gradient flow
        if ver == "original":
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



def oneD_Fourier_view(x_time: torch.Tensor) -> torch.Tensor:
    """
    1D Fourier Transform
    x_time: (batch_size, channel, length) -> (B, 1, n) real value 
    Return a real spectrum (magnitude) with same length n (two sided), 
    normalized per sample.  
    TODO: what exactly does it mean normalized pre sample?
    """

    # The data has an extra dimension
    data = x_time.squeeze(1) # (B, 1, n) -> (B, n)
    Xf = torch.fft.fft(data) # complex values 
    mag = torch.abs(Xf).unsqueeze(1) # (B, n) -> (B, 1, n)
    # normalisation -> the highest number is one and other values are scaled appropriatly 
    mag = mag / (mag.amax(dim = -1, keepdim = True) + 1e-8)  #TODO: im not sure what does this do ???
    return mag 


class BasicBlock1D(nn.Module): 
    def __init__(self, input_size, output_size, kernes_size = 3, strides = 2, p = None):
        super().__init__()
        # convolution, batchNorm, Relu 
        # convolution, batchNorm 
        # shortcut connection 

    def forword(self, x):
        # shortcut connection for the RESNET
        idt = x # if down is none 
        out = None # activation1(batchNorm1(conv(1)))
        out = None  #batchNorm1(conv(1))
        out = None # activation( out + idt )
        return out 
    

class BackBone1D(nn.Module):
    def __init__(self, feat_dim = 128):   # TODO: steel dont know how to choose dim 
        super().__init__()
        # Intial convolution + BN + Relu
        self.init = nn.Sequential()

        # Stage 1
        self.layer1 = BasicBlock1D()
        # Stage 2 
        self.layer2 = BasicBlock1D()
        # Stage 3 
        self.layer3 = BasicBlock1D()
     
        # Final BN + ReLU 
        # TODO: make sure that the BN and ReLU before average pooling and classifier
        # Global avarage pooling and classifier 
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(feat_dim,feat_dim)

    def forward(self,x):
        x = self.init(x)
        # TODO : rest of implementation 


class Projector_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim = 128, out_size = 16):
        super().__init__()
        self.net = nn.Sequential(
            # 2* linear layers 
        )

    def forward( self, x): return self.net(x)
        


class Predictor_MLP(nn.Module):
    def __init__(self, in_dim = 16, hidden_dim = 64, out_size = 16):
        super().__init__()
        self.net = nn.Sequential(
            # 2* linear layers (basicly the same as projector???)
        )

    def forward( self, x): return self.net(x)


class Encoder(nn.Module):
    """
    Encoder = Backbone (1D ResNet 1001) + Projection MLP (to m-dim feature).
    """
    def __init__(self,
                in_channel = 1,
                out_channel = 16,
                stride = 2,
                kernes_size = 3, 
                backbone = BackBone1D(feat_dim=128),
                projector = Projector_MLP(128, hidden_dim=128,
                )):

        super(Encoder, self).__init__()

        self.in_channel  = in_channel
        self.output_channel = out_channel

        # Hyper parameters 
        self.stide = stride # controlls downsampling rate: how much we mode per step
        self.kernel_size = kernes_size # controls the "receptive field": how many tiemsteps each filter looks at once.

        # Encoder parts 
        self.backbone = backbone
        self.projection = projector # shoud there be 128 hidden layers or 64 ??

    def forward( self, x): 
        h = self.backbone(x) # ( B, 1, n)
        z = self.projection(h) # (B, backbone_dim)
        z = F.normalize(z, dim=1) # L2-norm before similarity 
        # TODO: Why do we normilise 

        return z 