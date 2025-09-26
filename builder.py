import torch
import torch.nn as nn
import torch.nn.functional as F 

#from torchvision.models import resnet50 
#from torchvision.ops import MLP
#from  torch_geometric.nn import MLP as MLP2

class BasicBlock1D(nn.Module):
    """
    ResNet layer with 1D convolution, optional batch normalization, and activation ReLU.
    The Order of the operations dependt on the conv_first, making it more flexible.
    conv-> bn -> relu - conv -> bn -> shortcut? -> relu 
    """ 

    expantion = 1  # TODO : what and why????

    def __init__(self, in_channel = 1, output_channel= 128, kernes_size = 3, stride = 2, conv_first = True ):
        super().__init__()
        self.conv_first = conv_first
        
        pad = kernes_size//2

        self.conv_1 = nn.Conv1d(in_channels=in_channel, out_channels=output_channel, kernel_size=kernes_size, stride=stride, padding=pad)
        self.relu_1 = nn.ReLU(inplace=True)
        self.bn_1 = nn.BatchNorm1d(output_channel)
        self.bn = nn.BatchNorm1d(in_channel)

        self.conv_2 = nn.Conv1d(in_channels=output_channel, out_channels=output_channel, kernel_size=kernes_size, stride=1, padding=pad)
        self.relu_2 = nn.ReLU(inplace=True)
        self.bn_2 = nn.BatchNorm1d(output_channel) 

        # shortcut path (identity unless shape changes)
        self.downsample = None
        if stride != 1 or in_channel != output_channel:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels=in_channel, out_channels=output_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(output_channel)
            ) 
        

    def forward(self, x):
        identity = x 
        if self.conv_first:

            out = self.conv_1(x)
            out = self.bn_1(out)
            out = self.relu_1(out)

            out = self.conv_2(out)
            out = self.bn_2(out)

            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            out = self.relu_2(out)
        else:
            out = self.bn(x)
            out = self.relu_1(out)
            out = self.conv_1(out)
            out = self.bn_2(out)
            out = self.relu_2(out)
            out = self.conv_2(out)
            # implementation is not finished 
            raise ConnectionError


        

        return out
    

class BackBone1D(nn.Module):
    """
    BackBone of the SimSiam model.
    """
    def __init__(self, in_channel=1, out_channel = 256 ):
        super().__init__()

        self.init = BasicBlock1D(in_channel=in_channel, output_channel=64, kernes_size=3, stride=2) # (B,1,1024) -> (B,64,512)
        # Stage (512 ->256)
        self.layer1 = BasicBlock1D(64, 80, 3, 2) 
        # Stage (256->128)
        self.layer2 = BasicBlock1D(80, 94, 3, 2)
        # Stage (128->64)
        self.layer3 = BasicBlock1D(94, 112, 3, 2)
        # Stage (64->32)
        self.layer4 = BasicBlock1D(112, 128, 3, 2)
        # Stage (32->16)
        self.layer5 = BasicBlock1D(128, out_channel, 3, 2) #(256,64)
        
        self.extra = BasicBlock1D(out_channel, out_channel, 3, 1)

        self.final_bn = nn.BatchNorm1d(out_channel)
        self.final_relu = nn.ReLU(inplace=True)

        # Stage (256, 16) -> (4096)
        self.flatten = nn.Flatten(1) 
        self.out_dim = out_channel

        # Global avarage pooling and classifier 
        self.pool = nn.AdaptiveAvgPool1d(1) # -> output_channel(256) x 1
        self.fc = nn.Linear(out_channel, out_channel)

    def forward(self,x):

        x = self.init(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.extra(x)
        x = self.final_bn(x)
        x = self.final_relu(x)
        x = self.flatten(x) # flatten (B, )
        #x = self.pool(x)

        #x = self.fc(x)
        return x 

class Projector_MLP(nn.Module):
    # Here should be the implementation of the mlp (multi layer perceptron) with two layers 
    # nn.Linear().to(device), but i dont yet understand what does to(device) mean 
    # or using 
    # pyg_nn.MLP
    # im not sure how many layers there are either 
    def __init__(self, in_dim=4096, hidden_dim=256, out_dim=16):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 256-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)  # Changes this from hiddel layer !!!!!!!!!!!!!!!!!!
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 

    

class Predictor_MLP(nn.Module):
    # two layers perceptron 
    # the output of this presictor should be the same at the outpur of the projector+gradient stop
    def __init__(self, in_dim=16, hidden_dim=4, out_dim=16): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h_s input and output (z and p) is d = 16, 
        and h_s hidden layers dimension is 4, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


       
#cosine similarity 
def cosine_similarity(p1,f_i, ver= "simple"):
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
    
def loss_func(z_1, z_2, p_1, p_2):
    return  0.5 * cosine_similarity(p_1, z_2) + 0.5 * cosine_similarity(p_2, z_1)

        

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



class SimSiam(nn.Module):
    """
    Encoder = Backbone (1D ResNet 1001) + Projection MLP (to m-dim feature).
    """
    def __init__(self,
                in_channel = 1,
                out_channel = 16,
                backbone = BackBone1D(),
                projector = Projector_MLP()
                ):

        super().__init__()

        self.in_channel  = in_channel
        self.output_channel = out_channel

        # Encoder parts 
        self.backbone = backbone
        self.projector = projector 

        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )
        self.predictor = Predictor_MLP()

    def forward( self, x_1, x_2):  # x_1 satands for raw data and x_2 stands for frequency 
        f, h = self.encoder, self.predictor

        # Encoder step
        z_1, z_2 = f(x_1), f(x_2)
        # Predictor step 
        p_1, p_2 = h(z_1), h(z_2)
        Loss = loss_func(z_1, z_2, p_1, p_2)
        return {'loss': Loss}
    




def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH = 8
    N = 1024

    x_time = torch.randn(BATCH, 1, N, device=device)
    x_freq = oneD_Fourier_view(x_time)

    backbone = BackBone1D(in_channel=1, out_channel=256).to(device)
    projector = Projector_MLP(in_dim=4096, hidden_dim=256, out_dim=16).to(device)
    model = SimSiam(in_channel=1, out_channel=16, backbone=backbone, projector=projector).to(device)
    model.train()

    out = model(x_time, x_freq)
    assert "loss" in out, "Model.forward should return {'loss': ...}"
    loss = out["loss"]
    print("Initial loss:", float(loss))
    print("loss backwards",loss.backward())

    # simple optimizer step to verify gradients flow
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    opt.step()
    opt.zero_grad()

    # Check tensor shapes through encoder alone
    with torch.no_grad():
        z = nn.Sequential(backbone, projector)(x_time)
    print("z shape:", tuple(z.shape))  # should be (BATCH, 16)

    # invariants
    assert z.shape == (BATCH, 16), f"Expected (BATCH,16), got {z.shape}"

    print("Smoke test passed ✅")









if __name__ == "__main__":

    main()

    model = SimSiam()
    x1 = torch.randn(2,1,1024)
    x2 = torch.randn_like(x1)
    x_freq = oneD_Fourier_view(x1)

    model.forward(x1, x_freq)
    print("forward check")

    z1 = torch.randn((200, 2560))
    z2 = torch.randn_like(z1)
    import time
    tic = time.time()
    print(cosine_similarity(z1, z2, ver='original'))
    toc = time.time()
    print(toc - tic)
    tic = time.time()
    print(cosine_similarity(z1, z2, ver='simple'))
    toc = time.time()
    print(toc - tic)