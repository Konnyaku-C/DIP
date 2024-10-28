import torch.nn as nn
class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(8,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(64,256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        ### FILL: add more CONV Layers
        self.deconv3=nn.Sequential(
            nn.ConvTranspose2d(256,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv2=nn.Sequential(
            nn.ConvTranspose2d(64,8,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.deconv1=nn.Sequential(
            nn.ConvTranspose2d(8,3,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function


    def forward(self, x):
        # Encoder forward pass
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        # Decoder forward pass
        x=self.deconv3(x)
        x=self.deconv2(x)
        x=self.deconv1(x)
        ### FILL: encoder-decoder forward pass

        output = x
        
        return output
    