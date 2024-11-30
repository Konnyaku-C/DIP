import torch
import torch.nn as nn
class CGAN_Discriminator(nn.Module):
    def __init__(self,img_channels, condition_channels):
        super(CGAN_Discriminator,self).__init__()
        self.disc=nn.Sequential(
            nn.Conv2d(img_channels+condition_channels, 16,kernel_size=3,padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(16, 32,kernel_size=4,stride=2,padding=1),
            #128*128*128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(32, 64, kernel_size=4,stride=2,padding=1),
            #256*64*64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Flatten(),
            nn.Linear(64*64*64, 70),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(70, 8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )


    def forward(self, x, condition):
       
        x=torch.cat((x,condition),dim=1)
        #图片维度为batch_size*3*256*256
        #按第一维度拼接
        return self.disc(x)
class CGAN_Generator(nn.Module):
    def __init__(self,img_channels,condition_channels):
        super(CGAN_Generator,self).__init__()
        self.gen=nn.Sequential(
            nn.Conv2d(img_channels+condition_channels,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #256*64*64
            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
         #   nn.Conv2d(512,1024,kernel_size=4,stride=2,padding=1),
          #  nn.BatchNorm2d(1024),
           # nn.ReLU(inplace=True),
            #nn.ConvTranspose2d(1024,512,kernel_size=3,padding=1),
            # nn.BatchNorm2d(512),
           # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,3,kernel_size=3,padding=1),
            #3*256*256
            nn.Tanh()
        )

    def forward(self,x,condition):
        x=torch.cat((x,condition),dim=1)
        return self.gen(x)
