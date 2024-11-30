import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)
#参数设置
device='cuda'if torch.cuda.is_available() else 'cpu'
input_dim=64
output_dim=28*28*1
#学习率
lr=0.0002
betas=(0.5,0.999)
epochs=50
batch_size=32
disc=Discriminator(output_dim).to(device)
gen=Generator(input_dim,output_dim).to(device)
#损失函数
criterion=nn.BCELoss()
#优化器
disc_optim=optim.Adam(disc.parameters(),lr=lr,betas=betas)
gen_optim=optim.Adam(gen.parameters(),lr=lr,betas=betas)
#随机生成噪声图像
fix_noise=torch.randn(batch_size, input_dim).to(device)
#对输入数据进行处理
transforms=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
#下载数据和定义数据流
train_data=datasets.MNIST(root='../data',train=True,download=True,transform=transforms)
loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
#训练
writer_fake=SummaryWriter(f"../sun/GAN_MINST/fake")
writer_real=SummaryWriter(f"../sun/GAN_MNIST/real")
step=0
for epoch in range(epochs):
    for batch_idx,(real,_) in enumerate(loader):
        real=real.view(-1,output_dim).to(device)
        #生成噪声图像
        noise=torch.randn(real.shape[0],input_dim).to(device)
        #生成假图像
        fake=gen(noise)
        disc_real=disc(real).view(-1)
        loss_d_real=criterion(disc_real,torch.ones_like(disc_real))
        disc_fake=disc(fake).view(-1)
        loss_d_fake=criterion(disc_fake,torch.zeros_like(disc_fake))
        loss_d=loss_d_real+loss_d_fake
        disc.zero_grad()
        loss_d.backward(retain_graph=True)
        disc_optim.step()
        #生成器训练
        output=disc(fake).view(-1)
        loss_g=criterion(output,torch.ones_like(output))
        gen.zero_grad()
        loss_g.backward()
        gen_optim.step()

        if batch_idx==0:
            print(
                f"Epoch[{epoch}/{epochs}]\
                Batch{batch_idx}/{len(loader)}"\
                f"Loss D:{loss_d:.4f},\
                    Loss G:{loss_g:.4f}"
            )
            with torch.no_grad():
                fake=gen(fix_noise).reshape(-1,1,28,28)
                data=real.reshape(-1,1,28,28)
                img_grid_fake=torchvision.utils.make_grid(fake,normalize=True)
                img_grid_real=torchvision.utils.make_grid(data,normalize=True)
                writer_fake.add_image("Mnist Fake Images",img_grid_fake,global_step=step)
                writer_real.add_image("Mnist Real Images",img_grid_real,global_step=step)
                step+=1