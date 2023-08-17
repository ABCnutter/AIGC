# 导入软件包
import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, z_dim=20, image_size=256):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, image_size * 32,
                               kernel_size=4, stride=1),
            nn.BatchNorm2d(image_size * 32),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(image_size * 32, image_size * 16,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size * 16),
            nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(image_size * 16, image_size * 8,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(image_size * 8, image_size *4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(inplace=True))
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(image_size * 4, image_size * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(inplace=True))
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(image_size * 2, image_size,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True))
        self.last = nn.Sequential(
            nn.ConvTranspose2d(image_size, 3, kernel_size=4,
                               stride=2, padding=1),
            nn.Tanh())
        # 注意:因为是黑白图像，所以只有一个输出通道

    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.last(out)

        return out
    


class Discriminator(nn.Module):

    def __init__(self, z_dim=20, image_size=256):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, image_size, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))
       #注意:因为是黑白图像，所以输入通道只有一个

        self.layer2 = nn.Sequential(
            nn.Conv2d(image_size, image_size*2, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer3 = nn.Sequential(
            nn.Conv2d(image_size*2, image_size*4, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(image_size*4, image_size*8, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(image_size*8, image_size*16, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(image_size*16, image_size*32, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))
        
        self.last = nn.Conv2d(image_size*32, 1, kernel_size=4, stride=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.last(out)

        return out

    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    G = Generator(z_dim=20, image_size=256)

    # 输入的随机数
    input_z = torch.randn(1, 20)

    # 将张量尺寸变形为(1,20,1,1)
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)

    #输出假图像
    fake_images = G(input_z)
    print(fake_images.shape)
    img_transformed = fake_images[0].detach().numpy().transpose(1, 2, 0)
    plt.imshow(img_transformed)
    plt.show()


    #确认程序执行
    D = Discriminator(z_dim=20, image_size=64)

    #生成伪造图像
    input_z = torch.randn(1, 20)
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
    fake_images = G(input_z)

    #将伪造的图像输入判别器D中
    d_out = D(fake_images)

    #将输出值d_out乘以Sigmoid函数，将其转换成0～1的值
    print(torch.sigmoid(d_out))