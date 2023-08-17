#导入软件包
import torch
import torch.nn as nn

class Self_Attention(nn.Module):
    """ Self-AttentionのLayer"""

    def __init__(self, in_dim):
        super(Self_Attention, self).__init__()

        #准备1×1的卷积层的逐点卷积
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        #创建Attention Map时归一化用的SoftMax函数
        self.softmax = nn.Softmax(dim=-2)

        #原有输入数据x与作为Self−Attention Map的o进行加法运算时使用的系数
        # output = x +gamma*o
        #刚开始gamma=0，之后让其进行学习
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        #输入变量
        X = x

       #先计算卷积，再对尺寸进行变形，将形状由B、C'、W、H变为B、C'、N
        proj_query = self.query_conv(X).view(
            X.shape[0], -1, X.shape[2]*X.shape[3]) #尺寸 ：B、C'、N
        proj_query = proj_query.permute(0, 2, 1)  #转置操作
        proj_key = self.key_conv(X).view(
            X.shape[0], -1, X.shape[2]*X.shape[3])  #尺寸 ：B、C'、N

        #乘法运算
        S = torch.bmm(proj_query, proj_key)  #bmm是以批次为单位进行的矩阵乘法运算

        #归一化
        attention_map_T = self.softmax(S)  #将行i方向上的和转换为1的SoftMax函数
        attention_map = attention_map_T.permute(0, 2, 1)  #进行转置

        #计算Self-Attention Map
        proj_value = self.value_conv(X).view(
            X.shape[0], -1, X.shape[2]*X.shape[3])  #尺寸 ：B、C、N
        o = torch.bmm(proj_value, attention_map.permute(
            0, 2, 1))  #对Attention Map进行转置并计算乘积

        #将作为Self−Attention Map的o的张量尺寸与x对齐，并输出结果
        o = o.view(X.shape[0], X.shape[1], X.shape[2], X.shape[3])
        out = x+self.gamma*o

        return out, attention_map


class Generator(nn.Module):

    def __init__(self, z_dim=20, image_size=64):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            #添加频谱归一化处理
            nn.utils.spectral_norm(nn.ConvTranspose2d(z_dim, image_size * 8,
                                                      kernel_size=4, stride=1)),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
           #添加频谱归一化处理
            nn.utils.spectral_norm(nn.ConvTranspose2d(image_size * 8, image_size * 4,
                                                      kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(
            #添加频谱归一化处理
            nn.utils.spectral_norm(nn.ConvTranspose2d(image_size * 4, image_size * 2,
                                                      kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(inplace=True))

        #添加Self−Attentin网络层
        self.self_attntion1 = Self_Attention(in_dim=image_size * 2)

        self.layer4 = nn.Sequential(
            #添加频谱归一化处理
            nn.utils.spectral_norm(nn.ConvTranspose2d(image_size * 2, image_size,
                                                      kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True))

        #添加Self−Attentin网络层
        self.self_attntion2 = Self_Attention(in_dim=image_size)

        self.last = nn.Sequential(
            nn.ConvTranspose2d(image_size, 1, kernel_size=4,
                               stride=2, padding=1),
            nn.Tanh())
        #注意 ：由于是黑白图像，因此输出的通道数为1

        self.self_attntion2 = Self_Attention(in_dim=64)

    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out, attention_map1 = self.self_attntion1(out)
        out = self.layer4(out)
        out, attention_map2 = self.self_attntion2(out)
        out = self.last(out)

        return out, attention_map1, attention_map2


class Discriminator(nn.Module):

    def __init__(self, z_dim=20, image_size=64):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            #追加Spectral Normalization
            nn.utils.spectral_norm(nn.Conv2d(1, image_size, kernel_size=4,
                                             stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True))
       #注意 ：由于是黑白图像，因此输入的通道数为1

        self.layer2 = nn.Sequential(
           #追加Spectral Normalization
            nn.utils.spectral_norm(nn.Conv2d(image_size, image_size*2, kernel_size=4,
                                             stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer3 = nn.Sequential(
            #追加频谱归一化
            nn.utils.spectral_norm(nn.Conv2d(image_size*2, image_size*4, kernel_size=4,
                                             stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True))

        #追加Self-Attentin层
        self.self_attntion1 = Self_Attention(in_dim=image_size*4)

        self.layer4 = nn.Sequential(
            #追加频谱归一化
            nn.utils.spectral_norm(nn.Conv2d(image_size*4, image_size*8, kernel_size=4,
                                             stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True))

        #追加Self-Attentin层
        self.self_attntion2 = Self_Attention(in_dim=image_size*8)

        self.last = nn.Conv2d(image_size*8, 1, kernel_size=4, stride=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out, attention_map1 = self.self_attntion1(out)
        out = self.layer4(out)
        out, attention_map2 = self.self_attntion2(out)
        out = self.last(out)

        return out, attention_map1, attention_map2


if __name__ == "__main__":
    #动作确认
    import matplotlib.pyplot as plt

    G = Generator(z_dim=20, image_size=64)

    # 输入的随机数
    input_z = torch.randn(1, 20)

    # 将张量尺寸变形为(1,20,1,1)
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)

    # 输出伪造图像
    fake_images, attention_map1, attention_map2 = G(input_z)

    img_transformed = fake_images[0][0].detach().numpy()
    plt.imshow(img_transformed, 'gray')
    plt.show()

    # 确认程序执行
    D = Discriminator(z_dim=20, image_size=64)

    # 生成伪造图像
    input_z = torch.randn(1, 20)
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
    fake_images, _, _ = G(input_z)

    # 将伪造的图像输入判别器D中
    d_out, attention_map1, attention_map2 = D(fake_images)

    # 将输出值d_out乘以Sigmoid函数，将其转换成0～1的值
    print(nn.Sigmoid()(d_out))
