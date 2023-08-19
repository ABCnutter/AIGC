# 导入软件包

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
import time
from PIL import Image
import torch
import torch.utils.data as data
import torch.nn as nn

from torchvision import transforms
from model.DCGAN import Generator, Discriminator
from matplotlib import pyplot as plt


def make_datapath_list(root):
    """创建用于学习和验证的图像数据及标注数据的文件路径列表。 """

    train_img_list = list() #保存图像文件的路径

    for img_idx in range(200):
        img_path = f"{root}/img_7_{str(img_idx)}.jpg"
        train_img_list.append(img_path)

        img_path = f"{root}/img_8_{str(img_idx)}.jpg"
        train_img_list.append(img_path)

    return train_img_list


class ImageTransform:
    """图像的预处理类"""

    def __init__(self, mean, std):
        self.data_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    def __call__(self, img):
        return self.data_transform(img)


class GAN_Img_Dataset(data.Dataset):
    """图像的 Dataset 类，继承自 PyTorchd 的 Dataset 类"""

    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        '''返回图像的张数'''
        return len(self.file_list)

    def __getitem__(self, index):
        '''获取经过预处理后的图像的张量格式的数据'''

        img_path = self.file_list[index]
        img = Image.open(img_path)  # [ 高度 ][ 宽度 ] 黑白

        # 图像的预处理
        img_transformed = self.transform(img)

        return img_transformed


# 创建DataLoader并确认执行结果

# 创建文件列表
root = "./img_78"
train_img_list = make_datapath_list(root)

# 创建Dataset
mean = (0.5)
std = (0.5)
train_dataset = GAN_Img_Dataset(
    file_list=train_img_list, transform=ImageTransform(mean, std)
)

# 创建DataLoader
batch_size = 2
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

# 确认执行结果
batch_iterator = iter(train_dataloader)  # 转换为迭代器
imges = next(batch_iterator)  # 取出位于第一位的元素
print(imges.size())  # torch.Size([64, 1, 64, 64])


# 网络的初始化处理
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Conv2d和ConvTranspose2d的初始化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm2d的初始化
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


G = Generator(z_dim=20, image_size=64)
D = Discriminator(z_dim=20, image_size=64)
# 开始初始化
G.apply(weights_init)
D.apply(weights_init)

print("网络已经成功地完成了初始化")

# 定义误差函数
criterion = nn.BCEWithLogitsLoss(reduction='mean')


def train_model(G, D, dataloader, num_epochs):
    # 确认是否能够使用GPU加速
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    # 设置最优化算法
    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9
    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])

    # 定义误差函数
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # 使用硬编码的参数
    z_dim = 20
    mini_batch_size = 8

    # 将网络载入GPU中
    G.to(device)
    D.to(device)

    G.train()  # 将模式设置为训练模式
    D.train()  # 将模式设置为训练模式

    # 如果网络相对固定，则开启加速
    torch.backends.cudnn.benchmark = True

    # 图像张数
    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    # 设置迭代计数器
    iteration = 1
    logs = []

    # epoch循环
    for epoch in range(num_epochs):
        # 保存开始时间
        t_epoch_start = time.time()
        epoch_g_loss = 0.0  # epoch的损失总和
        epoch_d_loss = 0.0  # epoch的损失总和

        print('-------------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-------------')
        print('（train）')

        # 以minibatch为单位从数据加载器中读取数据的循环
        for imges in dataloader:
            # --------------------
            # 1.判别器D的学习
            # --------------------
            # 如果小批次的尺寸设置为1，会导致批次归一化处理产生错误，因此需要避免
            if imges.size()[0] == 1:
                continue

            # 如果能使用GPU，则将数据送入GPU中
            imges = imges.to(device)

            # 创建正确答案标签和伪造数据标签
            # 在epoch最后的迭代中，小批次的数量会减少
            mini_batch_size = imges.size()[0]
            label_real = torch.full((mini_batch_size,), 1).to(device)
            label_fake = torch.full((mini_batch_size,), 0).to(device)

            # 对真正的图像进行判定
            d_out_real = D(imges)

            # 生成伪造图像并进行判定
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)

            # 计算误差
            d_loss_real = criterion(d_out_real.view(-1), label_real.to(torch.float))
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake.to(torch.float))
            d_loss = d_loss_real + d_loss_fake

            # 反向传播处理
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            d_loss.backward()
            d_optimizer.step()

            # --------------------
            # 2.生成器G的学习
            # --------------------
            # 生成伪造图像并进行判定
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)

            # 计算误差
            g_loss = criterion(d_out_fake.view(-1), label_real.to(torch.float))

            # 反向传播处理
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # --------------------
            # 3.记录结果
            # --------------------
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            iteration += 1

        # epoch的每个phase的loss和准确率
        t_epoch_finish = time.time()
        print('-------------')
        print(
            'epoch {} || Epoch_D_Loss:{:.4f} ||Epoch_G_Loss:{:.4f}'.format(
                epoch, epoch_d_loss / batch_size, epoch_g_loss / batch_size
            )
        )
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

    return G, D


# 执行学习和验证操作
# 大约需要执行 6 分钟
num_epochs = 200
G_update, D_update = train_model(
    G, D, dataloader=train_dataloader, num_epochs=num_epochs
)


# 将生成的图像和训练数据可视化
# 反复执行本单元中的代码，直到生成感觉良好的图像为止

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 生成用于输入的随机数
batch_size = 8
z_dim = 20
fixed_z = torch.randn(batch_size, z_dim)
fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)

# 生成图像
fake_images = G_update(fixed_z.to(device))

# 训练数据
batch_iterator = iter(train_dataloader)  # 转换成迭代器
imges = next(batch_iterator)  # 取出位于第一位的元素


# 输出结果
fig = plt.figure(figsize=(15, 6))
for i in range(0, 5):
    # 将训练数据放入上层
    plt.subplot(2, 5, i + 1)
    plt.imshow(imges[i][0].cpu().detach().numpy(), 'gray')

    # 将生成数据放入下层
    plt.subplot(2, 5, 5 + i + 1)
    plt.imshow(fake_images[i][0].cpu().detach().numpy(), 'gray')
