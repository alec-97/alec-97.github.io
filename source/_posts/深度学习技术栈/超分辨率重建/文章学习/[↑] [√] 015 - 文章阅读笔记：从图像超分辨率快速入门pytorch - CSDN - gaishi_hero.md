---
title: 015 - 文章阅读笔记：从图像超分辨率快速入门pytorch - CSDN - gaishi_hero
tags:
  - pytorch入门
  - 超分辨率
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301082102026.png
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 653120721
date: 2023-01-08 20:59:28
---

> 链接：
>
> [从图像超分辨率快速入门pytorch - CSDN - gaishi_hero（√）](https://blog.csdn.net/gaishi_hero/article/details/102925507)
>
> 编辑于2019-11-06 16:10:30

## [√] 前言

---

最近又开始把[pytorch](https://so.csdn.net/so/search?q=pytorch&spm=1001.2101.3001.7020)拾起来，学习了github上一些项目之后，发现每个人都会用不同的方式来写深度学习的训练代码，而这些代码对于初学者来说是难以阅读的，因为关键和非关键代码糅杂在一起，让那些需要快速将代码跑起来的初学者摸不着头脑。

所以，本文打算从最基本的出发，只写关键代码，将完成一次深度学习训练需要哪些要素展现给各位初学者，以便你们能够快速上手。等到能够将自己的想法用最简洁的方式写出来并运行起来之后，再对自己的代码进行重构、扩展。我认为这种学习方式是较好的循序渐进的学习方式。

本文选择超分辨率作为入门案例，一是因为通过结合案例能够对训练中涉及到的东西有较好的体会，二是超分辨率是较为简单的任务，我们本次教程的目的是教会大家如何使用pytorch，所以不应该将难度设置在任务本身上。下面开始正文。。。

## [√] 正文

---

#### [√] 单一图像超分辨率（SISR）

---

简单介绍一下图像超分辨率这一任务：超分辨率的任务就是将一张图像的尺寸放大并且要求失真越小越好，举例来说，我们需要将一张256*500的图像放大2倍，那么放大后的图像尺寸就应该是512*1000。用深度学习的方法，我们通常会先将图像缩小成原来的1/2，然后以原始图像作为标签，进行训练。训练的目标是让缩小后的图像放大2倍后与原图越近越好。所以通常会用L1或者L2作为损失函数。

> alec：
>
> - 通常使用L1或者L2作为损失函数

#### [√] 训练4要素

---

一次训练要想完成，需要的要素我总结为4点：

- 网络模型

- 数据
- 损失函数
- 优化器

这4个对象都是一次训练必不可少的，通常情况下，需要我们自定义的是前两个：网络模型和数据，而后面两个较为统一，而且pytorch也提供了非常全面的实现供我们使用，它们分别在torch.nn包和torch.optim包下面，使用的时候可以到pytorch官网进行查看，后面我们用到的时候还会再次说明。

> alec：
>
> - 一般而言，需要自定义的网络模型和数据

#### [√] 网络模型

---

在网络模型和数据两个当中，网络模型是比较简单的，数据加载稍微麻烦些。我们先来看网络模型的定义。

自定义的网络模型都必须继承`torch.nn.Module`这个类。

里面有两个方法需要重写：初始化方法`__init__(self)`和`forward(self, *input)`方法。

在初始化方法中一般要写我们需要哪些层（卷积层、全连接层等），而在`forward`方法中我们需要写这些层的连接方式。

举一个通俗的例子，搭积木需要一个个的积木块，这些积木块放在`__init__`方法中，而规定将这些积木块如何连接起来则是靠`forward`方法中的内容。

> alec：
>
> - 自定义的网络模型都必须继承`torch.nn.Module`这个类
> - module，单元、单位
> - 里面有两个方法需要重写：初始化方法`__init__(self)`和`forward(self, *input)`方法
> - 在初始化方法中一般要写我们需要哪些层（卷积层、全连接层等），而在`forward`方法中我们需要写这些层的连接方式

```python
import torch.nn as nn
import torch.nn.functional as F


class VDSR(nn.Module):

    # 初始化方法中定义需要的模块
    def __init__(self):
        super(VDSR, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv9 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv10 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv11 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv13 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv14 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv15 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv16 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv17 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv18 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv19 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv20 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True)

        # 前向传播方法中定义网络的连接方式
    def forward(self, x):
        ori = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))
        x = F.relu(self.conv19(x))
        x = self.conv20(x)

        # 全局残差连接
        return x + ori


```

上面代码中展示的是我们要用到的模型VDSR，这个模型很简单，就是连续的20层卷积，外加一个跳线连接。结构图如下：

![image-20230108211515354](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301082305306.png)

在写网络模型时，用到的各个层都在`torch.nn`这个包中，在写自定义的网络结构时可以自行到[pytorch官网的文档](https://pytorch.org/docs/stable/nn.html#)中进行查看。

> alec：
>
> - 写网络模型的时候，用到的各个层都在`torch.nn`这个包里面，比如二维卷积层，就在里面。



#### [√] 数据

---

定义了网络模型之后，我们再来看“数据”。“数据”主要涉及到`Dataset`和`DataLoader`两个概念。

> alec：
>
> - “数据”主要涉及到`Dataset`和`DataLoader`两个概念。

Dataset是数据加载的基础，我们一般在加载自己的数据集时都需要自定义一个Dataset，自定义的Dataset都需要继承torch.utils.data.Dataset这个类，当实现了__getitem__()和__len__()这两个方法后，我们就自定义了一个Map-style datasets，Dataset是一个可迭代对象，通过下标访问的方式就能够调用__getitem__()方法来实现数据加载。

> alec：
>
> - Dataset是数据加载的基础，我们一般在加载自己的数据集时都需要自定义一个Dataset
> - 自定义的Dataset都需要继承torch.utils.data.Dataset这个类
> - 当实现了__getitem__()和__len__()这两个方法后，我们就自定义了一个Map-style datasets，Dataset是一个可迭代对象，通过下标访问的方式就能够调用__getitem__()方法来实现数据加载。

这里面最关键的就算是`__getitem__()`如何来写了，我们需要让`__getitem__()`的返回值是一对，包括图像和它的label，这里我们的任务是超分辨率，那么图像和label分别是经过下采样的图像和与其对应的原始图像。所以我们`Dataset`的`__getitem__()`方法返回值就应该是两个`3D Tensor`，分别表示两种图像。

这里需要重点说明一下__getitem__()方法的返回值为什么应该是3D Tensor。根据pytorch官网的说法，二维卷积层只接受4D Tensor，它的每一维表示的内容分别是nSamples x nChannels x Height x Width，我们最后需要用批量的方式将数据送到网络中，所以__getitem__()方法的返回值就应该是后面三维的内容，即便是我们的通道数为1，也必须有这一维的存在，否则就会报错。后面代码中用到的`unsqueeze(0)`方法的作用就是如此。
> alec：
>
> - Dataset类的getitem()方法的返回值是一对，返回数据x和标签y
> - getitem()方法的返回值是两个`3D Tensor`。因为根据pytorch官网的说法，二维卷积层只接受4D Tensor，它的每一维表示的内容分别是`nSamples x nChannels x Height x Width`。
> - 我们最后需要用批量的方式将数据送到网络中，所以__getitem__()方法的返回值就应该是后面三维的内容，即便是我们的通道数为1，也必须有这一维的存在，否则就会报错。
> - 后面代码中用到的`unsqueeze(0)`方法的作用就是如此。unsqueeze()方法的作用是扩展维度/增加维度。、
> - 二维卷积只接受4D的数据，因此Dataset类的getitem方法返回3D的数据，然后再由Dataloader类组装成4D的数据。

前面是说了为什么应该是3D的，为什么应该是Tensor呢？Tensor是跟NumPy中ndarray类似的东西，只是它能够被用于GPU中来加速计算。

> alec：
>
> - Tensor是一种类似于Numpy中的ndarray的数据结构，但是Tensor能够被用到GPU中来加速计算，因此数据应该是Tensor类型的。

下面来看一下我们的代码：

```python
import os
import random

import cv2
import torch
from torch.utils.data import Dataset

patch_size = 64

def getPatch(y):
    h, w = y.shape
    randh = random.randrange(0, h - patch_size + 1)
    randw = random.randrange(0, w - patch_size + 1)
    lab = y[randh:randh + patch_size, randw:randw + patch_size]
    resized = cv2.resize(lab, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    rresized = cv2.resize(resized, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # 随机返回一个patch大小的输入x（rresized）和标签y（lab）
    return rresized, lab


class MyDateSet(Dataset):
    def __init__(self, imageFolder):
        # 图像所在的文件夹
        self.imageFolder = imageFolder
        self.images = os.listdir(imageFolder)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        # 拿到图片的地址
        name = os.path.join(self.imageFolder, name)
        imread = cv2.imread(name)
        # 转换颜色空间
        # 转换到y空间
        ycrcb = cv2.cvtColor(imread, cv2.COLOR_RGB2YCR_CB)
        # 提取y通道
        y = ycrcb[:, :, 0]
        # 裁剪成小块
        # 对于一个图像，不是返回整个图像大小的数据，而是返回一个patch的大小的数据，比如一个400x400的图像，可以返回128x128的patch
        img, lab = getPatch(y)
        # 转为3D Tensor
        # 图像是二维的，通过工具方法扩展一维，返回一个三维的数据
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(lab).unsqueeze(0)

```

其中MyDateSet的内容也不长，包括了初始化方法、\_\_getitem\_\_()和\_\_len\_\_()两个方法。\_\_getitem\_\_()有一个输入值是下标值，我们根据下标，利用OpenCV，读取了图像，并将其转换颜色空间，超分训练的时候我们只用了其中的y通道。还对图形进行了裁剪，最后返回了两个3D Tensor。

在写自定义数据集的时候，我们最需要关注的点就是`__getitem__()`方法的返回值是不是符合要求，能不能够被送到网络中去。至于中间该怎么操作，其实跟pytorch框架也没什么关系，根据需要来做。

> alec：
>
> - 在写自定义数据集的时候，我们最需要关注的点就是`__getitem__()`方法的返回值是不是符合要求，能不能够被送到网络中去。至于中间该怎么操作，其实跟pytorch框架也没什么关系，根据需要来做。



#### [√] 训练

---

写好了`Dataset`之后，我们就能够通过下标的方式获取图像以及它的label。

> alec：
>
> - 写好了`Dataset`之后，我们就能够通过下标的方式获取图像以及它的label。

但是离开始训练还有两个要素：损失函数和优化器。前面我们也说了，这两部分，pytorch官方提供了大量的实现，多数情况下不需要我们自己来自定义，这里我们直接使用了提供的torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')作为损失函数和torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)作为优化器。

训练示例代码：

```python
import torch
import torch.nn as nn
import torch.optim as optim

import date
import model

date_set = date.MyDateSet("Train/")
# 此处使用的是VDSR
model = model.VDSR()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
mse_loss = nn.MSELoss()
# 将模型的参数交给优化器，用于梯度下降更新参数
adam = optim.Adam(model.parameters())


for epoch in range(100):
    running_loss = 0.0
    for i in range(len(date_set)):
        rresized, y = date_set[i]
        # 梯度清零
        adam.zero_grad()
        # 前向传播
        out = model(rresized.unsqueeze(0).to(device, torch.float))
        # 计算损失
        loss = mse_loss(out, y.unsqueeze(0).to(device, torch.float))
        # 损失反向传播计算梯度
        loss.backward()
        # 梯度下降更新参数
        adam.step()

        running_loss += loss
        if i % 100 == 99:  # print every 100
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0


print('Finished Training')

```

整个训练代码非常简洁，只有短短几行，定义模型、将模型移至GPU、定义损失函数、定义优化器（模型移动至GPU一定要在定义优化器之前，因为移动前后的模型已经不是同一个模型对象）。

> alec：
>
> - 先将模型移动到GPU，然后再把模型参数交给优化器
> - 训练时，先用`zero_grad()`来将上一次的梯度清零

训练时，先用`zero_grad()`来将上一次的梯度清零，然后将数据输入网络，求误差，误差反向传播求每个`requires_grad=True`的Tensor（也就是网络权重）的梯度，根据优化规则对网络权重值进行更新，在一次次的更新迭代中，网络朝着loss降低的方向变化着。

值的注意的是，图像数据也需要移动至GPU，并且需要将其类型转换为与网络模型的权重相同的`torch.float`

> alec：
>
> - 模型需要移到GPU，训练数据也要移到GPU



#### [√] DataLoader

---

到前面为止，其实已经能够实现训练的过程了，但是，通常情况下，我们都需要：

1. 将数据打包成一个批量送入网络
2. 每次随机将数据打乱送入网络
3. 用多线程的方式加载数据（这样能够提升数据加载速度）

> alec：
>
> - 通常做法：
>     - 将数据打包成一个批量送入网络
>     - 每次随机将数据打乱送入网络
>     - 用多线程的方式加载数据（这样能够提升数据加载速度）

这些事情不需要我们自己实现，有`torch.utils.data.DataLoader`来帮我们实现。完整声明如下：

```python
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None)

```

> alec：
>
> - 将数据打乱、成batch的取出、多线程的方式加载数据提高效率，这几个功能，不需要自己实现，工具方法已经实现好了。
> - 只需要将Dataset对象传给DataLoader方法，然后设置batch_size参数、设置是否打乱、设置num_workers加载数据的线程数量。

其中的`sampler`、`batch_sampler`、`collate_fn`都是可以有自定义实现的。我们简单的使用默认的实现来构造`DataLoader`。使用了`DataLoader`之后的训练代码稍微有些不同，其中也添加了保存模型的代码（只保存参数的方式）：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import date
import model

date_set = date.MyDateSet("Train/")
dataloader = DataLoader(date_set, batch_size=128,
                        shuffle=True, drop_last=True)

model = model.VDSR()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
mse_loss = nn.MSELoss()
# 先将模型移动到GPU，然后将模型的参数交给优化器
adam = optim.Adam(model.parameters())

def train():
    for epoch in range(1000):

        running_loss = 0.0

        # 从dataloader中成batch的取数据
        for i, images in enumerate(dataloader):
            # 此处的数据是四维的数据
            rresized, y = images
            # 优化器的梯度清零
            adam.zero_grad()
            # 前向传播
            out = model(rresized.to(device, torch.float))
            # 计算损失
            loss = mse_loss(out, y.to(device, torch.float))
            # 损失反向传播计算梯度
            loss.backward()
            # 优化器梯度下降法更新参数
            adam.step()

            running_loss += loss

        if epoch % 10 == 9:
            # 每10个epoch记录一次模型参数
            PATH = './trainedModel/net_' + str(epoch + 1) + '.pth'
            torch.save(model.state_dict(), PATH)

        print('[%d] loss: %.3f' %
              (epoch + 1, running_loss / 3))

    print('Finished Training')


if __name__ == '__main__':
    train()

```







