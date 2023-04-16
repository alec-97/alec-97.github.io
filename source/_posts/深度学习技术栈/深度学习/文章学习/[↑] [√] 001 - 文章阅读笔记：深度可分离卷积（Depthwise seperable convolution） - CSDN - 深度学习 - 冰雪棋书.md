---
title: 001 - 文章阅读笔记：深度可分离卷积（Depthwise seperable convolution） - CSDN - 深度学习 - 冰雪棋书
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301201905563.png
tags:
  - 轻量化网络
categories:
  - 深度学习技术栈
  - 深度学习
  - 文章学习
abbrlink: 125018155
date: 2023-01-20 17:20:37
---

> 原文链接：
>
> [深度可分离卷积（Depthwise seperable convolution） - CSDN - 深度学习 - 冰雪棋书](https://blog.csdn.net/zml194849/article/details/117021815)
>
> 于 2022-08-25 15:53:15 发布

> alec：
>
> - 深度可分离卷积的目的是为了轻量化网络
> - 组卷积的目的也是为了轻量化网络

## [√] 一、深度可分离卷积（Depthwise separable convolution）

---

> alec：
>
> - separable，adj，可分离的、可分的
> - Depthwise，逐深度的
> - pointwise，逐点的

一些轻量级的网络，如mobilenet中，会有深度可分离卷积depthwise separable convolution，由depthwise(DW)和pointwise(PW)两个部分结合起来，用来提取特征feature map。相比常规的卷积操作，其参数数量和运算成本比较低。

> alec：
>
> - 深度可分离卷积能够在参数量和运算成本两个方面低于常规的卷积操作

## [√] 二、常规卷积操作

---

​    对于5x5x3的输入，如果想要得到3x3x4的feature map，那么卷积核的[shape](https://so.csdn.net/so/search?q=shape&spm=1001.2101.3001.7020)为3x3x3x4；如果padding=1，那么输出的feature map为5x5x4。

![很清晰的常规卷积解释图](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301252229400.png)

卷积层共4个Filter，每个Filter包含了3个Kernel，每个Kernel的大小为3×3。因此卷积层的参数数量可以用如下公式来计算(即：卷积核W x 卷积核H x 输入通道数 x 输出通道数)：

> alec：
>
> - 卷积层的参数量 = 卷积核W x 卷积核H x 输入通道数 Cin x 输出通道数 Cout

N_std = 4 × 3 × 3 × 3 = 108

计算量(即：**卷积核W x 卷积核H x (图片W-卷积核W+1) x (图片H-卷积核H+1) x 输入通道数 x 输出通道数**，以padding= 0，不填充进行演示，输出为3*3*4，如果填充**卷积核W x 卷积核H x (图片W-卷积核W+2P+1) x (图片H-卷积核H+2P+1) x 输入通道数 x 输出通道数**)：

C_std =3*3*(5-2)*(5-2)*3*4=972



## [√] 三、深度可分离卷积

---

深度可分离卷积主要分为两个过程，分别为逐通道卷积（Depthwise Convolution）和逐点卷积（Pointwise Convolution）。

> alec：
>
> - 深度可分离卷积的两个过程：逐通道卷积、逐点卷积

#### [√] 逐通道卷积（Depthwise Convolution）

---

Depthwise Convolution的一个卷积核负责一个通道，一个通道只被一个卷积核卷积，这个过程产生的feature map通道数和输入的通道数完全一样。

一张5×5像素、三通道彩色输入图片（shape为5×5×3），Depthwise Convolution首先经过第一次卷积运算，DW完全是在二维平面内进行。卷积核的数量与上一层的通道数相同（通道和卷积核一一对应）。所以一个三通道的图像经过运算后生成了3个Feature map(如果有same padding则尺寸与输入层相同为5×5)，如下图所示。（卷积核的shape即为：卷积核W x 卷积核H x 输入通道数）

![0](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301252229401.png)

其中一个Filter只包含一个大小为3×3的Kernel，卷积部分的参数个数计算如下（即为：**卷积核Wx卷积核Hx输入通道数**）：

N_depthwise = 3 × 3 × 3 = 27

计算量为（即：**卷积核W x 卷积核H x (图片W-卷积核W+1) x (图片H-卷积核H+1) x 输入通道数**）

C_depthwise=3x3x(5-2)x(5-2)x3=243

Depthwise Convolution完成后的Feature map数量与输入层的通道数相同，无法扩展Feature map。而且这种运算对输入层的每个通道独立进行卷积运算，没有有效的利用不同通道在相同空间位置上的feature信息。因此需要Pointwise Convolution来将这些Feature map进行组合生成新的Feature map。

#### [√] 逐点卷积（Pointwise Convolution）

---

Pointwise Convolution的运算与常规卷积运算非常相似，它的卷积核的尺寸为 1×1×M，M为上一层的通道数。所以这里的卷积运算会将上一步的map在深度方向上进行加权组合，生成新的Feature map。有几个卷积核就有几个输出Feature map。（卷积核的shape即为：1 x 1 x 输入通道数 x 输出通道数）

![0](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301252229402.png)

> alec：
>
> - 所谓的逐点卷积，就是1x1卷积

由于采用的是1×1卷积的方式，此步中卷积涉及到的参数个数可以计算为(即为：**1 x 1 x 输入通道数 x 输出通道数**）：

N_pointwise = 1 × 1 × 3 × 4 = 12

计算量(即为：**1 x 1 x 特征层W x 特征层H x 输入通道数 x 输出通道数**）：

C_pointwise = 1 × 1 × 3 × 3 × 3 × 4 = 108

经过Pointwise Convolution之后，同样输出了4张Feature map，与常规卷积的输出维度相同。

> alec：
>
> - 深度可分离卷积和普通卷积的区别是：
>     - 普通卷积输出的每个特征图，是对一层的所有的（C个）特征图使用C个卷积核，卷积之后，将C个结果在通道维度上相加得到一张特征图，C个卷积核的参数是不一致的，且相加的时候，是直接相加，这样得到一个特征图；C个卷积核是1组，如果输出有y个特征图，那么就使用y组C个卷积。那么卷积核一共有C x Y组。
>     - 深度可分离卷积，是对C个输入特征图，也是使用C个二维卷积核，然后得到C个输出特征图，但是这个时候不直接执行通道方向上的相加操作（这是depth-wise conv）；如何得到y个输出特征呢？输出的一个特征图，是对刚刚的C个特征图使用1x1xC维度的卷积核合并得到，然后使用y组形状为1x1xC的卷积核，那么就得到了y个输出特征图。大大节省了参数量。
> - 举个例子，输入特征图为3通道，输出为5通道：
>     - 普通卷积：使用5组卷积核，每组卷积核的形状为w x h x 3（其中w和h为卷积核的长和宽）。总的维度为：w x h x 3 x 5
>     - 深度可分离卷积：使用1组w x h x 3的卷积核，得到3个中间状态特征图，然后使用5组1 x 1 x 3的卷积得到5个特征图。总的维度为：（w x h x 3 x 1） + （1 x 1 x 3 x 5）

## [√] 四、参数对比

---

回顾一下，常规卷积的参数个数为：

N_std = 4 × 3 × 3 × 3 = 108

Separable Convolution的参数由两部分相加得到：

N_depthwise = 3 × 3 × 3 = 27（因为C个通道只使用一个卷积核，所以此处的4没有了）

N_pointwise = 1 × 1 × 3 × 4 = 12

N_separable = N_depthwise + N_pointwise = 39

相同的输入，同样是得到4张Feature map，Separable Convolution的参数个数是常规卷积的约1/3。因此，在参数量相同的前提下，采用Separable Convolution的神经网络层数可以做的更深。

> alec：
>
> - 在相同的参数量的前提下，使用深度可分离卷积的神经网络可以做的更深。
> - 组卷积则是将C个输入通道的卷积分成几组，分别卷积，然后再concat。



## [√] 五、计算量对比

---

回顾一下，常规卷积的计算量为：

C_std =3*3*(5-2)*(5-2)*3*4=972

Separable Convolution的计算量由两部分相加得到：

C_depthwise=3x3x(5-2)x(5-2)x3=243

C_pointwise = 1 × 1 × 3 × 3 × 3 × 4 = 108

C_separable = C_depthwise + C_pointwise = 351

相同的输入，同样是得到4张Feature map，Separable Convolution的计算量是常规卷积的约1/3。因此，在计算量相同的情况下，Depthwise Separable Convolution可以将神经网络层数可以做的更深。
------------------------------------------------
版权声明：本文为CSDN博主「冰雪棋书」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/zml194849/article/details/117021815