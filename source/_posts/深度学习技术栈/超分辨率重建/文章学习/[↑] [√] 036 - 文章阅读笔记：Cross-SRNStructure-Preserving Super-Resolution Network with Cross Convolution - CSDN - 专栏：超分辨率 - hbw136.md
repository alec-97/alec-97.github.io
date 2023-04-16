---
title: >-
  036 - 文章阅读笔记：Cross-SRN:Structure-Preserving Super-Resolution Network with
  Cross Convolution - CSDN - 专栏：超分辨率 - hbw136
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301162148849.png
tags:
  - 计算机视觉
  - 深度学习
  - CNN
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 4087855987
date: 2023-01-16 21:30:50
---

> 转载自：
>
> 【√】[Cross-SRN:Structure-Preserving Super-Resolution Network with Cross Convolution - CSDN - 专栏：超分辨率 - hbw136](https://blog.csdn.net/hbw136/article/details/123794787)

# [Cross-SRN:Structure-Preserving Super-Resolution Network with Cross Convolution

# Cross-SRN：基于交叉卷积的结构保持超分辨率网络



## [√] 论文信息

---

***\*Yuqing Liu, Qi Jia, Xin Fan, Senior Member, IEEE, Shanshe Wang, Siwei Ma, Member, IEEE,\****

***\*and Wen Gao, Fellow, IEEE\****

论文地址：[[2201.01458\] Cross-SRN: Structure-Preserving Super-Resolution Network with Cross Convolution (arxiv.org)](https://arxiv.org/abs/2201.01458)

项目地址：暂无

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301171851895.png)

Cross-SRN的网络结构。整体网络Cross-SRN如下图所示，包括特征提取、层次特征利用和图像恢复三个步骤。红色的多尺度特征融合组(MFFG)的细节和深蓝色的交叉卷积块(CCB)的细节分别显示在左上角和右上角。

> alec：
>
> - Hierarchical Feature Exploitation
> - 层次特征利用

## [√] 面对的问题

---

将低分辨率(LR)图像恢复为具有正确和清晰细节的超分辨率(SR)图像，具有挑战性。

## [√] 问题引出

---

现有的深度学习工作几乎忽略了图像固有的结构信息，但structure对SR结果的视觉感知（visual perception）质量起着重要的作用。

## [√] 解决思路

---

- 设计了一种层次特征利用网络，以多尺度特征融合的方式探测和保存结构信息。
- 首先，作者在传统的边缘检测器上提出了一种基于交叉卷积的方法来定位和表示边缘特征。
- 然后，设计了具有特征归一化和通道注意性的交叉卷积块(CCBs)，以考虑特征之间的内在相关性。最后，作者利用多尺度特征融合组(MFFG)嵌入交叉卷积块，并在不同尺度上分层发展结构特征的关系，调用一个名为Cross-SRN的轻量级结构保护网络。

> alec：
>
> - 问题：交叉卷积是如何卷积的？

## [√] 贡献

---

1. 受边缘检测方法的启发，作者设计了一种具有有效的结构信息探索效果的交叉卷积方法。
2. 设计了一个交叉卷积块(CCB)来学习边缘特征之间的关系，并将ccb嵌入多尺度特征融合方式(MFFG)来探索不同特征尺度下的层次特征。
3. 提出的Cross-SRN实现了持平或更好的性能与最先进的方法与更准确的边缘恢复。特别是，作者的网络比具有丰富结构信息的选定基准具有显著的优势。

> alec：
>
> - 本文交叉卷积方法，是受边缘检测方法的启发
> - 是如何得到不同尺度的特征的，不同尺度的特征，是指的什么样的尺度

## [√] 具体内容

---

受边缘检测器的启发，本文提出了一种新的交叉卷积来探索特征的结构信息，该信息由两个因素分解的非对称fitter组成。

同时应用两个fitter来增加矩阵的秩和保留更多的结构信息。

在交叉卷积的基础上，设计了具有特征归一化(F-Norm)和CA的交叉卷积块(CCB)，以考虑特征之间的内在相关性，分别是关注空间和通道信息。

ccb以多尺度的方式分组以进行特征探索，称为MFFG。

MFFG考虑分层边缘信息，并逐步探索结构信息，其中利用填充结构来充分探索特征，并考虑残差连接来保持信息。

MFFG模块被级联以构成最终的Cross-SRN。

实验结果表明，Cross-SRN具有更准确的结构信息，具有竞争力或更好的性能。

图1显示了各种图像SR方法的视觉质量比较，其中Cross-SRN恢复了更正确的线和边缘纹理。

为了定量地展示恢复性能，作者从具有丰富结构信息的数值现有纹理HR图像中建立了一个选定的基准，这表明作者的网络可以有效地保存结构纹理。



> alec：
>
> - 本文的交叉卷积是受边缘检测器的启发。



#### [√] 1）网络设计

---

网络设计的概述如上图所示。在Cross-SRN中有三个步骤，分别称为特征提取、层次特征探索和图像恢复，用不同的虚线框分隔。设ILR和IHR分别表示LR和HR实例。

> alec：
>
> - i = instance

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301171851896.png)

其中作者可以用fFE（·）表示特征提取步骤。该卷积扩展了输入实例的通道数，并将该实例映射到一个包含比RGB空间更多的潜在信息的特定空间中。

> alec：
>
> - 本文的点之一：多尺度 + 交叉卷积提取边缘信息 + 通道注意力 + FN采用空间信息

然后，作者试图在不同尺度的特征中保留有价值的信息，同时强调边缘特征。因此，作者设计了G级联MFFGs用于全局残差学习的分层特征探索。可表示为Hg![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301171851897.png)

其中，fgMFFG（·）表示第g个MFFG，Hg−1和Hg分别表示MFFG模块的输入和输出特征。级联MFFGsHG的最终输出被输入到残差模块中，该模块由两个卷积层和LeakyReLU层组成。具有残差学习的填充结构被设计为：![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301171851898.png) 

其中，fPAD（·）表示padding。

最后，通过一次卷积和一次超像素卷积来恢复HR图像，如，

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301171851899.png)

###### [√] 1.交叉卷积

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301171851900.png)



> alec：
>
> - 上图中，左图为普通卷积，由图为交叉卷积。

如上图所示，与普通卷积不同的是，交叉卷积采用了两个非对称垂直fitter，分别表示为k1×m和km×1，感受域分别为1×m和m×1。假设，这是输入和输出特征， 

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301171851901.png)

其中⊗表示卷积，b为偏差项

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301171851902.png)

- 交叉卷积通过平行地利用垂直和水平梯度信息来强调边缘信息。并行设计表明计算复杂度和参数比传统fitter低。同时，并行开发也可以比顺序设计保存更多的信息。
- 上图展示了顺序卷积和交叉卷积之间关于信息保存的差异。输入特征包含不同方向的渐变信息，蓝色箭头表示每个像素的渐变方向。通过两个非对称fitter进行垂直和水平处理后，利用了不同方向的梯度。最后，具有序列卷积的输出特征只关注该特征的主梯度方向，而交叉卷积可以保留更多的梯度方向。

- 所提出的交叉卷积的优点可以通过它所持有的信息量来验证。对于卷积fitterkm×m，秩为秩(km×m)≤m。对于k1×m和km×1的顺序组合，排名为秩(k1×m·km×1)=1。低等级的fitter比高等级的fitter保存的信息更少。交叉卷积交叉的秩是秩(kcross)≤2，它可以比顺序交叉的信息保存更多的潜在信息。![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301171851903.png)

事实上，交叉卷积与传统的边缘检测器具有相似的公式。上图显示了一个由不同filter提取的结构纹理的示例。作者比较了交叉卷积与Sobel算子，它检测从垂直和水平方向的边缘。图5(b)和(f)分别是Sobel和交叉卷积的结果。对比表明，交叉卷积可以用显式和锐边角保留大部分的边缘信息，验证了结构纹理探索的能力。特别地，作者还将提取的边缘图与垂直、水平filter和顺序卷积进行了比较，这表明交叉卷积比其他方法可以提取更多的边缘信息。

###### [√] 2.多尺度特征融合组

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301171851904.png)



 为了获得准确的边缘信息，作者基于基本的交叉卷积构建了交叉卷积块(CCB)。利用与Leaky ReLU激活的交叉卷积来探索结构信息。除了卷积外，F-Norm和CA还被认为分别强调重要的空间信息和通道信息。上图显示了CA的操作，其中全局平均池化用于挤压信息，两个具有ReLU激活的完整连接层探索了每个通道的非线性注意。F-Norm专注于空间信息的多样性，它可以表述为，![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301171851905.png)

其中F(i)in，F(i)out是第i个通道的输入和输出特征，k(i)和b(i)是第i个通道的filter和偏差。

CCB在残余块设计中组织交叉卷积。残块设计在先进网络中被广泛考虑以提高性能。除了残差连接外，还利用信道注意和特征归一化来提高探索性能。

> alec：
>
> - 信道注意 = CA
> - 特征归一化 = F-Norm

> alec：
>
> - fuse = 融合

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301171851906.png)

> alec：
>
> - 多尺度中的尺度指的是什么？

由于边缘信息对尺度变化很敏感，因此在MFFG中以多尺度特征融合的方式进行分组，以探索不同尺度的特征。如上图所示，对于第g个MFFG，输入特征Hg−1平均分为几组，通道数相同。如图7所示，设fjCCB（·）为MFFG中的第j个CCB，输入特征Hg−1从HC00到HC03分为四组。多尺度的特征融合可以表现为，

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301171851907.png)

其中，HCjk表示第j个CCB之后的第k组，[·]表示组组合。

层次特征由残差块结构聚合，该结构由两个卷积层、Leaky ReLU和一个CA层组成。最后，MFFG的输出为，![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301171851908.png)

其中fFuse（·）为融合结构。

MFFG保留了原始信息，并分层地强调了结构性信息。输入特征Hg−1被分为四组。ccb依次对三组进行分层结构信息开发。为了保留在边缘探索过程中丢失的潜在信息，最后一组保持了相同的原始特征。在ccb之后，利用具有通道注意的普通残差块结构进行有效的特征探索和梯度传输。

## [√] 结果

---

#### [√] 1）消融实验

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301171851909.png)

三种不同的PSNR/SSIM卷积设计与BI×4降采方法的比较。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301171851910.png)

通道注意(CA)和特征归一化（F-Norm）的研究。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301171851911.png)

在BI×4降采的条件下，PSNR/SSIM上的[多尺度特征融合](https://so.csdn.net/so/search?q=多尺度特征融合&spm=1001.2101.3001.7020)的研究。

#### [√] 2）结果对比

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301171851912.png)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301171851913.png)

## [√] 启发

---

所使用的cross_conv能够较为准确的保留边缘信息对于提升帮助较大，MFFG的设计也很新颖，有利于于增强边缘结构信息，但可能会减小psnr。







