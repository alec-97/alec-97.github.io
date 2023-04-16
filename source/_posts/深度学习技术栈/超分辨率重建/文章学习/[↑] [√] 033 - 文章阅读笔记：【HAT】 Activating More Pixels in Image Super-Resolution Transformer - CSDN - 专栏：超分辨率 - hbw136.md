---
title: >-
  033 - 文章阅读笔记：【HAT】 Activating More Pixels in Image Super-Resolution
  Transformer - CSDN - 专栏：超分辨率 - hbw136
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301161256731.png
tags:
  - transformer
  - 深度学习
  - 计算机视觉
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 2859212987
date: 2023-01-16 12:31:44
---

> 转载自：
>
> 【√】[【HAT】 Activating More Pixels in Image Super-Resolution Transformer - CSDN - 专栏：超分辨率 - hbw136](https://blog.csdn.net/hbw136/article/details/124692907)
>
> 于 2022-05-10 17:37:15 修改

# Activating More Pixels in Image Super-Resolution Transformer

# （在图像超分辨率transformer中激活更多的像素）

---

## [√] 文章简介

---

![HAT](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301161519682.png)

作者：Xiangyu Chen1,2 , Xintao Wang3 , Jiantao Zhou1 , and Chao Dong2,4

单位：1University of Macau 2Shenzhen Institute of Advanced Technology,

Chinese Academy of Sciences 3ARC [Lab](https://so.csdn.net/so/search?q=Lab&spm=1001.2101.3001.7020), Tencent PCG 4Shanghai AI Laboratory

代码：[GitHub - chxy95/HAT: Activating More Pixels in Image Super-Resolution Transformer](https://github.com/chxy95/hat)

论文地址：https://arxiv.org/pdf/2205.04437



## [√] 一、问题与动机

---

尽管现阶段作者发现一些基于transformer的SR模型获得了更高的指标性能，但由于使用信息的范围有限，在某些情况下它产生的结果不如 RCAN。 这些现象说Transformer对局部信息的建模能力更强，但其利用信息的范围有待扩大。

> alec：
>
> - transformer在SR上获得了高的性能，但是因为利用的信息少，信息没有被充分利用，所以有的时候还不如RCAN这个残差通道注意力网络性能好。因此需要扩大transformer超分模型对于信息的利用范围。



## [√] 二、思路和亮点

---

为了解决上述问题并进一步发挥 Transformer for SR 的潜力，作者提出了一种 Hybrid Attention Transformer（混合注意力），即 HAT。

作者的HAT结合了channel attention和self-attention两种方案，以利用前者对全局信息的利用能力和后者强大的代表能力。 

此外，为了更好地聚合跨窗口信息，作者还引入了一个重叠的交叉注意模块。

 作者还另外探索了预训练对 SR 任务的影响，并提供了相同任务的预训练策略。 实验结果表明，该策略可以比多相关任务预训练执行得更好。



> alec：
>
> - 通道注意力CA有强大的全局信息利用能力
> - 自注意力SA有强大的代表能力
> - 通过重叠的交叉注意模块，能够更好的聚合跨窗口信息
> - 相同任务上的预训练策略，能够让模型发挥的更好

## [√] 三、具体内容

---

#### [√] 1、模型结构

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301161519683.png)

> alec：
>
> - 模型主要的模块是RHAG模块，即残差混合注意力模块，其中有残差、通道注意力、自注意力、重叠的交叉注意力这四个主要成分。然后模型的主干部分是多个RHAG串联组成的。
> - 然后RHAG模块中，包含HAB和OCAB，其中HAB是混合注意力模块，是由STL模块改造的，STL = swin transformer layer。stl = 层规范化 + 多头自注意力模块MSA + 跳跃连接 + 层规范化 + MLP + 跳跃连接。HAB将STL中的MSA换成了通道注意力模块和多头自注意力模块的并联。

#### [√] 2、相关模块

---

###### [√] 残差混合注意力组（RHAG）

---

![RHAG](https://img-blog.csdnimg.cn/eaf372b289ca4023a403837d2c71d69c.png)

 如上图所示，每个 RHAG 包含 M 个混合注意力块 (HAB)、一个重叠的交叉注意力块 (OCAB) 和一个 3×3 卷积层。 具体来说，对于第 i 个 RHAG，可以表示为

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301161519684.png)

其中 Fi−1,0 表示第 i 个 RHAG 的输入特征，Fi−1,j 表示第 i 个 RHAG 中第 j 个 HAB 的第 j 个输出特征。 在一系列 HAB 的映射之后，作者插入一个 OCAB 来扩大基于窗口的 self-attention 的感受野并更好地聚合跨窗口信息。 在 RHAG 结束时，作者保留之后的卷积层。 还添加了残差连接以稳定训练过程。

> alec：
>
> - OCAB模块用来扩大基于窗口的自注意力的感受野，并更好的聚合夸窗口的信息。

###### [√] 混合注意力模块（HAB）

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301161519685.png)

如图上图所示，当采用通道注意力时，会激活更多像素，因为计算通道注意力权重涉及全局信息。

此外，许多工作表明卷积可以帮助 Transformer 获得更好的视觉表示或实现更容易的优化。

> alec：
>
> - 通道注意力涉及全局信息，能够激活更多的像素。
> - W-MSA = 基于窗口的多头自注意力模块。W = window M = multi S = self A = attention

因此，作者将基于通道注意力的卷积块合并到标准的 Transformer 块中，以进一步增强网络的表示能力。如图 3 所示，通道注意块 (CAB) 与基于窗口的多头自注意 (W-MSA) 模块并行插入到标准 Swin Transformer 块中的第一个 LayerNorm (LN) 层之后。

请注意，在连续 HAB 中，每隔一段时间就会采用基于移位窗口的自我注意 (SW-MSA)。为了避免 CAB 和 MSA 在优化和视觉表示上可能发生的冲突，将一个小的常数 α 乘以 CAB 的输出。对于给定的输入特征 X，HAB 的整个过程计算为

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301161519687.png)

其中 XN 和 XM 表示中间特征。 Y 代表 HAB 的输出。 特别是，作者将每个像素视为嵌入的标记（即，在之后将patch大小设置为 1 以进行patch嵌入）。 MLP 表示多层感知器。

###### [√] 通道注意力块（CAB）

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301161519688.png)

由两个标准卷积层组成，它们之间有一个 GELU 激活函数 和一个通道注意 (CA) 模块，如上图所示。

由于基于 Transformer 的结构通常需要大量通道来进行token嵌入 ，直接使用具有恒定宽度的卷积会产生很大的计算成本。 因此，作者通过两个卷积层之间的常数 β 来压缩通道数。 对于具有 C 个通道的输入特征，第一个卷积层后输出特征的通道数被压缩到 Cβ，然后通过第二层将特征扩展到 C 个通道。 

> alec：
>
> - 标准的CA模块能够自适应地调整通道特征

接下来，利用标准 CA 模块来自适应地重新调整通道特征。 整个过程被表述为

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301161519689.png)

其中Xin、Xout、Conv1、Conv2分别表示输入特征、输出特征、第一卷积层和第二卷积层。

###### [√] 重叠交叉注意块（OCAB）

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301161519690.png)

作者引入OCAB直接建立跨窗口连接，增强窗口self-attention的代表能力。 作者的 OCAB 由一个重叠交叉注意力 (OCA) 层和一个类似于标准 Swin Transformer 块的 MLP 层组成。

但是对于 OCA，如图 4 所示，作者使用不同的窗口大小来划分投影特征。 具体来说，对于输入特征 X 的 XQ, XK, XV ∈ RH×W×C，XQ 被划分为大小为 M×M 的 HW M2 个非重叠窗口，而 XK, XV 被展开为大小为 HW M2 个重叠窗口 Mo × Mo。计算公式为

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301161519691.png)

​    其中γ是控制重叠大小的常数。为了更好地理解这个操作，可以将标准窗口分区看作是一个内核大小和步幅都等于窗口大小 M 的滑动分区。相反，重叠窗口分区可以看作是一个内核大小的滑动分区等于 Mo，而步幅等于 M。使用大小为 γM 的零填充来确保重叠窗口的大小一致性。注意矩阵计算为 Equ。如图7所示，也采用了相对位置偏差B∈RM×Mo。与 WSA 的查询、键和值是从相同的窗口特征计算的不同，OCA 从更大的字段计算键/值，在该字段中可以将更多有用的信息用于查询。

###### [√] Pre-training on ImageNet

---

相比之下，作者基于相同的任务直接在更大规模的数据集（即 ImageNet）上执行预训练。 例如，当作者要训练一个 ×4 SR 的模型时，作者首先在 ImageNet 上训练一个 ×4 SR 模型，然后在特定数据集上对其进行微调，例如 DF2K。

值得一提的是，足够的预训练训练迭代次数和合适的微调学习率对于预训练策略的有效性非常重要。 作者认为这是因为 Transformer 需要更多的数据和迭代来学习任务的一般知识，但需要很小的学习率来进行微调以避免过度拟合到特定的数据集。

> alec：
>
> - 在大的数据集上预训练，在小的数据集上微调



## [√] 四、实验

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301161519692.png)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301161519693.png)

#### [√] 消融实验

---

###### [√] 1）window size的影响

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301161519694.png)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301161519695.png)

> alec：
>
> - 更大的窗口尺寸能够感受到更大的信息，从而指标也越好。

###### [√] 2）OCAB的影响

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301161519696.png)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301161519697.png)

> alec：
>
> - 定量和定性实验结果可以看出，带有交叉重叠注意力模块和通道注意力模块的模型效果最好。



###### [√] 3）overlapping ratio的影响（in ocab）

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301161519698.png)







###### [√] 4）CA的影响（以及权重参数的研究）

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301161519699.png)



> alec：
>
> - 权重参数指的是：因为担心通道注意力模块和带有窗口的多头自注意力模块会冲突，因此对通道注意力模块添加了权重。

> alec：
>
> - 结果表明，CA模块完全使用和完全不使用，效果不如对CA模块使用但是进行衰减好。其中上述实验中，CA模块的衰减系数为0.01的时候，效果最好。



## [√] 五、总结

---

***\*在本文中，作者提出了一种新颖的混合注意力转换器 HAT，用于图像超分辨率。 作者的模型结合了通道注意力和自注意力来激活更多像素以重建高分辨率结果。 此外，作者提出了一个重叠的交叉注意力模块，它计算不同窗口大小的特征之间的注意力，以更好地聚合交叉窗口信息。 此外，作者引入了相同任务的预训练策略，以进一步激活所提出模型的潜力。\****











