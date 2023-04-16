---
title: >-
  051 - 文章阅读笔记：SANet|融合空域与通道注意力，南京大学提出置换注意力机制 - AIWalker（SA-NET:SHUFFLE
  ATTENTION FOR DEEP CONVOLUTIONAL NEURAL NETWORKS） - HappyAIWalker
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301271648181.jpg
tags:
  - 深度学习
  - 注意力机制
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 1235864834
date: 2023-01-27 13:59:52
---

> 原文链接：
>
> [SANet|融合空域与通道注意力，南京大学提出置换注意力机制 - AIWalker - HappyAIWalker](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651683459&idx=1&sn=dbea516a199697990ef3d05989f474b1&scene=21#wechat_redirect)
>
>  2021-02-04 22:00
>
> ---
>
> ==论文简介==
>
> 提出了一种新的注意力机制：置换注意力机制。它在空域注意力与通道注意力的基础上，引入了特征分组与通道置换，得到了一种超轻量型的注意力机制。



> alec：
>
> - shuffle，混合、挪动、洗牌、变换位置



## [√] 论文信息

---

英文标题：SA-NET:SHUFFLE ATTENTION FOR DEEP CONVOLUTIONAL NEURAL NETWORKS

中文标题：SA-Net：用于深度CNN的混合注意力机制

论文链接：https://arxiv.org/abs/2102.00240

论文代码：https://github.com/wofmanaf/SA-Net

论文刊物：ICASSCP 2021

![标题与作者团队](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301271648179.jpg)

> 本文是南京大学的杨育彬等人提出了一种新的注意力机制：置换注意力机制。它在空域注意力与通道注意力的基础上，引入了特征分组与通道置换，得到了一种超轻量型的注意力机制。所提方案在ImageNet与MS-COCO数据集上均取得了优于SE、SGE等性能，同时具有更低的计算复杂度和参数量。



## [√] 摘要

---

- 注意力机制已成为提升CNN性能的一个重要模块
- 一般来说，常用注意力机制有两种类型：spatial attention与channel attention，它们分别从pixel与channel层面进行注意力机制探索。
- 尽管两者组合(比如BAM、CBAM)可以获得更好的性能，然而它不可避免的会导致计算量的提升。



- 本文提出了一种高效置换注意力(Shuffle Attention，SA)模块以解决上述问题，它采用置换单元高效组合上述两种类型的注意力机制。

- 具体的说，SA首先将输入沿着通道维度拆分为多组，然后对每一组特征词用置换单元刻画特征在空域与通道维度上的依赖性，最后所有特征进行集成并通过通道置换操作进行组件特征通信。

- 所提SA模块计算高效且有效，以ResNet50为蓝本，其参数增加为300(基准为25.56M)，计算量增加为2.76e-3GFLOPs(基准为4.12GFLOPs)，而top1精度提升则高达1.34%。



- 最后作者在公开数据集(包含ImageNet、MS-COCO)上对所提注意力机制进行了验证，所提方案取得了优于SOTA的效果且具有更低的计算复杂度。下图给出了不同注意力机制的性能对比。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301271648180.jpg)

本文的主要贡献有如下两点：

- 引入一种轻量型且有效的注意力模块SA用于提升CNN的性能；
- 在公开数据集(ImageNet, MS-COCO)上验证所提注意力机制的优异性能，更高的性能、更低的计算复杂度。



## [√] 方法

---

接下来，我们将从SA的构成模块出发对其进行介绍；然后介绍如何将其嵌入到现有CNN中；最后可视化并验证所提SA的有效性。



#### [√] 置换注意力

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301271648181.jpg)

> alec：
>
> - 第一印象本文的idea：输入分组处理，每组再分成两份，一份使用通道注意力、一份使用空间注意力，然后将二者的输出concat，然后再将所有组的输出aggregate聚合。然后使用channel shuffle来融合信息。

![image-20230127144908771](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301271648182.png)

在与CNN的组合方面，我们**采用了与SE完全相同的方式进行嵌入集成**。而且SA的实现非常简单，这里给出了核心实现code：

> alec：
>
> - SE是通道注意力的实现方式之一。
> - chunk = 大块，矮胖的人或物

```python
def ShuffleAttention(x, cw, cb, sw, sb, G): # 置换注意力
    N, _, H, W = x.shape
    x = x.reshape(N*G, -1, H, W)
    x_0, x_1 = x.chunk(2, dim=1)
    
    xn = avg_pool(x_0)
    xn = cw * xn + cb
    xn = x_0 * sigmoid(xn)
    
    xs = GroupNorm(x_1)
    xs = sw * xs + sb
    xs = x_1 * sigmoid(xs)
    
    out = torch.cat([xn, xs], dim=1)
    out = out.reshape(N, -1, H, W)
    out = channel_shuffle(out, 2)
    return out
```

#### [√] 可视化

---

为验证SA是否可以改善特征的语义表达能力，我们在ImageNet上训练了两个模型：SANet50B（即无通道置换）与SANet50（有通道置换）。下图给出了SA_5_3(即最后一个阶段最后一个bottleneck)后不同组的top1精度统计，结果见下图。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301271648183.jpg)

正如上图所示，(1) 在SA之后，top1精度表示出了统计上的提升(平均提升约0.4%)，这也就意味着特征分组可以显著提升特征的语义表达能力;(2) 不同类别的分布在前面的层中非常相似，这也就意味着在前期特征分组的重要性被不同类别共享；(3) 随着深度加深，不同的特征激活表现出了类别相关性。

为更好的验证SA的有效性，我们基于ImageNet，采用GradCAM对其进行了可视化，见下图。可以看到：SA使得分类模型聚焦于目标信息更相关的区域，进而使得SA模块可以有效的提升分类精度。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301271648184.jpg)





## [√] 实验

---

![image-20230127163226530](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301271648185.png)

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301271648186.jpg)

上表给出了不同注意力机制在ImageNet上的性能对比，从中可以看到：

- SANet具有与原始ResNet相同的几乎相同计算复杂度和参数量，但取得了1.34%的top1精度提升@ResNet50；
- 在ResNet101上，所提SANet取得了0.76%的top1精度提升；
- 相比其他SOTA注意力机制，所提方案具有更高精度、更低计算复杂度。比如，在R而是Net101基础上，SE导致了4.778M参数量提升，14.34Flops提升，0.268%top1精度提升，而SA仅提升了0.002M参数量，5.12Flops提升，0.76%top1精度提升。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301271648187.jpg)

上表给出了COCO数据集上目标检测性能对比，可以看到：

- 无论是单阶段还是双阶段目标检测，SE与SA均能显著提升其性能；
- SA以更低的计算复杂度取得了优于SE的效果。具体来说，以Faster R-CNN为基准，SA能够以1.0%优于SE@ResNet50;如果采用RetinaNet作为基准，性能提升可以达到1.5%。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301271648188.jpg)

上表给出了COCO数据集上实例分割性能对比，可以看到：

- 相比原始ResNet，SA可以取得显著的性能提升；
- 相比其他注意力机制，SA可以取得更好的性能，同时具有更低的计算复杂度。
- 特别的，SA在小目标分割方面的增益更为显著。





## [√] 与SGE、BAM的对比

---

看到这里肯定有不少同学会觉得SA与SGE、BAM非常相似。确实非常相似，下面给出了SGE与BAM的核心模块示意图。SA采用了类似SGE的特征分组思想，SA同时还采用了与BAM类似的双重注意力机制(空域注意力+通道注意力)。

![sge](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301271648189.jpg)

<center>SGE</center>

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301271648190.jpg)

<center>BAM</center>

---

我们对这三中注意力机制进行更细节的拆分，SGE可以视作多组SE，即SGE= Group+SE；而BAM可以视作空域注意力与通道注意力的组合(注：这里采用加法进行两种注意力融合)，即BAM=SPA+CA；(为了与SA区分，这里空域注意力表示为SPA)；如果是Concat进行两种注意力融合呢，我们可以表示为Cat(SPA, CA)，那么SA则可以表示为：Group+Cat(SPA, CA)。另外还有一点：BAM在分别进行SPA与CA计算时未进行分组，而SA则是进行分组。

全文到此结束，更多消融实验与分析建议各位同学查看原文。











