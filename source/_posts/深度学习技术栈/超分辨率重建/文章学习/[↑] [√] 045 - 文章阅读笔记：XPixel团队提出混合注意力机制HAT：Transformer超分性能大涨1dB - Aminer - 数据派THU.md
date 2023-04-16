---
title: 045 - 文章阅读笔记：XPixel团队提出混合注意力机制HAT：Transformer超分性能大涨1dB - Aminer - 数据派THU
index_img: 'https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301210946513'
tags:
  - 注意力机制
  - 超分辨率重建
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 3434417572
date: 2023-01-21 09:41:24
---

> 原文链接：
>
> [XPixel团队提出混合注意力机制HAT：Transformer超分性能大涨1dB - Aminer - 数据派THU](https://www.aminer.cn/research_report/62a174007cb68b460fce055a?download=false)
>
> 2022-06-09 12:16
>
> 关键词: 注意力，性能，窗口尺寸，像素，训练策略

## [√] 论文信息

---

该方法结合了通道注意力，自注意力以及一种新提出的重叠交叉注意力等多种注意力机制。

本文介绍作者提出的一种基于混合注意机制的Transformer（Hybrid Attention Transformer, HAT）。

> alec：
>
> - 本文结合了通道注意力、自注意力、重叠交叉注意力。其中重叠交叉注意力是本文新提出的。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240417)

近日，来自澳门大学、中国科学院深圳先进技术研究院等机构的XPixel团队研究人员通过分析和实验指出，目前的方法无论是在模型设计，还是预训练策略上，都仍存在较大的提升空间。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240418)

为此，作者提出了一种基于混合注意机制的Transformer （Hybrid Attention Transformer, HAT）。该方法结合了通道注意力，自注意力以及一种新提出的重叠交叉注意力等多种注意力机制。此外，还提出了使用更大的数据集在相同任务上进行预训练的策略。

论文链接：https://www.aminer.cn/pub/6279c9c65aee126c0fdae979

项目链接：https://github.com/chxy95/HAT

实验结果显示，本文提出的方法在图像超分辨率任务上大幅超越了当前最先进方法的性能（超过1dB），如图1所示。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240419)

<center>图1. HAT与当前SOTA方法SwinIR和EDT的性能对比</center>

HAL-L表示在HAT的基础上深度增加一倍的更大容量的模型。



## [√] 分析

---

本文首先对不同方法的LAM [4]结果进行了对比。

LAM是一种为SR任务设计的归因方法，它能够显示模型在进行超分辨率重建的过程中哪些像素起到了作用。

如下图2(a)所示，LAM图中红色标记点表示：模型在重建左上图红框标记块时，对重建结果会产生影响的像素（LAM结果下的值为DI值[4]，它可以定量地反映被利用像素的范围。DI值越大，表示重建时利用的像素范围越大）。一般来说，被利用像素的范围越大，重建的效果往往越好[4]，该结论在对比基于CNN的方法EDSR与RCAN时可以得到明显体现。然而，当对比RCAN与基于Transformer的SwinIR方法时，却出现了结论相反的现象。SwinIR取得了更高的PSNR/SSIM，但相比RCAN并没有使用更大范围的像素信息，并且由于其有限的信息使用范围，在蓝色框区域恢复出了错误的纹理。这与以往普遍认为Transformer结构是通过更好地利用long-range信息来取得性能优势的直觉是相悖的。通过这些现象，本文认为：

- SwinIR结构拥有更强的局部表征能力，能够使用更少的信息来达到更高的性能；
- SwinIR依然有较大提升空间，如果更多的像素能够被利用，那么应该会取得更大的性能提升。

> alec：
>
> - SwinIR这个方法是利用的transformer，LAM分析结果表示，虽然这个模型用的信息少一些，但是效果仍然不错。这表明SwinIR结构有更强的局部表征能力，能够用较少的信息来达到更高的性能。
> - 同时，如果SwinIR能够利用更多的像素，那么这个应该能够取得更大的性能提升。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240420)

<center>图2. (a) 不同网络结构的LAM结果对比；（b）SwinIR网络产生的块效应</center>

除此之外，本文发现在SwinIR网络前几层产生的中间特征会出现明显的块状效应。这是由于模型在计算自注意力时的窗口划分导致的，因此本文认为现有结构进行跨窗口信息交互的方式也应该被改进。

## [√] 方法网络结构设计

---

HAT的整体架构采用了与SwinIR相似的Residual in Residual结构，如下图3所示。

主要的不同之处在于混合注意力模块（Hybrid Attention Block， HAB）与重叠的交叉注意力模块（Overlapping Cross-Attention Block， OCAB）的设计。

其中对于HAB，本文采用了并联的方式来结合通道注意力和自注意力。通道注意力能够利用全局信息；自注意力具有强大的表征能力。HAB模块的目的在于能够同时结合这两者的优势。

> alec：
>
> - MSA = 多头自注意力模块

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240421)

对于OCAB的设计，本文使用了一种重叠的窗口划分机制，如下图4所示。相对于原始基于窗口的self-attention中Q、K和V来自于同一个窗口特征，OCA中的K/V来自更大的窗口特征，这允许attention能够被跨窗口地计算，以增强相邻窗口间信息的交互。

> alec：
>
> - OCAB = 重叠的交叉注意力模块
> - 在一般的MSA中，自注意力的QKV三者都是来自于同一个窗口特征；但是这样可能容易导致块效应，就像SwinIR一样；这可能是窗口之间缺少交流导致的。因此本文设计了OCAB模块，在这个模块中，自注意力中的K和V都是采用的一个扩大的窗口，Q是采用的标准的窗口。通过这种方法计算的QKV，能够充分的利用窗口之间的信息，增强窗口间的交互。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240422)

<center>图4. OCAB的网络结构</center>



## [√] 预训练策略

---

本文提出了一种直接使用相同的任务，但是使用更大的数据集（比如ImageNet）进行预训练的策略。相比于之前用于超分任务的预训练方案，该策略更简单，但却能带来更多的性能增益。实验结果后面给出。

> alec：
>
> - 本文的方案是在更大的数据集上对相同的任务进行预训练。
> - 这种方式简单，但是能够带来更多的性能增益。









## [√] 实验

---

## [√] 更大的窗口尺寸

---

直接增加计算self-attention的窗口尺寸可以使模型能够利用更多的像素，并得到显著的性能提升。表1和图5给出了对于不同窗口尺寸的定量和定性比较，可以看到16窗口尺寸有明显提升，HAT使用窗口尺寸16作为默认设置。

> alec：
>
> - 增大窗口尺寸可以提升性能

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240423)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240424)

<center>图5. 不同窗口尺寸的定性比较</center>

从上图定性的实验结果也能够看出，增大窗口的尺寸，能够利用更多的信息。

## [√] 消融实验

---

本文提供了消融实验来验证CAB和OCAB的影响，定量和定性分析结果如下表2和图6所示。可以看到文中所提的两个模块在定量指标上均带来了不小的提升，在LAM和视觉效果上相对于Baseline也具有明显改善。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240425)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240426)

<center>图6. 消融实验的定性比较</center>

## [√] 主实验结果

---

在基准数据集上进行定量对比实验的结果如下表6所示。

从定量指标上看，没有使用ImageNet预训练策略的HAT的性能已经明显超越SwinIR，甚至在很多情况下超越了经过ImageNet预训练的EDT。

使用了ImageNet预训练的HAT则更是大幅超越了SwinIR与EDT的性能，在2倍超分的Urban100数据集上，超越SwinIR 1dB。更大容量的模型HAT-L带来了更大的性能提升，最高在2倍超分的Urban100数据集上超越SwinIR达1.28dB，超越EDT达0.85dB。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240427)

视觉效果对比如下图7所示。可以看出HAT能够恢复更多更清晰的细节，由于对于重复纹理较多的情况，HAT具有显著优势。在文字的恢复上，HAT相比其他方法也能够恢复出更清晰的文字边缘。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240428)

<center>图7. 视觉效果对比</center>

本文还提供了HAT与SwinIR的LAM对比，如下图8所示。可以看出HAT能够明显使用更多的像素进行超分辨率重建，并因此取得了更好的重建效果。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240429)

<center>图8. HAT与SwinIR的LAM结果对比</center>

> alec：
>
> - 可以看出，HAT比SwinIR能够利用更多的像素信息



## [√] 预训练策略对比

---

本文对于不同的预训练策略进行了对比，如下表7所示。相对于EDT [3]提出使用相关任务进行预训练的策略，本文提出的使用相同任务进行预训练的策略无论是在预训练阶段还是微调后的结果，性能都要更优。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240430)

## [√] 总结

---

- 在结构上，本文设计的HAT结合了通道注意力与自注意力，在以往Transformer结构的基础上进一步提升了模型利用输入信息的范围。同时设计了一个重叠交叉注意力模块，对Swin结构利用跨窗口信息的能力进行了有效增强。

- 在预训练策略上，本文提出的在相同任务上做预训练的方法，使得模型的性能进一步增强。

- HAT大幅超越了当前超分方法的性能，这表明该任务或许远没有达到上限，可能依然还有很大的探索空间。







