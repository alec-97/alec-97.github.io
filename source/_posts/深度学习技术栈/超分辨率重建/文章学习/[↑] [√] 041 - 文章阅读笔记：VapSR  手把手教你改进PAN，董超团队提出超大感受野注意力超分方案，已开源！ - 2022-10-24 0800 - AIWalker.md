---
title: >-
  041 - 文章阅读笔记：VapSR | 手把手教你改进PAN，董超团队提出超大感受野注意力超分方案，已开源！ - 2022-10-24 08:00 -
  AIWalker
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301202040748.jpg
tags:
  - 超分辨率重建
  - 注意力机制
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 2130194975
date: 2023-01-19 22:15:46
---

> 原文链接：[VapSR | 手把手教你改进PAN，董超团队提出超大感受野注意力超分方案，已开源！ - 2022-10-24 08:00 - AIWalker](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651689655&idx=1&sn=54ea18956870f2c5d9b4b468d8aea23d&chksm=f3c9dc9ac4be558c9f4413c205276fafeb4847d0cf37cc465a72d7a0daee72f5bdf673baa4c0&scene=178&cur_album_id=1338480951000727554#rd)
>
> 2022-10-24 08:00

## [√] 论文信息

---

中文名称：使用超大感受野的注意力的高效图像超分

paper https://arxiv.org/abs/2210.05960

code https://github.com/zhoumumu/VapSR

![图片](D:\坚果云\Alec - backup files\typora pictures\640-1674138061838.jpg)

注意力机制是深度学习领域非常重要的一个研究方向，在图像超分领域也有不少典型的应用案例，比如基于通道注意力构建的RCAN，基于二阶注意力机制构建的SAN，基于像素注意力机制构建的PAN，基于Transformer自注意力机制构建的SwinIR，基于多尺度大核注意力的MAN等。

本文则以PAN为蓝本，对其进行逐步改进以期达到更少的参数量、更高的超分性能。该方案具体包含以下几个关键点：

- 提升注意力分割的感受野，类似大核卷积注意力VAN；
- 将稠密卷积核替换为深度分离卷积，进一步降低参数量；
- 引入像素规范化(Pixel Normalization)技术，其实就是Layer Normalization，但出发点不同。

上述关键技术点为注意力机制的设计提供了一个清晰的演变路线，最终得到了本文的VapSR，即大感受像素注意力网络(VAst-receptive-field Pixel attention Network)。

实验结果表明：相比其他轻量超分网络，VapSR具有更少的参数量。比如，项目IMDB与RFDN，VapSR仅需21.68%、28.18%的参数即可取得与之相当的性能。

## [√] 本文动机

---

通过引入像素注意力，PAN在大幅降低参数量的同时取得了非常优秀的性能。相比通道注意力与空域注意力，像素注意力是一种更广义的注意力形式，为进一步的探索提供了一个非常好的基线。

受启发于自注意力的发展，我们认为：基于卷积操作的注意力仍有进一步改进的空间。因此，作者通过以下三个像素注意力中的设计原则展示了改善超分注意力的过程：

- 首先，在注意力分支引入大核卷积具有明显的优势；
- 其次，深度分离卷积可以降低大核卷积导致的巨大计算复杂度问题；
- 最后，引入像素规范化操作让训练更高效、更稳定。

![图片](D:\坚果云\Alec - backup files\typora pictures\640-1674203582629.jpg)

> alec：
>
> - 增大注意力分支的卷积核大小从而减少感受野，同时减少其它的卷积模块的卷积核的大小来精简参数。
> - 在模块的后端增加像素规范化操作，同时调整注意力模块的位置将注意力机制模块放到模块的中间，而不是后端。

**Large Kernel **以上图i中的baseline为基础，作者首先对注意力分支进行感受野扩增：将1x1提升到9x9(将图示ii)，性能提升0.15dB，但参数量从846K提升到了4123K。



#### [√] Parameter Reduction（轻量化）

---

为降低参数量，我们尝试尽可能移除相对不重要的部分。作者提出了三个方案：(1) 将非注意力分支的卷积尺寸从3x3下调到1x1；(2) 将大核卷积注意力分支替换为深度深度分离卷积；(3) 将深度分离卷积中的深度卷积进行分解为深度卷积+带扩张因子的深度卷积(该机制可参考下图，将卷积11x11拆分为5x5+3x3，其中后者的扩张因子为3)。此时，模型性能变为28.48dB，但参数量降到了240K，参数量基本被压缩到了极限。

> alec：
>
> - 什么是{深度分离卷积}？
> - 扩张卷积是将卷积核变为间隔离散的，而不是连续的？

#### [√] Pixel Normalization(PN) （像素规范化）

---

注意力机制的元素乘操作会导致训练不稳定问题：小学习率收敛不够好，大学习率又会出现梯度异常。前面的注意力改进导致所得方案存在性能下降问题。为解决该问题，作者经深入分析后提出了像素规范化技术(可参考下图不同规范化技术的可视化对比)。

![图片](D:\坚果云\Alec - backup files\typora pictures\640-1674204418803.jpg)

![image-20230120164742013](D:\坚果云\Alec - backup files\typora pictures\image-20230120164742013.png)



#### [√] Switch Attention to Middle（模块位置调整）

---

在上述基础上，作者进一步将注意力的位置进行了调整，放到了两个1x1卷积中间。此时，模型性能得到了0.03dB提升，达到了28.95dB，参数量仍为241K。



## [√] 本文方案

---

前面的探索主要聚焦在微观层面，基于此，作者进一步在宏观层面进行了更多设计与提炼，进而构建了VapSR，取得了更佳的性能，同时具有更少的参数量。

> alec：
>
> - dilated，加宽的、扩大的、膨胀的
> - PAN中的像素注意力分支是先通过1x1卷积，然后通过sigmoid激活函数，得到一个逐像素的注意力权重。

![图片](D:\坚果云\Alec - backup files\typora pictures\640-1674217213424.jpg)

![image-20230120202120092](D:\坚果云\Alec - backup files\typora pictures\image-20230120202120092.png)



> alec：
>
> - 深度扩张卷积，也称之为空洞卷积。
> - 空洞卷积，通过给卷积核插入空洞来变相的增大感受野。

![image-20230120202911180](D:\坚果云\Alec - backup files\typora pictures\image-20230120202911180.png)

```python
class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.depthwise_dilated = nn.Conv2d(dim, dim, 5, 1, padding=6, groups=dim, dilation=3)

    def forward(self, x):
        u = x.clone()
        attn = self.pointwise(x)
        attn = self.depthwise(attn)
        attn = self.depthwise_dilated(attn)
        return u * attn
      
class VAB(nn.Module):
    def __init__(self, d_model, d_atten):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_atten, 1)
        self.activation = nn.GELU()
        self.atten_branch = Attention(d_atten)
        self.proj_2 = nn.Conv2d(d_atten, d_model, 1)
        self.pixel_norm = nn.LayerNorm(d_model)
        default_init_weights([self.pixel_norm], 0.1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.atten_branch(x)
        x = self.proj_2(x)
        x = x + shorcut

        x = x.permute(0, 2, 3, 1) #(B, H, W, C)
        x = self.pixel_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous() #(B, C, H, W)

        return x
```



## [√] 本文实验

---

在实验部分，作者构建了VapSR与VapSR-S两个版本的轻量型超分方案：

- VapSR：包含21个VAB模块，主干通道数为48；
- VapSR-S：包含11个VAB模块，主干通道数为32。

此外，需要注意的是：对于X4模型，重建模块并未采用常规的轻量方案(Conv+PS)，而是采用了类EDSR的重方案(Conv+PS+Conv+PS)。

> alec：
>
> - 理论计算量通常只考虑乘加操作(Multi-Adds)的数量,而且只考虑CONV和FC等参数层的计算量,忽略BatchNorm和PReLU等等。
> - multi-adds表示的是计算量。

![图片](D:\坚果云\Alec - backup files\typora pictures\640-1674218056022.jpg)

![图片](D:\坚果云\Alec - backup files\typora pictures\640-1674218065202.jpg)

![图片](D:\坚果云\Alec - backup files\typora pictures\640-1674218083589.jpg)

上表&图给出了不同方案的性能与可视化效果对比，从中可以看到：

- 所提VapSR取得了SOTA性能，同时具有非常少的参数量。
- 在X4任务上，相比RFDN与IMDN，VapSR仅需21.68%/28.18%的参数量，即可取得平均0.187dB指标提升；
- VapSR-S取得了与BSRN-S相当的性能，后者是NTIRE2022-ESR模型复杂度赛道冠军。
- 在线条重建方面，VapSR具有比其他方案更精确的重建效果。







































