---
title: 065 - 文章阅读笔记：BSRGAN超分辨网络 - 知乎 - CV论文分析 - 无为
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302121604238.jpg
tags:
  - 超分辨率重建
  - 图像处理
  - 深度学习
  - 盲超分
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 1100682133
date: 2023-02-12 15:44:04
---

> 原文链接：
>
> [BSRGAN超分辨网络 - 知乎 - CV论文分析 - 无为](https://zhuanlan.zhihu.com/p/379876494)
>
> 2021-04-01
>
> ps：本文为依据个人日常阅读习惯，在原文的基础上记录阅读进度、记录个人想法和收获所写，关于原文一切内容的著作权全部归原作者所有。



## [√] 正文开始

---

论文：[Designing a Practical Degradation Model for Deep Blind Image Super-Resolution](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2103.14006)

参考：[BSRGAN超分辨网络](https://link.zhihu.com/?target=https%3A//blog.csdn.net/weixin_46515047/article/details/117820008)

2021年新出炉的文章，张凯大哥等人写的。

## [√] 重点提要

---

这篇文章的目的是：构建一个能够实际应用的超分模型；

核心议题：如何构建一个实际的图像降级模型；

超分网络backbone：ESRGAN；

主要对比方法是：2019年的模型FSSR、2020年的模型Real-SR；

图像质量评价指标：有参：PSNR、SSIM、LPIPS。无参：NIQE、NRQM、PI；

核心思路：
$$
y = (x \otimes k)↓_s + n
$$
围绕着上述退化模型的3个因子：K为模糊核、S为降采样核、N为噪声，随机安排各因子的执行顺序（例如KSN、NKS、SNK、SKN、NSK、KNS）。同时，每个因子又有不同的方法（例如：降采样核S可以采用以下任一种方式：双三次、最近邻、双线性等等），可以从这些方法中为每个因子随机选取一种。此时，便可通过两种随机过程构建出退化模型。

要点1：忽略模糊核在构建HR-LR対时的影响，能够注入符合实际情况的噪声对构建HR-LR对是至关重要的。



## [√] 退化模型的构建

---

为遵循论文的表述，我们将模糊核记为B（blur），降采样核记为D（downsample），噪声记为N（noise）。各因子及各因子所包含的方法如下：

> 模糊核B：各向同性的高斯模糊核iso、各向异性的高斯模糊核aniso；
> 降采样核D：最近邻插值nearest、双线性插值bilinear、双三次插值bicubic、上下缩放up-down；
> 噪声N：高斯噪声G、JPEG压缩噪声JPEG、传感器噪声S。

对于三种因子的一些说明：

1. 降采样核D的上下缩放up-down方法中，包含两次缩放，（例如：欲完成2倍的缩小，可先进行1/3倍down，再进行3/2的up，即可完成1/2的缩小。）每次插值都在双三次和双线性插值中随机选取一个，先放大还是先缩小的顺序不限。
2. JPEG压缩噪声在退化模型中有两次添加。第一次按照上述随机顺序添加，第二次是在退化步骤的最后一步额外添加一次。这也在一定程度上说明了JPEG噪声的重要性。

退化模型如下：

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132248410.jpg)

作者对于退化模型的随机化描述如下：

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132248411.jpg)

另外需要说明的一点是，上述退化模型是真的2倍缩小的图像的。如果要进行4倍退化，需要在所有随机退化（Degradation Shuffle）步骤之前先通过双三次或双线性对图像进行2倍缩小，然后在进行模型退化，便可得到4倍退化结果。

通过上述退化模型获取到数据集后，输入以ESRGAN为backbone的网络进行调整、训练。作者将BSRGAN的生成网络称为BSRnet。整个BSRGAN在Tesla V100上需要训练10天。。。

以下是一些效果图：

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132248412.jpg)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132248413.jpg)









