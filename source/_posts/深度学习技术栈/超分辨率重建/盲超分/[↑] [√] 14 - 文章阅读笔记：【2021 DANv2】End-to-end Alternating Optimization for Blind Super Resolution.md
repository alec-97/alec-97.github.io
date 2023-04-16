---
title: >-
  文章阅读笔记：【2021 DANv2】End-to-end Alternating Optimization for Blind Super
  Resolution
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252354841.jpg
tags:
  - 盲超分
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 盲超分
abbrlink: 1100471768
date: 2023-02-25 22:51:35
---

> 原文链接：
>
> （1）【√】每日五分钟一读# Blind Super Resolution - 知乎 - Andy（[link](https://zhuanlan.zhihu.com/p/373082386)）
>
> 发布于 2021-05-18 11:03
>
> ps：本文为依据个人日常阅读习惯，在原文的基础上记录阅读进度、记录个人想法和收获所写，关于原文一切内容的著作权全部归原作者所有。



#  [√] 文章信息

---

论文标题：【2021 DANv2】End-to-end Alternating Optimization for Blind Super Resolution

中文标题：基于端到端交替优化进行盲超分

论文链接：https://arxiv.org/pdf/2105.06878.pdf

论文代码：https://github.com/greatlog/DAN

论文发表：TPAMI 2021



# [√] 文章1

---

> 总结：
>
> 略，见DANv1



## [√] 概述

---

End-to-end Alternating Optimization for Blind Super Resolution

论文地址：[https://arxiv.org/pdf/2105.06878.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2105.06878.pdf)

代码地址：[https://github.com/greatlog/DAN.git](https://link.zhihu.com/?target=https%3A//github.com/greatlog/DAN.git)

关键词：交替优化、端到端、盲图像超分

## [√] 解决的问题

---

解决的问题：如何迭代优化方法，实现核估计和图像超分交替进行

---> 单张图像超分的目的是从退化LR图像中恢复出HR图像。具体数学模型如下：

![image-20230225225821052](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252357070.png)

--->过去五年里，大部分基于DNNs的图像超分都是假设 模糊核为bicubic插值核。但实际中内核是复杂的，未知的。因此，需要探究盲SR方法。但是盲SR问题中模糊核是未知变量，优化也变得更加困难。目前的方法可分解为两步单独训练，再结合：1) 从LR图像估计模糊核，2）基于估计模糊核恢复SR图像。 但仍存在一定的缺陷：1）第一步估计的小错误可能会导致第二步性能严重下降，2）第一步只能从LR图像使用限制信息，这使其难以预测正确的模糊核。

-->根据上述难点，本文提出了一种交替优化方法，能够在一个模型中交替估计模糊核k和复原SR图像。

## [√] 论文的贡献

---

1. 采用一种交替优化算法在单个网络（DAN）中来估计模糊核和恢复SR图像，这有助于使两个模块相互良好兼容，从而获得比以前的两步解决方案 更好的最终结果。
2. 计了两个卷积神经模块，这些模块可以反复交替，然后展开以形成端到端的可训练网络，而无需任何前/后处理。 与以前的两步式解决方案相比，它更易于训练并且速度更高。 据我们所知，所提出的方法是第一个用于盲目SR的端到端网络。
3. 对合成图像和真实世界图像进行的大量实验表明，我们的模型可以大大优于最新方法，并以更高的速度产生更直观的视觉效果。

## [√] 网络结构

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252357071.jpg)

#### [√] 1、基本模型

---

> 可以基于MAP框架，交替优化模糊核和图像：

![image-20230225232922771](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252357072.png)

> 进一步的，使用额外的先验知识进行优化：

![image-20230225232947698](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252357073.png)

> 在构建图像模型和强假设后，上述优化问题可以被解决。但是，上述图像模型或假设难以直接用到实际应用中，并且，如果确实强假设，上述优化问题难以被解决。



#### [√] 2、两步法

---

> 将上述问题分解为两个子问题：

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252357074.jpg)

> 但是，单单的两步法，具有以下的缺点：1） 需要训练两个甚至更多的模型；2）M函数仅仅从退化图像y中获取信息；3）非盲SR模型在第二步优化中，需要真实的模糊核。



#### [√] 3、交替优化方法展开

---

> 为了解决上述问题，将优化问题展开有以下交替优化的形式：

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252357075.jpg)

> 定义两个子网络，Estimator 和 Restorer分别用于估计模糊核和恢复图像，命名为深度交替网络（DAN）。首先，初始化模糊核为Dirac函数，并进行变形与PCA转换，输入到DAN中。最后的监督信息采用L1损失。



#### [√] 4、网络具体结构

---

> Estimator 的输入是LR图像和SR图像， Restorer的输入为LR图像和模糊核。基本结构和子网络
> 1） 基本结构。条件残差模块（CRB）具有以下的缺点：a）在Restorer中，条件输入（即他估计必须在空间上扩展内核才能与LR功能串联），这大大增加了计算成本。 b）实验表明，CRB中的通道注意层（CALayer）很耗时，并且容易导致梯度爆炸，从而减慢了推理速度并使训练变得不稳定。 c）网络中的所有块都具有相同的功能，这可能会限制整个网络的表示能力。
> 因此，本文提出了双路条件模块（DPCB）用于Restorer的基本结构。
> 提出了双路条件组（DPCG）用于Estimator 的基本结构。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252357076.jpg)



## [√] 实验

---

**训练数据：**DIV2K [[1\]](https://zhuanlan.zhihu.com/p/373082386#ref_1)and Flickr2K[[2\]](https://zhuanlan.zhihu.com/p/373082386#ref_2)

度量标准: PSNR、SSIM （YCbCr空间的Y通道）

#### [√] 1、对比实验

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252357077.jpg)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252357079.jpg)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252357080.jpg)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252357081.jpg)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252357082.jpg)

#### [√] 2、消融实验

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252357083.jpg)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252357084.jpg)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252357085.jpg)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252357086.jpg)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252357087.jpg)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252357088.jpg)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252357089.jpg)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302252357090.jpg)



