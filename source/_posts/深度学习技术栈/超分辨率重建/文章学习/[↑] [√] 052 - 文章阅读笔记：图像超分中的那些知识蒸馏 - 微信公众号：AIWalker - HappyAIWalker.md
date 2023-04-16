---
title: 052 - 文章阅读笔记：图像超分中的那些知识蒸馏 - 微信公众号：AIWalker - HappyAIWalker
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302031754424.jpg
tags:
  - 知识蒸馏
  - 超分辨率重建
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 650134279
date: 2023-01-27 17:26:22
---

> 原文链接：
>
> [图像超分中的那些知识蒸馏 - 微信公众号：AIWalker - HappyAIWalker](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651683255&idx=1&sn=49d357567760d4fe1a7413deccff1a85&scene=21#wechat_redirect)
>
>  2021-01-21 22:00

> alec：
>
> - distill = 蒸馏
> - dis，till（直到没了）

## [√] 概述

---

本文对三篇知识蒸馏在图像超分中的应用进行了简单的总结，主要包含：

- SRKD：它将最基本的知识蒸馏直接应用到图像超分中，整体思想分类网络中的蒸馏方式基本一致，整体来看属于应用形式；
- FAKD：它在常规知识蒸馏的基础上引入了特征关联机制，进一步提升被蒸馏所得学生网络的性能，相比直接应用有了一定程度的提升；
- PISR：它则是利用了广义蒸馏的思想进行超分网络的蒸馏，通过充分利用训练过程中HR信息的可获取性进一步提升学生网络的性能。
- 注：在公众号后台回复：KDSR，即可获得上述论文下载链接。



## [√] SRKD

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302031755921.jpg)

上图给出了SRKD的蒸馏示意图，它采用了最基本的知识蒸馏思想对老师网络与学生网络的不同阶段特征进行蒸馏。考虑到老师网络与学生网络的通道数可能是不相同的，SRKD则是对中间特征的统计信息进行监督。该文考虑了如下四种统计信息：

![image-20230203172318518](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302031755922.png)

![image-20230203172555926](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302031755923.png)

下图给出了所提蒸馏方案在不同倍率、不同测试集上的性能对比。总而言之，SRKD确实可以取得比直接训练更好的性能。

![image-20230203172639423](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302031755924.png)

> alec：
>
> - 可以看出，使用SRKD方式进行蒸馏，能够取得比直接训练更好的性能。

## [√] FAKD

---

![image-20230203173106344](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302031755925.png)

> FAKD：它在常规知识蒸馏的基础上引入了特征关联机制，进一步提升被蒸馏所得学生网络的性能，相比直接应用有了一定程度的提升；

上图给出了FAKD的整体架构示意图，在整个蒸馏过程上来看，它与SRKD比较类似。区别在于蒸馏损失部分。它提出了特征关联思想用于知识蒸馏，所提的特征关联定义如下：

![image-20230203174417207](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302031755926.png)

看上述公式可能比较难以理解，看这面这个图则比较容易理解了。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302031755927.jpg)

为了说明所提特征关系思想的有效性，作者对比了不同形式的特征关联机制，见下表对比。

![image-20230203174547776](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302031755928.png)

最后则给出了所提方法在RCAN与SAN两种优秀超分中的蒸馏效果对比。从下表可以看到：对于RCAN来说，蒸馏所提模型基本上可以提升0.05-0.15不等的性能，看来很不错哟。

![image-20230203174616369](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302031755929.png)

## [√] PISR

---

![image-20230203174812755](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302031755930.png)

上图给出了PISR的蒸馏示意图。相比SRKD与FAKD，PISR的创新点则更多，要不然也不至于能中ECCV了，对吧。

总体来说，PISR参考了广义蒸馏的思想，同时采用HR作为输入，通过Encoder模拟退化过程，并令Decoder与学生网络具有相同的结构。这种处理机制使得老师网络与学生网络在生成的特征结构信息方面具有了更好的“均等”性，而这个“均等性”是其他蒸馏方法很少去考虑的。

在损失函数方面，PISR参考了VID蒸馏方案中的“变分信息蒸馏”思想：最大化老师网络与学生网络之间的互信息。这里的蒸馏损失定义如下：

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302031755931.jpg)

下表对比了所提方案中不同模块的重要性说明，总而言之：各个模块都很重要。

![image-20230203174930514](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302031755932.png)

然后给出了FSRCNN的蒸馏性能对比，该文主要也是针对FSRCNN这种轻量型网络进行蒸馏，这是难能可贵的。当然也可以对其他网络进行蒸馏，比如VDSR、IDN、CARN等。

![image-20230203175059138](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302031755933.png)

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302031755934.jpg)

## [√] 小结

---

本文对图像超分领域中的知识蒸馏进行了简单总结，从目前的初步调研来看：图像超分中的知识蒸馏仍处于“莽荒”阶段，深入性稍显不足，如何将其他领域的知识蒸馏技术迁移到图像超分领域并进行针对性的“魔改”可能会是一个不错的点。

## [√] 参考

---

1. ACCV 2018. Image Super-Resolution using Knowledge Distillation.
2. ICIP 2020. FAKD: Feature-Affinity based Knowledge Distillation for Efficient Image Super Resolution.
3. ECCV 2020. Learning with Privileged Information for Efficient Image Super Resolution.
4. CVPR 2019. Variational Information Distillation for Knowledge Transfer.

































