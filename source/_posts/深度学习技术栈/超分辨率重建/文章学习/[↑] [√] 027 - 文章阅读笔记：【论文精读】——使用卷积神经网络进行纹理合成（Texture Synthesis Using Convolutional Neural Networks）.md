---
title: >-
  027 - 文章阅读笔记：【论文精读】——使用卷积神经网络进行纹理合成（Texture Synthesis Using Convolutional
  Neural Networks）
tags:
  - cnn
  - 人工智能
  - 深度学习
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 3072813515
date: 2023-01-14 21:59:14
---

> 转载自：
>
> 【√】[【论文精读】——使用卷积神经网络进行纹理合成（Texture Synthesis Using Convolutional Neural Networks）](https://blog.csdn.net/zency/article/details/127700229)
>
> 于 2022-11-08 16:48:13 发表

# 2015-Texture Synthesis Using Convolutional Neural Networks

---



## [√] 基本信息

---

**作者：** Leon A. Gatys，Alexander S. Ecker，Matthias Bethge
**期刊：** NIPS
**引用：** *

摘要： 本文介绍了一个基于卷积神经网络特征空间的自然纹理的新模型，该模型为目标识别而优化。该模型的样本具有很高的感知质量，显示了以纯粹的辨别方式训练的神经网络的生成能力。在该模型中，纹理是由网络中若干层的特征图之间的相关性来表示的。我们表明，跨层的纹理表示越来越多地捕捉到自然图像的统计特性，同时使物体信息越来越明确。该模型提供了一个新的工具来产生神经科学的刺激，并可能提供对卷积神经网络学习的深层表征的见解。
> alec：
>
> - 纹理是由网络中若干层的特征图之间的相关性来表示的。



## [√] 1.简介

---

**视觉纹理合成**的目标是从一个实例纹理中推断出一个生成过程，然后可以产生任意多的该纹理的新样本。对合成纹理质量的评价标准通常是人的检查，如果人的观察者不能从合成的纹理中分辨出原始纹理，那么纹理的合成就成功了。

**寻找纹理生成过程的两种方法：**

- 通过对原始纹理的像素或整个斑块进行重采样来生成新的纹理。它们并没有为自然纹理定义一个实际的模型，而是给出了一个机械化的程序，说明如何在不改变其感知特性的情况下随机化一个源纹理。
- 明确地定义一个参数化的纹理模型。该模型通常由一组在图像的空间范围内进行的统计测量组成。**目前最好的模型**：Portilla和Simoncelli提出的模型，基于一组精心手工制作的摘要统计，这些统计是在一个叫做Steerable Pyramid的线性滤波器组的响应上计算的。然而，尽管他们的模型在合成广泛的纹理方面显示出非常好的性能，但它仍然未能捕捉到自然纹理的全部范围。

**本文提出了一个新的参数化纹理模型来解决这个问题。****使用卷积神经网络作为我们纹理模型的基础。****将关于特征反应的空间汇总统计的概念框架与经过物体识别训练的卷积神经网络的强大特征空间相结合。通过这种方式得到一个纹理模型，它的参数是建立在卷积神经网络的分层处理结构上的空间不变的表征。**





## [√] 2.卷积神经网络

---

提取特征使用[VGG](https://so.csdn.net/so/search?q=VGG&spm=1001.2101.3001.7020)-19网络。使用16个卷积层和5个池化层，不使用全连接层，整个网络基于以下两种运算：

- 用大小为3x3xk的滤波器进行线性整流卷积，其中k为输入特征图的数量。卷积的stride和padding等于1，这样输出的特征图与输入的特征图具有相同的空间尺寸。
- 在不重叠的2x2个区域中进行最大池化，将特征图的采样率降低2倍。

将最大池化改为平均池化；重新调整网络权重。

## [√] 3 and 4.纹理模型&纹理生成

---

![image-20230114235823984](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301150001289.png)

## [√] 5.结果

---

结果证明，随着特征的深度，生成效果也越好，同时，“碎片化”的情况也越缓解

- 模型严重参数化，通过有规律地减少模型及参数，其效果虽然有降低，但是可以将参数降低很多。表明纹理模型可以大大压缩，而对纹理效果影响很小。
- 带有小型卷积滤波器的VGG网络的极深结构似乎特别适合于纹理生成的目的。假象网格可能来源于网络较大的感受野和步幅。

- 学到的特征空间对纹理生成同样重要，强调使用经过训练的网络的重要性
- 本网络效果在目标识别方面接近原网络，纹理表征不断地拆分物体身份信息。物体身份可以在各层中被解码得越来越好。事实上，从最后的集合层进行的线性解码几乎和原始网络一样好，这表明我们的纹理表示几乎保留了所有的高级信息。网络中的卷积表征是移位的，网络的任务（物体识别）与空间信息无关，因此我们预计物体信息可以独立于特征图中的空间信息被读出

## [√] 6.讨论

---

- 本文的纹理模型超过SOTA
- 本文的计算成本较大（参数多），还可以有更佳的改善

- 通过计算特征图上的格拉姆矩阵，纹理模型将卷积神经网络的表征转换为固定的特征空间。**这个一般的策略最近被用来提高物体识别和检测或纹理识别和分割的性能**

- 该模型为产生神经科学的刺激提供了一个新的工具，并可能为卷积神经网络学习的深度表征提供见解。

> alec：
>
> - 通过计算特征图上的格拉姆矩阵，纹理模型将卷积神经网络的表征转换为固定的特征空间。





## [√] 代码实现

---

https://github.com/leongatys/DeepTextures

## [√] 个人总结

---

1. **本文提出了纹理损失的雏形，可以捕捉更多物体特性，使物体信息更加明确，有助于物体识别等任务。**

















