---
title: '048-文章阅读笔记：CVPR2022-Restormer:刷新多个low-level任务指标-公众号：AIWalker-Happy'
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301261800508.jpg
tags:
  - transformer
  - 超分辨率重建
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 3593526722
date: 2023-01-25 16:55:12
---

> 原文链接：
>
> [CVPR2022 | Restormer: 刷新多个low-level任务指标 - 公众号：AIWalker - Happy](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651690247&idx=1&sn=9cbe82e50851130b5ec2031af9116495&chksm=f3c9db2ac4be523c57c42d2cb8ef565d11f8ec9add51af7d0ccb05afb3ee35c77bf292b7a5ac&scene=178&cur_album_id=1338480951000727554#rd)
>
> - [x] 整理到思维导图



## [√] 论文信息

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301261800506.jpg)

arXiv：https://arXiv.org/abs/2111.09881

code：https://github.com/swz30/Restormer

> 本文是MPRNet与MIRNet的作者在图像复原领域的又一力作，也是Transformer技术在low-level领域的又一个SOTA。针对Transformer在高分辨率图像复原中存在的难点，提出了两种MDTA与GDFN两种改进，极大程度上缓解了计算量与GPU缓存占用问题。所提方案刷新了多个图像复原任务的SOTA性能。



## [√] 摘要

---

通过MHSA与FFN进行改进，本文提出一种高效Transformer，它可以捕获长距离像素相关性，同时可适用于大尺寸图像。所提方案Restormer(Restoration Transformer)在多个图像复原任务上取得了SOTA性能，包含图像去雨、图像去运动模糊、图像去散焦模糊以及图像降噪(包含合成与真实噪声)，可参见下图。

> Restormer = rest + ormer = Restoration + Transformer = 恢复 + 转换器

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301261800507.jpg)

本文主要贡献包含以下几点：

- 提出了一种编解码Transformer用于高分辨率图像上多尺度local-global表达学习，且无需进行局部窗口拆分；
- 提出一种MDTA(Multi-Dconv head Transposed Attention)模块，它有助于进行局部与非局部相关像素聚合，可以高效的进行高分辨率图像处理；
- 提出一种GDFN(Gated-Dconv Feed-forward Network)模块，它可以执行可控特征变换，即抑制低信息特征，仅保留有用信息。



## [√] 方法

---

本文旨在设计一种高效Transformer模型，它可以处理复原任务中的高分辨率图像。为缓解计算瓶颈，我们对MHSA进行了关键性改进并引入多尺度分层模块，见下图。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301261800508.jpg)

> alec：
>
> - 这个 Restormer 是一个多尺度形式的高效transformer模型。
> - 这个transformer模型的核心模块包含两部分：MDTA和GDFN。
> - 其中MDTA执行夸通道的特征交互，而不是执行跨空间的特征交互。（多个Dconv头转换的注意力机制）
> - GDFN负责控制特征转换，允许使用有用的信息来前向传播。（门控Dconv的前馈网络）

![image-20230125221855918](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301261800509.png)

#### [√] Multi-Dconv Head Transposed Attention（多个Dconv头转换的注意力）

---

![image-20230125222118706](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301261800510.png)

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301261800511.jpg)

为解决该问题，我们提出了MDTA(见上图)，它具有线性复杂度，其关键成分在于：在通道维度(而非空间维度)执行自注意力计算跨通道的交叉协方差以生成关于全局上下文的隐式注意力特征图。

作为MDTA的另一个重要成分，在计算特征协方差生成全局特征图之前，我们引入了深度卷积以突出局部上下文。

> alec：
>
> - Dconv = 深度卷积
> - 深度卷积能够突出局部上下文。

![image-20230125225755983](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301261800512.png)



#### [√] Gated-Dconv Feed-forward Network

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301261800513.jpg)

![image-20230125225936202](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301261800514.png)



#### [√] Progressive Learning（渐进式学习）

---

基于CNN的复原模型通过采用固定尺寸图像块进行训练。然而，Transformer模型在较小块上训练可能无法进行全局统计信息编码，进而导致全分辨率测试时的次优性能。

针对该问题，我们提出了Progressive Learning机制：**在训练的初期，模型在较小图像块上进行训练；在训练的后期，模型采用更大图像块进行训练** 。由于更大的图像块会导致更长的计算耗时，我们随图像块提升降低batch参数以保持与固定块训练相当的耗时。

通过混合尺寸图像块训练的模型具有更优的性能。Progressive学习策略具有类似Curriculum学习策略相似的行为。



## [√] 实验

---

我们在不同的任务上进行了所提方案的性能验证，包含图像去雨、图像去运动模糊、图像去散焦模糊、图像降噪。



#### [√] Image Deraining

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301261800515.jpg)

上图&表给出了所提方案在去雨任务上的性能与效果对比，可以看到：

- 相比此前最佳SPAIR，**Restormer在所有数据集上取得了平均1.05dB指标提升** ；
- 在Rain100L数据集上，**性能增益甚至高达2.06dB** ；
- Restormer可以生成更好的无雨图像，且可以有效的保持结构内容。

#### [√] Motion Deblurring

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301261800516.jpg)

上表给出了所提方案在不同去模糊数据集上的性能对比，可以看到：

- 相比MIMO-UNet+，所提Restormer可以取得了平均0.47dB指标提升；
- 相比MPRNet，所提方案Restormer可以取得平均0.26dB指标提升；Restormer的FLOPs仅为MPRNet的81%；
- 相比IPT，所提方案Restormer取得了0.4dB指标提升，同时具有更少的参数量(4.4x)、更快的推理速度(29x)；
- 可视化效果见下图，很明显：Restormer重建结果更锐利清晰。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301261800517.jpg)



#### [√] Defocus Deblurring

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301261800518.jpg)

上图&表给出了去散焦模糊任务上的性能对比，从中可以看到：

- 无论是单帧还是双摄图像，所提方案大幅均优于其他方案；
- 在组合场景方面，相比此前最佳IFAN，所提方案取得了约0.6dB指标提升；相比Uformer，所提方案取得了1.01dB指标提升；
- 所提方案可以有效移除空间可变的散焦模糊。



#### [√] Image Denoising

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301261800519.jpg)

上表&图给出了不同图像降噪任务上的性能与效果对比，从中可以看到：

- Gaussian Denoising：所提方案在两种实验配置下均取得了SOTA性能。对于极具挑战性的50噪声水平的Urban100数据，Restormer取得了比DRUNet高0.37dB指标，比SwinIR高0.31dB。此外，相比SwinIR，Restormer计算量更少，速度更快。
- Real Denoising：所提方案是仅有的指标超过40dB的方案。相比此前最佳MIRNet与Uformer，所提方案分贝取得了0.3dB与0.25dB指标提升。
- Visual：从视觉效果上来看，所提方案可以重建更清晰的图像，同时具有更细粒度的纹理。

## [√] 消融实验

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301261800520.jpg)

> alec：
>
> - UNet，是一个多尺度模型？
> - 从消融实验的结果能够看出，使用MDTA+GDFN这个两个模块的模型，取得了最佳的效果。

上表给出了关于模块的消融实验，可以看到：

- 相比基线模型，MDTA可以带来0.32dB指标提升
- GDFN可以在MTA基础上取得0.26dB指标提升；
- GDFN与MDTA的组合可以取得了0.51dB指标提升。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301261800521.jpg)

从上表可以看到：

- 在提炼阶段添加Transformer可以进一步提升模型性能；
- Progressive学习机制可以取得更佳的指标，提升约0.07dB；
- 深而窄的模型比宽而浅的模型质保更高。











