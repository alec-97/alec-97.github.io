---
title: >-
  文章阅读笔记：【2021 MANet】Mutual Affine Network for Spatially Variant Kernel
  Estimationin Blind Image Super-Resolution
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241432069.jpg
tags:
  - 盲超分
password: 972274
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 盲超分
abbrlink: 997647599
date: 2023-02-23 22:33:12
---

> 原文链接：
>
> （1）【√】ICCV2021 盲图像超分 MANet：ETH团队提出空间可变模糊核估计新思路（[link](https://zhuanlan.zhihu.com/p/413939350)）
>
> 发布于 2021-09-25 20:29
>
> ps：本文为依据个人日常阅读习惯，在原文的基础上记录阅读进度、记录个人想法和收获所写，关于原文一切内容的著作权全部归原作者所有。



#  [√] 文章信息

---

论文标题：【2021 MANet】Mutual Affine Network for Spatially Variant Kernel Estimationin Blind Image Super-Resolution

中文标题：基于互仿射网络进行空间可变模糊核估计用于盲图像超分

论文链接：https://arxiv.org/abs/2108.05302

论文代码：https://github.com/JingyunLiang/MANet

论文发表：ICCV2021





# [√] 文章1

---

> 总结：
>
> 【文章思想】
>
> - 本文提出了一种卷积方式：MAConv，这种卷积方式能够和非盲图像超分方案组合，组合之后能够将盲超分的超分性能提升。
> - 本文认为现有的盲超分方案，往往认为模糊核是空间不变的。但是真实的模糊核由于目标运动、虚焦等因素影响是空间可变的。这就导致现有的盲超分方案的性能非常有限，甚至非常差。因此本文提出MANet用于空间可变的模糊核估计。
> - KernelGAN的方式是提取的空间不变的模糊核，无法对小图像块进行有效的估计。本文提出了直接从图像块来估计模糊核。
>
> 【本文贡献】
>
> - 提出名为MANet的核估计框架，能够在LR图像上估计模糊核。
> - 提出MAConv的卷积方式，这种卷积方式通过探索通道相关性来增强特征表达能力，非常适合用于模糊核特征提取；同时相比卷积层能够降低30%的参数量和计算量。
>
> 【网络结构】
>
> {核估计网络}
>
> ![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241437472.jpg)
>
> - 现在的神经网络为了保证感受野大，需要堆叠较深的网络。但是对于空间可变的核估计，因为要估计局部的模糊核，因此要保持退化的局部性。提出了带适度感受野的MANet。
> - 该网络的特征提取部分是一种类似于UNet的结构。
> - 通过将通道split，然后计算仿射变换，能够探索不同的通道之间的相互关系。
>
> {超分网络}
>
> ![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241437473.jpg)
>
> 【可用于自己的论文的话】
>
> - 本文提出的MAConv卷积层，可以提升特征表达能力，且不会造成感受野、模型大小以及计算复杂度的提升。
>
> 【可以用于自己论文的idea】
>
> - 是否可以将这个MAConv用到自己的模型中、然后稍作修改，但是自己的模型架构不能和本文的相同。可以起名为自适应降质卷积。
> - 这个降质公式![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241437474.png)只适用于模糊核是空间不变的情况。当模糊核具有空间可变性之后，问题的“病态性”变得非常严重。
> - 一般来讲，小感受野意味着小网络、弱表达能力。一种可能的方案是提升通道数量，但这会带来指数级的参数量与计算量提升。为解决该问题，我们提出了MAConv，见下图。
> - 本文的MAConv中用到了仿射变化模块，可以计算不同通道之间的相互关系，从而能够提升表达能力。这个仿射变化的过程有点像注意力计算。因此自己是不是可以根据这个结构来设计一种注意力机制，然后放在与这个类似的模块中，起名字叫某某某卷积。同时结构也是先将特征沿着空间维度均分，然后再做注意力机制。从而使得能够节省参数量和计算量。计算完每一份的相关关系之后，是否可以再叠一个通道注意力，来探索通道维度的优先级。（论证分析方式可以参考：MAConv通过互仿射变换探索了不同通道之间的相互关系，这种设计可以有提升特征表达能力，同时极大降低模型大小与计算复杂度。下表对比了卷积、组卷积以及MAConv在参数量、内存占用、FLOPs以及推理耗时方面的对比。注：由于仿射变换不会提升感受野，MAConv的感受野仍为3×3；而稠密与SE模块会导致感受野极大提升而不适合于核估计。）
> - 或者是将这里的仿射变换用到自己的模块中。
> - 这个自适应的核估计是一个很好的idea，能够在图像中不同区域探索不同的模糊核。
> - 或者可以将这种核估计方案结合其它的盲超分方法，比如结合高低通滤波器。
> - ![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241437475.jpg)
>
> 【问题记录】
>
> - 本文估计出来的模糊核，是怎么用到超分任务中的？
>
> - - 答：结合方式是在超分网络中利用SFT分散的在网络中添加kernel



## [√] 概述

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241437476.jpg)

arXiv [https://arxiv.org/pdf/2108.05302.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2108.05302.pdf)

code:[https://github.com/JingyunLiang](https://link.zhihu.com/?target=https%3A//github.com/JingyunLiang/MANet)

这篇文章是ETH团队在盲图像超分之**空间可变模糊核估计** 方面的工作，已被ICCV2021接收。针对实际应用场景中模糊核的空间可变性，提出一种新的空间可变模糊核估计方案MANet。从退化的局部性角度发出，对现有方案的局限性进行了分析，同时提出MAConv解决小模型的弱表达能力问题。相比已有模糊核估计方案，所提方案取得了显著性能提升；当与非盲图像超分方案组合后，将盲图像超分性能推到了新的高度。

## [√] Abstract

---

现有盲图像超分往往假设模糊核具有空间不变性，然而这种假设在真实图像中很好碰到：**真实图像中的模糊核由于目标运动、虚焦等因素通常是空间可变的** 。因此，现有盲超分方案在实际应用中的性能非常非常有限，甚至导致比较差的效果。

为解决上述问题，本文提出MANet(Mutual Affine Network)用于空间可变模糊核估计。具体来说，MANet具有两个固有特性：

- 它具有适度的感受野以确保退化的局部性；
- 它包含一个新的MAConv(Mutual Affine Convolution)层，该层通过提升特征表达能力且不会造成感受野、模型大小以及计算复杂度的提升。

合成数据与真实数据上的实验表明：MANet不仅优于空间可变与不变核估计，同时当与非常盲超组合后将盲超分性能提升到了新的高度。

## [√] Contribution

---

本文主要贡献包含以下几个方面：

- 提出一种称之为MANet的核估计框架。通过适度感受野(22×22)，它可以较小LR图像块上估计模糊核，其可精确估计的核尺寸为9×9。
- 提出MAConv通过探索通道相关性增强特征表达能力，且不会提升感受野，这使得其非常适合与模糊核特征提取。相比卷积层，它可以降低约30%参数量与计算量。
- 相比现有方案，MANet优于空间可变与不变核估计方案，当与非盲超分方案组合后取得了SOTA盲超分性能。在处理不同类型的图像块时，MANet表现出了非常好的属性：对于非平台块可以精确估计模糊核，对于平坦区域可以生成固定核。











## [√] Method

---

#### [√] Problem Formulation

---

LR图像I\^LR通过对HR图像I\^HR执行退化模型得到，当模糊核为空间不变类型时，两者之间的关系可以描述如下:

![image-20230224124950040](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241437477.png)

对于盲超分来说，HR图像I\^HR与模糊核k均是未知的。由于有很多对HR图像I\^HR与模糊核k可以生成相同的LR图像I\^LR，该问题是一种“病态(ill-posed)”问题。当模糊核具有空间可变性后，问题的“病态性”变得更为严重。



> alec：
>
> - formula，公式
> - forum，论坛

#### [√] Proposed Method

---

采用不同核模糊的图像块具有不同的分布特性。KernelGAN通过GAN方案对该属性进行了探索，然而它仅适用于空间不变核估计，对于小图像块无法进行有效核估计。再向前走一步，我们提出了直接从图像块估计模糊核。

###### [√] Overall Framework

---

现代神经网路通常堆叠多个层以构建具有更大感受野的深度模型。然而，对于空间可变核估计任务，我们需要保持退化的局部性。因此，我们提出带适度感受野的MANet。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241437478.jpg)

上图给出了MANet架构示意图，它包含特征提取与核重建两个模块。特征提取模块是一种类似UNet架构，由卷积、残差模块以及上/下采样构成；核重建模块由卷积、Softmax以及最近邻插值构成。预测得到的模糊核表示为K∈R\^hw×H×W。基于上述架构设计，MANet的感受野为22×22。

###### [√] Mutual Affine Convolution

---

一般来讲，小感受野意味着小网络、弱表达能力。一种可能的方案是提升通道数量，但这会带来指数级的参数量与计算量提升。为解决该问题，我们提出了MAConv，见下图。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241437479.jpg)

![image-20230224135405606](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241437480.png)

MAConv通过互仿射变换探索了不同通道之间的相互关系，这种设计可以有提升特征表达能力，同时极大降低模型大小与计算复杂度。下表对比了卷积、组卷积以及MAConv在参数量、内存占用、FLOPs以及推理耗时方面的对比。注：由于仿射变换不会提升感受野，MAConv的感受野仍为3×3；而稠密与SE模块会导致感受野极大提升而不适合于核估计。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241437481.jpg)

此外，从上表还可以看到：

- MAConv在LR图像上取得了最佳PSNR/SSIM指标，这说明所生成的模糊核可以更好的保持数据一致性；
- 提升通道数，MAConv的性能可以进一步提升，但同时也带来了参数量与FLOPs提升；
- 提升MAConv的split数同样会带来性能提升，这说明更多的split可以更好的探索通道相关性、提升特征表达能力。为平衡精度与推理耗时，我们将通道数与split数分别设置为[128,256,128],2。

> alec：
>
> 【仿射变换】
>
> **仿射变换**，又称**仿射映射**，是指在[几何](https://baike.baidu.com/item/几何/303227?fromModule=lemma_inlink)中，一个[向量空间](https://baike.baidu.com/item/向量空间/5936597?fromModule=lemma_inlink)进行一次[线性变换](https://baike.baidu.com/item/线性变换/5904192?fromModule=lemma_inlink)并接上一个[平移](https://baike.baidu.com/item/平移/2376933?fromModule=lemma_inlink)，变换为另一个向量空间。 [1] 
>
> 仿射变换是在几何上定义为两个[向量空间](https://baike.baidu.com/item/向量空间/5936597?fromModule=lemma_inlink)之间的一个仿射变换或者仿射[映射](https://baike.baidu.com/item/映射/410062?fromModule=lemma_inlink)（来自拉丁语，affine，“和…相关”）由一个非奇异的线性变换(运用一次函数进行的变换)接上一个平移变换组成。
>
> 在有限维的情况，每个仿射变换可以由一个矩阵A和一个向量b给出，它可以写作A和一个附加的列b。一个仿射变换对应于一个矩阵和一个向量的乘法，而仿射变换的复合对应于普通的[矩阵乘法](https://baike.baidu.com/item/矩阵乘法/5446029?fromModule=lemma_inlink)，只要加入一个额外的行到矩阵的底下，这一行全部是0除了最右边是一个1，而列向量的底下要加上一个1。

###### [√] Loss Function

---

在损失函数方面，我们采用了简单的MAE(即L1损失)：

![image-20230224140012824](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241437482.png)

## [√] Experiments

---

#### [√] Kernel Estimation

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241437483.jpg)

上图对比了测试图像上的核估计结果，可以看到：对于非平坦区域，MANet可以精确估计模糊核；而对于平坦区域，MANet倾向于生成固定核。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241437484.jpg)

上图给合成图像上的核估计对比，可以看到：MANet可以从9×9图像块上精确估计模糊核，当块尺寸提升后性能进一步提升。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241437485.jpg)

在真实应用场景，图像还可能存在噪声与压缩伪影。为测试在更复杂场景下的核估计性能，我们在训练过程中添加高斯与JPEG压缩噪声并在不同噪声水平下进行测试，参见上表。从表中可以看到：相比无噪情况，尽管出现了性能下降，但LR图像的PSNR范围仍为40.59-45.45dB，这无疑说明了所提方案在重度噪声干扰下的核估计性能。

#### [√] Spatially Variant SR

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241437486.jpg)

上表对比了不同盲超分方案的性能，从中可以看到：

- 对比不同类型的空间可变核，所提MANet均取得了最佳性能；
- 当模糊核出现差异后，极具代表的BicubicSR模型RCAN与HAN出现了严重性能下降；
- 类似地，DIP也难以生成令人满意的记过，因其模糊核是固定的；
- 通过逐块优化核，SRSVD可以处理空间可变SR问题，但无疑会极大提升运行耗时；
- IKC能够生成比其他方案更好的结果，但它对每个图像仅估计一个核，这无疑限制了其性能；
- MANet在每个位置预测一个核，因此它可以处理空间可变退化问题，取得了大幅优于IKC的性能

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241437487.jpg)

上图给出了几种方案在空间可变核与真实场景数据上的视觉效果对比，从中可以看到：**MANet可以生成具有最佳视觉效果的结果** ，而其他方案要么存在过度模糊，要么存在过度锐化问题。

在推理速度与内存占用方面，所提MANet仅需0.2s与0.3GB显存占用(Tesla V100 GPU)；相反，KernelGAN需要93s，占用1.3GB显存；IKC需要15.2s，占用2.0GB显存。

#### [√] Spatially Invariant SR

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241437488.jpg)

上表对比了空间不变超分方案的性能，从中可以看到：

- **在不同数据集、不同超分倍率下，所提MANet均取得了最佳性能** 。
- 尽管KernelGAN可以从LR图像估计模糊核，但其性能与HAN、DIP相近；
- IKC具有比其他方案更优的性能，但仍弱于所提MANet。





#### [√] Ablation Study

---

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241437489.jpg)



###### [√] Different Numbers of MAConv Layers

---

上图a与b对比了MAConv层数变化的影响，此时感受野从22×22提升到了38×38。从对比可以看到：**小感受野的MANet可以更精确的估计模糊核，而大感受野反而无法精确估计** 。这与我们的分析相一致：当模型具有大感受野时，它会将远离中心的像素纳入到核估计过程，造成核估计性能下降。**大感受野并非空间可变核估计的期望属性。**

###### [√] Kernel Loss vs LR Image Loss

---

上图a与c对比了两种损失的影响，从对比可以看到：**当采用KernelLoss训练时，MANet可以成功的进行模糊核估计；而采用ImageLoss训练时，MANet则无法进行有效估计。**



## [√] 内容补充

---

上面的内容主要针对模糊核估计MANet进行介绍，那么超分网络是什么样的呢？文章也只提到是RRDB的SFT变种，笔者从补充材料中找到了网络结构的图示，见下图。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241437490.jpg)

从上图可以看到：模糊核部分延续了SRMD的作风，采用PCA对估计到的模糊核进行降维；但在与超分网络结构方面则使用了SFT机制，而非SRMD中的处理机制。下图为SRMD的结构示意图，可以看到：模糊核与图像的结合仅在网络开头部分进行了一次结合。后来的一些可调制图像复原方案大多采用了多阶段融合思想，比如CFSNet、CResMD等。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302241437491.jpg)









