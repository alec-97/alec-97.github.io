---
title: 055 - 文章阅读笔记：【DASR】盲超分两大流派集成大者！- 微信公众号 - AIWalker
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051406174.jpg
tags:
  - 人工智能
  - 深度学习
  - 盲超分
  - 图像处理
  - 超分辨率重建
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 266288220
date: 2023-02-04 18:26:49
---

> 原文链接：
>
> [DASR: 盲超分两大流派集成大者！ - 微信公众号 - AIWalker](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651688180&idx=1&sn=99eb47d0bde2320ac677e438b5f823d8&chksm=f3c9c2d9c4be4bcfb588c5d0cb6fbe19c441e7896ea41cd215174afac4403b1dede763432915&scene=178&cur_album_id=1338480951000727554#rd)
>
>  2022-03-29 22:00
>
> ps：本文为依据个人日常阅读习惯，在原文的基础上记录阅读进度、记录个人想法和收获所写，关于原文一切内容的著作权全部归原作者所有。

> 【本文要点】
>
> 对每个输入图像自适应估计其退化信息
>
> 点1：退化自适应预测网络 - 采用一个很小的回归网络预测输入图像的退化参数
>
> 点2：带多专家知识的超分网络 - 骨干部分采用了类似CondConv的"多专家"方案进行处理

## [√] 文章信息

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409765.jpg)

论文标题：DASR：Efficient and Degradation-Adaptive Network for Real-World Image Super-Resolution

中文标题：DASR：用于真实世界图像超分辨率的高效和退化自适应网络

论文链接：https://arxiv.org/abs/2203.14216

论文代码：https://github.com/csjliang/dasr

论文刊登：ECCV 2022

---

真实场景图像未知且复杂的退化机制、实际应用中的有限计算资源等问题导致高效且实用的RealSR仍是一个极具挑战性的任务。

尽管近期的一些RealSR通过关于退化空间的建模取得极大的进展(比如BSRGAN、Real-ESRGAN)，但这些方案严重依赖于**重骨干网络**(比如RRDB)，对于处理不同退化强度的图像不够灵活(内容自适应性不足)。

本文提出一种高效且实用的方案DASR，**它对每个输入图像自适应估计其退化信息并用于网络参数调制**。

具体来说，**它采用一个很小的回归网络预测输入图像的退化参数，同时骨干部分采用了类似CondConv的"多专家"方案进行处理**。

这种"多专家"与退化自适应联合优化能够大幅扩展模型容量以处理不同强度的退化，同时保持高效的推理。

实验结果表明：相比有方案，在处理不同退化强度的真实场景图像时，所提方案不仅更有效，同时部署更高效。

> alec：
>
> - RealSR的挑战：退化未知、计算资源有限
> - 显示建模的盲超分的不足：严重依赖于重骨干网络，处理不同退化程度的图像不够灵活



## [√] 1 Method

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409766.jpg)

上图为本文所提DASR整体架构示意图，它包含如下两部分：(1) 退化预测网络; (2)带多专家知识的超分网络。

> alec：
>
> - 理解：给退化预测网络输入LR图像，预测出降质特征向量，然后将该预测向量输入到多专家网络，获得专家知识。在SR的时候，给自适应SR网络分别输入LR图像和专家知识，然后获得SR。
> - 其中降质预测网络能够预测出降质分布，用于辅助SR过程。将降质分布向量输入到专家知识网络获得专家知识，用于辅助SR。因为不同的图像的降质类型是不同的，所以SR网络成为基于专家知识的自适应SR网络。
>
> ---
>
> - 现在的问题是：
>     - 如何预测降质
>     - 专家知识网络如何根据降质得到专家知识
>     - 专家知识是如何帮助SR过程的

#### [√] DASR

---

为达成高效且退化自适应图像超分，我们采用回归网络![image-20230204214830403](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409767.png)对每个输入x预测其退化参数![image-20230204214537976](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409768.png)。为使得该估计过程更高效，我们设计了一种轻量型网络：它包含6个后接LeakyReLU的卷积以及全局均值池化。在参数优化方面，我们引入了回归损失，定义如下：

![image-20230204214844682](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409769.png)

为达成高效且实用Real-ISR，我们提出了一种退化自适应超分网络，它通过MoE(Mixture of Experts)策略提升模型容量。具体来说，我们采用N个"卷积专家"(Convolutional Experts)，表示为![image-20230205130330199](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409770.png)，每个"专家"![image-20230205130340373](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409771.png)均是一个轻量型超分网络（比如EDSR-M）且具有相同的架构，同时采用相同的损失进行优化。**我们初衷是期望通过隐式训练使得每个专家能处理退化子空间的图像，然后通过多专家协同处理整个退化空间的图像**。

用于多专家融合的加权因子![image-20230205130705127](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409772.png)具有退化自适应特性，我们采用另一个小网络![image-20230205130716676](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409773.png)(它仅包含两个全连接层)对回归网络的预测![image-20230205130734332](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409774.png)进行处理得到：![image-20230205130745541](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409776.png)。

基于前述多专家$E$与自适应权值因子$a$，我们能够以非线性方式进行混合:

![image-20230205130915198](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409777.png)

也就是说，**我们可以对所有专家网络进行自适应融合并到一个自适应网络E~A~**。

需要注意的是：在经典的动态卷积(如DyConv、CondConv)中，每一层的权值因子需要通过一个独立的网络预测，这会导致不可忽视的计算量；而在这里，**我们对所有层学习相同的加权因子**，故更为高效。



> alec：
>
> - 退化自适应超分网络 = 降质预测 + MOE（专家混合知识）

#### [√] Degradation Modeling

---

由于不对齐因素，真实场景的高质量HR-LR数据对难以收集，而退化模型对Real-ISR的训练又是那么重要。

**我们设计了一个退化空间S进行训练数据对制作与退化自适应优化，该退化空间S由退化参数![image-20230205131754810](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409778.png)控制，![image-20230205131813145](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409779.png)表示退化类型或者退化参数**。注：在DASR方案中，$v$同样作为GT对退化预测网络进行训练优化。

在退化方面，我们引入了如下退化操作：

- 模糊：包含各项异性高斯模糊、各项同性高斯模糊；
- resizing：包含由area、bilinear以及bicubic构成的上采样和下采样；
- noise：包含加性高斯噪声、泊松噪声；
- JPEG压缩。

在退化参数$v$中，**我们采用对退化类型采用one-hot编码，对退化强度进行归一化处理编码**，比如，对于模糊退化参数，我们采用核尺寸s、标准差![image-20230205132057860](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409780.png)以及旋转角度θ等进行量化。

通过这种处理，退化参数更具可解释性。

此外，**受益于该可解释性与退化空间的紧致性(33个参数)，我们可以在推理阶段显式控制退化参数**。



尽管BSRGAN的退化置换与Real-ESRGAN的二阶退化可以构建一个非常大的退化空间，但它们**无法训练出一个能够自适应处理不同强度退化图像的超分模型**。

我们期望：**通过让不同专家处理不同的强度的退化输入，进而利用DASR的多专家混合机制使其能够处理非常宽泛的真实输入**。

为此，我们通过指定退化参数将整个退化空间拆分为3个层次![image-20230205132445429](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409781.png)，![image-20230205132454891](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409782.png)分别表示小参数范围和大参数范围的一阶退化，![image-20230205132503486](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409783.png)表示二阶退化。更详细的退化参数信息见下表。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409784.jpg)

> alec：
>
> - 二阶退化即两阶段退化，先退化一次、然后再退化一次

#### [√] Training Losses

---

所提DASR包含三个子网络[$E,P,A$]，前面提到的回归损失用于对$P$进行优化，为优化整体架构，我们构建了如下损失：

![image-20230205133801971](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409785.png)

注：在本文实验中，![image-20230205133815936](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409786.png)。

## [√] 2 Experiments

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409787.jpg)

需要注意的是，本文所设计的退化空间比BSRGAN、Real-ESRGAN的多样性还要强，既包含轻度的退化，也包含重度退化。上图给出了不同程度的退化示意图(更多示意图可查看原文)，包含最基本的Bicubic退化，也包含轻度一阶退化，还包含重度一阶退化与重度二阶退化。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409788.jpg)

上表给出了不同方案在不同退化强度下的性能对比，从中可以看到：

- **已有方案仅能处理特定类型的退化**，如RRDB、ESRGAN只能处理bicubic退化；Real-ESRGAN、BSRGAN、Real-SwinIR对于重度退化数据效果很好，但在轻度退化数据上的效果出现了严重下降。
- 相比其他方案，**在三种类型退化数据上，所提DASR取得稳定且显著性能提升**。这说明，**所提DASR对不同的退化具有极强的泛化性能**。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409789.jpg)

为验证所提退化自适应策略的有效性，我们基于所构建退化空间对不同骨干进行了重训练，结果见上表，可以看到：

- **当采用相同骨干SRResNet时，所提方案取得大幅性能提升**，比如bicubic退化数据上的0.5dB、Level-II退化数据上的5%LPIPS。这说明：退化自适应混合机制可以极大扩展模型容量，同时保持推理高效性。
- **相比RRDB与SwinIR，所提DASR所需计算资源更少**，约为RRDB的1/3, SwinRI的1/12；同时，DASR具有更优的重建质量。这进一步证实了其退化自适应的有效性与部署高效性。
- 在推理效率方面，RRDB比SRResNet的FLOPs高7倍，推理速度慢4倍；SwinIR具有可接受的FLOPs与Params，但其实际推理速度反而最慢(attention与IO的缘故)。相反，**所提DASR具有超优异性能的同时具有非常快的推理速度、低FLOPs与参数量**。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409790.jpg)

上图给出了不同方案在不同退化下的重建效果对比，从中可以看到：**DASR能够更为稳定的重建锐利而真实的纹理**；而BSRGAN与Real-ESRGAN对于轻度退化数据无法生成令人满意的纹理细节。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051409791.jpg)

前面也提到了，DASR一个有意义的优势在于：**受益于可解释性与退化表达的紧致性，在推理阶段，所提DASR支持用户交互**。上图给出了一个用户交互调制效果对比，可以看到：**当把模糊相关的参数手动调大后，重建结果更为锐利(见Fig5-c)；当对噪声相关的参数进行调制时可以更灵活的处理细节与噪声之间的均衡(见Fig5-e，Fig5-f)**。



## [√] 3 个人理解

---

- 这篇论文可谓把盲图像超分的两个流派(IKC流派显示进行退化核建模和BSRGAN流派进行隐式建模)给打通，兼具两大盲超分流派的优势：
    - IKC流派的数据自适应性
    - BSRGAN流派的超大退化核空间。

- 此外，DASR还具备两者所不具备的退化信息人工交互调制性，而这这种退化核支持交互调制的方案最早可追溯到SRMD一文。

- 此外，SRMD、DASR这种支持人工交互调整超分结果的机制使其在图像编辑领域极具价值；同时对于某些特定无HR数据的领域，该方案非常适合用于制作数据。

- 总而言之，这篇论文非常值得称道，将BSRGAN类方案的重骨干做成了轻量型，为进一步的轻量型盲超分探索打开了一扇门。





## [√] 4 推荐阅读

---

1. [CVPR2021|超分性能不变，计算量降低50%，董超等人提出用于low-level加速的ClassSR](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651683607&idx=1&sn=2df5fd9f30b52321fa7a8786f7ba8b35&scene=21#wechat_redirect)
2. [让Dropout在图像超分领域重焕光彩！](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651687062&idx=1&sn=98cdf4f017c15f94b71db69b5ad64810&scene=21#wechat_redirect)
3. [图像增强领域大突破！以1.66ms的速度处理4K图像，港理工提出图像自适应的3DLUT](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651682108&idx=1&sn=a29346042bf42730a48901bac00ef317&scene=21#wechat_redirect)
4. [ETH开源业内首个广义盲图像超分退化模型，性能效果绝佳](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651683889&idx=1&sn=a602e3cdb44ea2901dd506630b5e3b70&scene=21#wechat_redirect)
5. [ICCV2021 FBCNN: 超灵活且强度可控的盲压缩伪影移除新思路](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651686061&idx=1&sn=3b2ca0b46d95a1d393449b8db8161581&scene=21#wechat_redirect)





