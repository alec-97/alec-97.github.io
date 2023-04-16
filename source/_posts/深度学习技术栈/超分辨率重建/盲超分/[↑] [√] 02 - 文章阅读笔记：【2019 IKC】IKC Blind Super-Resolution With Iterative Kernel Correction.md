---
title: 文章阅读笔记：【2019 IKC】IKC Blind Super-Resolution With Iterative Kernel Correction
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212231015.jpg
tags:
  - 盲超分
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 盲超分
abbrlink: 101675316
date: 2023-02-20 14:37:40
---

> 原文链接：
>
> （1）【盲图像超分】IKC解析与深度思考 - AIWalker - Happy - [link](https://mp.weixin.qq.com/s/_yCvxzN2ryBQSIawHI1M1Q) - 2021-09-12 22:00
>
> ps：本文为依据个人日常阅读习惯，在原文的基础上记录阅读进度、记录个人想法和收获所写，关于原文一切内容的著作权全部归原作者所有。

# ——————————————————————————————

# [√] 文章信息

论文标题：IKC: Blind Super-Resolution With Iterative Kernel Correction

中文标题：用于盲超分的迭代内核矫正

论文链接：https://arxiv.org/abs/1904.03377

论文代码：https://github.com/yuanjunchai/IKC

论文发表：2019 CVPR

# ——————————————————————————————

# [√] 文章1

> 总结：
>
> 零碎点：
>
> 已有研究往往采用各项同性高斯模糊核，此外，各项异性模糊核(可视作运动模糊+各项同性模糊核的组合)也开始得到关注。
>
> 
>
> 文章思想：
>
> （1）IKC  ---> 用于模糊核估计
>
> 本文提出一种迭代核估计方法用于盲超分中的模糊核估计。 本文思想源自：核不匹配会导致有规律的伪影(过度退化或者过度模糊)，而这种规律可以用于对不精确的模糊核进行校正 。因此，我们提出一种迭代校正机制IKC，它可以取得比直接核估计更好的结果。
>
> （2）SFTMD  ---> 用于图像超分
>
> 与此同时，我们还提出一种基于SFT(Spatial Feature Transformer)的超分网络SFTMD用于处理多模糊核。
>
> 
>
> 本文贡献：
>
> - 提出一种直观且有效的深度学习框架用于模糊核估计；
> - 提出一种基于SFT的非盲超分模型用于多模糊核图像超分；
> - 所提SFTMD+IKC在盲超分领域取得了SOTA性能。
>
> 
>
> 
>
> 评价：
>
> 估计出来的模糊核不匹配，会导致有规律的伪影。因此本文提出使用迭代模糊核修正的方式，修正模糊核，从而实现较高性能的超分。
>
> 
>
> 
>
> 
>
> 
>
> 方法：
>
> （1）IKC：
>
> 为解决核不匹配问题，我们提出了迭代校正模糊核以得到无伪影超分结果。为校正估计模糊核k，我们构建了一个Corrector度量估计核与真实核之间的差异。核心思想在于：**利用中间超分结果进行模糊核校正** 。Corrector可以通过最小估计核与真实核之间的l2损失优化。
>
> 迭代模糊核修正的过程：
>
> - 所提IKC方案包含超分模型F、预测器P以及校正器C。
> - 首先将LR输入预测器，得到预测的模糊核h。
> - 然后将LR + 模糊核一起输入超分模型F，得到SR。
> - 然后将h和SR一起输入校正器C输出矫正之后的模糊核偏移delta h’。
> - 然后将delta h’ + h = h’，得到修正之后的h‘。
> - 然后再将h’和LR输入到F，得到SR。循环往复的执行这个过程，不断的修正模糊核。一共循环t次。
>
> ![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233735)
>
> （2）SFT
>
> SRMD通过将LR和模糊核叠加作为输入的方式，不一定是最好的方式，因为模糊核信息的影响仅在网络的浅层，深层特征难以受到这个核信息的影响。
>
> 为解决这个问题，本文提出了一种基于SFT的超分模型SFTMD，SFT通过对特征执行仿射变换提升模糊核的影响，该仿射变化并不是直接包含在图像处理图像中，因而可以提供更好的性能。
>
> “本文这里的SFT使用了2018CVPR的超分文章SFTGAN的模块，是否可以借鉴这种方式？”
>
> ![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233736.jpg)
>
> （3）预测器和校正器的结构图
>
> ![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233737.jpg)
>
> “可能确实能提升模型效果，但是这个预测器和校正器是否真的是在预测和校正，还是只是因为复杂了网络结构导致网络的拟合能力更强，也不一定。”
>
> 
>
> 个人思考
>
> ![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233738.jpg)
>
> 上图简单梳理了自SRMD以来用于多模糊核退化的图像超分方案，SRMD、DPSR、USRNet、DPIR以及MANet是Kai Zhang及其团队成员的工作，IKC则是Jinjin Gu、Chao Dong团队的成果，DANv1&DANv2是中科院Tieniu Tan团队的成果。
>
> 
>
> SRMD首次成功的将核先验、噪声先验信息嵌入到超分模型中 ；而后续的工作则针对模糊核的迭代估计进行探索，后续的工作延续了两条不同的路线：
>
> - 路线一：基于MAP思想进行迭代估计，像DPSR、USRNet以及DPIR采用了类似的思路，将传统方法MAP逐渐嵌入到迭代优化中；
> - 路线二：基于CNN进行迭代估计，像IKC、DANv1以及DANv2均采用了深度学习的思想进行模糊核的迭代优化。
>
> 
>
> 作为路线二的探索者，IKC以核不匹配造成的伪影 作为切入点，深入分析了估计核与真实核之间过渡时的现象，提出了模糊核迭代优化机制IKC。针对SRMD中核先验与LR图像的拼接处理方式可能存在弊端(核信息只影响一次、对深层难产生影响)，引入SFT以加深核先验的影响。
>
> 
>
> 当然，作为“吃螃蟹”的工作，它肯定会留下一些“坑”留给后来者去填。这些坑是啥呢？感兴趣的可以先去看一下DANv2，或者等待笔者的解读亦可。

【AI侃侃】知道IKC 一文有一年多，但一直没有深入看过论文，code也未曾仔细看过，潜意识中认为IKC太复杂了，所以一直拖、一直拖，直到看了DAN的两个版本，看到了DAN中附带了IKC的code，才觉得IKC可能是与DAN相类似的方法。趁着周末，花了近一天时间去看了IKC的原理以及code。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233739.jpg)

arXiv:https://arxiv.org/abs/1904.03377
code: https://github.com/yuanjunchai/IKC



## [√] Abstract

---

因其优异的有效性与高效率，深度学习已成为图像超分领域主流方案。现有图像超分方案往往假设下采样过程中的模糊核是固定/已知(比如bicubic)。然而，**实际应用场景中的退化模糊核往往是复杂且未知的**，进而导致已有方案在实际应用中的严重性能退化。

**本文提出一种迭代核估计方法用于盲超分中的模糊核估计。** 本文思想源自：**核不匹配会导致有规律的伪影(过度退化或者过度模糊)，而这种规律可以用于对不精确的模糊核进行校正** 。因此，我们提出一种迭代校正机制IKC，它可以取得比直接核估计更好的结果。与此同时，我们还提出一种基于SFT(Spatial Feature Transformer)的超分网络SFTMD用于处理多模糊核。

合成数据与真实场景上的实验表明：所提SFTMD+IKC可以生成视觉友好效果，同时在盲超分领域取得了SOTA性能。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233740.jpg)

本文主要贡献包含以下几点：

- 提出一种直观且有效的深度学习框架用于模糊核估计；
- 提出一种基于SFT的非盲超分模型用于多模糊核图像超分；
- 所提SFTMD+IKC在盲超分领域取得了SOTA性能。



## [√] Method

---

#### [√] Problem Formulation

---

盲图像超分问题可以描述如下：

![image-20230220160554627](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233741.png)

已有研究往往采用各项同性高斯模糊核，此外，各项异性模糊核(可视作运动模糊+各项同性模糊核的组合)也开始得到关注。为简单起见，**本文主要聚焦于各项同性模糊核** 。延续SRMD，我们采用了**高斯模糊+bicubic下采样** 退化方式。在真实场景中，LR图像往往还存在加性噪声退化。噪声假设同样延续了SRMD中的高斯分布。



#### [√] Motivation

---

接下来，我们将思考**正确模糊核在超分过程中的重要性** 。假设![image-20230220202935343](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233742.png)为带核信息输入的预训练超分模型，当输入正确模糊核，生成的超分图像不会存在伪影。**盲超分问题就等价于寻找合适的模糊核以使得超分模型生成视觉友好的结果**$I^{SR}$。一种直接的方案是采用预测器(Predictor)$k’=p(I^{LR})$直接从LR估计模糊核k，该预测器可通过最小化l2损失得到：

![image-20230220203054423](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233743.png)

然而，对模糊核k进行精确估计不太可能。此外，超分模型对于估计误差非常敏感，不精确的模糊核会导致生成的结果包含伪影。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233744.jpg)

![image-20230220203446703](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233745.png)

为解决核不匹配问题，我们提出了迭代校正模糊核以得到无伪影超分结果。为校正估计模糊核k，我们构建了一个Corrector度量估计核与真实核之间的差异。核心思想在于：**利用中间超分结果进行模糊核校正** 。Corrector可以通过最小估计核与真实核之间的l2损失优化：

![image-20230221204126873](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233746.png)

Corrector基于超分结果的特征对模糊核进行调整，调整后的模糊核又将优化超分模型以得到具有更少伪影的结果。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233747.jpg)

上图给出了迭代次数与性能的对比，可以看到：

- 仅仅一次校正的结果并不是非常好；
- 多次迭代可以有效提升PSNR/SSIM指标，直到达到饱和。

#### [√] Proposed Method

---

###### [√] Overall Framework

---

所提IKC方案包含超分模型F、预测器P以及校正器C。下图给出了IKC的实现伪代码。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233748.jpg)

> alec：
>
> - 所提IKC方案包含超分模型F、预测器P以及校正器C。
> - 首先将LR输入预测器，得到预测的模糊核h。
> - 然后将LR + 模糊核一起输入超分模型F，得到SR。
> - 然后将h和SR一起输入校正器C输出矫正之后的模糊核偏移delta h’。
> - 然后将delta h’ + h = h’，得到修正之后的h‘。
> - 然后再将h’和LR输入到F，得到SR。循环往复的执行这个过程，不断的修正模糊核。一共循环t次。

###### [√] Network Architecture of SR Model F

---

作为最成功的处理多模糊核退化的超分方案，SRMD将输入图像与退化信息拼接到一起作为模型输入，然后通过级联卷积与PixelShuffle进行图像超分。然而，SRMD中的拼接方式并非仅有的、也并非最优选择，原因有二：

- 核map并不包含图像信息，直接采用聚氨基对其处理可能会引入与图像无关的干扰；
- 核信息的影响仅在第一层得到了体验，深层特征难以收到该核信息的影响。

为解决上述问题，我们提出了一种基于SFT的超分模型SFTMD，SFT通过对特征执行仿射变换提升模糊核的影响，该仿射变化并不是直接包含在图像处理图像中，因而可以提供更好的性能。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233749.jpg)

上图给出了所提SFTMD架构示意图，它通过引入SFT对SRResNet进行扩展。SFT则基于模糊核特征H对于特征F进行仿射变换：

![image-20230221210642719](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233750.png)

注：仿射变换系数γ,β通过另一个轻量CNN计算得到。

###### [√] Network Architecture of Predictor p and Corrector c

---

 预测器与校正器的网络架构见下图。预测器由4个卷积层(后接LeakyReLU)+GAP组成；校正器则同时将超分图像与已有估计h作为输入。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233751.jpg)

> alec：
>
> - GAP = 全局池化
> - “可能确实能提升模型效果，但是这个预测器和校正器是否真的是在预测和校正，还是只是因为复杂了网络结构导致网络的拟合能力更强，也不一定。”

> alec：
>
> - stretch
>     - vt. 伸展,张开
>     - vi. 伸展
>     - adj. 可伸缩的
>     - n. 伸展，延伸
>
> 

## [√] Experiments

---

我们按照前述退化模型合成训练数据集，各项同性高斯模块的核宽分别为![image-20230221213225869](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233752.png)以对应x2、x3以及x4，核尺寸固定为21x21；当应用于真实图像时，我们添加了![image-20230221213248046](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233753.png)的加性高斯噪声。训练数据为DIV2K+Flickr2K。

为定量评估所提方案，我们还提供了一个测试集Gaussian8：它包含8个各项同性模糊核，核宽范围分别为![image-20230221213314252](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233754.png)。

SFTMD与IKC均在合成训练数据集上进行训练。首先，采用MSE训练SFTMD；然后，固定SFTMD参数，交替训练预测器与校正。

#### [√] Experiments of SFTMD

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233755.jpg)

上表对比了所提SFTMD与其他盲超分方案的性能，从中可以看到：

- 相比SRCNN-CAB与SRMD，所提SFTMD在所有配置与数据集上均取得了显著性能提升；
- 相比两个基于SRResNet的基线模型，所提SFTMD同样取得了最佳结果。

#### [√] Experiments on Synthetic Test Images

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233756.jpg)

上表给出了Gaussian8数据集上不同方案的性能对比，从中可以看到：

- 当退化核非bicubic时，在bicubic下采样退化下表现好的模型出现了严重的性能下降；
- 尽管无核校正的的方案已经取得了与现有方案相当的结果，但是，提升迭代次数仍可极大提升模型性能。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233757.jpg)

上图对比了模糊核迭代校正过程中的超分结果，可以看到：

- 直接采用预测器估计的模糊核生成的结果并不好，或者过于模糊或者存在振铃伪影；
- 随着迭代次数提升，PSNR指标逐渐提升，同时视觉效果也逐渐变好。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233758.jpg)

上表对所提方案的泛化性能进行了验证，从中可以看到：

- **所提IKC仍可保持其性能** ，说明IKC具有良好的泛化性；
- 移除PCA会造成性能下降，说明PCA有助于提升IKC的泛化性。

#### [√] Experiments on Real Image Set

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233759.jpg)

上图对比了不同方案在真实图像上的超分效果，可以看到：**尽管退化模糊核未知，IKC仍可生成无伪影、边缘锐利的超分结果** 。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233760.jpg)

上图提供了通过网格搜索优化模糊核+SRMD与IKC在Chip图像上的超分结果对比，从中可以看到：

- 尽管SRMD具有更锐利边缘、高对比度，但存在轻度伪影；
- IKC可以自动生成视觉友好的超分结果，尽管对比度稍低，但仍具有锐利而自然的边缘。



## [√] 个人思考

---

因为最近一年确实看过不少盲超分的paper，所以第一遍看完IKC后只感觉不过如此。在做笔记时，思考了IKC这一类方案的时间线时才真的意识到IKC的巧妙之处。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302212233761.jpg)

上图简单梳理了自SRMD以来用于多模糊核退化的图像超分方案，SRMD、DPSR、USRNet、DPIR以及MANet是Kai Zhang及其团队成员的工作，IKC则是Jinjin Gu、Chao Dong团队的成果，DANv1&DANv2是中科院Tieniu Tan团队的成果。

**SRMD首次成功的将核先验、噪声先验信息嵌入到超分模型中** ；而后续的工作则针对模糊核的迭代估计进行探索，后续的工作延续了两条不同的路线：

- 路线一：**基于MAP思想进行迭代估计**，像DPSR、USRNet以及DPIR采用了类似的思路，将传统方法MAP逐渐嵌入到迭代优化中；
- 路线二：**基于CNN进行迭代估计**，像IKC、DANv1以及DANv2均采用了深度学习的思想进行模糊核的迭代优化。

作为路线二的探索者，IKC以**核不匹配造成的伪影** 作为切入点，深入分析了估计核与真实核之间过渡时的现象，提出了模糊核迭代优化机制IKC。针对SRMD中核先验与LR图像的拼接处理方式可能存在弊端(核信息只影响一次、对深层难产生影响)，引入SFT以加深核先验的影响。

当然，作为“吃螃蟹”的工作，它肯定会留下一些“坑”留给后来者去填。这些坑是啥呢？感兴趣的可以先去看一下DANv2，或者等待笔者的解读亦可。







