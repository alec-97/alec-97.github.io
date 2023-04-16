---
title: >-
  文章阅读笔记：【2020 DAN_v1】Unfolding the Alternating Optimization for Blind Super
  Resolution
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251105553.png
tags:
  - 盲超分
password: 972274
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 盲超分
abbrlink: 2340942422
date: 2023-02-24 20:01:55
---

> 原文链接：
>
> （1）【√】DAN | 新型端到端交替优化盲超分方案（[link](https://mp.weixin.qq.com/s/1g5cCqDWTCj4p2C_RAe6dg)）
>
> 2022-09-19 08:00
>
> ps：本文为依据个人日常阅读习惯，在原文的基础上记录阅读进度、记录个人想法和收获所写，关于原文一切内容的著作权全部归原作者所有。



#  [√] 文章信息

---

论文标题：【2020 DAN】Unfolding the Alternating Optimization for Blind Super Resolution

中文标题：展开盲超分辨率的交替优化

论文链接：https://arxiv.org/abs/2010.02631

论文代码：https://github.com/greatlog/DAN

论文发表：NeurIPS 2020



# [√] 文章1

---

> 总结：
>
> 【本文思想】
>
> - 本文设计了两个模块，Restorer用于超分得到SR图像，Estimator用于利用SR图像估计模糊核。反复的交替这两个模块，可以形成一个端到端的网络。
> - 本文由于先前IKC论文的描述：但是，IKC 的问题在于：每次迭代过程的第1步只能利用有限的 LR 图像信息，难以对模糊核有一个高精度的预测。因此，尽管这2步可以很好地单独完成，但是当它们组合在一起时，最终结果可能不是最好的。盲超分的端到端交替优化方法 DAN 把这2步合并在一起, 在同一个模型中迭代地估计模糊核  和 利用模糊核  复原 SR 图。
> - 重建器利用估计的模糊核和LR得到SR。估计器利用SR得到模糊核。迭代这两个过程。其中因为SR是从LR而来的，因此估计器总的来说利用了来自LR和SR的信息，因此这使得对于模糊核K的估计更加的准确。
> - 不同于IKC中需要训练多个模型，本文只需要训练一个端到端的模型，且IKC中估计模糊核只用到了LR图像，没有利用到HR图像的信息。
> - DAN 模型的 Estimator 既输入 LR 图片, 又输入 SR 图片, 这使得对于模糊核  的估计更加容易。
>
> 【本文贡献】
>
> 【网络结构】
>
> ![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251108508.jpg)
>
> - DAN 模型的 Estimator 既输入 LR 图片, 又输入 SR 图片, 这使得对于模糊核  的估计更加容易。
> - Restorer 的输入是LR图像和估计得到的模糊核k，Estimator 的输入是LR图像和SR图像。
>
> ![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251108509.png)
>
> - 其中CRB称之为条件残差块，用于将基本输入和条件数据concat到一起。这里CRB的结构是卷积层+通道注意力+残差。
>
> {DANv2}
>
> ![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251108510.jpg)
>
> 【可以用于自己论文的话】
>
> - 双三次下采样的训练样本和真实图像之间存在一个域差。以双三次下采样为模糊核训练得到的网络在实际应用时，这种域差距将导致比较糟糕的性能。这种退化核未知的超分任务我们称之为盲超分任务 (Blind Super Resolution)。
>
> 【可以用于自己论文的idea】
>
> - 本文交替的估计模糊核和超分图像，其实这个思想和DegradationGAN的思想很像，只不过后者是直接估计的带有降质因子的LR图像，而这里估计的是模糊核。这个思想是否可以加上MANet中的自适应模糊核大小从而让估计出来的模糊核更加的准确。即使用MANet的模糊核估计方法，替换本文的模糊核估计器。二者做一个加法。
> - DAN的网络结构可以很好的用来参考。
> - 本文还有许多其它的文章，都是将矢量模糊核reshape成二维的，然后和LR图像concat之后进行超分。这里有个疑问，将矢量进行reshape的做法是否影响效果。是否可以换一种方式将矢量模糊核和LR图像进行融合？
> - 可以看看DANv2如何在DANv1的基础上改进，然后再看看自己可以在v2的基础上做些什么？
>
> 【问题记录】
>
> 【零碎点】
>
> - DAN = Deep Alternating Network = 深度迭代网络，即不断地迭代模糊核估计和超分这两个过程。



## [√] 导读

---

具体来说，我们设计了两个卷积神经模块，即 Restorer 和 Estimator。Restorer 基于预测核恢复 SR 图像，Estimator借助恢复的 SR 图像估计模糊核。我们反复交替这两个模块并展开这个过程以形成一个端到端的可训练网络。

**论文名称：** Unfolding the Alternating Optimization for Blind Super Resolution (NeurIPS 2020)

**拓展版本：** End-to-end Alternating Optimization for Blind Super Resolution (TPAMI)

**论文地址：**https://arxiv.org/pdf/2010.02631.pdf



## [√] 1.1 盲超分任务介绍

---

作为基本的 low-level 视觉问题，单图像超分辨率 (SISR) 越来越受到人们的关注。SISR 的目标是从其低分辨率观测中重建高分辨率图像。目前已经提出了基于深度学习的方法的多种网络架构和超分网络的训练策略来改善 SISR 的性能。顾名思义，SISR 任务需要两张图片，一张高分辨率的 HR 图和一张低分辨率的 LR 图。超分模型的目的是根据后者生成前者，而退化模型的目的是根据前者生成后者。经典超分任务 SISR 认为：**低分辨率的 LR 图是由高分辨率的 HR 图经过某种退化作用得到的，这种退化核预设为一个双三次下采样的模糊核 (downsampling blur kernel)。** 也就是说，这个下采样的模糊核是预先定义好的。但是，在实际应用中，这种退化作用十分复杂，不但表达式未知，而且难以简单建模。双三次下采样的训练样本和真实图像之间存在一个域差。以双三次下采样为模糊核训练得到的网络在实际应用时，这种域差距将导致比较糟糕的性能。这种**退化核未知的超分任务我们称之为盲超分任务 (Blind Super Resolution)**。

令x和y分别代表HR和LR图片, 退化模型为:

![image-20230224203644656](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251108511.png)

式中, $\otimes$ 代表卷积操作, 模型主要由3部分组成: 模糊核s, 下采样操作↓s和附加噪声n。前人工作中最广泛采用的模糊核是各向同性高斯模糊核 (Isotropic Gaussian Blur Kernel)。n一般为加性白高斯噪声 (Additive White Gaussian Noise, AWGN)。Blind SISR 任务就是从 LR 图片恢复 HR 图片的过程。

在[底层任务超详细解读 (一)：模糊核迭代校正盲超分方法 IKC](http://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247617444&idx=1&sn=2200d9cbf62daa1fc6e3fb34dc7678a7&chksm=ec1de65ddb6a6f4b0e4d53074e3b3d93ce59484aaeccf6bcba006f994e4eb30ca113c36bc2d9&scene=21#wechat_redirect) 中，我们介绍了一种模糊核迭代校正的盲超分方法 IKC。IKC 发现只有当我们预设的模糊核与图片真实的模糊核相差不大的时候，超分的结果才显得自然，没有伪影和模糊。因此，IKC 提出了一种退化核的迭代校正方法。它的每次迭代都可以分成2步：

第1步：从 LR 图片中估计模糊核k。

第2步：根据估计得到的模糊核k复原 SR 图片。

这样做的缺点是：第1步带来的微小偏差或者错误将会对第2步的结果带来较大的影响。所以，IKC 为了准确地估计模糊核k，设计一个校正函数c，它测量估计的模糊核 k和真值之间的差异。先训练好超分模型，之后迭代训练预测器和校正器若干次，得到模糊核的一个较为准确的估计。最后借助这个模糊核完成超分的任务。

## [√] 1.2 盲超分的端到端交替优化方法 DAN

---

但是，IKC 的问题在于：每次迭代过程的第1步只能利用有限的 LR 图像信息，难以对模糊核有一个高精度的预测。因此，尽管这2步可以很好地单独完成，但是当它们组合在一起时，最终结果可能不是最好的。

盲超分的端到端交替优化方法 DAN 把这2步合并在一起, 在同一个模型中**迭代地估计模糊核k和利用模糊核k复原 SR 图。** 具体而言, 作者设计了两个模块, 分别是 Restorer 和 Estimator。Restorer 可以根据 Estimator 估计得到的模糊核k复原 SR 图，而复原得到的 SR 图又进一步输入 Estimator 以更好地取估计模糊核k。

一旦模糊核k被初始化，这两个模块可以很好地相互协作, 形成一个闭环, 反复迭代优化。这个 迭代过程被展开成一个端到端的可训练网络, 称为 **Deep Alternating Network (DAN)**。通过这种 方式, Estimator 可以利用来自 LR 和 SR 图像的信息, 这使得模糊核 的估计更加容易。Restorer 和 Estimator 这两个模块可以在迭代优化中性能不断提升, 性能明显优于之前的2步独立 的盲超分方法。

盲超分任务的数学表达式是：

![image-20230224213221601](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251108512.png)

式中前一项是重建 HR 图片 \mathbf{x}\mathbf{x} ，后一项是 HR 图片的先验。很多盲超分方法将这个问题分解成两个连续的步骤：

![image-20230224213347843](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251108513.png)

式中, M(·)代表 LR 图片 y 来估计模糊核 k 的函数。这种两步法有3个主要的缺点：

- 这种方法一般需要训练好两个或者三个模型 (IKC), 模型越多就使得系统越复杂。
- 这种方法 M(·) 只能从 LR 图片 y 中提取信息, 也就是把模糊核 k 当做LR图片 y 的先验。但是事实上，若缺乏 HR 图片的信息, 模糊核 k 就没法很好地被估计。
- 这种方法的超分模型是通过 GT 模糊核 k 训练得到的。但是, 在实际测试或者使用的时候, 模糊核 k 是通过 M(·) 估计得到的, 显然不是真值。那这个差距就会导致盲超分模型的性能下降。

针对这些缺点，DAN 提出了一种端到端交替优化方法，DAN 仍然将其拆分为两个子问题，但不是按顺序求解，而是采用交替优化算法，交替恢复 SR 图像和估计相应的模糊核。数学表达式是：

![image-20230224215006958](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251108514.png)

上式4的这两步作者提供2个卷积神经网络来分别求解，这2个网络分别叫做 Estimator 和 Restorer，它们形成端到端的可训练网络，称为 **Deep Alternating Network (DAN)**。

## [√] 1.3 DAN 框架

---

如下图1所示是 DAN 的框架, 通过 Dirac function 初始化模糊核 , 即模糊核 的中心值为 1 , 其余值为 0 。遵循 IKC 的做法, 模糊核 依次进行 Reshape 操作和 PCA 降维。送入 Restorer 中进行图片的复原。之后, 两个模块 Estimator 和 Restorer 交替4轮, 并且在模型最后使用 L1 Loss 作为监督信息。由于两个模块的参数在不同的迭代之间共享, 所以整个网络可以很好地训练, 而对中间结果没有任何限制。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251108515.jpg)

另外值得注意的是, DAN 模型的 Estimator 既输入 LR 图片, 又输入 SR 图片, 这使得对于模糊核 的估计更加容易。更重要的是, Restorer 是用 Estimator 估计得到的模糊核 来训练的, 而不是像以前的方法那样用 GT 模糊核。因此, 在测试过程中, 恢复器可以更好地容忍估计器的估计误差。在 scale factor s = 1 的情况下, DAN 变成去模糊网络。





## [√] 1.4 Restorer 和 Estimator 的模型架构

Restorer 和 Estimator 都有两个输入。Estimator 以 LR 和 SR 图像为输入, Restorer 以 LR 图像和模糊核 k 为输入。二者都会以 LR 图像作为输入。那么若把 LR 图像定义为 Basic Input, 把另一个输入定义为 Conditional Input, 则 Estimator 的 Conditional Input 就是 SR 图像, 而 Restorer 的 Conditional Input 就是模糊核 k 。

我们可以发现，两个模块的 Basic Input 都始终是 LR 图片不变，但是两个模块的 Conditional Input 都在不断迭代中变化。而且另外值得注意的一点是，每个模块的 Output 都必须与 Input 密切相关。否则当一个模块的输出固定时，另一个模块的输出也是固定的，则不管迭代多少次，结果都没有变化，就没有迭代的意义了。为了确保 Restorer 和 Estimator 的输出与其条件输入密切相关，作者提出了一个 Conditional Residual Block (CRB)，它把 Basic Input 和 Conditional Input concat 起来：

![image-20230224220148720](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251108516.png)

式中, R(·)代表残差映射函数, 具体是 2 个 3x3 卷积加上一个 channel attention 模块组成, R(·)对于 Restorer 和 Estimator 而言都是一样的。

Restorer 和 Estimator 的详细模型架构如下图2所示。

**Estimator：** 输入的 LR 图像首先通过一个k=3,s=1的卷积层, 再送入 CRB 中。输入的 SR 图像首先通过一个 k=9,s=scale 的卷积层进行下采样, 得到和 LR 一样尺寸的图片大小, 再送入 CRB 中。在网络的末尾，作者通过 GAP 来压缩模糊核。在实际应用中, Estimator 有5个 CRB, 每个 CRB 的 Basic Input 和 Conditional Input 都有32个 channel。

**Restorer：** 输入的 LR 图像首先通过两个k=3,s=1的卷积层, 再送入 CRB 中。输入的模糊核 k 为首先通过 strech 操作进行拉伸, 得到和 LR 一样尺寸的图片大小, 再送入 CRB 中。在网络的末尾, 作者通过PixelShuffle 层和一个 k=3,s=1 的卷积层进行上采样。在实际应用中, Restorer 有40个 CRB, 每个 CRB 的 Basic Input 和 Conditional Input 分别有64和10个 channel。

![图2：DAN 模型的 Restorer 和 Estimator 的架构](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251108517.jpg)

## [√] 1.5 DAN 训练过程

---

作者收集了 DIV2K 和 Flickr2K 的3450张 HR 图像作为训练集。为了与其他方法进行公平的比较，使用两种不同的退化设置 (degradation settings) 来训练模型。

**Setting1：** 根据 IKC 的设置，kernel size k 设置为21，在训练期间，对于 scale factor 4,3,2，kernel width 分别在 [0.2, 4.0]，[0.2, 3.0] 和 [0.2, 2.0]中均匀采样。

测试集为 Set5 , Set14 , Urban100 , BSD100 和 Manga109。测试图像时，使用的是8个不同的核，即 Gaussian8。对于 scale factor 4,3,2，kernel width 分别在 [1.8, 3.2], [1.35, 2.40] 和 [0.80, 1.60]中均匀采样。HR 图像首先被选定的模糊核模糊，然后被下采样以形成合成测试图像。

**Setting2：** 根据 "Blind super-resolution kernel estimation using an internal-gan" 的设置，kernel size k 设置为11，首先生成各向异性高斯核。两个轴的长度均匀分布在 (0.6, 5) 中，旋转一个随机角度均匀分布在 [-π, π] 中。为了偏离规则高斯，作者进一步应用均匀乘法噪声 (高达核的每个像素值的25%)，并将其归一化。为了测试，使用了基准数据集 DIV2KRK。

对于所有的 scale factor，输入图片的尺寸都设置为64×64，batch size 设置为64，每个模型都训练 400000 个 iteration。



## [√] 1.6 DAN 实验结果

---

对于 Setting1，作者与 ZSSR，IKC 和 CARN 模型做了对比， YCbCr 空间 Y 通道上的 PSNR 和 SSIM 结果如下图3所示。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251108518.jpg)

尽管 CARN 模型在双三次下采样的情况下取得了非常不错的效果，但是当应用于具有未知模糊核的图像时，性能就会严重下降。当 CARN 模型后面接上去模糊方法时，它的性能得到了很大的改善，但是仍然不如盲超分辨方法。Manga109 scale ×3 条件下的 DAN 模型比 IKC 高了0.495个 dB。对于其他规模和数据集，DAN 也在很大程度上优于 IKC。

如下图4所示为 Urban100 的 img5 的可视化结果，模糊核的宽度是1.8。可以看到，CARN 和 ZSSR 甚至不能恢复窗户的边缘。IKC 表现更好，但边缘严重模糊。而 DAN 可以恢复锐利的边缘，产生更加视觉愉悦的效果。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251108519.jpg)

对于 Setting2，它涉及不规则模糊核，比较难解决。作者主要比较三种不同类别的方法：

1. 在双三次下采样图像上训练的 SOTA SR 算法，如 EDSR 和 RCAN。
2. NTIRE 比赛设计的 Blind SR 方法，如 PDN 和 WDSR。
3. 之前的两步法，即核估计方法和 SR 方法的组合，如 Kernel-GAN 和 ZSSR。

YCbCr 空间 Y 通道上的 PSNR 和 SSIM 结果如下图5所示。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251108520.jpg)

对于类别1的方法，即在双三次下采样图像上训练的方法的结果只比插值的结果稍好。对于类别2的方法，即在 NTIRE 比赛中获得优异成绩的方法，仍然不能很好地推广到不规则模糊核。类别3的方法带给我们很多启发。具体而言，USRNet 在提供 GT 核的情况下取得了显著的效果，KernelGAN 在核估计上也有很好的表现。然而，当它们组合在一起时，如图5所示，最终的 SR 结果比所有其他方法都要差。这表明模糊核估计的模型和图像复原的模型相互兼容是很重要的。对于 scale factor 为 ×2 和 ×4 的情况，DAN 的性能分别比 KernelGAN + ZSSR 的组合高2.20 dB 和 0.74 dB。

如下图6所示为 DIV2KRK 的 img892 的可视化结果。可以看到，KernelGAN + ZSSR 的组合 可以产生比插值稍微整形的边缘，但是它受到严重伪影的影响。DAN的 SR 图像显然要干净得多，细节也更可靠。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251108521.jpg)



#### [√] Real Images Set 实验结果

---

除了上述在合成测试图像上的实验之外，作者还在真实图像上进行了实验，以验证所提出的 DAN 模型的有效性。真实图像 (Real Images Set) 没有 GT 的 HR 图片，因此只提供视觉对比。对于真实图像，退化核是未知的且比较复杂，因此 Blind SISR 的性能会受到严重影响。为了简单起见，作者通过在训练期间将加性白高斯噪声 AWGN 添加到 LR 图像来重新训练模型。如下图7所示是不同方法的视觉对比结果。可以看到，KernelGAN + ZSSR 的结果略好于双三次插值，但仍然严重模糊。CARN 的结果是过度平滑，边缘不够锐利。IKC 产生了更干净的结果，但仍有一些伪影。相比之下，DAN 模型的结果要自然很多。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251108522.jpg)

## [√] 1.7 DAN v2 拓展版本

---

以上就是 NeurIPS 版本的 DAN v1 模型的介绍，之后作者有提出了 DAN 模型的拓展 TPAMI 版本 DAN v2。

#### [√] DAN v2 相比于 DAN v1，作了3处改动：

---

- 把 Estimator 和 Restorer 的基本架构 Conditional Residual Block (CRB) 换成了 Dual-Path Conditional Block (DPCB)。与原始条件残差块 (CRB) 相比，DPCB 的优势是能够同时处理 Basic Input 和 Conditional Input。DPCB 能够模拟两个输入之间更深的相关性，并有助于提高估计器和恢复器的性能。DPCB 中的 Dual-Path 摒弃了 CRB 的扩展和级联操作，节省了大量计算量。实验表明，DPCB 使整个网络加速28%。

- 当前版本 Estimator 是由完全模糊核而不是降维之后的模糊核训练的。一方面，更强的监督可能有助于估计器得到更好的优化。另一方面，完整的模糊核很容易用于其他任务，而简化的模糊核只能用于 Restorer。

- 当前版本补了一些实验和分析。

DAN v2 认为 Estimator 和 Restorer 的基本架构 Conditional Residual Block (CRB) 主要有3个缺点：

1. 在 Restorer 中，模糊核 需要拉伸成与 LR 图特征一样的大小之后再进行 Concat 的连接，这大大增加了计算的复杂度。
2. 实验表明，Conditional Residual Block (CRB) 的通道注意力层 (CALayer) 耗时长，容易导致梯度爆炸，降低推理速度，使训练不稳定。
3. 网络的所有 Block 的 Conditional Input 都是一样的，可能会影响模型的表达能力。



###### [√] Dual-Path Conditional Block (DPCB)

---

针对以上缺点，作者提出了一种 Dual-Path Conditional Block，如下图8所示。DPCB 中有两条路径，作者不把 Basic Input 和 Conditional Input concat 在一起，而是分开处理，并在最后相乘。而且两条路径都有残差连接， 它可以提高整个模块的表达能力，增强最终的效果。DAN v2 还去除了通道注意力层 (CALayer)，以加速推理和稳定训练。

![图8：DAN v2 模型的 Restorer 和 Estimator 的架构](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251108523.jpg)

###### [√] Dual-Path Conditional Group (DPCG)

---

把几个 DPCB 组合在一起得到 DPCG，作为 DAN v2 的基本模型单元。其中，两条路径有着不同的 kernel size 和 stride。

Restorer 和 Estimator 的详细模型架构如上图8所示。

**Estimator:** 输入的 LR 图像首先通过一个 k=3,s=1 的卷积层, 再送入 DPCG 中。输入的 SR 图像首先通过一个 k=9,s=scale 的卷积层进行下采样, 得到和 LR 一样尺寸的图片大小, 再 送入 DPCG 中。DPCB 中两条路径的卷积核大小都是 3x3的。在网络的末尾, 作者通过 GAP + Conv + Softmax 来压缩模糊核。在实际应用中, Estimator 有1个 DPCG, 每个 DPCG 包含5个 DPCB。每个 DPCB 的 Basic Input 和 Conditional Input 都有32个 channel。

值得一提的是，在 DAN v1 中，Estimator 的输出是 reduced kernel。这样即使最终的超分辨率结果足够好，我们也不知道模糊核是什么样子的。所以在 DAN v2 中，Estimator 的输出不再是 reduced kernel，而是完整的模糊核。

**Restorer：** 输入的 LR 图像和模糊核 k 为首先通过1个卷积层, 送入 DPCG 中。模糊核的路径一直保持 1x1 的空间尺寸, 这也节约了很多的计算。在网络的末尾, 作者通过PixelShuffle 层和一个 k=3,s=1 的卷积层进行上采样。在实际应用中, Restorer 有5个 DPCG, 每个 DPCG 包含 10 个 DPCB。每个 DPCB 的 Basic Input 和 Conditional Input 都有64个 channel。

DANv2 直接预测完整的模糊核的结果如下图9所示，这里作者对比了几种不同模型的结果。KernelGAN 估计的核可能是各向同性的，看起来与 GT 的模糊核非常不同。与这两种方法相比，DAN 可以更精确地估计各向异性的模糊核。

![图9：不同模型的模糊核预测结果](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251108524.jpg)

## [√] 总结

---

盲超分方法一般可以分为2步，即：从 LR 图片中估计模糊核 k 和根据估计得到的模糊核 k 复原 SR 图片。盲超分的端到端交替优化方法 DAN 把这2步合并在一起，在同一个模型中迭代地估计模糊核 k 和利用模糊核 k 复原 SR 图。具体而言，作者设计了两个模块，分别是 Restorer 和 Estimator。Restorer 可以根据 Estimator 估计得到的模糊核 k 复原 SR 图，而复原得到的 SR 图又进一步输入 Estimator 以更好地取估计模糊核 k 。

一旦模糊核 k 被初始化，这两个模块可以很好地相互协作，形成一个闭环，反复迭代优化。这个迭代过程被展开成一个端到端的可训练网络，称为 Deep Alternating Network (DAN)。通过这种方式，Estimator 可以利用来自 LR 和 SR 图像的信息，这使得模糊核 k 的估计更加容易。Restorer 和 Estimator 这两个模块可以在迭代优化中性能不断提升，性能明显优于之前的2步独立的盲超分方法。



















