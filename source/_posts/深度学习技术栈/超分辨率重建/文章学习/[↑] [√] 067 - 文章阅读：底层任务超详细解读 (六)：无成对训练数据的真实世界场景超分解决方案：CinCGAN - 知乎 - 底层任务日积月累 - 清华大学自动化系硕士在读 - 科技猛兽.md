---
title: >-
  067 - 文章阅读：底层任务超详细解读 (六)：无成对训练数据的真实世界场景超分解决方案：CinCGAN - 知乎 - 底层任务日积月累 -
  清华大学自动化系硕士在读 - 科技猛兽
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132245361.jpg
tags:
  - 深度学习
  - 超分辨率重建
  - 盲超分
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 776310902
date: 2023-02-13 14:53:34
---

> 原文链接：
>
> [底层任务超详细解读 (六)：无成对训练数据的真实世界场景超分解决方案：CinCGAN - 知乎 - 底层任务日积月累 - 清华大学自动化系硕士在读 - 科技猛兽](https://zhuanlan.zhihu.com/p/471483181)
>
> 发布于 2022-09-29 16:33
>
> ps：本文为依据个人日常阅读习惯，在原文的基础上记录阅读进度、记录个人想法和收获所写，关于原文一切内容的著作权全部归原作者所有。
>
> - [x] 整理

> alec：
>
> ![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249346.png)
>
> 【CinC-GAN1】
>
> 这个第一个CycleGAN的目的是生成不含噪声的LR图像，然后再通过第二个CycleGAN进行超分。
>
> （和Degradation的不同之处在于，后者是生成大量含有真实噪声的图像，从而让超分模型能够拟合降噪真实噪声的能力。而这里则是将LR图像去掉真实噪声，然后再超分。后者是希望SR模型能够降噪，而这里的CycleGAN则是希望G1先降噪，然后再超分。）
>
> 
>
> 【CinC-GAN2】
>
> （1）GAN loss：
>
> 用来判别得到的HR图像是否是真实的还是伪造的
>
> （2）Cycle Consistency Loss：
>
> CycleGAN 的 **Cycle Consistency Loss** 保证不成对数据也能有效训练：即LR变成SR，然后再变成LR'，这样LR'和LR做监督，进行训练。
>
> （3）Identity Loss：
>
> 这个损失用来训练SR模型，训练方式是将干净的HR图像z，通过双三次插值得到z'，然后用配对的Z和Z'来进行预定义的配对的训练。用来使得SR模型具有合理的SR能力。
>
> （4）Total Variation (TV) Loss：
>
> 对超分得到的HR进行计算，帮助增加空间平滑度。
>
>  
>
> 真实世界场景由于下采样和退化函数复杂且相互耦合，因此很难像传统的盲超分那样进行精确估计。而且真实世界场景 HR 图片是未知的，没有 HR 图片作为训练集。我们就没法构造真实世界的 HR-LR 数据对进行有监督训练。利用循环一致性的特性，作者提出了一种 Cycle-in-Cycle GAN (CinCGAN) 来解决无成对训练数据的真实世界场景超分问题。整个方法包含两套 CycleGAN，第一套把输入的带有未知且复杂的退化作用的 LR 图片转化为干净的 LR 图片；第二套把输入的干净的 LR 图片用 CycleGAN 的方式超分成干净的 HR 图片。作者的方法实现了与最先进的有监督 CNN 的算法相当的性能。

## [√] 本文目录

---

**6 无成对训练数据的真实世界场景超分解决方案：CinCGAN**
6.1 真实世界场景超分任务介绍
6.2 使用不成对的数据训练盲超分模型
6.3 CinCGAN 方法训练数据
6.4 第一套 CycleGAN
6.5 第二套 CycleGAN
6.6 模型架构
6.7 CinCGAN 训练过程
6.8 CinCGAN 实验结果







## [√] 文章信息

---

- 中文标题：无成对训练数据的真实世界场景超分解决方案：CinCGAN
- 论文名称：Unsupervised Image Super-Resolution using Cycle-in-Cycle Generative Adversarial Networks (CVPRW 2018)
- 论文地址：https://arxiv.org/abs/1809.00437
- 论文代码：https://github.com/sangyun884/CinCGAN-pytorch（非官方）



## [√] 1.真实世界场景超分任务介绍

---

作为基本的 low-level 视觉问题，单图像超分辨率 (SISR) 越来越受到人们的关注。SISR 的目标是从其低分辨率观测中重建高分辨率目前已经提出了基于深度学习的方法的多种网络架构和超分网络的训练策略来改善 SISR 的性能。顾名思义，SISR 任务需要两张图片，一张高分辨率的 HR 图和一张低分辨率的 LR 图。超分模型的目的是根据后者生成前者，而退化模型的目的是根据前者生成后者。经典超分任务 SISR 认为：**低分辨率的 LR 图是由高分辨率的 HR 图经过某种退化作用得到的，这种退化核预设为一个双三次下采样的模糊核 (downsampling blur kernel)。**也就是说，这个下采样的模糊核是预先定义好的。但是，在实际应用中，这种退化作用十分复杂，不但表达式未知，而且难以简单建模。 双三次下采样的训练样本和真实图像之间存在一个域差。以双三次下采样为模糊核训练得到的网络在实际应用时，这种域差距将导致比较糟糕的性能。这种**退化核未知的超分任务我们称之为盲超分任务 (Blind Super Resolution)**。

盲超分任务的退化方式有很多种，比如**模糊 (blur)**，**噪声 (noise)** 或者**下采样 (downsampling)** 等。令 x 和 z 分别代表 HR 和 LR 图片，退化模型为：

![image-20230213155034019](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249347.png)

![image-20230213155059066](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249348.png)

![image-20230213155653795](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249349.png)



## [√] 2.使用不成对的数据训练盲超分模型

---

![image-20230213160100547](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249350.png)

缺点是，DualGAN 和 CycleGAN 都处理相同大小的输入和输出图像，而超分任务要求输出图像比输入图像大几倍。利用循环一致性的特性，作者提出了一种 Cycle-in-Cycle GAN (CinCGAN) 来解决无成对训练数据的真实世界场景超分问题。作者的方法实现了与最先进的有监督 CNN 的算法相当的性能。

**问1：为什么要使用无监督方式训练真实世界超分模型？**

**答：**由于下采样和退化函数复杂且相互耦合，因此很难像传统的盲超分那样进行精确估计。而且真实世界场景 HR 图片是未知的，没有 HR 图片作为训练集。我们就没法构造真实世界的 HR-LR 数据对进行有监督训练。

**问2：SR 任务和图像翻译任务有什么区别？**

**答：** SR 接受 LR 图像并输出分辨率高得多的 HR 图像。 此外，SR 要求输出是高质量的，而不仅仅是不同的风格。如果我们直接应用图像到图像的转换方法，我们需要首先通过插值对 LR 图像进行上采样得到高分辨率的图像，但这也将放大噪声。 直接套用 CycleGAN 之类的现有方法无法去除这种被放大的噪声，训练变得非常不稳定。

> alec：
>
> - 作者的方法实现了与最先进的有监督 CNN 的算法相当的性能。
> - 缺点是，DualGAN 和 CycleGAN 都处理相同大小的输入和输出图像，而超分任务要求输出图像比输入图像大几倍。利用循环一致性的特性，作者提出了一种 Cycle-in-Cycle GAN (CinCGAN) 来解决无成对训练数据的真实世界场景超分问题。

## [√] 3.CinCGAN 方法训练数据

---

![image-20230213160627144](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249351.png)![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249352.jpg)

## [√] 4.第一套 CycleGAN

---

模型组成：G1 G2 D1

功能：把输入的带有未知且复杂的退化作用的 LR 图片转化为干净的 LR 图片。

分析：由于真实世界场景的输入图片一般带有未知的退化作用，所以我们需要首先想办法把退化作用去掉。但是又受限于没有成对的训练数据，所以只能以 CycleGAN 的方式来学习生成器 G1 完成去退化 (Degredation) 的作用。根据 CycleGAN 的原理，我们同时还需要生成器 G2 的对称设计来辅助 G1 的训练。同时，判别器 D1 用来判断输入判别器的图片是否有退化的作用。

运行过程：给定输入图像 x ，生成器 G1 学习生成干净的 LR 图像 y ，以欺骗判别器 D1 。同时， D1 学习区分生成的样本 G1(x) 和真实的样本 y 。

**GAN Loss:**

![image-20230213162727779](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249353.png)

CycleGAN 的 **Cycle Consistency Loss** 保证不成对数据也能有效训练：

![image-20230213162820961](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249354.png)

CycleGAN 中，作者引入了一个 **Identity Loss**，以保持输入和输出图像之间的颜色组成。作者发现，Identity Loss 可以帮助保存输入图像的颜色。所以这里作者也添加了 Identity Loss 来避免输出的颜色变化：

![image-20230213162840567](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249355.png)

**Total Variation (TV) Loss** 帮助增加空间平滑度：

![image-20230213162920301](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249356.png)

式中， ∇ℎ 和 ∇w 分别用来计算 G1(xi) 的水平和垂直梯度。

**第一套 CycleGAN 总的损失函数：**

![image-20230213163002763](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249357.png)

式中， w1,w2,w3 是不同损失的权重。

> alec：
>
> - 第一套cycleGAN的G1用来生成没有退化的LR图像

## [√] 5.第二套 CycleGAN

---

**模型组成：**G1,SR,G3,D2。

**功能：**把输入的干净的 LR 图片用 CycleGAN 的方式超分成干净的 HR 图片。

**分析：**第二套 CycleGAN 里面， G1,SR 相当于是第一个 Generator， G3 相当于是第二个 Generator。同时，判别器 D2 用来判断输入判别器的 HR 图片是否有退化的作用。

**问：为什么要以 CycleGAN 的方式超分成干净的 HR 图片，而不是以普通 GAN 的方式超分成干净的 HR 图片？**

**答：**普通 GAN 方式需要成对的训练数据，真实世界超分场景是没有的。

**GAN Loss：**

![image-20230213164246544](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249358.png)

CycleGAN 的 **Cycle Consistency Loss** 保证不成对数据也能有效训练：

![image-20230213164335322](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249359.png)

对于 **Identity Loss**，作者考虑确保 SR 网络可以生成足够质量的超分辨率图像，而不是保持输入和输出之间的色调一致性，所以将新的 Identity Loss 定义为:

![image-20230213164553374](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249360.png)

![image-20230213180625902](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249361.png)

**Total Variation (TV) Loss** 帮助增加空间平滑度：

![image-20230213180911466](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249362.png)

![image-20230213180837564](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249363.png)

**第二套 CycleGAN 总的损失函数：**

![image-20230213180939349](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249364.png)

式中， λ1,λ2,λ3 是不同损失的权重。



## [√] 6.模型架构

---

生成器 G1,G2,G3 和判别器 D1,D2 的架构如下图2所示，与 CycleGAN 结构相似。Conv 代表卷积层， k,n,s 分别指的是是 kernel size，number of filters 和 stride。

对于生成器 G1 和 G2 ，作者在头部和尾部使用3个卷积层，在中间使用6个残差块。生成器 G3 的架构与生成器 G1 和 G2的相同，只是第二和第三卷积层的 stride 设置为2，以执行下采样。

对于判别器 D1 和 D2 ，作者使用了70×70的 PatchGAN，70×70的 PatchGAN 的意思是：把输入图像处理为有重叠的70x70的 patches，然后每个 patch 被处理为标量，最终输出结果是30x30x1。

LR 图片和 HR 图片的大小分别是32×32和128×128， D1 的前三层卷积 stride 设为1。这样一来，判别器 D1 的感受野就是16×16，判别器 D2 的感受野就是70×70。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249365.jpg)



## [√] 7.CinCGAN 训练过程

---

**数据集：**NTIRE2018 track 2 dataset。

- track 2 数据集从 DIV2K 数据集退化得到，DIV2K包含800张 HR 的训练数据 和 100张 HR 的验证集。track 2 数据集的退化方式包含下采样 (Down-sampling), 模糊 (Blurring), 像素偏移 (Pixel shifting) 和 噪声 (Noises)。尽管退化算子的参数对于所有图像都是固定的，但是模糊核是随机生成的，并且它们所产生的像素偏移因图像而异。因此，track 2 数据集中图像的退化核是未知且多样的。

- 由于作者的目的是在没有成对的 LR-HR 数据的情况下无监督地训练网络，所以作者从训练 LR 集中取前400幅图像 (编号从1到400) 作为输入图像 X ，从 HR 集中取另外400幅图像 (编号从401到800) 作为要求的 HR 图像 Z 。图像 Y 直接从 HR 图像 Z 经过双三次下采样得到。 注意，尽管 DIV2K 包含成对的训练数据集，但作者依然不使用成对的数据进行监督训练。

- 训练的过程一共分为2步：

**第1步：**训练第一套 CycleGAN，即： G1,G2,D1 。完成 LR → clean LR 的训练。

使用 Adam 优化器， β1=0.5,β2=0.999,$\epsilon $=10−8 超参数设置为 w1=10,w2=5,w3=0.5 。

学习率初始值为 2×10−4 ，按照每 40000 iteration 衰减一倍，一共训练 400000 iterations。

**第2步：**同时训练第一套 CycleGAN 和第二套 CycleGAN。完成 LR → HR 的训练。

![image-20230213204723666](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249366.png)

## [√] 8.CinCGAN 实验结果

---

- 作者将提出的 CinCGAN 模型的性能与几种最先进的 SISR 方法进行了比较：FSRCNN，EDSR 和 SRGAN。
- 我们使用公开可用的 FSRCNN 和 EDSR 模型，它们用成对的 LR 和 HR 图像训练，其中输入是从 HR 图像向下采样的干净的 LR 图像。
- 为了使结果更具可比性，我们还用配对的 track 2 数据集微调了 EDSR 和 SRGAN (分别标记为EDSR+ 和 SRGAN+)。
- 为了强调 CinCGAN 结构的有效性，作者还尝试首先对输入的 LR 图像进行去噪，然后对去噪后的图像进行超分辨率处理以进行比较。
- BM3D 是目前最先进的图像去噪方法之一，是一种高效而强大的去噪工具。因此，我们首先使用 BM3D 预处理测试 LR 图像，然后使用 EDSR (标记为 BM3D+EDSR) 进行超分辨率处理，结果如下图3所示。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249367.jpg)

图3显示了不同的模型在 NTIRE 2018 track 2 dataset 上的实验结果的 PSNR 和 SSIM 值。结果表明：

- 如果模糊和噪声在训练过程中未知，FSRCNN 和 EDSR 就不能很好地完成任务。在通过成对的 track 2 数据集进行微调之后，EDSR+ 和 SRGAN+ 结果有了改进，但是这个改进是建立在有成对训练数据的情况下的。
- CinCGAN 可以在没有成对的训练数据的情况下在 PSNR 和 SSIM 方面与 SRGAN+ 性能相当。
- 虽然 BM3D 模型可以去除噪声，但它也会过度平滑输入图像。BM3D+EDSR 的 PSNR 和 SSIM 值低于所提出的方法。

不同方法超分可视化的结果如下图4所示。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249368.jpg)

**对比实验：**

为了验证 CinCGAN 结构的有效性，作者尝试了3种不同的训练范式：

第1种训练的范式仅使用一个 CycleGAN 将 LR 图像 X 恢复为 HR 图像 Z ，即同时对 LR 图像进行去噪、去模糊和超分。直接去优化 $L_{total}^{HR}$ 。但是在训练过程中发现，结果 $\tilde{z}$ 总是不稳定，并且存在许多不期望的伪像，这说明单个网络很难同时去噪、去模糊和超分任务，尤其是当退化核因图像而异时，以及在无监督学习的情况下。

![image-20230213224032809](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249369.png)

第2种训练的范式在 CinCGAN 的基础上去掉 D2 和 G3 ，先使用第一套 CycleGAN 把退化的 LR 图转化为 clean 的 LR 图，即对 LR 图像进行去噪、去模糊。再直接通过 SR 模型进行超分。但是结果发现这样做会放大一些噪声，并且会进一步被 SR 模型放大，影响了视觉质量。

![image-20230213224138772](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249370.png)

![image-20230213224248424](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302132249371.png)











## [√] 总结

---

真实世界场景由于下采样和退化函数复杂且相互耦合，因此很难像传统的盲超分那样进行精确估计。而且真实世界场景 HR 图片是未知的，没有 HR 图片作为训练集。我们就没法构造真实世界的 HR-LR 数据对进行有监督训练。利用循环一致性的特性，作者提出了一种 Cycle-in-Cycle GAN (CinCGAN) 来解决无成对训练数据的真实世界场景超分问题。整个方法包含两套 CycleGAN，第一套把输入的带有未知且复杂的退化作用的 LR 图片转化为干净的 LR 图片；第二套把输入的干净的 LR 图片用 CycleGAN 的方式超分成干净的 HR 图片。作者的方法实现了与最先进的有监督 CNN 的算法相当的性能。