---
title: >-
  文章阅读笔记：【2020 FS-SRGAN】Guided frequency separation network for real-world
  super-resolution
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222049168.jpg
tags:
  - 盲超分
password: 972274
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 盲超分
abbrlink: 904285378
date: 2023-02-22 16:40:48
---

> 原文链接：
>
> （1）CVPR2020 ｜ 高低频分离超分方案 - AIWalker - Happy（[link](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651680623&idx=1&sn=4ef95ca6014e6218aec6c80bee4136a3&chksm=f3c92142c4bea85478ae07b895c9a25d396c675aa936933bf85ea615065c2b90a321afc74b74&scene=178&cur_album_id=1338480951000727554#rd)）
>
> 2020-06-18 21:06
>
> （2）[论文速度] 超分系列：基于频率分离的图像超分辨率算法 两篇 ICCVW 2019 和 CVPRW 2020（[link](https://blog.csdn.net/u014546828/article/details/112343872)） - CSDN - Phoenixtree_DongZhao
>
> 于 2021-01-31 09:05:25 发布
>
> ps：本文为依据个人日常阅读习惯，在原文的基础上记录阅读进度、记录个人想法和收获所写，关于原文一切内容的著作权全部归原作者所有。



# ——————————————————————————————

# [√] 文章信息

论文标题：【2020 FS-SRGAN】Guided frequency separation network for real-world super-resolution

中文标题：频率分离引导的真实世界超分

论文链接：https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Zhou_Guided_Frequency_Separation_Network_for_Real-World_Super-Resolution_CVPRW_2020_paper.pdf

论文代码：https://github.com/fzuzyb/2020NTIRE-Guided-Frequency-Separation-Network-for-RWSR

论文发表：CVPR2020

论文评价：

# ——————————————————————————————

# [√] 文章1

> 总结：【零碎点】
>
> - 噪声包括手工噪声（例如JPEG压缩噪声）、传感器噪声
> - DS-GAN是FSSR论文中的LR数据生成GAN网络
> - 本文实验对比的是FSSR中的SDSR和TDSR
>
> 
>
> 【要点】
>
> - 为了实现无监督超分，本文提出无监督方案，分为两个阶段：（1）无监督域变换（2）有监督超分
>
> - - 作者提出采用一种颜色引导的域映射网络缓解域变换过程中的色差问题。
>     - 提出了一种颜色注意力残差模块作为该网络的基础单元，它模块可以动态的根据输入数据调节参数，因此该网络具有更好的泛化性能。
>     - 更进一步，作者还修改了超分阶段的判别器以使网络不仅关注高频特征，同时还关注低频特征。
>     - 最后作者构建了一种边缘损失提升纹理细节。
>
> 
>
> 【文章结构图】
>
> ![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055076.jpg)
>
> ![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055077.jpg)
>
> ![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055078.jpg)
>
> ![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055079.jpg)
>
> 
>
> 
>
> 【观文idea】
>
> - 是否可以设计一个基础单元，该单元是一种注意力单元，同时里面可以放上高低通滤波器，用来根据输入图像的深浅，来自适应的提取不同频次的信息。浅层的特征图提取低频信息用于L1颜色引导，深层的特征图提取高频信息用于对抗损失。或者用于高频引导。
> - 本文FS-SRGAN和自己的baseline Degradation GAN的总体框架几乎是一样的，只不过加入了所谓的高低通滤波器，然后将HR直接生成Z’的步骤变成了先下采样，然后再生成的两步。
> - 本文和FSSR的结构差不多，稍微复杂了FSSR的网络结构和损失部分。可以借鉴这种性能提升方式。
> - 本文提出了CARB模块作为与变换网络的基础单元，有效的对图像进行域迁移的同时保持内容与色彩信息。自己是否可以将色彩提取模块替换为MDN，用于提取别的内容，然后参考本文的描述方式。
> - 自己的文章也可以参考本文的高低频分开提取然后进行损失计算的思想。或者在网络之外，单独分出一个分支，专门自适应的提取高低频信息，然后concat或者加到网络中，作为网络增强手段。
> - 本文提出了一种有效的near-real数据的制作方案，这种作用要比网络架构上的魔改要更加有意义。
>
> 【观文收获】
>
> - 本文和FSSR一样，在真实LR数据生成过程中，是先将HR下采样到LR，然后再将LR变成真实LR，而不是将HR直接变为真实LR。
> - 直接使用DS-GAN生成LR图像，会存在色差问题。利用这种存在色差问题的数据训练SR模型，会导致最终的数据模糊。因此本文修正了这一点，修正方式是：在L->L'的生成过程中，提出了颜色引导网络以动态输出颜色特征。
> - 本文SR部分模型的生成器使用了内容损失和边缘损失，判别器则分别对高频信息和低频信息进行了损失计算
> - FS = frequency separation
>
> 【问题记录】
>
> - 什么是AdaIN？
> - 颜色引导网络得到的均值和方差如何组合进CARB，该均值和方差的形状是什么样的？
> - 什么是LS-GAN策略？





## [√] 引文

---

![标题&作者团队](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055080.jpg)

> 该文是福州大学的老师提出的一种用于真实世界图像超分的方案，在NTIRE2020图像超分竞赛中取得了前五的好成绩，该文的思路与其他NTIRE2020竞赛的优胜方案思考角度有一些不同之处，故笔者稍微花了点时间进行总结。文末附文章与code下载链接。





## [√] Abstract

---

真实世界训练数据对往往难以获得，因此在学术界往往采用bicubic下采样方式生成训练数据对，然而这种数据生成方式会噪声图像的特性发生变化(比如artifacts, sensor noise, 以及其他特征)，这也是为何诸多学术界SOTA超分方案在真实世界图像上的效果差强人意的原因。

为了解决前述问题，作者提出一种无监督超分方案，它可以分为两个阶段：(1)无监督域变换；(2)有监督超分。

作者提出采用一种颜色引导的域映射网络缓解域变换过程中的色差问题。提出了一种颜色注意力残差模块作为该网络的基础单元，它模块可以动态的根据输入数据调节参数，因此该网络具有更好的泛化性能。更进一步，作者还修改了超分阶段的判别器以使网络不仅关注高频特征，同时还关注低频特征。最后作者构建了一种边缘损失提升纹理细节。作者所提方案在NTIRE2020超分竞赛中取得了具有竞争力的性能。

## [√] Method

---

正如大家所知道的，基于深度学习的方案可以直接根据LR估计HR，但它需要大量的LR-HR数据对。已有的方案(比如SRCNN、EDSR) 往往采用人工合成的方式制作上述数据对，但是这种人工合成训练数据与真实测试存在很大的差异。下图给出了两者差异，经由双三次插值后的图像的特征出现了明显的差别。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055081.jpg)

为解决上述问题，作者提出一种无监督真实世界超分方案。它包含两个阶段:(1)无监督SR数据生成阶段；(2)有监督SR阶段。下面从三个角度来介绍一下这篇文章的内容。

#### [√] Problem formulation

---

（1）在无监督SR数据生成阶段，假设y表示真实世界HR图像域，x表示真实HR图像下采样后的LR图像域，z表示真实世界LR图像域。该阶段聚焦于寻找一个映射f1将x映射到z，同时f1(x)与z尽可能具有相似的特征，同时保持内容的一致性。

（2）在监督SR阶段，利用前述生成的数据对训练SR模型。此时，我们需要寻找另一个映射f2将Z’映射到Y，同时fx(Z’)应与Y尽可能的相似。

总而言之，只要我们可以使得Z’与Z足够的相似(理想情形下Z’=Z)，上述问题就可以简化为有监督SR问题。该文所提整体方案流程如下所示。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055082.jpg)

#### [√] Unsupervised SR Data Generation

---

为进行域变换，作者采用了GAN的思路。特别的，采用DSGAN对LR图像进行迁移，但其存在色差问题，见下图。如果采用这样的数据对去训练SR模型，会导致SR模型生成过渡模糊的图像。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055083.jpg)

作者经过分析后认为，产生上述问题的原因在于：Instance Normalization，它缺失了关于颜色独立的先验信息。为解决该问题，作者提出一种颜色引导网络以动态输出图像的颜色特征。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055084.jpg)

上图给出了作者所涉及的颜色引导生成器网络，上半部分是引导参数网络，用于生成CARB的均值(bias，表示全局信息)与方差(weight)信息。而CARB是一个残差模块，作者组合空域注意力与AdaIN增强空域感知。因此原始图像的内容与颜色得以保留。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055085.jpg)

上图给出了判别器的网络架构头，它采用了`Frequence Separation`的思路。采用高斯高斯滤波器提取高频信息，这使得判别器仅进行图像的高频成分的真实性判别，同时也会使得整个GAN的训练更稳定、更快收敛。

对于每个真实图像y∈$y$，其双三次下采样结果为x∈$X$。那么![image-20230222193612844](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055086.png)表示真实LR图像。而z’,z将被送入判别器![image-20230222193639497](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055087.png)进行判别真假。

为确保生成器更有效的进行域变换，作者组合了三种损失：低频损失、感知损失以及高频损失，他们分别定义如下所述。低频损失定义如下：

![image-20230222194140671](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055088.png)

> alec：
>
> - 注意这里存在生成器的高频损失和判别器的高频损失。



#### [√] Supervised SR

---

在完成域变换后，生成图像z’，y数据对可用于有监督训练SR模型。此时仅需要解决z’到y的映射关系f2，为更好的提升生成图像的视觉质量，作者提出采用LSGAN。生成器Gz’->y仅仅包含9个RRDB模块；而判别器Dy包含两个网络：高频网络与低频网络。这样可以更好确保生成器不仅关注高频特征，同时更好的保持低频特征。这里的判别器与无监督数据生成阶段的判别器架构见下图。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055089.jpg)

作者发现：在超分模型训练过程中，感知损失会带来轻微的色差问题。因此作者移除了感知损失而添加了一种边缘损失以更好的确保颜色一致性。因此生成器的损失包含：内容损失Lc、边缘损失Le以及对抗损失Ladv。注：这里的内容损失指的是L1损失。

对于边缘损失而言，作者希望训练过程可以聚焦于图像的边缘细节，进而有效的增强视觉质量。在这里，作者采用Canny操作提取图像的边缘，边缘损失定义如下：

![image-20230222200826115](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055090.png)

在对抗损失方面，作者采用了LSGAN中的对抗损失，定义如下：

![image-20230222200849292](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055091.png)

生成器的总损失定义为：

![image-20230222200913576](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055092.png)

类似的，判别器的损失定义如下：

![image-20230222200937061](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055093.png)

## [√] Experiments

---

如前面所述，本文所提方法包含两个阶段，那么训练也会包含两个阶段。

在无监督数据生成阶段，对HR图像进行双三次下采样得到x，图像块大小裁剪为128x128，batch=8，优化器为Adam，学习率为1e-4，合计迭代30W。损失函数中的参数为![image-20230222203400760](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055094.png)。

在有监督超分模型阶段，首先将图像z’裁剪为120x120，图像y裁剪为480x480。在训练阶段，输入随机裁剪为64x64，batch=12，优化器为Adam，学习率为2e-4，合计迭代30W，损失函数中的参数为![image-20230222203608963](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055095.png).

首先，我们先来看一下域变换的效果，见下图。可以看到DSGAN存在色差问题，而所提方案则基本缓解了该色差问题。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055096.jpg)

然后，我们再来看一下超分的效果，见下表与图。可以看到：所提方法无论是LPIPS指标还是视觉效果均有了非常大的提升。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055097.jpg)

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055099.jpg)

其次，我们来看一下所提方法在NTIRE2020超分竞赛中的表现，见下图。从图中可以看到：相比已有方法，所提方法具有更优的视觉效果。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055100.jpg)

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055101.jpg)

最后，给出作者设计的几组消融实验结果对比。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055102.jpg)

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302222055103.jpg)

从上表与图中的对比，可以得出这样几个结论：

- Bicubic-vs-GAN：基于双三次插值合成数据训练的GAN存在严重的失真现象；
- CARB-vs-GAN：采用标准GAN生成的数据训练的超分模型存在假性纹理；
- CARB-vs-GANFS：采用GAN-FS方式生成数据训练训练的超分模型同样存在假性纹理有所改善，但细节也丢失了；
- 总而言之，本文所提方案最接近GroundTruth。

## [√] Conclusion

---

这篇论文提出了一种CARB模块作为域变换网络的基础单元，它可以有效的对图像进行域迁移，同时保持内容与色彩信息；作者同时还对ESRGAN的判别器进行了修改使其同时关注高频与低频信息；最后作者基于Canny提出一种边缘损失用于增强图像的边缘细节。所提方法在真实世界数据上取得了优于ESRGAN的效果，并在NTIRE2020超分竞赛中取得了优异成绩。

从笔者角度来看，这篇论文最大的价值点在于：无监督域变换模块。它提供了一种有效的`Near-Real`数据制作方案，它的作用要比网路架构上的“魔改”更有意义。

## [√] 推荐阅读

---

1. [思维的碰撞｜稀疏表达偶遇深度学习](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651680491&idx=1&sn=6f5abd049a17b13d6dc7984fc64f4512&scene=21#wechat_redirect)
2. [超越RCAN，图像超分又一峰：RFANet](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651680373&idx=1&sn=d7180cb4ef8e8d6724c6a1e92dd94f00&scene=21#wechat_redirect)
3. [NTIRE2020 图像降噪SSIM指标冠军方案](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651680548&idx=1&sn=8223b8316c06f445c2572ecd4ac38c14&scene=21#wechat_redirect)
4. [NTIRE2020冠军方案RFB-ESRGAN：带感受野模块的超分网络](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651680158&idx=1&sn=2b37109a504138552cbd8cf59d1f1c60&scene=21#wechat_redirect)
5. [显著提升真实数据超分性能，南大&腾讯开源图像超分新方案，获NTIRE2020双赛道冠军](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651680463&idx=2&sn=6a16768b75114c79ae5ad81a5f6799af&scene=21#wechat_redirect)









# ——————————————————————————————

# [] 文章2

> 总结：
>
> 

摘要：



提出问题：

解决方案：

实验结果：



方法：



颜色引导的域映射网络：

颜色注意剩余块 (CARB)：

判别器进行了改进：

边缘损失来改善纹理细节：





