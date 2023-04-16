---
title: >-
  文章阅读笔记：【2021 USR_DA】Unsupervised Real-World Super-Resolution：A Domain
  Adaptation Perspective
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302272044875.jpg
tags:
  - 盲超分
password: 972274
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 盲超分
abbrlink: 973933338
date: 2023-02-27 19:11:47
---

> 原文链接：
>
> （1）【√】Unsupervised Real-World Super-Resolution: A Domain Adaptation Perspective 阅读笔记 - 知乎 - liemer（[link](https://zhuanlan.zhihu.com/p/445268270)）
>
> 文章阅读笔记：【2021 USR_DA】Unsupervised Real-World Super-Resolution: A Domain Adaptation Perspective
>
> ps：本文为依据个人日常阅读习惯，在原文的基础上记录阅读进度、记录个人想法和收获所写，关于原文一切内容的著作权全部归原作者所有。

#  [√] 文章信息

---

论文标题：Unsupervised Real-World Super-Resolution: A Domain Adaptation Perspective

中文标题：从域适应的视角进行盲超分

论文链接：https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Unsupervised_Real-World_Super-Resolution_A_Domain_Adaptation_Perspective_ICCV_2021_paper.html

论文代码：https://github.com/anse3832/USR_DA

论文发表：ICCV 2021



# [√] 文章1

---

> 总结：
>
> 【本文思想】
>
> 1. 目前方法的问题所在
>
> 现有的方法大都在尝试如何从HR图像生成更加逼真的LR图像，也就是去尽可能地拟合真实世界中的退化过程。但是这样的方法很难适应所有情况，一旦出现了新的未知的退化（真实场景中的退化是非常复杂的），模型的效果就可能大打折扣。
>
> 1. 本文的做法
>
> 所以这篇文章就尝试使用UDA的方法来解决这一问题。
>
> 1. 文章的主要思想
>
> （a）引导Encoder能学习到退化无关的特征（这样即使出现了新的退化，Encoder也能有效地提取特征），将源域和目标域的图像投影到同一特征空间当中。
>
> （b）让共享空间向目标域靠近以保留更多目标域的信息，这样模型能更好地从Encoder的特征中恢复目标域的图像。
>
> 
>
> 【本文贡献】
>
> 【网络结构】
>
> ![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302272047748.jpg)
>
> 源域的图像是配对的，但是源域的LR图像没有自然噪声。通过生成对抗网络，让网络学习到能够将自然噪声嵌入到源域的LR图像中，这样源域的LR和HR就能成对的来训练了。
>
> 【可以用于自己论文的话】
>
> 【可以用于自己论文的idea】
>
> （1）第一步：Feature Distribution Alignment（特征分布对齐）（将源域和目标域的特征提取到同一个特征空间当中（通过GAN））（引导Encoder能学习到退化无关的特征（这样即使出现了新的退化，Encoder也能有效地提取特征），将源域和目标域的图像投影到同一特征空间当中。）（这一步让这个编码器具有提取图像特征、忽略各种噪声的能力）
>
> - 通过生成对抗网络，引导编码器能够学习到和降质无关的特征，然后LR图像重建为SR。并通过L1+perception loss+GAN loss来约束。
> - 源域是配对的LR图像，目标域是自然LR图像空间，通过GAN将配对的LR图像从其特征空间变换到自然LR图像特征空间。（这是一个降质图像生成过程）
> - 目标域的LR图像：自然噪声 + 图像特征；源域的LR图像：人工噪声 + 图像特征。通过编码器+GAN网络，让编码器能够提取出二者的共同点：图像特征，忽略自然噪声和人工噪声。
>
> （2）第二步：Feature Domain Regularization（域空间规范化）（让共享空间向目标域靠近以保留更多目标域的信息，这样模型能更好地从Encoder的特征中恢复目标域的图像。）
>
> 
>
> 【问题记录】
>
> 【零碎点】



## [√] 引文

---

![Unsupervised Real-World Super-Resolution: A Domain Adaptation Perspective 阅读笔记](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302272047749.jpg)

这是一篇使用UDA思想解决盲超分辨率的文章，发表在ICCV2021上，看起来模型很复杂但是背后的思想感觉还是比较好理解的。

## [√] Background

---

盲超分辨率的目的是从具有未知退化参数的LR图像中回复HR图像。现有的方法大都在尝试如何从HR图像生成更加逼真的LR图像，也就是去尽可能地拟合真实世界中的退化过程。但是这样的方法很难适应所有情况，一旦出现了新的未知的退化（真实场景中的退化是非常复杂的），模型的效果就可能大打折扣。所以这篇文章就尝试使用UDA的方法来解决这一问题。

*突然觉得传统的方法就像是DG，希望训练好的模型在其他域中有很好的泛化能力，而使用UDA比DG的方法好似乎是理所应当的。*

## [√] Method

---

这篇文章的主要思想有两个：

（a）引导Encoder能学习到退化无关的特征（这样即使出现了新的退化，Encoder也能有效地提取特征），将源域和目标域的图像投影到同一特征空间当中。

（b）让共享空间向目标域靠近以保留更多目标域的信息，这样模型能更好地从Encoder的特征中恢复目标域的图像。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302272047750.jpg)

在训练阶段，会有三种数据，来自源域的成对的LR图像和HR图像，和来自目标域的LR图像。训练的流程图如下：

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302272047751.jpg)

首先，为了获得退化的不可区分特征（degradation-indistinguishable feature），需要让源域的特征分布S和目标域的特征分布T尽可能相似。为此作者使用了GAN结构，鉴别器Df 负责判断输入图像是来自源域还是目标域。作者在这里参照了LSGAN的方法来优化：

![image-20230227201433916](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302272047752.png)

然后为了重构源域的HR图像，将源域的特征送入 G_SR （Decoder），通过常见的L1+perception loss+GAN loss来约束，也就是图中的 L_rec ：

![image-20230227202016600](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302272047753.png)

到这里已经完成了第一个目标，将源域和目标域的特征提取到同一个特征空间当中（通过GAN）。然后就是完成上述的第二个目标，这一过程作者称之为Feature Domain Regularization.

为此作者引入了一个新的Decoder G_t ，通过下面的损失约束Encoder和G_t，保证编码器提取的特征f不会丢失图片的内容信息和目标域的特征（图中未画出）

![image-20230227203327685](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302272047754.png)

> 然用了一个新的鉴别器 �� （注意和上面的鉴别器不是同一个，功能也不同）鉴别输入图片（ ��→� 和��→�）是否来自目标域。�� 和 ���� 就要求编码器生成具有目标域退化的LR图像。
>
> ![image-20230227203428072](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302272047755.png)

> 然后通过Encoder得到��→�的特征。因为��→�和��**有一样的内容，但一个是遵循目标域的退化方式，一个遵循源域的退化方式**。但是！因为前面说到要让Encoder学到的是**退化无关的特征**，所以这两个特征在理想状态下应该是一致的，所以作者加了一个损失函数：
>
> ![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302272047756.jpg)
>
> 然后将两个特征 �� 和 ��~ 送入���得到恢复的高清图片。����和����一样目的是为了让模型能够恢复图像内容：
>
> ![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302272047757.jpg)
>
> ![image-20230227203816772](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302272047758.png)

> 不得不说，这篇文章虽然背后的两个思想很简单，但是实际采用的方法确实有点复杂，最后总的损失是上述所有损失的加权和。我尝试对该框架的思路做一个简单总结，如果有理解有偏差的地方欢迎大家讨论。
>
> - 鉴别器�� 和 ���� 的作用是让Encoder学习的退化无关的特征，让源域和目标域的特征投影到同一特征空间。
> - 鉴别器�� 和 ���� 的作用是让退化无关的特征空间向目标域靠近，与此同时����还保证了Encoder提取的特征不会丢失内容信息。因为对于一般任务而言，可以通过丢弃一些纹理特征来提取域无关特征，但是对于超分问题这样是不行的，所以要保证内容信息不会丢失。
> - 最后����和����不用多说就是为了更好地恢复图像内容。
>
> 在推理阶段应该只用到Encoder和Decoder���
>
> ![image-20230227204037727](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302272047759.png)

> alec：
>
> - 源域：成对的LR图像和HR图像，其中HR是清晰的图像
> - 目标域：自然LR图像
> - 因此这里目标域是自然噪声域、源域是成对的LR、HR域
> - x\_t = 含有自然噪声的目标域的LR图像
> - x\_s = 预定义降质的源域的LR图像
> - 源域的图像是配对的，但是源域的LR图像没有自然噪声。通过生成对抗网络，让网络学习到能够将自然噪声嵌入到源域的LR图像中，这样源域的LR和HR就能成对的来训练了。



## [√] Experiment

---

简单放一下结果

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302272047760.jpg)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302272047761.jpg)

作者还做了一些消融实验，感兴趣的小伙伴可以去看原文。本人学识有限，有理解错误的地方还请大家指正。





















