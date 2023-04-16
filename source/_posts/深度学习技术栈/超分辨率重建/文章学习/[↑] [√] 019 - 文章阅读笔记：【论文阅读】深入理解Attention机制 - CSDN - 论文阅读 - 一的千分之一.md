---
title: 019 - 文章阅读笔记：【论文阅读】深入理解Attention机制 - CSDN - 论文阅读 - 一的千分之一
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301102141002.png
tags:
  - 深度学习
  - 计算机视觉
  - 机器学习
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 2353765967
date: 2023-01-10 21:39:35
---

> 转载自：
>
> [【论文阅读】深入理解Attention机制 - CSDN - 论文阅读 - 一的千分之一（√）](https://blog.csdn.net/yideqianfenzhiyi/article/details/79422857)
>
> 于 2018-03-16 21:32:57 发布

## [√] 1 - 什么是Attention机制？

---

其实我没有找到attention的具体定义，但在计算机视觉的相关应用中大概可以分为两种：

> alec：
>
> - 输入数据或特征图上的不同部分，对应的专注度不同。
>
> ---
>
> - 软注意力：保留了所有的分量，以概率的方式做加权
> - 硬注意力：选取部分分量，以01的方式做加权
>
> ---
>
> - recurrent，经常发生的，周期性的
>
> ---
>
> - 加权可以作用在原图上，也可以作用在特征图上
> - 加权可以作用在空间尺度上、也可以作用在channel尺度上，也可以作用在特征图上的每个元素尺度上
> - 这个加权还可以作用在不同时刻历史特征上，如Machine Translation。（时间维度上的加权）

1）**学习权重分布：输入数据或特征图上的不同部分对应的专注度不同**，对此Jason Zhao在[知乎回答](https://www.zhihu.com/question/68482809/answer/264070398)中概括得很好，大体如下：

- 这个加权可以是保留所有分量均做加权（即soft attention）；也可以是在分布中以某种采样策略选取部分分量（即hard attention），此时常用RL来做。
- 这个加权可以作用在原图上，也就是《Recurrent Model of Visual Attention》（RAM）和《Multiple Object Recognition with Visual Attention》（DRAM）；也可以作用在特征图上，如后续的好多文章（例如image caption中的《 Show, Attend and Tell: Neural Image Caption Generation with Visual Attention》)。
- 这个加权可以作用在空间尺度上，给不同空间区域加权；也可以作用在channel尺度上，给不同通道特征加权；甚至特征图上每个元素加权。
- 这个加权还可以作用在不同时刻历史特征上，如Machine Translation。

**2） 任务聚焦：通过将任务分解，设计不同的网络结构（或分支）专注于不同的子任务，重新分配网络的学习能力，从而降低原始任务的难度，使网络更加容易训练。**



## [√] 2 - Attention机制应用在了哪些地方？

---

针对于1部分中的attention的两大方式，这里主要关注其在视觉的相关应用中。

#### [√] 2.1 - 方式1：学习权重分布

---



###### [√] 2.1.1 - 精细分类

---

【相关论文】

Jianlong Fu, Heliang Zheng, Tao Mei (Microsoft Research), Look Closer to See Better: Recurrent Attention Convolutional Neural Network for Fine-grained Image Recognition, CVPR2017 （这个文章很有意思）

> alec：
>
> - proposal，提议、建议
> - zoom in 放大
> - zoom out 缩小

在关注的每一个目标尺度上，都采用一个分类的网络和一个产生attention proposal 的网络(APN)。

本文最有趣的就是这个APN。这个APN由两个全连接层构成，输出3个参数表示方框的位置，接下来的尺度的分类网络只在这个新产生的方框图像中提特征进行分类。

怎么训练呢？**本文定义了一个叫做rank Loss，用这个loss来训练APN，并强迫finer的尺度得到的分类结果要比上一个尺度的好，从而使APN更提取出更有利于精细分类的目标局部出来。**通过交替迭代训练，APN将越来越聚焦目标上的细微的有区分性的部分。

当然这里有一个问题，那就是精细尺度只能聚焦到最显著的部位（如鸟头），但其他部分（如羽毛、鸟爪）就关注不到了。

> alec：
>
> - 在分类网络中通过一个注意力网络提取出图像中重要的位置，然后对这部分区域方法再次进行分类。

![image-20230110222458866](D:\坚果云\Alec - backup files\typora pictures\image-20230110222458866.png)

![image-20230110222535353](D:\坚果云\Alec - backup files\typora pictures\image-20230110222535353.png)

###### [√] 2.1.2 - 图像分类

---

【相关论文】

（图像分类）Fei Wang, etc. (SenseTime Group Limited). Residual Attention Network for Image Classification，CVPR2017



本文是在分类网络中，增加了Attention module。这个模块是由两支组成，一支是传统的卷积操作，另一支是两个下采样加两个上采样的操作，目的是获取更大的感受野，充当attention map。因为是分类问题，所以高层信息更加重要，这里通过attention map提高底层特征的感受野，突出对分类更有利的特征。相当于变相地增大的网络的深度。



![image-20230110223052619](D:\坚果云\Alec - backup files\typora pictures\image-20230110223052619.png)



###### [√] 2.1.3 - 图像分割

---

【相关论文】

*Liang-Chieh Chen，etc. (UCLA) Attention to Scale: Scale-aware Semantic Image Segmentation, CVPPR2016*（权重可视化效果有点意思）

【分析】

通过对输入图片的尺度进行放缩，构造多尺度。

传统的方法是使用average-pooling或max-pooling对不同尺度的特征进行融合。

> alec：
>
> - 在某种程度上，最大池化和平均池化，也是一种注意力机制。只关注最重要的那部分信息。最大池化是硬注意力，平均池化是软注意力。
>
> ---
>
> - 本文的注意力模块是由两个卷积层构成的。

而**本文通过构造Attention model（由两个卷积层构成）从而自动地去学不同尺度的权重，进行融合（效果提升1到2个点吧，不同的数据集不一样）**。

从论文中的权重可视化的结果，能发现大尺寸输入上，对应网络关注于small-scale objects，而在稍微小一点的尺寸输入上，网络就关注于middle-scale，小尺寸输入则关注background contextual information。可视化效果感觉非常有意思。

> alec：
>
> - 注意力模块计算出的注意力权重和特征图相乘，得到result
> - 通过得到的注意力可视化可以看出，对于大尺寸输入，注意力关注小目标；对于中等尺寸的输入，注意力关注中等目标；对于小尺寸的输入，注意力关注背景纹理信息。
> - 能够看出注意力确实是在关注目标，大图关注各个目标，小图没有目标则去关注背景纹理。

![image-20230110223738263](D:\坚果云\Alec - backup files\typora pictures\image-20230110223738263.png)



###### [√] 2.1.4 - Image Caption看图说话

---

【相关论文】

（Image Caption看图说话）Kelvin Xu，etc. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention，ICML2015



因为不做NLP，所以这个论文技术细节并没有看懂。大意是对一个图像进行描述时，生成不同的单词时，其重点关注的图像位置是不同的，可视化效果不错。

![image-20230110224456702](D:\坚果云\Alec - backup files\typora pictures\image-20230110224456702.png)

#### [√] 2.2 - 方式2：任务聚焦/解耦

---

###### [√] 2.2.1 - Instance Segmentation实例分割

---

【相关论文】

（Instance Segmentation）Kaiming He, etc. Mask R-CNN（非常好的一篇文章）（何凯明的Mask-RCNN）



Kaiming大神在Mask R-CNN中，将segment branch的损失函数由softmax loss换成了binary sigmoid loss。即是，**将分类和分割任务进行解耦，当box branch已经分好类时，segment branch 就不用再关注类别，只需要关注分割，从而使网络更加容易训练。**

> alec：
>
> - segment，分割、部分
> - box branch关注分类、segment branch关注分割
>
> ---
>
> - segment会得到mask
>
> ---
>
> - 不同的卷积核负责将不同的类别的像素凸显出来。属于狗的卷积核，只会凸显狗的区域，而不会凸显别的样本的分类。

具体到训练中，假设分狗、猫、马三类，segment branch会得到3个mask，当训练样本是狗类，那么这个类别的loss才会被反传，猫类和马类对应的mask都不用管。

也就是说，**生成狗mask的那部分网络连接（卷积核）只需要聚焦于狗类的样本，然后将属于狗的像素目标凸显出来出来，训练其他类别时不会对这些连接权重进行更新。\**通过这个任务解耦，分割的结果得到了很大的提升\**（5%-7%）**。

Kaiming大神在文中也指出，当只输出一个mask时，分割结果只是略差，从而进一步说明了将分类和分割解耦的作用。



###### [√] 2.2.2 - 图像分割

---

【相关论文】

（图像分割）Lin etc. Fully Convolutional Network with Task Partitioning for Inshore Ship Detection in Optical Remote Sensing Images



针对靠岸舰船，本文通过任务解耦的方法来处理。

因为高层特征表达能力强，分类更准，但定位不准；底层低位准，但分类不准。

为了应对这一问题，本文利用一个深层网络得到一个粗糙的分割结果图（船头/船尾、船身、海洋和陆地分别是一类）即Attention Map；利用一个浅层网络得到船头/船尾预测图，位置比较准，但是有很多虚景。

**训练中，使用Attention Map对浅层网络的loss进行引导，只反传在粗的船头/船尾位置上的loss，其他地方的loss不反传。**相当于，深层的网络能得到一个船头/船尾的大概位置，然后浅层网络只需要关注这些大概位置，然后预测出精细的位置，图像中的其他部分（如船身、海洋和陆地）都不关注，从而降低了学习的难度。

![image-20230110233435969](D:\坚果云\Alec - backup files\typora pictures\image-20230110233435969.png)



## [√] 3 - 感想

---

总的来说，我觉得attention这个概念很有趣，使用attention也可以做出一些有意思的工作。相比于方式一，个人更喜欢方式二任务解耦，因为其对所解决的任务本身有更深刻的认识。当然上述介绍的论文，主要是关于high-level的任务，还没看到attention在low-level的任务中的应用（也可能是自己查得不全），当然如何应用，这值得思考。



## [√] 4 - 参考资料

---

除了上面的一些论文，其他的参考资料：

知乎问题：目前主流的attention方法都有哪些？

知乎问题：Attention based model 是什么，它解决了什么问题？

知乎专栏总结：计算机视觉中的注意力机制

CSDN博客总结：Attention Model（mechanism） 的 套路

CSDN专栏：从2017年顶会论文看 Attention Model

CSDN专栏：模型汇总24 - 深度学习中Attention Mechanism详细介绍：原理、分类及应用











