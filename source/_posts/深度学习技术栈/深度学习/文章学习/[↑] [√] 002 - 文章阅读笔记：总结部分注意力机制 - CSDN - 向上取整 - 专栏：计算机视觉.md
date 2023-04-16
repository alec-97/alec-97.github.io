---
title: 002 - 文章阅读笔记：总结部分注意力机制 - CSDN - 向上取整 - 专栏：计算机视觉
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301202325783.png
tags:
  - 计算机视觉
  - 注意力机制
  - 深度学习
categories:
  - 深度学习技术栈
  - 深度学习
  - 文章学习
abbrlink: 681746266
date: 2023-01-20 23:07:12
---

> 参考：
>
> [总结部分注意力机制 - CSDN - 向上取整 - 专栏：计算机视觉](https://blog.csdn.net/qq_42782782/article/details/127425795?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-127425795-blog-123270536.pc_relevant_recovery_v2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-127425795-blog-123270536.pc_relevant_recovery_v2&utm_relevant_index=6)
>
> 于 2022-10-20 14:00:55 

## [√] 部分注意力机制

---

#### [√] 1 - 空间注意力

---

###### [√] 1.1 - 自注意力

---

![image-20230120231340949](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301252229927.png)

> alec：
>
> - 自注意力机制，是通过QK相乘，然后通过softmax激活函数，得到注意力权重，然后再乘上V得到注意力加权后的数据。

自注意力计算时通常分为三步：

1. 第一步是将query和每个key进行相似度计算得到权重，常用的相似度函数有点积，拼接，感知机等；
2. 第二步一般是使用一个softmax函数对这些权重进行归一化，转换为注意力；
3. 第三步将权重和相应的键值value进行加权求和得到最后的attention。

> alec：
>
> - 注意力机制中，需要进行相似度计算得到权重，常用的相似度函数有[点积]\[拼接]\[感知机]
> - softmax激活函数的目的是对权重进行归一化，以便于转化为注意力

> alec：
>
> - 自注意力机制中，在将输入变成QKV三部分之前，一般将输入分别通过1x1卷积
> - 然后QK相乘，然后通过激活函数归一化，然后就得到了注意力分布，然后再乘上V

###### [√] 1.2 - 非局部注意力

---

![在这里插入图片描述](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301252229928.png)

1. 首先对输入的feature map X 进行线性映射（1x1x1 卷积，来压缩通道数），然后得到θ，Φ，g特征；
2. 然后对θ，Φ进行相似度计算，对自相关特征以列或以行（具体看矩阵g 的形式而定） 进行Softmax 操作，得到0~1的权重，这里就是我们需要的Self-attention 系数；
3. 最后将attention系数，对应乘回特征矩阵g 中，然后加上原输入的特征图，获得non-local block的输出。

> alec：
>
> - 非局部注意力，类似于在自注意力的基础上，自注意力的输出加上原输入的特征图。

#### [√] 2 - 通道注意力

---

通道域注意力类似于给每个通道上的特征图都施加一个权重，来代表该通道与关键信息的相关度的话，这个权重越大，则表示相关度越高。在神经网络中，越高的维度特征图尺寸越小，通道数越多，通道就代表了整个图像的特征信息。

> alec：
>
> - 通道域注意力类似于给每个通道上的特征图都施加一个权重，来代表该通道与关键信息的相关度的话，这个权重越大，则表示相关度越高。在神经网络中，越高的维度特征图尺寸越小，通道数越多，通道就代表了整个图像的特征信息。

![在这里插入图片描述](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301252229929.png)

> alec：
>
> - 通道注意力要给每个通道加权，所以需要将注意力的权重变为1x1xc





