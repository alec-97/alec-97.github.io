---
title: 061 - 文章阅读笔记：CNN与Transformer相互促进，助力ACT进一步提升超分性能 - AIWalker
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302082240702.jpg
tags:
  - CNN
  - transformer
  - 深度学习
  - 超分辨率重建
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 3802034691
date: 2023-02-08 20:30:19
---

> 原文链接：
>
> [CNN与Transformer相互促进，助力ACT进一步提升超分性能 - AIWalker](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651687821&idx=1&sn=605624f3ae6289b5f2f0e10420ead7db&scene=21#wechat_redirect)
>
> 2022-03-21 22:00
>
> ps：本文为依据个人日常阅读习惯，在原文的基础上记录阅读进度、记录个人想法和收获所写，关于原文一切内容的著作权全部归原作者所有。



## [√] 文章信息

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302082240802.jpg)

论文题目：【ACT】Rich CNN-Transformer Feature Aggregation Networks for Super-Resolution

中文题目：用于图像超分的nb的CNN-transformer特征聚合网络

论文链接：https://arxiv.org/abs/2203.07682

代码链接：https://github.com/jinsuyoo/act

论文发表：WACV 2023

本文提出一种用于图像超分的混合架构，它同时利用了CNN局部特征提取能力与Transformer的长程建模能力以提升超分性能。

具体来说，**该架构由CNN与Transformer两个分支构成，并通过信息互融合补偿各自特征表达进一步提升性能**。

更进一步，**本文提出一种跨尺度token注意力模块，它使得Transformer可以更高效的探索不同尺度token的信息相关性**。所提方案在多个图像超分数据集上取得了SOTA性能。



## [√] Method

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302082240803.jpg)

上图给出了所提方案ACT的整体架构示意图，很明显，它也是一种类EDSR的架构，其核心在于body部分的组成。因此，我们主要对这两部分进行相似介绍，其他部分略过。ACT的body部分由于CNN与Transformer两个分支以及FusionBlock构成：

- CNN branch: 该分支采用了RCAB模块。具体来说，我们堆叠N个RCAB模块生成如下特征。

![image-20230208213628416](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302082240804.png)

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302082240805.jpg)

> alec：
>
> - MHSA = 多头自注意力
> - CSTA = cross-scale token attention = 跨尺度的token注意力
> - token = 象征，代表
> - leverage = 杠杆、促使、影响

- Transformer branch: 在该部分，我们基于MHSA进行构建。此外，我们还添加了CSTA(Cross-Scale Token Attention)模块以探索跨尺度相关性，见上图。我们首先将浅层特征![image-20230208223042258](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302082240806.png)序列化为非重叠token![image-20230208223056419](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302082240807.png)(注：d表示每个token的维度，n表示token的数量)。此外，我们发现：位置信息对于超分而言并不重要，故我们并未对token添加位置嵌入信息。经由上述处理得到的token将被送入到Transformer分支进行处理，描述如下：

![image-20230208223108077](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302082240808.png)

正如Fig2a所示，每个Transformer模块包含两个连续的注意力操作：MHSA与CSTA。



通过对不同尺度的QKV信息进行混合处理，CSTA可以对跨尺度信息进行探。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302082240809.jpg)

- Multi-branch Feature Aggregation 如前面Fig1所示，我们需要对不同分支的特征进行融合。我们采用了上图所示的双向融合方案，该融合方式可以描述成如下公式：

## [√] Experiments

---

在模型配置方面，ACT有四个CNN模块、四个Transformer模块，每个CNN模块包含12个RCAB模块，通道数为64，Transformer模块的维度为576，融合模块堆叠了四个$1 \times1 $残差模块。

在模型训练方面，本文参考IPT，采用ImageNet进行训练。输入图像块尺寸固定为$48 \times48$,常规数据增强，Adam优化器，batch=512，训练150epoch。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302082240810.jpg)

上表&图给出了不同方案的性能与效果对比，从中可以看到：

- 相比其他方案，在所有尺度下，ACT与ACT均取得最/次最佳的PSNR/SSIM指标；
- 相比IPT，受益于多尺度特征提取与融合，ACT取得了显著性能提升；
- 相比SwinIR，ACT在Urban100上的指标高出0.3dB，证实了CSTA模块可以成功探索多尺度特征(话说，这个指标提升真的来自CSTA吗？与ImageNet训练无关吗？)。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302082240811.jpg)

上表从参数量与FLOPs方面对不同方案进行了对比，可以看到：**尽管ACT的参数量多于EDSR、RCAN与SwinIR，但FLOPs是所有方案中最低者**。





## [√] 推荐阅读

---

1. [超越SwinIR，Transformer再一次占领low-level三大任务](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651687009&idx=1&sn=87a7cde28f2ba0f5cd602aed2928e065&scene=21#wechat_redirect)
2. [让Dropout在图像超分领域重焕光彩！](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651687062&idx=1&sn=98cdf4f017c15f94b71db69b5ad64810&scene=21#wechat_redirect)
3. [AdaDM: 让超分网络不仅可以用BN，性能还可以更优](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651686986&idx=1&sn=b6b4f687d75729b65be5cf30ba8b3dfa&scene=21#wechat_redirect)
4. [视频超分新标杆 | BasicVSR&IconVS](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651682755&idx=1&sn=64944ce382f1d3928160189f97b5c383&scene=21#wechat_redirect)
5. [HINet | 性能炸裂，旷视科技提出适用于low-level问题的Half Instance Normalization](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651684871&idx=1&sn=372665b85e3ed9ea90c7c5a67a17ac61&scene=21#wechat_redirect)
6. [图像增强领域大突破！以1.66ms的速度处理4K图像，港理工提出图像自适应的3DLUT](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651682108&idx=1&sn=a29346042bf42730a48901bac00ef317&scene=21#wechat_redirect)
7. [ETH开源业内首个广义盲图像超分退化模型，性能效果绝佳](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651683889&idx=1&sn=a602e3cdb44ea2901dd506630b5e3b70&scene=21#wechat_redirect)
8. [ICCV2021 FBCNN: 超灵活且强度可控的盲压缩伪影移除新思路](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651686061&idx=1&sn=3b2ca0b46d95a1d393449b8db8161581&scene=21#wechat_redirect)