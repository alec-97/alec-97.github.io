---
title: 021 - 文章阅读笔记：【论文阅读】CVPR2021|超分辨率重建相关论文整理与阅读（持续更新） - CSDN - 一的千分之一
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301131936266.png
tags:
  - 深度学习
  - CNN
  - 超分辨率重建
  - CVPR
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 742256945
date: 2023-01-11 19:20:14
---

> 转载自：
>
> [【论文阅读】CVPR2021|超分辨率重建相关论文整理与阅读（持续更新） - CSDN - 一的千分之一（√）](https://blog.csdn.net/yideqianfenzhiyi/article/details/114675699?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-114675699-blog-84288591.pc_relevant_recovery_v2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-114675699-blog-84288591.pc_relevant_recovery_v2&utm_relevant_index=3)
>
> 于 2021-03-11 17:44:59 发布

本文主要对CVPR2021中超分辨率重建相关论文进行整理与阅读。

## [√] CVPR2021 Super-resolution papers

---

#### [√] 1.ClassSR: A General Framework to Accelerate Super-Resolution Networks by Data Characteristic

---

论文标题：ClassSR：通过数据特征加速超分辨率网络的通用框架

论文链接：https://arxiv.org/abs/2103.04039

论文代码：https://github.com/Xiangtaokong/ClassSR

![image-20230111192827695](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301132007850.png)

本文idea巨简单，根据LR图像块重建的难易程度进行分类，难的用深的模型，容易的用小模型，用来进行加速。之前其实就有类似的思想。

![image-20230111193326182](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301132007851.png)

> alec：
>
> - 评价：图像的不同区域，超分重建的难度是不一样的。对于简单的区域（低频），使用浅一些的网络进行超分。对于复杂的区域，使用深层的网络进行超分。相比于统一全都使用深层的网络，这样能够节省计算量、提高计算效率。



#### [√] 2.SMSR：Learning Sparse Masks for Efficient Image Super-Resolution

---

论文标题：学习稀疏掩模以实现高效的图像超分辨率

论文链接：https://arxiv.org/abs/2006.09603

论文代码：https://github.com/LongguangWang/SMSR

![image-20230111193637859](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301132007852.png)

该文也是从计算资源的问题来进行思考的，平滑的区域应该赋予更少的计算资源。为了本文会生成一些sparse mask，来重新分配网络的计算资源。

![image-20230111200003891](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301132007853.png)

> alec：
>
> - 本文设计了一个稀疏掩模，通过稀疏掩模来针对不同的区域进行对应的计算。平滑、简单的区域，则赋予少一些计算资源。







#### [√] 3.Learning Continuous Image Representation with Local Implicit Image Function

---

论文标题：使用局部隐式图像函数学习连续图像表示

论文链接：https://arxiv.org/abs/2012.09161

论文代码：https://github.com/yinboc/liif

主页：https://yinboc.github.io/liif/

![image-20230111201841488](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301132007854.png)

人们通过视觉观察的世界是连续的，但是机器去“识别”图像是以2D矩阵的方式，离散地存储、处理图像像素。对此，本文旨在探索图像的一种连续的表达。近期，implicit function在3D重建大为流行，受此启发，本文提出Local Implicit Image Function (LIIF) 来实现对图像的连续性表达。LIIF是以图像坐标和周围的深度特征作为输入，预测该坐标位置所对应的RGB像素值。因为坐标是连续的，所以LIIF可以以任意分辨率进行表示。

![image-20230111202029569](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301132007855.png)

这个论文采用了最近大火的implicit neural representation。



#### [√] 4.Interpreting Super-Resolution Networks with Local Attribution Maps

---

论文标题：使用局部归因图解释超分辨率网络

论文链接：https://arxiv.org/abs/2011.11036

论文代码：https://colab.research.google.com/drive/1ZodQ8CRCfHw0y6BweG9zB3YrK_lYWcDk?usp=sharing

![image-20230111202303458](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301132007856.png)

这是一个很有趣的论文，做超分辨网络的可解释性，旨在寻找对SR结果有重要影响的输入图像的像素，之前SR领域有人关注可解释相关的问题。具体地，本文提出了局部归因图（Local Attribution Map, LAM）来分析超分辨网络，同时基于LAM分析得到比较容易理解的现象，如具有规则条纹或网格的纹理更容易被SR网络捕捉到，而复杂的语义则难以被利用。

![image-20230111202447886](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301132007857.png)

相关ref: https://zhuanlan.zhihu.com/p/363139999



#### [√] 5.GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution

---

论文标题：用于大因子图像超分辨率的生成潜在库、

论文链接：https://ckkelvinchan.github.io/papers/glean.pdf

![image-20230111202741344](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301132007858.png)

本文主要采用GAN inversion-based方法来实现大尺度的图像超分辨。但之前GAN inversion方法，如PULSE，需要在测试阶段进行image-specific的优化，而本文提出的GLEAN方法则只需要一次前向计算。

![image-20230111203150683](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301132007859.png)



#### [√] 6.Unsupervised Degradation Representation Learning for Blind Super-Resolution

---

论文标题：盲超分辨率的无监督退化表示学习

论文链接：https://arxiv.org/abs/2104.00416

论文代码：https://github.com/LongguangWang/DASR

![image-20230111210652271](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301132007860.png)

针对盲超分辨(blind super-resolution)问题，本文提出了一种非监督退化表示学习的方法来处理退化模型未知的情况。具体而言，本文利用近期在表示学习中大火的对比学习（contrastive learning），来学习不同图片的退化表示，这里假设的是在同一张图片内退化方式是相同的（退化表示相接近），但不同的图片之间是有区别的（退化表示相远离）。接着，本文提出Degradation-Aware SR（DASR）模型，将学习到的退化表示融入到超分辨模型中，以更有效地完成未知退化方式的图像重建。

> alec：
>
> - 本文使用对比学习来学习不同图像的退化表示。认为同一张图片中的退化方式是相同的，不同的图片之间的退化方式是有区别的。
> - 然后利用本文提出的DASR模型，将学习到的退化表示融入到超分模型中。

![image-20230113193327509](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301132007862.png)

![image-20230113193623355](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301132007863.png)



#### [√] 7.Cross-MPI: Cross-scale Stereo for Image Super-Resolution using Multiplane Images

---

论文标题：使用多平面图像实现图像超分辨率的跨尺度立体

论文链接：https://arxiv.org/abs/2011.14631

论文代码：http://www.liuyebin.com/crossMPI/crossMPI.html

![image-20230113194434935](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301132007864.png)

> alec：
>
> - 本文提出的方法属于基于参考的超分
> - 本文主要是利用multiplane image （MPI,多平面图像），提出plane-aware attention-based MPI mechanism，并利用参考图像进行多尺度引导。本文在实验部分都是针对放大倍数x8进行的。

本文提出的方法属于 reference-based super-resolution (RefSR)。RefSR方法最开始被用来解决跨尺度重建问题，其中数据为hybrid multiscale imaging camera systems所得到的不同分辨率的。相比于常规数据驱动的SISR方法，RefSR能更好地应用放大倍数很大的情况（如x8），因为有参考图像的信息。而本文主要是利用multiplane image （MPI,多平面图像），提出plane-aware attention-based MPI mechanism，并利用参考图像进行多尺度引导。本文在实验部分都是针对放大倍数x8进行的。

（关于MPI的介绍，可参考 https://blog.csdn.net/qq_40723205/article/details/114455424）

![image-20230113195243492](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301132007865.png)



#### [√] 8.MASA-SR: Matching Acceleration and Spatial Adaptation for Reference-Based Image Super-Resolution

---

论文标题：基于参考的图像超分辨率的匹配加速度和空间自适应

论文链接：https://jiaya.me/papers/masasr_cvpr21.pdf

论文代码：https://github.com/Jia-Research-Lab/MASA-SR

![image-20230113195415743](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301132007866.png)

本文提出一种名为MASA-SR的RefSR方法。MASA-SR方法主要有两个核心：1）是Match & Extraction Module(MEM), 该模块通过coarse-to-fine的方式寻找到Ref图像和LR图像之间的匹配关系；2）是Spatial Adaption Module (SAM), 该模块用来处理Ref图像到LR图像的特征变换，并适合处理两个图像之间存在color和luminance分布存在比较大的差异的时候。

![image-20230113195547186](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301132007867.png)

![image-20230113195905357](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301132007868.png)



## [√] Reference

---

[1] github: [Awesome-CVPR2021-CVPR2020-Low-Level-Vision](https://github.com/Kobaayyy/Awesome-CVPR2021-CVPR2020-Low-Level-Vision/blob/master/CVPR2021.md#1.超分辨率)

