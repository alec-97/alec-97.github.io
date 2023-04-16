---
title: >-
  053 - 文章阅读笔记：Transformer再下一城！low-level多个任务榜首被占领，北大华为等联合提出预训练模型IPT - 微信公众号 -
  AIWalker
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032003804.jpg
tags:
  - 图像超分辨率重建
  - 深度学习
  - transformer
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 4022388376
date: 2023-02-03 17:59:53
---

> 原文链接：
>
> [Transformer再下一城！low-level多个任务榜首被占领，北大华为等联合提出预训练模型IPT - 微信公众号 - AIWalker](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651682736&idx=1&sn=d8f48cacf9dcf82efb66f687d6d1f6f0&scene=21#wechat_redirect)
>
> 2020-12-02 22:30
>
> ps：本文为依据个人日常阅读习惯，在原文的基础上记录阅读进度、记录个人想法和收获所写，关于原文一切内容的著作权全部归原作者所有。

## [√] 导读

---

来自Transformer的降维打击！北京大学等最新发布论文，联合提出图像处理Transformer。通过对low-level计算机视觉任务，如降噪、超分、去雨等进行研究，提出了一种新的预训练模型IPT，占领low-level多个任务的榜首。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004217.jpg)

本文是北京大学&华为诺亚等联合提出的一种图像处理Transformer。Transformer自提出之日起即引起极大的轰动，BERT、GPT-3等模型迅速占用NLP各大榜单；后来Transformer被用于图像分类中同样引起了轰动；再后来，Transformer在目标检测任务中同样引起了轰动。**现在Transformer再出手，占领了low-level多个任务的榜首，甚至它在去雨任务上以1.6dB超越了已有最佳方案。**

论文链接: https://arxiv.org/abs/2012.00364

> alec：
>
> - IPT:预训练模型IPT，使用transformer在多个low-level任务上取得最佳效果。



## [√] 摘要

---

随机硬件水平的提升，在大数据集上预训练的深度学习模型(比如BERT，GPT-3)表现出了优于传统方法的有效性。transformer的巨大进展主要源自其强大的特征表达能力与各式各样的架构。

在这篇论文中，**作者对low-level计算机视觉任务（比如降噪、超分、去雨）进行了研究并提出了一种新的预训练模型：IPT(image processing transformer)**。为最大挖掘transformer的能力，作者采用知名的ImageNet制作了大量的退化图像数据对，然后采用这些训练数据对对所提IPT(它具有多头、多尾以适配多种退化降质模型)模型进行训练。此外，**作者还引入了对比学习以更好的适配不同的图像处理任务**。经过微调后，预训练模型可以有效的应用不到的任务中。仅仅需要一个预训练模型，IPT即可在多个low-level基准上取得优于SOTA方案的性能。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004218.jpg)

上图给出了所提方案IPT与HAN、RDN、RCDNet在超分、降噪、去雨任务上的性能对比，IPT均取得了0.4-2.0dB不等的性能提升。

> alec：
>
> - 预训练模型：在大数据集上预训练的深度学习模型
> - transformer具有强大的特征表征能力和各种各样的架构
> - IPT = image processing transformer
> - 在大数据集ImageNet上预训练，具有多头、多尾的结构，通过替换头尾能够适用于不同的low-level任务
> - 作者还引入了对比学习以适配不同的图像处理任务
> - 2021年提出的这个IPT比2020年的HAN这个混合注意力的模型，性能提升不少。

## [√] 方法

---

为更好的挖掘Transformer的潜力以获取在图像处理任务上的更好结果，作者提出了一种ImageNet数据集上预训练的图像处理Transformer，即IPT。



#### [√] IPT architecture

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004219.jpg)

上图给出了IPT的整体架构示意图，可以看到它包含四个成分：

- 用于从输入退化图像提取特征的Heads；
- encoder与decoder模块用于重建输入数据中的丢失信息；
- 用于输出图像重建的Tails。



> alec：
>
> - 结构：对于特征图，以通道维度为方向进行线性flatten，拿到一维向量，然后通过transformer编码器编码，然后输入到transformer解码器解码，然后将输出reshape为HWC形状，得到输出。



#### [√] Heads

---

![image-20230203183133392](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004220.png)

> alec：
>
> - 多头机制中的每个头是采用的三个卷积层

#### [√] Transformer encoder

---

![image-20230203190839051](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004221.png)

为编码每个块的位置信息，作者还在encoder里面添加了可学习的位置编码信息![image-20230203191648094](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004222.png)。这里的encoder延续了原始Transformer，采用了多头自注意力模块和前向网络。

encoder的输出表示为![image-20230203191745728](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004223.png),它与输入块尺寸相同，encoder的计算过程描述如下：

![image-20230203191855130](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004225.png)

> alec：
>
> - f~pi~表示原输入的块，E~pi~表示可学习的位置编码信息
> - 将输入通过LN层，得到QKV。（LN = Layer Normalization）
> - QKV输入多头自注意力模块MSA得到y’，然后将y‘通过前馈前向网络FFN得到y。
> - 上面公式的计算过程就是STL
> - MSA是STL模块的一部分
>
> ![image-20230203192430353](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004226.png)



> alec：
>
> - 将特征输入到transformer之前，需要将HWC的特征拆分成块，每个块称为word。
> - 理解：这里的块类似于NLP中的一个单词对应的向量。将这些“单词”送到transformer中，transformer能够对这些单词之间的关联进行建模。
> - transformer编码：多头自注意力模块 + 前向网络

其中l表示encoder的层数，MSA表示多头自注意力模块，FFN表示前馈前向网络(它仅包含两个全连接层)。

#### [√] Transformer decoder

---

decoder采用了与encoder类似的架构并以encoder的输出作为输入，它包含两个MSA与1个FFN。它与原始Transformer的不同之处在于：采用任务相关的embedding作为额外的输入，这些任务相关的embedding![image-20230203193348093](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004227.png)用于对不同任务进行特征编码。decoder的计算过程描述如下：

![image-20230203193354877](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004228.png)

其中![image-20230203193428085](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004229.png)表示decoder的输出。decoder输出的N个尺寸为![image-20230203193437569](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004230.png)的块特征将组成特征![image-20230203193449241](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004231.png)。

> alec：
>
> - 解码的时候，加上了任务相关的embedding，将任务相关的embedding加到Q和K里面，用于计算注意力。
>
> - 解码的计算过程，和编码的STL相比，将STL的MSA部分重复了1次，然后加入了任务相关的embedding。



#### [√] Tails

---

![image-20230203193631899](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004232.png)

#### [√] Pre-training on ImageNet

---

除了transformer的自身架构外，成功训练一个优化transformer模型的关键因素为：大数据集。而图像处理任务中常用数据集均比较小，比如图像超分常用数据DIV2K仅仅有800张。针对该问题，作者提出对知名的ImageNet进行退化处理并用于训练所提IPT模型。

这里的退化数据制作采用了与图像处理任务中相同的方案，比如超分任务中的bicubic下采样，降噪任务中的高斯噪声。图像的退化过程可以描述如下：

![image-20230203194042431](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004233.png)

其中f表示退化变换函数，它与任务相关。对于超分任务而言，![image-20230203194148036](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004234.png)表示bicubic下采样；对于降噪任务而言，![image-20230203194200413](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004235.png)。IPT训练过程中的监督损失采用了常规的![image-20230203194207830](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004236.png)损失，描述如下：

![image-20230203194230527](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004237.png)

上式同样表明：**所提方案IPT同时对多个图像处理任务进行训练**。也就说，对于每个batch，随机从多个任务中选择一个进行训练，每个特定任务对应特定的head和tail。在完成IPT预训练后，我们就可以将其用于特定任务的微调，此时可以移除掉任务无关的head和tail以节省计算量和参数量。

【对比学习】

除了上述监督学习方式外，作者还引入了对比学习以学习更通用特征以使预训练IPT可以应用到未知任务。对于给定输入![image-20230203194559050](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004238.png)(随机从每个batch中挑选)，其decoder输出块特征描述为![image-20230203194611095](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004239.png)。作者期望通过对比学习最小化同一图像内的块特征距离，最大化不同图像的块特征距离，这里采用的对比学习损失函数定义如下：

![image-20230203194620112](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004240.png)

其中![image-20230203194709597](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004241.png)表示cosine相似性。为更充分的利用监督与自监督信息，作者定义了如下整体损失：

![image-20230203194727617](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004242.png)

> alec：
>
> - 数据和标签的损失是监督损失
> - 对比学习的损失是自监督损失

> alec：
>
> - 除了transformer的架构之外，成功训练一个优化transformer模型的关键因素为：大数据集。数据集的数据量一定要多才能充分发挥transformer的数据拟合能力。
>
> - corrupt，腐败的、腐烂的、玷污的
> - 本文的方案同时选择多个任务进行预训练，每个batch随机选择一个任务的batch进行训练。在预训练完之后，再在目标任务上进行微调。

## [√] 实验

---

#### [√] Datasets

---

作者采用ImageNet数据制作训练数据，输入图像块大小为![image-20230203195053772](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004243.png)，大约得到了10M图像数据。采用了6种退化类型：x2、x3、x4、noise-30、noise-50以及去雨。

#### [√] Training&Fine-tuning

---

作者采用32个NVIDIA Tesla V100显卡进行IPT训练，优化器为Adam，训练了300epoch，初始学习率为![image-20230203195155667](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004244.png)，经200epoch后衰减为![image-20230203195211476](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004245.png)，batch=256。在完成IPT预训练后，对特定任务上再进行30epoch微调，此时学习率为![image-20230203195214292](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004246.png)。

#### [√] Super-resolution

---

下表&下图给出了超分方案在图像超分任务上的性能与视觉效果对比。可以看到：

- IPT取得了优于其他SOTA超分方案的效果，甚至在Urban100数据集上以0.4dB优于其他超分方案；
- IPT可以更好重建图像的纹理和结构信息，而其他方法则会导致模糊现象。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004247.jpg)

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004248.jpg)













#### [√] Denoising

---

下表&下图给出了所提方法在降噪任务上的性能与视觉效果对比，可以看到：

- 在不同的噪声水平下，IPT均取得了最佳的降噪指标，**甚至在Urban100数据上提升高达2dB**。
- IPT可以很好的重建图像的纹理&结构信息，而其他降噪方法则难以重建细节信息。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004249.jpg)

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004250.jpg)

#### [√] Deraining

---

下表&下图给出了所提方法在图像去雨任务上的性能与视觉效果对比。可以看到：

- 所提方法取得了最好的指标，甚至取得了1.62dB的性能提升；
- IPT生成的图像具有与GT最详尽，且具有更好的视觉效果。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004251.jpg)

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004252.jpg)





#### [√] Generalization Ability

---

为说明所提方法的泛化性能，作者采用了未经训练的噪声水平进行验证，结果见下表。可以看到：尽管未在该其噪声水平数据上进行训练，所提IPT仍取得了最佳的指标。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004253.jpg)







#### [√] Ablation Study

---

下图对比了IPT与EDSR在不同数量训练集上的性能对比，可以看到：当训练集数量较少时，EDSR具有更好的指标；而当数据集持续增大后，EDSR很快达到饱和，而IPT仍可持续提升并大幅超过了EDSR。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004254.jpg)

> alec：
>
> - 在数据量少的时候，transformer模型的性能没有被充分的激发，这个时候CNN模型的性能高于transformer，随着数据量的上升，transformer模型的性能得到发挥，此时transformer的性能高于CNN模型。

下表给出了对比损失对于模型性能影响性分析(x2超分任务)。当仅仅采用监督方式进行训练时，IPT的指标为38.27；而当引入对比学习机制后，其性能可以进一步提升0.1dB。这侧面印证了对比学习对于IPT预训练的有效性。

![image-20230203200235670](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302032004255.png)

> alec：
>
> - 对比学习机制的加入就是加入了一个对比损失？

全文到此结束，对此感兴趣的同学建议阅读原文。







