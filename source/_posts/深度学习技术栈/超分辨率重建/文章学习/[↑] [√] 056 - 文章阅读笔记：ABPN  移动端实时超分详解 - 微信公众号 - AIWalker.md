---
title: 056 - 文章阅读笔记：【ABPN】移动端实时超分详解 - 微信公众号 - AIWalker
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051559239.png
tags:
  - 超分辨率重建
  - 深度学习
  - 移动端超分
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 958041100
date: 2023-02-05 14:13:11
---

> 原文链接：
>
> [ABPN | 移动端实时超分详解 - 微信公众号 - AIWalker](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651688192&idx=1&sn=4011c8d194fa83e5f2a2a3700b1397f2&chksm=f3c9c32dc4be4a3bface7de370a861b5530beca182f9ab131315379297b2b00eb3786ef41c19&cur_album_id=1338480951000727554&scene=189#wechat_redirect)
>
> 2022-03-30 22:00
>
> ps：本文为依据个人日常阅读习惯，在原文的基础上记录阅读进度、记录个人想法和收获所写，关于原文一切内容的著作权全部归原作者所有。



## [√] 论文信息

---

今天要介绍的MobileAI2021的图像超分竞赛的最佳方案，无论是PSNR指标还是推理速度均显著优于其他方案，推理速度**达到了手机端实时(<40ms@1080P)**。

![标题&作者团队](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051626459.jpg)



> alec：
>
> - anchor，锚

## [√] Abstract

---

尽管基于深度学习的图像超分取得前所未有的进展，但实际应用要求i越来越高的性能、效率，尤其是移动端推理效率。智能手机的升级迭代、5G的盛行，用户能感知到的图像/视频分辨率越来越高，从早期的480过度到720p，再到1080p，再到最近的1k、4k。高分辨率需要更高的计算量，占用更多的RAM，这就导致了端侧设备的部署问题。

本文旨在设计一种8-bit量化版高效网络并将其部署到移动端，整个设计过程如下：

- 首先，我们通过将轻量型超分架构分解并分析每个节点的推理延迟，进而确定可利用的算子；
- 然后，我们深入分析了何种类型的架构便于进行8-bit量化并提出了ABPN(Anchor-Based Plain Network)；
- 最后，我们采用量化感知训练(Quantization-Aware Training, QAT)策略进一步提升模型的性能。

我们所设计的模型能以2dB指标优于8-bit量化版FSRCNN，同时满足实际速度需求。

## [√] Method

---

接下来，我们从节点延迟测试开始，然后引出本文方案背后的思考，最后构建所提ABPN。

#### [√] Meta-node Latency（算子的延迟）

---

由于我们的目标在于构建一种实时量化模型用于真实场景(比如实时视频超分)。我们需要做的第一件事就是构建可移植算子集并统计每个算子的耗时。

- 我们将当前轻量型网络(如EDSR、CARN、IMDN、IDN、LatticeNet)进行分解构建初始算子集；
- 我们在Synaptics Dolphin平台(专用NPU)上测试每个算子的延迟。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051626461.jpg)

上述算子可以分为四大类：张量操作、卷积算子、激活算子、resize，见上表。从上表可以得出四个发现：

- **近期的SOTA轻量架构使用的技术似乎难以在移动端部署**。

- - EDSR采用了大量的ResBlock，每个ResBlock会引入`元素加`，该操作甚至比高速优化的卷积还要慢；
    - CARN采用了全局与局部特征集成，每个集成过程包含一个concat与一个卷积，仅仅带来了0.09dB指标提升；
    - 由于大量的特征分离与拼接，IDN与IMDN同样存在端侧部署问题；
    - LatticeNet的部署问题更为严重，它采用了16个CA模块，每个CA模块包含一个元素加、一个元素乘、两个池化层，四个卷积，导致了过高的计算负担。
    - 另一个常见问题：它们都需要保存前面层的特征并采用控制数据流动。这种长距离依赖会导致RAM的低频处理，这是因为端侧内存非常有限。
    - 因此，**我们将不考虑特征融合、特征蒸馏、组卷积以及注意力机制**。

- 尽管卷积的参数量是卷积的9倍，但由于并行计算的缘故，两者的推理速度差别并不大。因此，**我们采用卷积以得到更大感受野**。

- **在激活函数方面，我们选择ReLU**。这是因为它要比Leaky ReLu速度更快，而且i两者导致的性能差异非常小；

- 由于HR与LR之间的坐标映射导致**resize操作的推理速度过慢**。



> 【CPU、GPU、NPU的区别】
>
> 　　1、定义不同，CPU是中央处理器，GPU是图形处理器，而npu则是人工智能处理器。
>
> 　　2、负责内容不同， CPU主要是负责低精度，各种普通的数据，GPU是高精度处理图像数据，npu则是人工智能算法上面运行效率要高于另外两者。
>
> 　　3、工作模式不同， CPU是顺序执行运算，需要一件一件事情来完成。GPU是可以并发执行运算，可以几件事情同时运作。而npu是具备智能的特性， NPR也可以被称之为是神经网络处理器，也就是说这个处理器它是会模仿人的大脑神经网络的。

> tensor = 张量 = 多维向量

#### [√] Anchor-based Residual Learning（基于锚的残差学习）

---

正如前一节所讨论的，能用的算子非常有限。为得到一个好的解决方案，**我们深入分析了架构设计与INT8量化之间的相关性**。

据我们所知，其难度主要在于I2I(Image-to-Image, I2I)映射的高 动态范围，**最直接的想法是生成低标准差权值与激活**。有两种方式可以达成该目的：

- 添加BN层：BN往往被集成在ResBlock中，尽管不会导致额外耗时与内存占用，但会导致0.2dB的性能下降。

- 残差学习：近邻像素往往具有相似的值，很自然的一种选择就是学习残差。残差学习又可以分为以下两种：

- - ISRL：图像空间的残差学习
    - FSRL：特征空间的残差学习。

图像空间的残差学习在早期的工作(如VDSR, DRRN)中有得到应用，而特征空间的残差学习则更多在近期的SOTA方案(如SRGAN、IDN、IMDN)中得到应用并取得了稍优的性能。然而，我们认为：**ISRL更适合于INT8量化**。

从前面Table1中可以看到：**图像空间插值存在不可接受的推理耗时，甚至仅仅一次resize都无法满足实时需求**。为解决该问题，我们提出了ABRL(Anchor-Based Residual Learning)：它直接在LR空间复制每个像素9次为HR空间的每个像素生成锚点。受益于PixelShuffle层，所提ABRL可以通过一个concat+一个元素加操作实现。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051626462.jpg)

上图给出了四种类型残差学习的区别所在，从推理耗时角度来看：

- FSRL仅需要一个元素加操作，它的耗时为5.2ms；
- ABRL包含一个通道拼接与一个元素加，总结耗时15.6ms，约为最近邻插值的四分之一。

所提ABRL有这样两个优点：

- 相比FSRL，ABRL可以显著提升INT8量化模型的性能，提升高达0.6dB；
- 多分枝架构可以通过并行加速，因此ABRL与FSRL的实际推理耗时相当。**ABRL与FSRL的主要耗时源自RAM的访问速度慢**。

> alec：
>
> - 残差学习：近邻像素往往具有相似的值，很自然的一种选择就是学习残差。残差学习又可以分为以下两种：
>     - ISRL：图像空间的残差学习
>     - FSRL：特征空间的残差学习。
> - alec：图像空间的残差学习和特征空间的残差学习不是一个东西吗？
> - alec：图像空间的残差学习指的是全局残差学习，特征空间的残差学习指的是基本块内部的残差学习，在网络的特征提取部分学习残差
> - IDN = 信息蒸馏网络，IMDN = 信息多蒸馏网络

> alec：
>
> - 全局残差学习的时候，需要将LR图像的尺寸提升到HR图像一样大，然后再相加。但是一般是使用最近邻、双线性、双三次插值。由上图可知插值速度很慢，因此本文提出将上述的残差方法替换为ABRL：受益于PixelShuffle层，所提ABRL可以通过一个concat+一个元素加操作实现。

#### [√] Network Architecture

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051626463.jpg)

上图给出了本文所提架构示意图，它包含四个主要模块：

- 浅层特征提取：该过程由![image-20230205153846820](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051626464.png)卷积+ReLU构成，定义如下：

![image-20230205153900435](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051626465.png)

- 深层特征提取：该过程采用多个Conv-ReLU组合构成，描述如下：

为充分利用并行推理，我们设置Conv-ReLu的数量为5以匹配上分支的开销，这意味着**当Conv-ReLU数量小于5时推理速度不变**。最后，我们采用一个卷积将前述特征变换到HR图像空间：

![image-20230205154452268](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051626466.png)

然后再采用本文所提ABRL得到超分特征：

![image-20230205154511465](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051626467.png)

- 重建模块：该模块采用PixelShuffle进对前述所得超分超分进行像素重排得到超分图像。

![image-20230205154757952](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051626468.png)

- 后处理模块：该模块采用Clip操作约束超分输出，即输出最大值不超过255，最小值不小于0。移除该操作会导致输出分布偏移，进而导致量化误差。

> alec：
>
> - 参数量对应内存、计算量对应电量消耗、推理时间对应速度
> - depth to space = 亚像素卷积 = 将通道数量转换为图像尺寸大小

#### [√] Loss Function

---

在损失函数方面，我们采用了简单的L1损失



## [√] Experiments

---

在训练方面，图像块尺寸为$64\times64$，batch=16，优化器为Adam,初始学习率0.001,每200epoch减半，合计训练1000epoch。训练数据为DIV2K，在RGB空间评估性能。

QAT是一种流程的提升模型性能的量化技术且无额外推理耗时。我们设置初始学习率为0.0001，每50epoch减半，合计训练200epoch。QAT可以进一步提升0.06的B性能，此时INT8模型仅比FP32性能低0.07dB。

#### [√] Residual Learning

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051626469.jpg)

上表对比了残差学习的性能、耗时。从中可以看到：

- 对于FP32模型而言，FSRL模型取得了最佳性能，其他模型性能相当；
- 对于INT8模型而言，不带残差的模型会出现严重性能下降(-1.93dB)，FSRL模型会下降0.78dB，而ISRL则则仅仅下降0.13dB。因此，**残差学习可以极大缓解INT8量化过程中的高动态范围问题，而ISRL变现优于FSRL**。



#### [√] Test on Snapdragon 820

---

我们在Snapdragon 820的手机平台上，采用AIBenchmark软件测试了所提方案的CPU、GPU以及NNAPI耗时，结果见下表。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051626470.jpg)

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051626471.jpg)



#### [√] MAI2021 SISR Challenge

---

本文起初用于参加MAI2021图像超分竞赛，结果见下表。注：首次的提交的模型在模型尾部没有添加Clip操作，导致量化性能非常差(小于20dB)；在竞赛结束后才解决了该问题并提交了校正后模型。受益于素体ABRL，所提方案取得了最佳PSNR指标，同时具有更快的推理速度。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051626472.jpg)