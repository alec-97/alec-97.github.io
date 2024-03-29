---
title: 文章阅读笔记：【2020 USRnet】Deep Unfolding Network for Image Super-Resolution
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232046161.png
tags:
  - 盲超分
password: 972274
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 盲超分
abbrlink: 973653310
date: 2023-02-23 17:32:30
---

> 原文链接：
>
> （1）【√】CVPR2020：USRNet - AIWalker - Happy（[link](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651679715&idx=1&sn=615b3992e07595ee313082318c0ff48a&chksm=f3c925cec4beacd8a48985dcfd0be9525422012717d7edc52e68aaa016899251c8a4c02d5baf&scene=178&cur_album_id=1338480951000727554#rd)）
>
> 2020-05-13 20:02
>
> ps：本文为依据个人日常阅读习惯，在原文的基础上记录阅读进度、记录个人想法和收获所写，关于原文一切内容的著作权全部归原作者所有。





#  [√] 文章信息

---

论文标题：【2020 USRnet】Deep Unfolding Network for Image Super-Resolution

中文标题：用于图像超分的可展开网络

论文链接：https://arxiv.org/abs/2003.10428

论文代码：https://github.com/cszn/USRNet

论文发表：2020CVPR



# [√] 文章1

---

> 总结：
>
> 【阅读收获】
>
> - 不同于基于建模的方法可以在统一的MAP框架下处理不同尺度、模糊核以及噪声水平的图像超分，基于学习的图像超分缺乏上述灵活性。
> - 本文是首个将降质过程和超分过程合在一起构成一个端到端的盲超分模型。且本文方法结合基于建模的方法的灵活性和基于学习方法的优势。
> - 本文验证数据使用了12个模糊核（4个各向同性、4个各向异性、4个运动模糊核）
> - 各项同性指的是对二维图像的各个方向的影响相同，各项一定则是对二维图像的各个方向影响不同。
>
> 
>
> 【阅读灵感】
>
> - 超参数模块起滑动条的作用去控制数据模块和先验模块。该超参数模块包含三个全连接层，前两个后接ReLU，最后一个后接Softplus，隐含层的通道数为64。具体代码实现见Reference部分。
> - 本文使用了12个、3组模糊核生成验证数据用来验证模型，自己也可以这样做来充分验证，且同时能够重试工作量。
>
> - - ![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049901.png)
>
> 
>
> 【中心思想】
>
> - 为解决上述问题，作者提出一种端到端可训练展开的网络，它集成了基于学习与基于建模的方法。通过half-quadratic splitting算法将MAP推理进展展开，通过固定次数的迭代求解数据子问题与先验子问题。上述两个子问题可以通过神经网络模块进行求解，从而得到一个可端到端训练的迭代网络。因此，所提网络不仅具有建模方法的灵活性(可处理不同尺度、模糊、噪声降质问题)，同时具有学习方法的高性能。最后作者通过实验证实了所提方法在灵活性、高效性以及泛化性能方面的优势。
> - 本文的贡献主要包含以下几点：
>
> - - 提出一种端到端可训练展开的图像超分网络，它是首个采用单个端到端模型解决经典降质问题(不同尺度、模糊核以及噪声水平)的方法；
>     - `USRNet`集成了建模方法的灵活性与学习方法的优势，为缩小两者差异构建一条通路；
>     - `USRNet`从根本上将降质约束与先验约束信息嵌入到求解过程中；
>     - `USRNet`在不同降质配置下的LR图像均具有极好的性能，在实际应用中具有极大的潜力。
>
> - Zhang Kai之前的DPSR和本USRnet，都是传统方法和深度学习方法相结合而产生的的，利用了建模方法中的概率论等的知识。
>
> 
>
> 【网络架构】
>
> ![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049902.png)
>
> - 数据模块用于寻求清晰的HR图像得到z_k
> - 先验模块用于对清晰的HR图像z_k进行降噪得到的清晰的HR图像x_k



## [√] 简介

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049903.png)

paper：https://arxiv.org/abs/2003.10428 code：https://github.com/cszn/USRNet

> 本文作者是`Kai Zhang`，也许你对这个名字不是那么熟悉，但是`DPSR`,`SRMD`,`IRCNN`, `FFDNet`,`CBDNet`，`DnCNN`这几个图像复原领域知名方法你一定听过至少一个。个人非常推崇的一位大神，专注用深度学习技术解决实际画质问题。

## [√] Abstract

---

相比传统方法，受益于端到端训练，基于学习的图像超分方法取得了越来越好的性能(无论是性能还是计算效率)。然而，不同于基于建模的方法可以在统一的MAP框架下处理不同尺度、模糊核以及噪声水平的图像超分，基于学习的图像超分缺乏上述灵活性。

为解决上述问题，作者提出一种端到端可训练展开的网络，它集成了基于学习与基于建模的方法。通过`half-quadratic splitting`算法将MAP推理进展展开，通过固定次数的迭代求解数据子问题与先验子问题。上述两个子问题可以通过神经网络模块进行求解，从而得到一个可端到端训练的迭代网络。因此，所提网络不仅具有建模方法的灵活性(可处理不同尺度、模糊、噪声降质问题)，同时具有学习方法的高性能。最后作者通过实验证实了所提方法在灵活性、高效性以及泛化性能方面的优势。

本文的贡献主要包含以下几点：

- 提出一种端到端可训练展开的图像超分网络，它是首个采用单个端到端模型解决经典降质问题(不同尺度、模糊核以及噪声水平)的方法；
- `USRNet`集成了建模方法的灵活性与学习方法的优势，为缩小两者差异构建一条通路；
- `USRNet`从根本上将降质约束与先验约束信息嵌入到求解过程中；
- `USRNet`在不同降质配置下的LR图像均具有极好的性能，在实际应用中具有极大的潜力。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049904.png)

> alec：
>
> - MAP = 最大后验概率

## [√] Method

---

#### [√] Degradation Model

---

一般而言，广义降质过程可以通过如下公式进行刻画：

![image-20230223180854061](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049905.png)

其中研究最多的当属双三次插值降质，事实上双三次插值降质也可以通过上述公式配合合适模糊近似。与此同时，我们可以采用数据驱动的方式求解上述核估计问题，优化目标如下：

![image-20230223180900597](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049906.png)

下图给出了不同尺度下的双三次核近似估计。注：由于下采样操作是选择每个块的左上角像素，因此所估计的双三次核分别偏离中心0.5,1,1.5个像素。

![image-20230223180921044](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049907.png)









#### [√] Unfolding optimization

---

从MAP框架的角度来看，HR图像的可以通过最小化如下目标函数进行估计：

![image-20230223181057510](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049908.png)

为得到上述公式的展开推理，作者选择了`half-quadratic spliting, HQS`算法(因其简洁性与快速收敛性)。`HQS`通过引入辅助变量z对上述公式进行求解，从而有如下等价近似：

![image-20230223181104661](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049909.png)

其中为惩罚性参数。上述优化目标可以通过迭代求解x与z进行解决：

![image-20230223181112809](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049910.png)

从上述公式推导可以看到：数据项与先验项可以进行解耦。对于数据项而言，可以采用快速傅里叶变换进行求解；对于先验项而言，它等价于降噪问题。

#### [√] Deep unfolding Network

---

一定展开优化方案得到确定，下一步的任务是如何一种有效的展开超分网络。由于展开优化主要包含迭代优化数据子问题与先验子问题，`USRNet`需要在数据模块与先验模块之间进行交替执行。与此同时，子问题的求解同样需要将超分数α\_k，β\_k纳入输入中；更进一步，作者引入了一个超参数模块H。下图给出了包含K次迭代的`USRNet`架构图，为均衡速度与精度，作者设置K=8。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049911.png)

![image-20230223181630049](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049912.png)

![image-20230223181950241](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049913.png)

![image-20230223182214001](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049914.png)









#### [√] End-to-End training

---

![image-20230223182545329](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049915.png)

## [√] Experiments

---

在验证阶段，作者选用广泛采用的`BSD68`数据集，为合成LR图像，需要提供模糊核以及噪声水平等信息。相关模糊核（包含4个不同核宽度的各项同性高斯核，4个各向异性高斯核，4个运动模糊核共计12个模糊核，作者认为12个模糊核足以覆盖非常大的核空间）示意图以及模型`PSNR`性能对比见下表。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049916.png)

尽管各项异性高斯核杜宇超分任务而言已经充足，但是在实际应用中可以处理更复杂模糊核的超分方法会更受欢迎。因此，很有必要进一步分析不同方法对于模糊核的鲁棒性。

#### [√] PSNR results

---

上表给出了不同方法在不同模糊核以及噪声水平的性能对比(对比方法包含`RCAN，ZSSR， IKC，IRCNN`，其中`RCAN`在双三次降质方面具有`SOTA`指标，`ZSSR`为用于处理各项异性高斯核模糊的少样本学习方法，`IKC`是一种适合于处理各项同性高斯核模糊的盲迭代核校正方法,`IRCNN`是一种非盲深度降噪方法，同时为公平对比，作者对`IRCNN`进行任务适配)。

从上表可以得出以下几点发现：

- `USRNet`在不同尺度因子、模糊核以及噪声水平降质问题中取得了优于其他方法的性能；
- `RCAN`在降质过程类似双三次的配置下可以取得很好的效果，但是当降质过程存在较大差异时性能急剧下降；
- `ZSSR`在小尺度且各项异性与各项同性高斯核降质方面表现良好，在其他类型降质表现较差；
- `IKC`在各项异性高斯核与运动核方面泛化性并不好。

尽管`USRNet`并非为双三次降质设计，但有意思的是：通过采用近似双三次核作为输入，它仍可取得比较好的性能，见下表。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049917.png)

#### [√] Visual results

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049918.png)

上图给出了不同方法在无噪4倍超分下的视觉效果对比。从中可以看到：(1) `USRNet`与`USRGAN`具有更好的视觉效果，其中`USRGAN`具有最佳视觉效果。(2)`RankSRGAN`对于双三次降质表现良好，对于运动模糊核降质性能较差，而`USRGAN`则可以灵活处理不同降质问题。

#### [√] Analysis

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049919.png)

上图给出了不同迭代阶段的HR重建效果对比图。可以看到：在前面几次迭代过程中`USRNet`与`USRGAN`具有相似的结果，但最后几次迭代`USRGAN`可以重建更好的结构与纹理。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049920.png)

上图给出了超参数模块在不同尺度与噪声组合下的输出。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049921.png)

正如前面所提到的：由于数据项与先验的解耦，所提方法具有较好的泛化性能。上图给出了所提方法在更大核下的图像复原效果，可以看到`USRNet, USRGAN`均能重建视觉效果良好的结果。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232049922.png)

上图给出了所提方法在真实降质下的超分效果对比，可以看到超分重建效果是真的非常赞。



## [√] Conclusion

---

作者聚焦于经典的图像超分问题并提出一种深度展开超分网络。受启发于传统建模方法的展开优化，作者设计了一种端到端的深度展开网络，它集成了建模方法与学习方法的优势。该方法的主要优势在于：单模型处理多种降质问题。更具体的说，所提模型包含三个可解释模块：(1)数据模块，它可以使HR更清晰；(2) 先验模块，它可以更精确的估计先验信息；(3)超参模块，它可以更精确的控制上述两个模块的输出。

总而言之，所提方法将降质模型与先验信息约束到解空间中，从而具有更好的性能与泛化性能。与此同时，作者还通过实验验证了所提方法的灵活性、高效性以及泛化性能。











## [√] Postscript

---

这两天前前后后大概花费了近四个小时研究论文的思想与代码实现，只能说被作者的硬核公式推导震撼到了。尤其是快速傅里叶变换部分的`pytorch`实现，虽然基本看懂了作者的实现与细节信息。但是，我自己是写不出来那些代码的，虽然每一个代码功能都曾经看到过matlab版本或者C++版本的，但确实做不到抽丝剥茧一层层的将其还原出来。

`USRNet`的代码实现需要的知识点：复数理论、傅里叶变换、psf2otf等。所以想看到作者提供的源码真是不容易，目前非常期待作者的预训练模型开源。

最后不得不配合的就是作者将传统方法与深度学习相结合的深厚功底了，作者在之前DPSR一文中就已经将传统方法同深度学习相结合过，这篇论文可以说是更进一步的结合，这篇论文可以视作前一篇论文的进阶版。

尽管作者目前仅开源了`USRNet`的实现部分代码，尚未开源预训练模型。不过既然知道`USRNet`与DPSR两者之间的关联性，那么数据制作可以直接采用作者提供的DPSR数据制作方法，最后剩下的就是模型训练了，应该就更容易了。期待诸君能尽快复现并开源`USRNet`。











