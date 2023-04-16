---
title: 063 - 文章阅读笔记：【RLFN】NTIRE2022-ESR 冠军方案RLFN解析 - 微信公众号 - AIWalker
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302092203060.jpg
tags:
  - 轻量化
  - 深度学习
  - 人工智能
  - 超分辨率重建
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 1733414390
date: 2023-02-09 18:13:19
---

> 原文链接：
>
> [NTIRE2022-ESR 冠军方案RLFN解析 - 微信公众号 - AIWalker](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651690304&idx=1&sn=86373aec71a75125007b6a3e51973ea5&chksm=f3c9db6dc4be527b9a1db97183924ee2af1811850936fcd3dc0676c6e373b58f43e3d861a84f&scene=178&cur_album_id=1338480951000727554#rd)
>
> 2022-12-04 08:00
>
> ps：本文为依据个人日常阅读习惯，在原文的基础上记录阅读进度、记录个人想法和收获所写，关于原文一切内容的著作权全部归原作者所有。



## [√] 文章信息

---

近年来，Efficient Super-Resolution(ESR)的研究主要聚焦于参数量与FLOPs的降低，这些方案往往通过复杂的层连接策略进行特征聚合(比如IMDN与RFDN中的特征蒸馏与聚合)。但是，这种复杂的结构不利于高推理速度需求，进而导致这些方案难以部署到资源有限的设备上。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302092203224.jpg)

论文标题：RLFN：Residual Local Feature Network for Efficient Super-Resolution

中文标题：用于高效超分辨率的局部特征残差网络

论文链接：https://arxiv.org/pdf/2205.07514

论文代码：https://github.com/fyan111/RLFN

论文发表：NTIRE 2022

论文简介：

> 【痛点】
>
> 近年来，Efficient Super-Resolution(ESR)的研究主要聚焦于参数量与FLOPs的降低，这些方案往往通过复杂的层连接策略进行特征聚合(比如IMDN与RFDN中的特征蒸馏与聚合)。但是，这种复杂的结构不利于高推理速度需求，进而导致这些方案难以部署到资源有限的设备上。
>
> 
>
> 【本文要点】
>
> （1）本文提出了一种新的ESR方案RLFN(Residual Local Feature Network)，**它采用三个卷积层进行残差局部特征学习以简化特征聚合**，这种处理机制有助于达成更优的性能-推理耗时均衡。
>
> （2）【改进对比损失】与此同时，本文对主流的对比损失(Contrastive Loss)进行回顾并发现：**特征提取器的中间特征选择对于性能有极大影响，其中浅层特征可以保持更精确的细节与纹理**。
>
> （3）【提出训练策略】此外，**本文提出一种新颖的多阶段热启动(warm-start)训练策略**。
>
> 
>
> 【RLFB】
>
> 为了提高推理速度，RLFB去除了RFDB中的蒸馏机制，为了补充性能损失，RLFB采用更大的通道数，从48提升到了52.（轻量化降低了性能损失，可以通过增大通道数来补足性能损失。）
>
> 
>
> 【重新思考对比损失】
>
> 为进一步改进对比损失，我们将特征提取器中的ReLU激活替换为Tanh。
>
> 【多阶段的热启动训练策略】
>
> X3与X4模型训练采用X2模型参数作为预训练参数已成为一种常用trick。但是，这种好处我们只能享受一次，因为预训练模型与目标模型的尺度因子不一致。
>
> 为解决上述局限性，本文提出了一种多阶段热启动训练策略，它可以进一步改善模型性能。
>
> - 在第一个阶段，我们从头开始训练RLFN；
> - 在下一个阶段，我们以前一阶段训练的RLFN进行初始化(此为热启动)。
>
> 【零碎点】
>
> - ESA是RFDB中使用的用于计算空间注意力的模块。
> - 本文RLFB模块是对RFDN模型中的RFDB模块的简化。去除了蒸馏的机制。
> - RLFN的主要结构是引入了注意力模块ESA，实现了高效轻量的超分。
> - RFDB尽管大幅度降低参数量，但是严重降低了推理速度。

本文提出了一种新的ESR方案RLFN(Residual Local Feature Network)，**它采用三个卷积层进行残差局部特征学习以简化特征聚合**，这种处理机制有助于达成更优的性能-推理耗时均衡。与此同时，本文对主流的对比损失(Contrastive Loss)进行回顾并发现：**特征提取器的中间特征选择对于性能有极大影响，其中浅层特征可以保持更精确的细节与纹理**。此外，**本文提出一种新颖的多阶段热启动(warm-start)训练策略**。

在改进对比损失与训练策略加持下，所提RLFN取得了比其他SOTA ESR方案更快的推理速度，同时具有相当的PSNR与SSIM指标。值得一提的是，所提方案RLFN取得了NTIRE2022 ESR竞赛主赛道冠军。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302092203225.jpg)



## [√] 本文方案

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302092203226.jpg)

上图给出了本文所提RLFN整体架构示意图，它主要包含 三部分：

- 浅层特征提取：该部分由一个卷积构成；
- 深层特征提取：该部分由多个堆叠RLFB(Residual Local Feature Block)构成；
- 图像重建模块：该部分由卷积与PixelShuffle构建。

总体来说，RLFN是一种类EDSR的架构。RLFN的核心模块在于其所设计的RLFB模块(见上图b)。RLFN是在RFDN的基础上演变而来，关于RFDN的介绍可以参考：[**AIM2020-ESR冠军方案解读：引入注意力模块ESA，实现高效轻量的超分网络**](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651682112&idx=1&sn=690a05c704e875d298b66e386ecc9239&scene=21#wechat_redirect)。

RFDB采用渐进式特征提炼与特征蒸馏方式提取更强力特征，其特征蒸馏通过卷积实现，特征聚合通过Concat完成。尽管RFDB的这种处理方式可以大幅降低参数量，但同时严重影响了推理速度。

为此，本文提出了RLFB(见Figure3-b)，它可以大幅减少推理耗时，同时保持模型容量。从图示可以看到：**RLFB消除了特征蒸馏链接，仅通过堆叠Conv-ReLU进行局部特征提取**。此外，**RLFB保留了RFDB中的ESA模块**。为补充性能损失，**RLFB采用更大的通道数**，从48提升到了52.



> alec：
>
> - RLFN的主要结构是引入了注意力模块ESA，实现了高效轻量的超分。
> - RFDB尽管大幅度降低参数量，但是严重降低了推理速度。
> - 为了提高推理速度，RLFB去除了RFDB中的蒸馏机制，为了补充性能损失，RLFB采用更大的通道数，从48提升到了52.（轻量化降低了性能损失，可以通过增大通道数来补足性能损失。）

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302092203227.jpg)

为进一步降低推理耗时，本文采用剪枝敏感性分析工具对ESA模块的冗余性进行了分析，可以看到：ConvGroups中的三个卷积的冗余性排名1、3、4。因此，**本文将ESA中的ConvGroups的卷积数减少到1**。

#### [√] Revisiting the Contrastive Loss

---

对比学习已在自监督学习领域表现出了惊人的性能，在超分领域也开始有所探索，其损失定义如下：

![image-20230209195914603](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302092203228.png)

CSD与AECR-Net提取VGG19的1、3、5、9以及13层的特征。但是，我们发现：当采用上述CL时，PSNR会出现下降现象。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302092203229.jpg)

本文通过特征可视化对此差异进行了探究并发现：**深层提取的特征具有更强语义信息，但缺乏精确的细节**。总而言之，**深层特征有助于改善感知质量，而浅层特征特征则有助于提供更精确的细节与纹理(而这对于PSNR导向的模型非常重要)\**。也就是说，\**我们需要采用浅层特征以改善模型的PSNR指标**。为进一步改进对比损失，**我们将特征提取器中的ReLU激活替换为Tanh**。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302092203230.jpg)

由于VGG19是采用ReLU激活训练而来，直接进行激活函数替换无法确保其性能。而近期的一些研究表明：随机初始化的正确架构已足以捕获感知细节信息。受此启发，本文构建一个随机初始化的两层特征提取器(Conv_k3s1-Tanh-Conv-K3s1)。从上图可以看到：**本文所提特征提取器具有更强的响应，可以捕获更多细节与纹理**(见上图b)。也就是说，随机初始化的特征器已可以捕获结构信息，预训练并非必要的。

> alec：
>
> - CL = 对比损失

#### [√] Warm-Start Strategy

---

X3与X4模型训练采用X2模型参数作为预训练参数已成为一种常用trick。但是，这种好处我们只能享受一次，因为预训练模型与目标模型的尺度因子不一致。

为解决上述局限性，本文提出了一种多阶段热启动训练策略，它可以进一步改善模型性能。

- 在第一个阶段，我们从头开始训练RLFN；
- 在下一个阶段，我们以前一阶段训练的RLFN进行初始化(此为热启动)。

## [√] 消融实验

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302092203231.jpg)

上图给出了从RFDB到RLFB的模块优化对比，这里主要对比了两种RFDB变种(移除了特征蒸馏链接)。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302092203232.jpg)

从下表2可以看到：**相比RFDB，RLFB具有同等复原性能，同时具有明显的速度优势**。从上表3可以看到：**移除ESA ConvGroups中的两个卷积并不会牺牲性能，但会加速模型推理速度**。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302092203233.jpg)

上表对所提对比损失的有效性进行了对比，可以看到：**在四个基准数据集上，所提CL均可一致的提升模型性能**。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302092203234.jpg)

上表对多阶段热启动训练策略的有效性进行了对比，可以看到：**多阶段热启动确实可以提升模型性能**。这意味着：该训练策略有助于跳出局部最优并改进模型整体性能。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302092203235.jpg)

上表对所提对比损失与热启动策略的泛化性进行了验证，可以看到：**所提方案具有普适性，可以用于其他SISR方案**。

> alec：
>
> - 运行时间的计算是在DIV2K数据集上运行10次的平均时间。
> - halve = 把什么分成两半



## [√] SOTA方案对比

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302092203236.jpg)

上表给出了不同方案在X2与X4任务上的性能对比，可以看到：

- 相比其他方案，**RLFN-S与RLFN取得了更优的PSNR与SSIM指标**；
- 相比RFDN，RLFN-S取得相当的性能，同时参数量更少；
- 总而言之，**RLFN取得了更优的性能-推理耗时均衡**。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302092203237.jpg)

在NTIRE2022-ESR竞赛中，所提RLFN取得了主赛道第一，sub-track2赛道第二的成绩。竞赛所选用的RLFN_cut只有4个RLFB模块，通道数为48，ESA中的通道数为16。上表给出了NTIRE2022-ESR竞赛的结果对比，可以看到：**相比基线IMDN与RFDN，所提方案取得了全方位的性能提升，同时具有最快推理速度**。





## [√] 推荐阅读

---

1. [NAFNet ：无需非线性激活，真“反直觉”！但复原性能也是真强！](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651688451&idx=1&sn=fa0b1a57284affcdf13a1d0d12521fab&scene=21#wechat_redirect)
2. [真实用！ETH团以合成数据+Swin-Conv构建新型实用盲图像降噪](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651688528&idx=1&sn=8d11efd50d221bd97572a82e0e67069d&scene=21#wechat_redirect)
3. [ELAN | 比SwinIR快4倍，图像超分中更高效Transformer应用探索](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651688234&idx=1&sn=3665e1eb2a0b8cd5d60453ff56a1e2c1&scene=21#wechat_redirect)
4. [AIM2020-ESR冠军方案解读：引入注意力模块ESA，实现高效轻量的超分网络](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651682112&idx=1&sn=690a05c704e875d298b66e386ecc9239&scene=21#wechat_redirect)
5. [CVPR 2022 Oral | MLP进军底层视觉！谷歌提出MAXIM模型刷榜多个图像处理任务](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651688367&idx=1&sn=3d3097322bd56bf9d08b3ec8520f4091&scene=21#wechat_redirect)
6. [CVPR 2022 Oral | MLP进军底层视觉！谷歌提出MAXIM模型刷榜多个图像处理任务，代码已开源](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651688367&idx=1&sn=3d3097322bd56bf9d08b3ec8520f4091&scene=21#wechat_redirect)
7. [ELAN | 比SwinIR快4倍，图像超分中更高效Transformer应用探索](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651688234&idx=1&sn=3665e1eb2a0b8cd5d60453ff56a1e2c1&scene=21#wechat_redirect)
8. [CNN与Transformer相互促进，助力ACT进一步提升超分性能](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651687821&idx=1&sn=605624f3ae6289b5f2f0e10420ead7db&scene=21#wechat_redirect)
9. [CVPR2022 | Restormer: 刷新多个low-level任务指标](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651687716&idx=1&sn=0963c52eec68b51c023ec05b53c948cc&scene=21#wechat_redirect)
10. [Transformer在图像复原领域的降维打击！ETH提出SwinIR：各项任务全面领先](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651685845&idx=1&sn=d52f8937588231c8efc3eaa6c5662d06&scene=21#wechat_redirect)