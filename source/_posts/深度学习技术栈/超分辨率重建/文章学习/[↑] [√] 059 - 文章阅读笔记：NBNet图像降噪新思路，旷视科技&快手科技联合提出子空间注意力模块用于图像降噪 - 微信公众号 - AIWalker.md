---
title: 059 - 文章阅读笔记：NBNet|图像降噪新思路，旷视科技&快手科技联合提出子空间注意力模块用于图像降噪 - 微信公众号 - AIWalker
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302081713486.jpg
tags:
  - 深度学习
  - 图像降噪
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 2364689269
date: 2023-02-08 15:17:51
---

> 原文：
>
> [NBNet|图像降噪新思路，旷视科技&快手科技联合提出子空间注意力模块用于图像降噪 - 微信公众号 - AIWalker](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651683040&idx=1&sn=239836014fe7ce34d8f5073ad935755e&scene=21#wechat_redirect)
>
> 2021-01-03 22:15
>
> ps：本文为依据个人日常阅读习惯，在原文的基础上记录阅读进度、记录个人想法和收获所写，关于原文一切内容的著作权全部归原作者所有。

## [√] 文章信息

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302081713023.jpg)

文章标题：NBNet: Noise Basis Leaning for Image Denosing with Subspace Projection

中文标题：NBNet：基于子空间投影的图像降噪

论文链接：https://arxiv.org/abs/2012.15028

> 该文是旷视科技&快手&电子科技联合提出的一种图像降噪方案，该方案从一种新的角度(子空间投影)对图像降噪问题进行了分析并提出了一种新颖的子空间注意力模块。所提方案在多个公开数据集上取得SOTA指标与更好的视觉效果。

## [√] Abstract

---

该文提出一种新颖的框架NBNet用于图像降噪，它从新的角度出发设计：通过图像自适应投影进行降噪。具体来说，NBNet通过训练这样的网络进行信号与噪声的分离：在特征空间学习一组重建基；然后，图像降噪可以通过将输入图像映射到特征空间并选择合适的重建基进行噪声重建。

该文的关键洞察在于：投影可以自然的保持输入信号的局部结构信息。这种特性尤其适合于low-light区域/弱纹理区域。为此，作者提出了一种新颖的子空间注意力模块(SubSpace Attention, SSA)显示的进行重建基生成、子空间投影。与此同时，作者进一步将SSA与NBNet(一种UNet改进)相结合进行端到端图像降噪。

作者在公开数据集(包含SIDD与DND)上对所提方案进行了评估，在PSNR与SSIM指标方面，NBNet以更少的计算量取得了SOTA性能，见下图。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302081713024.jpg)

该文的主要贡献包含以下几点：

- 从子空间投影角度出发对图像降噪问题进行了分析，设计了一种简单而有效的SSA(即插即用)模块用于学习子空间投影；
- 提出NBNet(UNet与SSA的组合)用于图像降噪；
- NBNet在多个主流基准数据集上取得了SOTA性能(PSNR与SSIM指标)；
- 对基于投影的图像降噪问题进行了深入分析并指明这是一个很有价值的方向。





## [√] Method

---

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302081713025.jpg)

上图给出了本文所提方案NBNet的网络架构示意图，很明显，它是UNet架构的一种扩展，而其关键核心在于SSA模块。所以这里主要针对SSA部分进行介绍。

#### [√] Subspace Projection with Neural Network

---

正如前面图示，SSA模块包含两个关键步骤：

- Basis Generation：用于根据图像特征生成子空间基向量；
- Projection：用于将图像特征变换到信号子空间。

![image-20230208163307033](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302081713026.png)

#### [√] Basis Generation

---

假设![image-20230208163512678](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302081713027.png)表示基生成函数，基生成过程描述如下：

![image-20230208163936772](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302081713028.png)

![image-20230208164008403](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302081713029.png)

#### [√] Projection

---

给定了上述矩阵![image-20230208164522595](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302081713030.png)，我们可以将图像特征X1通过正交线性投影投影到上述空间。假设![image-20230208164539782](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302081713031.png)表示信号子空间的正交投影，而P可以通过如下公式计算得到：

![image-20230208164549848](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302081713032.png)

注：由于基生成过程无法确保基向量存在正交关系，故而![image-20230208164607008](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302081713033.png)是有必要的。最后图像特征X1可以在信号子空间重建为Y，表示如下：

![image-20230208164619110](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302081713034.png)

注：上述投影过程就是简单的线性矩阵操作，可以通过合适的reshaping达成。



#### [√] NBNet Architecture and Loss Function

---

前面Fig3给出了本文所提出的NBNet的网络架构示意图，它基于经典Unet架构得到，它包含4个encoder和4个decoder，下采样操作通过stride=2的![image-20230208164712598](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302081713035.png)的卷积达成，上采样操作通过2x2的反卷积达成，同时对应的encoder与decoder之间还存在跳过连接。encoder与decoder中的基础模块见Fig3(b)，作者在每个卷积之后采用了LeakyReLU激活函数。

所提SSA模块至于每个encoder-decoder之间的跳过连接中，由于low-level特征包含更多原始图像信息，故而将其视作X1，将high-level特征视作X2，并将两者送入到SSA模块。也就是说，将low-level特征投影到由high-level特征引导的信号子空间中，投影所得特征进一步与原始的high-level特征融合并送入下一个decoder。

相比常规UNet架构(直接对low-level和high-level特征进行融合)，NBNet的主要区别在于：low-level在融合之前先通过SSA模块进行投影处理。

最后一个decoder模块的输出经由3x3卷积处理并作为全局残差与噪声输入相加得到最终的降噪结果。

该网络可以通过端到端的方式进行训练，作者采用了简单的$l1$损失函数：

![image-20230208165112138](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302081713036.png)

其中，$x,y,G(·)$分别表示无噪图像、噪声图像以及NBNet。

## [√] Experiments

---

为验证所提方案的有效性，作者在合成数据与真实数据上将其与其他SOTA方案进行了对比。

训练超参数信息：网络采用kaiming初始化，优化器为Adam，初始学习率为$2\times10^{-4}$，余弦退化方式衰减，合计训练700000次迭代。

训练数据信息：输入块大小为$128\times128$,batch=32,数据增广为随机旋转、随机裁剪、随机镜像。

#### [√] Synthetic Gaussian Noise

---

合成数据信息：训练数据包含BSD(432)以及ImageNet(400源自验证集)以及WaterlooExploration(4744)；验证集包含Set5、LIVE1以及BSD68。在合成数据集上对比所提方法与SOTA方案的性能对比，结果见下表。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302081713037.jpg)

从上表可以看到：所提NBNet取得了比VDN更好的结果。尽管NBNet不依赖于噪声的先验分布，但它仍取得了最佳结果。这也就意味着：**所提投影方案可以有效的将噪声与信息进行分离**。

#### [√] SIDD Benchmark

---

SIDD数据信息：它包含10个场景、不同亮度条件、5款智能机拍摄的30000噪声图像，SIDD数据集可以用于评价智能机camera的降噪性能。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302081713038.jpg)

上图&上表给出了所提方案与其他SOTA方案在SIDD数据上的性能对比，可以看到：**所提方案NBNet取得了最佳指标，同时具有更好的视觉感知效果**。相比MIRNet，NBNet仅需11.25%的计算复杂度和41.82%的参数量即可取得同等PSNR指标，而SSIM指标则提升了0.01。





#### [√] DND Benchmark

---

DND数据信息：它50对真实噪声图像以及对应的GT图像。该数据同时提供了bbox用于提取图像块，合计得到了1000图像块。注：DND数据并未提供训练数据，故而作者采用了SIDD与Renoir的组合进行训练。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302081713039.jpg)



上图&上表给出了所提方案与其他SOTA方案在DND你上的性能对比。可以看到：**所提方法通那样取得了最佳的PSNR指标**。

更多消融实验分析与结果，建议各位同学查看原文。



## [√] 推荐阅读

---

1. [RealSR新突破|中科大提出全频带区域自适应图像超分|ORNet](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651682902&idx=1&sn=7dff47c77f6a01484fbeecec74717304&scene=21#wechat_redirect)
2. [真正的无极放大！30x插值效果惊艳，英伟达等开源LIIF：巧妙的图像超分新思路](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651682814&idx=1&sn=8095788d9136c438ce20a774c1d68eab&scene=21#wechat_redirect)
3. [Transformer再下一城！low-level多个任务榜首被占领，北大华为等联合提出预训练模型IPT](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651682736&idx=1&sn=d8f48cacf9dcf82efb66f687d6d1f6f0&scene=21#wechat_redirect)
4. [图像/视频超分之BackProjection](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651682198&idx=1&sn=b2b2efcefef9a946f80953c58f4639db&scene=21#wechat_redirect)
5. [计算高效，时序一致，超清还原！清华&NYU 提出 RRN：视频超分新型递归网络](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651681974&idx=1&sn=dbf2700a594aec04a095ebcaf181cefa&scene=21#wechat_redirect)
6. [OverNet | 速度快&高性能&任意尺度超分](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651681935&idx=1&sn=1b75da5c0ff3025e1a224e9194475d34&scene=21#wechat_redirect)
7. [ECCV2020|真实世界图像超分CDC｜分而治之思想的探索应用](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651681898&idx=2&sn=8b7449f1ab310c8c6aa51357f04c2911&scene=21#wechat_redirect)
8. [董超大神新作MS3Conv｜多尺度卷积在图像超分中的应用探索](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651681887&idx=1&sn=d5f390d828d6267345016518caccbcdc&scene=21#wechat_redirect)
9. [FDRNet｜混合降质图像复原](https://mp.weixin.qq.com/s?__biz=MzIyMjIxNDk3OA==&mid=2651681746&idx=2&sn=771a26b36604439821ccfebee4196c75&scene=21#wechat_redirect)

















