---
title: >-
  文章阅读笔记：【2021 Real-ESRGAN】Training Real-World Blind Super-Resolution with Pure
  Synthetic Data
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251628318.jpg
tags:
  - 盲超分
password: 972274
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 盲超分
abbrlink: 2887697496
date: 2023-02-25 11:48:34
---

> 原文链接：
>
> （1）【√】底层任务超详细解读 (三)：只用纯合成数据来训练真实世界的盲超分模型 Real-ESRGAN（[link](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247619165&idx=2&sn=e976a6752098c184a215d19bf98e1eab&chksm=ec1dffa4db6a76b2b55b5caa86692cdf8f3985a887b86d9973ef5fb1590289e9aee06200a103&scene=21#wechat_redirect)） - 极市平台 - 知乎：科技猛兽
>
> 2022-08-31 22:00
>
> ps：本文为依据个人日常阅读习惯，在原文的基础上记录阅读进度、记录个人想法和收获所写，关于原文一切内容的著作权全部归原作者所有。



#  [√] 文章信息

---

论文标题：【2021 Real-ESRGAN】Training Real-World Blind Super-Resolution with Pure Synthetic Data

中文标题：使用纯净合成数据训练盲超分模型

论文链接：https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.html

论文代码：https://github.com/xinntao/Real-ESRGAN

论文发表：ICCV2021



# [√] 文章1

---

> 总结：
>
> 【本文思想】
>
> - 简单的退化不足以概括所有的情形，因为现实中的退化是复杂多样的。本文用了二阶退化来模拟真实退化。每一阶退化都是一次模糊核、下采样、噪声、传感器噪声的组合。
> - 本文主要是噪声多样性上的创新，而不是网络结构和训练方式上的创新。（本文的网络结构的创新是引入了U-Net形式的判别器）
>
> 【本文贡献】
>
> 【网络结构】
>
> 【可以用于自己论文的话】
>
> - 顾名思义，SISR 任务需要两张图片，一张高分辨率的 HR 图和一张低分辨率的 LR 图。超分模型的目的是根据后者生成前者，而退化模型的目的是根据前者生成后者。
> - 经典超分任务 SISR 认为：**低分辨率的 LR 图是由高分辨率的 HR 图经过某种退化作用得到的，这种退化核预设为一个双三次下采样的模糊核 (downsampling blur kernel)。** 也就是说，这个下采样的模糊核是预先定义好的。但是，在实际应用中，这种退化作用十分复杂，不但表达式未知，而且难以简单建模。双三次下采样的训练样本和真实图像之间存在一个域差。以双三次下采样为模糊核训练得到的网络在实际应用时，这种域差距将导致比较糟糕的性能。这种**退化核未知的超分任务我们称之为盲超分任务 (Blind Super Resolution)** 。
> -  JPEG 压缩, 这种操作一般用于真实世界场景图片。
> - 解决盲超分任务的方法主要有2种，一种是把退化核显式地表达出来，如：盲超分辨率超详细解读 (一)：模糊核迭代校正方法 IKC，盲超分辨率超详细解读 (二)：盲超分的端到端交替优化方法 DAN。另一种是直接获得或者人为生成尽可能接近真实数据的训练对，然后训练一个统一的网络来解决盲超分任务。接近真实数据的训练对可以通过特定的相机采集，或者借助 Cycle Loss 借助非成对数据，或者是通过一些估计的模糊核来生成。
> - **真实世界场景的复杂退化核**通常是不同退化过程的复杂的组合，比如：**1 相机的成像系统、2 图像编辑过程和3 互联网传输**等等多个过程的退化作用的结合。例如，当我们用手机拍照时，照片可能会有一些退化，如相机导致的模糊、传感器的噪声、锐化伪像和 JPEG 压缩。然后我们做一些编辑并上传到一个社交媒体应用程序，这引入了进一步的压缩和没办法预测的噪音。当图像在互联网上被多次共享时，上述过程就会变得更加复杂。
> - 通常将模糊退化建模为与线性模糊滤波器的卷积, 各向同性和各向异性高斯滤波器是最为常见的选 择。
> - 考虑两种常用噪声类型：1 加性高斯噪声 2 泊松噪声。加性高斯噪声具有与高斯分布相等的概率密度函数。噪音强度受高斯分布的标准偏差 (即 \sigma\sigma 值) 控制。当 RGB 图像的每个通道都独立的采样噪声时，合成噪声就是 color noise。当 RGB 图像的每个通道都采样相同噪声时，合成噪声就是 gray noise。泊松噪声遵循泊松分布。它通常用于近似模拟由统计量子波动引起的传感器噪声，即在给定曝光水平下感测到的光子数量的变化。泊松噪声具有与图像强度成比例的强度，并且不同像素处的噪声是独立的。
> - JPEG 压缩是一种常用的数字图像有损压缩技术。它首先将图像转换到 YCbCr 的色彩空间, 并对 色度通道进行下采样。然后, 图像被分成 8x8 的块, 每个块用二维离散余弦变换(DCT) 进行变 换, 随后是 DCT 系数的量化。JPEG 压缩算法通常会引入令人不愉快的块效应。压缩图像的质量 由质量因子q∈[0,100] 决定, 其中较低的q表示较高的压缩比和较差的质量。这里作者使用 PyTorch 实现：DiffJPEG。
> - 真实世界场景的复杂退化核通常是不同退化过程的复杂的组合，比如：1 相机的成像系统、2 图像编辑过程和3 互联网传输等等多个过程的退化作用的结合。例如，当我们用手机拍照时，照片可能会有一些退化，如相机导致的模糊、传感器的噪声、锐化伪像和 JPEG 压缩。然后我们做一些编辑并上传到一个社交媒体应用程序，这引入了进一步的压缩和没办法预测的噪音。当图像在互联网上被多次共享时，上述过程就会变得更加复杂。以上三个过程的复合作用导致的 Real-world 场景的复杂退化是没办法用一个简单的退化模型来准确表达或建模的。
>
> 【可以用于自己论文的idea】
>
> - 通过高阶退化来模拟更加真实的退化过程：当我们采用上述经典退化模型来合成训练对时，经过训练的模型确实可以处理一些真实的样本。然而，它很难模仿真实世界中的图像低分辨模糊情况。这是因为合成的低分辨率图像与真实的退化图像相比有很大的差距。因此，我们将经典退化模型扩展到高阶退化过程，以模拟更实际的退化。
> - 判别器中如何缓解训练不稳定问题：U-Net 结构和复杂的退化也增加了训练的不稳定性。通过加入 Spectral Normalization Regularization，可以缓和由于复杂数据集合复杂网络带来的训练不稳定问题。
> - 本文的训练方式是：先训练以PSRN为导致的Real-ESRNet，然后将这个网络的参数作为预训练参数，然后训练Real-ESRGAN。
> - 微调：Real-ESRNet 是从 ESRGAN Fine-tune 而来的，可实现更快的收敛。
> - JPEG压缩会导致振铃和过冲效应，第1组样本这个退化后的图包含一些过冲带来的伪影 (字母周围的白边)，直接上采样将不可避免地放大这些伪像 (例如 DAN 和 BSRGAN)。Real-ESRGAN 将这类常见伪像考虑在内，采用 sinc filters 来模拟常见的振铃和过冲伪像，从而有效去除它们。然后使用sinc filters作为高频滤波器模拟常见的振铃和过冲伪像，可以有效地去除。
> - 本文的验证部分，数据的类型和结果的论证比较充分，自己的实验可以参考这个。数据含有过冲伪影、复杂未知的退化作用的样本、真实世界的样本。
> - https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247619165&idx=2&sn=e976a6752098c184a215d19bf98e1eab&chksm=ec1dffa4db6a76b2b55b5caa86692cdf8f3985a887b86d9973ef5fb1590289e9aee06200a103&scene=21#wechat_redirect提供了一个很好的训练教程，且这个文章主要是噪声多样性上的创新，而不是网络结构的创新。因此自己刚好可以在这个基础上进行网络结构的创新，进一步地优化模型的性能。（本文的网络结构的创新是引入了U-Net形式的判别器）
>
> 【问题记录】
>
> 【零碎点】
>
> - 【振铃效应】的形成原因是信号传输过程中，在频域信号的高频成分被过滤掉了，就会在时域中引起波纹。
> - 【过冲伪像 (Overshoot)】通常与振铃伪像结合在一起，表现为一张图片里面物体边缘处的跳跃增加。
> - 振铃和过冲伪像一般是由于图像处理时的 JPEG 压缩造成的。
> - 本文采用了和ESRGAN相同的生成器。



## [√] 导读

---

本文介绍了一篇来自ICCVW 2021的关于盲超分任务的工作Real-ESRGAN，引入高阶退化模型来更准确地模拟真实世界场景的复杂退化作用。本文还详细说明了该模型的训练过程。

## [√] 3 只用纯合成数据来训练真实世界的盲超分模型 Real-ESRGAN

---

**论文名称：Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data (ICCVW 2021)**

**论文地址：**

https://arxiv.org/pdf/2107.10833.pdf

先放张图看下效果对比：

![图1：Real-ESRGAN 视觉效果](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251632044.jpg)





## [√] 3.1 盲超分任务介绍

---

作为基本的 low-level 视觉问题，单图像超分辨率 (SISR) 越来越受到人们的关注。SISR 的目标是从其低分辨率观测中重建高分辨率图像。目前已经提出了基于深度学习的方法的多种网络架构和超分网络的训练策略来改善 SISR 的性能。顾名思义，SISR 任务需要两张图片，一张高分辨率的 HR 图和一张低分辨率的 LR 图。超分模型的目的是根据后者生成前者，而退化模型的目的是根据前者生成后者。经典超分任务 SISR 认为：**低分辨率的 LR 图是由高分辨率的 HR 图经过某种退化作用得到的，这种退化核预设为一个双三次下采样的模糊核 (downsampling blur kernel)。** 也就是说，这个下采样的模糊核是预先定义好的。但是，在实际应用中，这种退化作用十分复杂，不但表达式未知，而且难以简单建模。双三次下采样的训练样本和真实图像之间存在一个域差。以双三次下采样为模糊核训练得到的网络在实际应用时，这种域差距将导致比较糟糕的性能。这种**退化核未知的超分任务我们称之为盲超分任务 (Blind Super Resolution)** 。

令x和y分别代表HR和LR图片, 退化模型为:

![image-20230225121409419](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251632045.png)

式中, D(·)代表退化图片, $\otimes$ 代表卷积操作, 模型主要由 3 部分组成：模糊核k, 下采样操作↓s和附加噪声 。[]JPEG 代表 JPEG 压缩, 这种操作一般用于真实世界场景图片。Blind SISR 任 务就是从 LR 图片恢复 HR 图片的过程。

真实世界图像的退化核绝不是理想的双三次下采样那么简单。因此，SR 领域的很多方法因为有一个预设的前提，即：退化核是理想的双三次下采样，所以这种退化核的不匹配使得这些方法在应用于 Real-world 场景时有些不切实际。

**真实世界场景的复杂退化核**通常是不同退化过程的复杂的组合，比如：**1 相机的成像系统、2 图像编辑过程和3 互联网传输**等等多个过程的退化作用的结合。例如，当我们用手机拍照时，照片可能会有一些退化，如相机导致的模糊、传感器的噪声、锐化伪像和 JPEG 压缩。然后我们做一些编辑并上传到一个社交媒体应用程序，这引入了进一步的压缩和没办法预测的噪音。当图像在互联网上被多次共享时，上述过程就会变得更加复杂。

以上三个过程的复合作用导致的 **Real-world 场景的复杂退化**是没办法用**一个简单的退化模型**来准确表达或建模的。



## [√] 3.2 经典退化模型：模糊，噪声，缩放，JPEG 压缩

---

经典退化模型包括模糊 (blur)，噪声 (noise)，缩放 (resize)，和 JPEG 压缩 (JPEG compression)等等。

#### [√] 1）模糊

---

通常将模糊退化建模为与线性模糊滤波器的卷积, 各向同性和各向异性高斯滤波器是最为常见的选 择。对于 kernel size 为2t+1的高斯模糊核k, 其中的第(i,j)∈[-t,t]个元素是从高斯分布 中采样, 形式上为:

![image-20230225122512019](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251632046.png)

式中∑是协方差矩阵,C是空间坐标,N是归一化常数。协方差矩阵可以进一步表示如下:

![image-20230225122626648](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251632047.png)

> 其中 和 是沿着两个主轴的标准偏差 (即, 协方差矩阵的特征值)。 为旋转角度。当 时, 是各向同性高斯模糊核, 否则 是各向异性核。
>
> 虽然高斯模糊核被广泛用于去模拟模糊退化作用, 但它们可能无法很好地逼近真实的相机模糊。为 了包括更多样的核形状, 我们进一步采用广义高斯模糊核, 其概率密度函数分别为 和 。其中 是形状参数。根据经验, 我们发现包含 这些模糊核可以为几个真实样本产生更清晰的输出。
>
> ![image-20230225122742268](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251632048.png)

#### [√] 2）噪声

---

考虑两种常用噪声类型：1 加性高斯噪声 2 泊松噪声。

加性高斯噪声具有与高斯分布相等的概率密度函数。噪音强度受高斯分布的标准偏差 (即 \sigma\sigma 值) 控制。当 RGB 图像的每个通道都独立的采样噪声时，合成噪声就是 color noise。当 RGB 图像的每个通道都采样相同噪声时，合成噪声就是 gray noise。

泊松噪声遵循泊松分布。它通常用于近似模拟由统计量子波动引起的传感器噪声，即在给定曝光水平下感测到的光子数量的变化。泊松噪声具有与图像强度成比例的强度，并且不同像素处的噪声是独立的。

#### [√] 3）缩放

---

下采样是在超分任务中合成低分辨率图像的基本操作。一般我们考虑下采样和上采样，即调整大小操作。有几种调整大小的算法：1 最近邻插值 2 面积缩放 3 双线性插值 4 双三次插值。不同的调整大小操作会带来不同的效果：有些会产生模糊的结果，而有些可能会输出带有伪影的过于尖锐的图像。为了包括更多样和复杂的调整大小效果，我们考虑从上述选择中随机调整大小的操作。

#### [√] 4）JPEG 压缩

---

JPEG 压缩是一种常用的数字图像有损压缩技术。它首先将图像转换到 YCbCr 的色彩空间, 并对 色度通道进行下采样。然后, 图像被分成 8x8 的块, 每个块用二维离散余弦变换(DCT) 进行变 换, 随后是 DCT 系数的量化。JPEG 压缩算法通常会引入令人不愉快的块效应。压缩图像的质量 由质量因子q∈[0,100] 决定, 其中较低的q表示较高的压缩比和较差的质量。这里作者使用 PyTorch 实现：DiffJPEG。

## [√] 3.3 借助高阶退化模型生成训练样本

---

3.3 和 3.4 小节介绍合成数据的制造过程。

当我们采用上述经典退化模型来合成训练对时，经过训练的模型确实可以处理一些真实的样本。然而，它很难模仿真实世界中的图像低分辨模糊情况。这是因为合成的低分辨率图像与真实的退化图像相比有很大的差距。因此，我们将经典退化模型扩展到高阶退化过程，以模拟更实际的退化。

如下图2所示，经典退化模型可以处理一些真实的样本 (图2左)，但是它们会放大噪声或为复杂的真实世界图像带来振铃效应 (图2右)。

![图2：经典退化模型可以处理一些真实的样本，但是它们会放大噪声或为复杂的真实世界图像带来振铃效应](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251632049.jpg)

经典的退化模型只包括一个固定的基本退化作用，比如模糊，噪声等等，可以被视为是退化作用的一阶建模。然而，现实生活中的退化过程非常多样，通常包括一系列过程，包括**相机的成像系统、图像编辑、互联网传输**等。例如，当我们有一张从互联网上下载的低质量图像，想完成盲超分任务时，**这张图像的退化作用其实是不同退化模型的复杂组合。** 具体来说，原始图像可能是很多年前用手机拍摄的，**不可避免地包含相机模糊、传感器噪声、低分辨率和 JPEG 压缩等退化作用。**然后对图像进行了**锐化**和**尺寸调整**操作，带来了**过冲和模糊的伪影**。之后，它被上传到一些社交媒体应用程序，这引入了进一步的压缩和不可预测的噪声。由于**数字传输**也会带来伪像，这个过程变得更加复杂。

如此复杂的退化作用不能用经典的一级模型来模拟。因此，作者提出了**高阶退化模型：**一个n阶模型包括n个重复的退化过程，如下式所示。

![image-20230225154419663](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251632050.png)

上式，其实就是对 First-order 进行多次重复操作，也就是每一个D都是执行一次完整的 First-order 退化。实际上作者采用的是二阶退化过程，因为它可以解决大多数真实情况，同时简单。下图3描绘了作者的纯合成数据的整个生成过程。

![图3：纯合成数据的整个生成过程](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251632051.jpg)

值得注意的是，改进的高阶退化过程并不完美，无法覆盖现实世界中的整个退化空间。**但是是比之前的盲超分方法 IKC，DAN 等等在合成数据时多了一个退化作用，**扩展了先前盲超分方法的可解退化边界。

其实这个改进本质上是传统的一阶模型只执行图3中的上面一行的4步，而本文的高阶模型是执行图3中的上面一行的4步两次。

## [√] 3.4 带有振铃和过冲伪像的训练样本

---

**振铃**和**过冲**效应的维基百科：

https://en.wikipedia.org/wiki/Ringing_artifacts

https://en.wikipedia.org/wiki/Overshoot_(signal)

**振铃伪像 (Ringing artifacts)** 是指一张图片里面的一些尖锐过渡的地方，这些地方附近的虚假边缘。如下图4就是 Ringing artifacts 的示意图 (来自上面的维基百科链接)。图4上面就是带有 Ringing artifacts 的图片，下面是不带 Ringing artifacts 的图片。就频域而言，造成振铃假象的主要原因是由于信号在频域中被带限而没有了高频分量 (bandlimited without high frequencies) 或通过了低通滤波器；就时域而言，这种振铃的原因是sin c函数中的波纹。具体而言，矩形函数的傅里叶变换是sin c函数，所以信号在频域中被加上窗函数丢失了高频的分量，在时域中就会引起波纹。

**过冲伪像 (Overshoot)** 通常与振铃伪像结合在一起，表现为一张图片里面物体边缘处的跳跃增加。

振铃和过冲伪像一般是由于图像处理时的 **JPEG 压缩**造成的。

<img src="https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251632052.jpg" alt="图4：Ringing artifacts 的示意图 (来自上面的维基百科链接)" style="zoom:50%;" />

所以作者在合成图片的时候，刻意来合成一些带有振铃和过冲现象的图片，形成训练对来使得训练出的模型能够更好地适应这种振铃和过冲伪像。

**怎么合成呢？** 无非就是通过sin c函数，因为sin c函数在频域中是窗函数，刚好能够过滤掉高频信号。因此作者使用了一种sin c滤波器，用于截断高频信号，为训练数据对合成振铃和过冲伪像。sin c滤波器可以写成：

![image-20230225155334697](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251632053.png)

式中, w\_c是截止频率, J\_1是一阶贝塞尔函数。

下图5 (底部) 显示了具有不同截止频率的sin c滤波器及其相应的滤波图像。我们观察到它可以很 好地合成振铃和过冲伪像。这些伪像在视觉上类似于图5 (顶部) 中的前两个真实样本。

![图片](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251632054.jpg)



作者在图3中合成数据的整个生成过程的第一步和最后一步使用了sinc滤波器，最后一步的 JPEG Compression 和 2D sinc filter 随机交换位置。



## [√] 3.5 Real-ESRGAN 模型架构

---

#### [√] Real-ESRGAN Generator

---

采用了与 ESRGAN 相同的生成器，即一个具有若干 Residual-in-Residual Dense Blocks (RRDB) 的深度网络。由于 ESRGAN 是一个较重的网络，作者首先通过 Pixel-Unshuffle 操作 (Pixel-Shuffle 的反操作，Pixel-Shuffle可理解为通过压缩图像通道而对图像尺寸进行放大来减少空间分辨率，并扩大 channel 数)。以降低图像分辨率为前提，对图像通道数进行扩充，然后将处理后的图像输入网络进行超分辨重建。因此，大部分计算是在较小的分辨率空间中执行的，这可以减少 GPU 内存和计算资源的消耗。

#### [√] Real-ESRGAN Discriminator

---

由于 Real-ESRGAN 旨在解决比 ESRGAN 大得多的退化空间，ESRGAN 中的 Discriminator 的原始设计就已经不再适用了。Real-ESRGAN 中的 Discriminator 对于复杂的训练输出需要更大的鉴别能力。而且之前的 ESRGAN 的 Discriminator 更多的集中在图像的整体角度判别真伪，而使用 U-Net Discriminator 可以在像素角度，对单个生成的像素进行真假判断，这能够在保证生成图像整体真实的情况下，注重生成图像细节。

U-Net 结构和复杂的退化也增加了训练的不稳定性。通过加入 Spectral Normalization Regularization，可以缓和由于复杂数据集合复杂网络带来的训练不稳定问题。

## [√] 3.6 Real-ESRGAN 训练过程

---

1. 首先，作者用 L1 Loss 训练一个 PSNR 导向的模型。得到的模型命名为 Real-ESRNet。
2. 再通过 Real-ESRNet 的网络参数进行网络初始化， 并用 L1 Loss，Perceptual Loss 和 GAN Loss 的组合来训练最终的网络 Real-ESRGAN。

训练集使用 DIV2K，Flickr2K，OutdoorSceneTraining。训练的 HR Patch size 是256，batch size 是48。Real-ESRNet 是从 ESRGAN Fine-tune 而来的，可实现更快的收敛。训练 Real-ESRNet 1000K iterations，训练 Real-ESRGAN 400K iterations。L1 Loss，Perceptual Loss 和 GAN Loss 的权重分别是 1.0，1.0，0.1。

3.3和3.4小节介绍的2组退化过程使用相同的参数。为了提高训练效率，所有的退化过程都是通过 PyTorch 实现的，具有 CUDA 加速功能，因此我们能够实时合成训练对。

此外，作者发现在训练过程中锐化 GT 图像能够在视觉上提高清晰度，同时不引入可见的伪像。将使用锐化 GT 图像训练的模型表示为 Real-ESRGAN+。

## [√] 3.7 Real-ESRGAN 实验结果

---

作者使用几个不同的测试数据集 (都是 real-world images)，包括 RealSR，DRealSR，OST300，DPED，ADE20K 和一些来自互联网的图像。

如图6所示是不同方法的生成图片的质量可视化。Real-ESRGAN 在去除伪像和恢复纹理细节方面都优于以前的方法。Real-ESRGAN+ (用锐化的地面事实训练) 可以进一步提高视觉锐度。

我们具体看下图6：

第1组样本这个退化后的图包含一些过冲带来的伪影 (字母周围的白边)，直接上采样将不可避免地放大这些伪像 (例如 DAN 和 BSRGAN)。Real-ESRGAN 将这类常见伪像考虑在内，采用 sinc filters 来模拟常见的振铃和过冲伪像，从而有效去除它们。

第2组样本这个退化后的图包含未知的复杂退化作用。大多数算法不能有效地消除它们，而用二阶退化过程训练的Real-ESRGAN 可以。

第3,4,5组样本是真实世界的样本，Real-ESRGAN 恢复更真实的纹理 (例如，砖、山和树纹理)，而其他方法要么无法消除退化，要么添加不自然的纹理 (例如，RealSR 和 BSRGAN)。

![图6：不同方法的生成图片的质量可视化](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302251632055.jpg)

## [√] 3.8 Real-ESRGAN 的局限性

---

1. 一些恢复出来的图像 (尤其是建筑和室内场景) 具有扭曲的线条。
2. GAN 训练在一些样本上引入了令人不愉快的伪像。
3. 现实世界中的退化作用更为复杂，高阶退化模型也不能完全建模。有些伪像甚至会被 Real-ESRGAN 再次放大。

## [√] 3.9 Real-ESRGAN 训练指南

---

**Real-ESRGAN 代码：**

https://github.com/xinntao/Real-ESRGAN

**训练官方指南：**

https://github.com/xinntao/Real-ESRGAN/blob/master/Training.md

训练被分为两个阶段。这两个阶段有相同的数据合成过程和训练流程，除了损失函数。具体来说。

1. 首先，用 L1 Loss 训练一个 PSNR 导向的模型。得到的模型命名为 Real-ESRNet。
2. 再通过 Real-ESRNet 的网络参数进行网络初始化， 并用 L1 Loss，Perceptual Loss 和 GAN Loss 的组合来训练最终的网络 Real-ESRGAN。

**1 数据准备**

**数据集：** DF2K (DIV2K and Flickr2K) + OST datasets，只需要 HR 图。

下载链接：

1. DIV2K: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
2. Flickr2K: https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
3. OST: https://openmmlab.oss-cn-hangzhou.aliyuncs.com/datasets/OST_dataset.zip

> **1.1 [可选]：** 创建 multi-scale 的图片。

对于 DF2K 数据集，我们使用多尺度策略，即对 HR 图像进行降样，以获得几个不同尺度的 GT 图像，代码是：https://github.com/xinntao/Real-ESRGAN/blob/master/scripts/generate_multiscale_DF2K.py

```python
python scripts/generate_multiscale_DF2K.py --input datasets/DF2K/DF2K_HR --output datasets/DF2K/DF2K_multiscale
```

> **1.2 [可选]：** 裁剪成子图像。

然后我们将 DF2K 图像裁剪成子图像，以加快 IO 和处理。如果你的 IO 足够或者你的磁盘空间有限，这个步骤是可选的。代码是：https://github.com/xinntao/Real-ESRGAN/blob/master/scripts/extract_subimages.py

```
 python scripts/extract_subimages.py --input datasets/DF2K/DF2K_multiscale --output datasets/DF2K/DF2K_multiscale_sub --crop_size 400 --step 200
```

> **1.3：** 准备一个 .txt 文件。

需要准备一个包含图像路径的 .txt 文件，下面是一个例子，代码是：https://github.com/xinntao/Real-ESRGAN/blob/master/scripts/generate_meta_info.py

DF2K_HR_sub/000001_s001.png
DF2K_HR_sub/000001_s002.png
DF2K_HR_sub/000001_s003.png
...

```python
 python scripts/generate_meta_info.py --input datasets/DF2K/DF2K_HR, datasets/DF2K/DF2K_multiscale --root datasets/DF2K, datasets/DF2K --meta_info datasets/DF2K/meta_info/meta_info_DF2Kmultiscale.txt
```

**2 训练 Real-ESRNet**

> **2.1：** 把预训练好的 ESRGAN (https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth) (~63M)下载到`experiments/pretrained_models`文件夹。

```python
 wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth -P experiments/pretrained_models
```

> **2.2：** 把文件`options/train_realesrnet_x4plus.yml`改成：

```yaml
train:
    name: DF2K+OST
    type: RealESRGANDataset
    dataroot_gt: datasets/DF2K  # modify to the root path of your folder
    meta_info: realesrgan/meta_info/meta_info_DF2Kmultiscale+OST_sub.txt  # modify to your own generate meta info txt
    io_backend:
        type: disk
```

> **2.3：** 如果想在训练时同时验证，那就再加上下面的 (把它们 Uncomment 掉)：

```yaml
  # Uncomment these for validation
  # val:
  #   name: validation
  #   type: PairedImageDataset
  #   dataroot_gt: path_to_gt
  #   dataroot_lq: path_to_lq
  #   io_backend:
  #     type: disk

...

  # Uncomment these for validation
  # validation settings
  # val:
  #   val_freq: !!float 5e3
  #   save_img: True

  #   metrics:
  #     psnr: # metric name, can be arbitrary
  #       type: calculate_psnr
  #       crop_border: 4
  #       test_y_channel: false
```

> **2.4：** 在正式训练之前，通过`--debug`模式确认：

```yaml
# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --launcher pytorch --debug

# single GPU
python realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --debug 
```

> **2.5：** 正式训练：

```python
# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --launcher pytorch --auto_resume

# single GPU
python realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --auto_resume
```

**3 训练 Real-ESRGAN**

现在有：`experiments/train_RealESRNetx4plus_1000k_B12G4_fromESRGAN/model/net_g_1000000.pth`

> **3.1：按照**2.2和2.3的指示修改文件`train_realesrgan_x4plus.yml` 。
> **3.2：** 在正式训练之前，通过--debug模式确认：

```python
# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/train_realesrgan_x4plus.yml --launcher pytorch --debug

# single GPU
python realesrgan/train.py -opt options/train_realesrgan_x4plus.yml --debug
```

> **2.5：** 正式训练：

```python
# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/train_realesrgan_x4plus.yml --launcher pytorch --auto_resume

# single GPU
python realesrgan/train.py -opt options/train_realesrgan_x4plus.yml --auto_resume
```



## [√] 总结

---

真实世界场景的复杂退化核通常是不同退化过程的复杂的组合，比如：1 相机的成像系统、2 图像编辑过程和3 互联网传输等等多个过程的退化作用的结合。例如，当我们用手机拍照时，照片可能会有一些退化，如相机导致的模糊、传感器的噪声、锐化伪像和 JPEG 压缩。然后我们做一些编辑并上传到一个社交媒体应用程序，这引入了进一步的压缩和没办法预测的噪音。当图像在互联网上被多次共享时，上述过程就会变得更加复杂。以上三个过程的复合作用导致的 Real-world 场景的复杂退化是没办法用一个简单的退化模型来准确表达或建模的。

Real-ESRGAN 引入高阶退化模型来更准确地模拟真实世界场景的复杂退化作用，为了合成更实际的退化，采用 sinc filters 来模拟常见的振铃和过冲伪像。此外，Real-ESRGAN 引入了 U-Net 形式的 Discriminator 在像素角度，对单个生成的像素进行真假判断，这能够在保证生成图像整体真实的情况下，注重生成图像细节。实验结果表明用合成数据训练的 Real-ESRGAN 能够增强细节，同时消除大多数真实世界图像中令人不愉快的伪像。











