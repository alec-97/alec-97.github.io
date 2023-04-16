---
title: 文章阅读笔记：【2021 DPIR】Plug-and-Play Image Restoration with Deep Denoiser Prior
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232203412.jpg
tags:
  - 盲超分
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 盲超分
abbrlink: 360577049
date: 2023-02-23 21:54:51
---

> 原文链接：
>
> （1）【√】ETH Zurich提出DPIR：具有Denoiser先验的即插即用图像恢复 - CVer计算机视觉（[link](https://zhuanlan.zhihu.com/p/243492602)）
>
> 编辑于 2020-09-14 23:06
>
> ps：本文为依据个人日常阅读习惯，在原文的基础上记录阅读进度、记录个人想法和收获所写，关于原文一切内容的著作权全部归原作者所有。







#  [√] 文章信息

---

论文标题：Plug-and-Play Image Restoration with Deep Denoiser Prior

中文标题：具有Denoiser先验的即插即用图像恢复

论文链接：https://arxiv.org/abs/2008.13751

论文代码：https://github.com/cszn/DPIR

论文发表：2021 TPAMI



# [√] 文章1

---

> 总结：
>
> 本文是一个UNet结构的去噪器，适用于多种图像恢复任务。

![ETH Zurich提出DPIR：具有Denoiser先验的即插即用图像恢复](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232226484.jpg)

在三大图像恢复任务（去模糊，超分辨率和去马赛克）上表现SOTA！代码刚刚开源！

**Plug-and-Play Image Restoration with Deep Denoiser Prior**

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232226485.jpg)

> 作者单位：ETH Zurich(张 凯, Luc Van Gool等), 哈工大
> 代码：[cszn/DPIR](https://link.zhihu.com/?target=https%3A//github.com/cszn/DPIR)
> 论文：[https://arxiv.org/abs/2008.1375](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2008.13751)

注：如果上述论文链接无法访问，可以看文末，论文已上传至百度云，方便下载。

关于即插即用图像恢复的最新工作表明，去噪器（Denoiser）可以隐式用作基于模型的方法解决许多inverse问题的先验图像。当通过具有大型建模的深度卷积神经网络（CNN）判别学习去噪器时，此属性为即插即用图像恢复（例如，整合基于模型的方法的灵活性和基于学习的方法的有效性）带来了很大的优势。但是，尽管更深，更大的CNN模型迅速流行，但由于缺少合适的去噪器，现有的即插即用图像恢复阻碍了其性能。为了突破即插即用图像恢复的局限性，我们通过训练高度灵活且有效的CNN去噪器来设置基准深度去噪器。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232226486.jpg)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232226487.jpg)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232226488.jpg)

然后，我们将深度去噪器作为模块的一部分插入到基于半二次分裂的迭代算法中，以解决各种图像恢复问题。同时，我们对参数设置，中间结果和经验收敛进行了全面分析，以更好地了解工作机制。

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232226489.jpg)

## 实验结果

对三个代表性图像恢复任务（包括去模糊，超分辨率和去马赛克）的实验结果表明，所提出的即插即用深度降噪器图像恢复功能不仅明显优于其他基于模型的最新方法，而且与最先进的基于学习的方法相比，它还具有竞争性甚至卓越的性能。

**在图像去模糊任务上表现SOTA！**

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232226490.jpg)

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232226491.jpg)

**在图像超分辨率任务上表现SOTA！**

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232226492.jpg)

**在去马赛克任务上表现SOTA！**

![img](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302232226493.jpg)

## 论文下载

> 链接：[https://pan.baidu.com/s/1IP6Jer6WYtXfVA3yfjSnWg](https://link.zhihu.com/?target=https%3A//pan.baidu.com/s/1IP6Jer6WYtXfVA3yfjSnWg)
> 提取码：7s4u

