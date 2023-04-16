---
title: 005 - 文章阅读笔记：综述  基于深度学习的人脸超分辨率：全面调研 - 知乎 - CVer计算机视觉
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 873487138
date: 2023-01-06 16:15:00
---

> 链接：
>
> [综述 | 基于深度学习的人脸超分辨率：全面调研 - 知乎 - CVer计算机视觉（√）](https://zhuanlan.zhihu.com/p/343790225)

![image-20230106161638978](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301081552650.png)

# 综述 | 基于深度学习的人脸超分辨率：全面调研

**40页综述，共计202篇参考文献！**本文对人脸超分辨率的深度学习技术进行了全面调研，对诸多算法进行分类和介绍，并盘点了代表性工作，以及常用的数据集和性能指标！

**Deep Learning-based Face Super-resolution: A Survey**

作者单位：哈工大（刘贤明团队），武汉大学(马佳义)
论文：[Deep Learning-based Face Super-resolution: A Survey](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2101.03749)

人脸超分辨率，也称为face hallucination，旨在提高一个低分辨率（LR）或一系列人脸图像的分辨率以生成相应的高分辨率（HR）人脸图像。

近年来，人脸超分辨率得到了极大的关注，并且见证了深度学习技术的惊人发展。迄今为止，关于基于深度学习的人脸超分辨率的研究的总结很少。

**盘点过去十年的人脸超分辨率的综述：**

![image-20230106161903924](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301081552651.png)

在本次调研中，我们以系统的方式对人脸超分辨率的深度学习技术进行了全面综述。

![image-20230106161930446](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301081552652.png)

首先，我们总结了人脸超分辨率的问题表述。

![image-20230106162304338](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301081552653.png)

![image-20230106162325805](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301081552654.png)

其次，我们比较了普通图像超分辨率和人脸超分辨率之间的差异。

第三，介绍了常用的数据集和性能指标。

PSNR、Structural Similarity、FID

![image-20230106162402943](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301081552655.png)

![image-20230106162417496](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301081552656.png)

第四，我们根据特定于人脸的信息对现有方法进行粗略分类。

在每个类别中，我们从设计原理的一般描述开始，给出代表性方法的概述，并比较各种方法之间的异同。

![image-20230106162724513](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301081552657.png)

![image-20230106162815000](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301081552658.png)

![image-20230106162917726](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301081552659.png)

![image-20230106163018428](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301081552660.png)

> alec：
>
> - PEN是先验估计网络，SRN是超分网络。FEN是特征提取网络。P是先验信息。

![image-20230106163225189](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301081552661.png)

![image-20230106163327814](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301081552662.png)

最后，我们展望了该领域进一步技术发展的前景。









