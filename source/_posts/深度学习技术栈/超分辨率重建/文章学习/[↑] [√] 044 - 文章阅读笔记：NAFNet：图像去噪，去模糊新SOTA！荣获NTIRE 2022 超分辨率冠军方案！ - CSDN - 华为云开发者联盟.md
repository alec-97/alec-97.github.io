---
title: 044 - 文章阅读笔记：NAFNet：图像去噪，去模糊新SOTA！荣获NTIRE 2022 超分辨率冠军方案！ - CSDN - 华为云开发者联盟
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301202242797.png
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 1043541079
date: 2023-01-20 21:01:53
---

> 原文链接：
>
> [NAFNet：图像去噪，去模糊新SOTA！荣获NTIRE 2022 超分辨率冠军方案！ - CSDN - 华为云开发者联盟](https://huaweicloud.csdn.net/638088b1dacf622b8df89b3b.html?spm=1001.2101.3001.6650.16&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-16-125057312-blog-125071858.pc_relevant_multi_platform_whitelistv4&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-16-125057312-blog-125071858.pc_relevant_multi_platform_whitelistv4&utm_relevant_index=21)
>
> 2022-05-30 14:00:02

导读：2022年4月，旷视研究院发表了一种基于图像恢复任务的全新网络结构，它在SIDD和GoPro数据集上进行训练和测试，该网络结构实现了在图像去噪任务和图像去模糊任务上的新SOTA。具体计算量与实验效果如下图所示：不仅如此，基于NAFNet，旷视还提出了一种针对超分辨率的NAFNet变体结构，该网络为NAFNet-SR。NAFNet-SR在NTIRE 2022 超分辨率...

## [√] 论文信息

---

![825c9abcccc6ed3af5891845b367af03.png](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240020.png)

**导读**：2022年4月，旷视研究院发表了一种基于图像恢复任务的全新网络结构，它在SIDD和GoPro数据集上进行训练和测试，该网络结构实现了在图像去噪任务和图像去模糊任务上的新SOTA。具体计算量与实验效果如下图所示：

> alec：
>
> - MACs, MAdds: (Multiply–Accumulate Operations) 即乘加累积操作数,常常与FLOPs概念混淆,实际上1MACs包含一个乘法操作与一个加法操作,大约包含2FLOPs。通常MACs与FLOPs存在一个2倍的关系。MACs和MAdds说的是一个东西。

![1b73bfe8bd3b83c3c8e0979e72f7d9db.png](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240021.png)

不仅如此，基于NAFNet，旷视还提出了一种针对超分辨率的NAFNet变体结构，该网络为NAFNet-SR。NAFNet-SR在NTIRE 2022 超分辨率比赛中荣获冠军方案。本文将从模型的组成、主要结构以及代码的训练和配置等方面进行详细介绍！

![c4627ce9da9f13888f3d577fbdd9a08e.png](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240022.png)

上图给出了三种主流的图像恢复主流网络架构设计方案，包含多阶段特征提取、多尺度融合架构以及经典的UNet架构。本文为了最大化减少模型每个模块间进行交互的复杂度，直接采用了含有Short Cut的UNet架构。NAFNet在网络架构上实现了最大精简原则！

**项目地址：https://github.com/murufeng/FUIR**





## [√] 核心模块与代码

---

![3a7916739cb836fffeaf355c57f50a3f.png](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240023.png)

基于Restormer的模块示意图，NAFNet设计另一种最简洁的模块方案，具体体现在：

1. 借鉴Trasnformer中使用LN可以使得训练更平滑。NAFNet同样引入LN操作，在图像去噪和去模糊数据集上带来了显著的性能增益。
2. 在Baseline方案中使用GELU和CA联合替换ReLU，GELU可以保持降噪性能相当且大幅提升去模糊性能。
3. 由于通道注意力的有效性已在多个图像复原任务中得到验证。本文提出了两种新的注意力模块组成即CA和SCA模块，具体如下所示：

![c7d0d927168703ef873a03bfef8cf893.png](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240024.png)

> alec：
>
> - LayerNorm可以使得训练更平滑
> - CA = 通道注意力，SCA = 简单通道注意力
> - SCA = 将CA的注意力分支中的卷积 + 激活函数的结构替换为1x1卷积
> - Simple Gate 可以替换一个激活函数，能够实现性能的提升

其中SCA（见上图b）直接利用1x1卷积操作来实现通道间的信息交换。而SimpleGate(见上图c)则直接将特征沿通道维度分成两部分并相乘。采用所提SimpleGate替换第二个模块中的GELU进行，实现了显著的性能提升。

![30c11bc8eedd24f1d385b26b21b3cdc7.png](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240025.png)

> alec：
>
> - 对比于PlainNet，可以看出，规范化方式LN、激活函数GELU、注意力机制CA是有效的，能够提升性能

![46d824a19fc382dd4713648b9bb088f5.png](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240026.png)

> alec：
>
> - NAFNet在baseline的基础上简化了（在模型简化的基础上还能提高性能）
> - 将GELU替换为了simple gate，将CA替换为了SCA

NAFBlock构成代码如下：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base

# 将x分开，然后再相乘
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

    
    
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)#LN

        x = self.conv1(x)#1x1
        x = self.conv2(x)#3x3
        x = self.sg(x)#simple gate
        x = x * self.sca(x)#SCA
        x = self.conv3(x)#1x1卷积

        x = self.dropout1(x)#随机失活层

        y = inp + x * self.beta#残差连接

        x = self.conv4(self.norm2(y))#LN + 1x1卷积
        x = self.sg(x)#SG
        x = self.conv5(x)#1x1卷积

        x = self.dropout2(x)

        return y + x * self.gamma#残差连接
```

## [√] 模型训练与实验结果

---

#### [√] 图像去噪任务（SIDD数据集）

---

![69bf0b5cd40b4fbf25335d1cb8e47045.png](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240027.png)

#### [√] 图像去模糊任务（GoPRO数据集）

---

![2860a87c62505c2abf5392656725245f.png](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301232240028.png)



















