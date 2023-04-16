---
title: 064 - 文章阅读笔记：【BSRGAN】针对真实图像退化的盲图像超分辨率BSRGAN复现 - CSDN - AI Studio
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302112337562.png
tags:
  - 计算机视觉
  - 深度学习
  - 人工智能
  - 超分辨率重建
  - 盲超分
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 1005743627
date: 2023-02-11 18:00:43
---

> 原文链接：
>
> [针对真实图像退化的盲图像超分辨率BSRGAN复现 - CSDN - AI Studio](https://blog.csdn.net/m0_63642362/article/details/126235539)
>
> 于 2022-08-08 20:40:22 发布
>
> ps：本文为依据个人日常阅读习惯，在原文的基础上记录阅读进度、记录个人想法和收获所写，关于原文一切内容的著作权全部归原作者所有。

> alec：
>
> **BSRGAN：Designing a Practical Degradation Model for Deep Blind Image Super-Resolution（ICCV 2021）**
>
> 【痛点】
>
> 现有的方法是针对某一种特定的退化方式，模型对于广泛的退化类型无法适应，导致模型在对非训练集的数据做测试的时候，效果不好。
>
> 【创新点】
>
> 作者选择对图像退化方式进行创新，手工设计了尽量能模拟真实世界中图像退化的退化模型，应用该退化模型处理高分辨率数据以生成成对的数据进行训练。
>
> 现有的方法大多是对某一特定的退化类型有效，本文的出发点是如何设计一个模型能够处理范围更广的退化呢？
>
> 解决思路是随机安排3种退化方式的顺序，同时每种退化方式设置几种不同的方法（比如下采样可以采用：双三次、最近邻、双线性等等）。这些顺序和因子的随机选取能够保证最终模型组成更加广义的退化模型。
>
> - 实现的步骤如下：
>
> - - 将模糊，降采样和噪声复杂化（实用化）。模糊：采用两种模糊，分别是各向同性高斯模糊和各向异性高斯模糊；降采样：nearest、bilinear、bicubic以及up-down scaling；噪声：3D高斯噪声、JPEG噪声、相机噪声。
>     - 随机打乱模糊，降采样和噪声的顺序

## [√] 1.项目背景

---

- [超分辨率](https://so.csdn.net/so/search?q=超分辨率&spm=1001.2101.3001.7020)模型大家应该都不会陌生，从董超老师等人2014年发表的深度学习做超分的模型SRCNN已经有8年了，期间有许多优秀的作品发表。本项目讲解的是张凯老师(超分与降噪方向知名DnCNN、IRCNN、FFDNet、SRMD、DPSR、USRNet、DPIR等极具影响力文章的作者)等人在ICCV2021上发表的论文[《Designing a Practical Degradation Model for Deep Blind Image Super-Resolution》](https://arxiv.org/abs/2103.14006)
- 该论文提出的超分辨率模型BSRGAN，目的是想构建一个能够**实际应用的超分模型**，作者选择对**图像退化方式**进行创新，手工设计了尽量能模拟真实世界中图像退化的退化模型，应用该退化模型处理高分辨率数据以生成成对的数据进行训练
- **本项目复现了手工设计的退化模型以及超分模型**，将pytorch训练好的BSRGAN权重转为paddle的权重，超分辨率的效果如下：



![image-20230211181721812](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302112340810.png)

- **效果是不是挺好的**？来看看这是怎么做到的吧！

## [√] 2.BSRGAN解读

---

- 注：这部分内容参考
    - 论文作者张凯老师本人的解读：[一种手工设计的广义盲图像超分退化模型](https://zhuanlan.zhihu.com/p/364062784)
    - [BSRGAN超分辨网络](https://zhuanlan.zhihu.com/p/379876494)
    - [业界首个针对广义盲图像超分的人工设计退化模型](https://www.cvmart.net/community/detail/4546)



#### [√] 2.1 BSRGAN论文出发点

---

- 通常图像退化模型的定义公式为：

$$
y = (x \otimes k)↓_s + n
$$

其中，$k$为模糊核，$↓_s$代表下采样，而$n$则是表示噪声。近年来已经有很多学者意识到**图像退化模型**对超分辨率的重要性，涌现出基于更广泛的退化模型的盲超分辨方法、基于不成对数据的超分辨方法、以及基于相机采集的成对训练数据的超分辨方法，但是**现有的方法大多只针对某一特定退化类型的图像有效**。

- 因此，文章的出发点是如何设计一个模型能够处理范围更广的退化呢？

#### [√] 2.2 解决思路

---

- 解决的思路依然是从图像退化模型的公式出发，围绕着上述退化模型的3个因子：$k、 ↓_s、 n$，随机安排各因子的执行顺序，同时，每个因子又有不同的方法（例如：$↓_s$可以采用以下任一种方式：双三次、最近邻、双线性等等），可以**从这些方法中为每个因子随机选取一种组成更广义的退化模型**



#### [√] 2.3 退化模型

---

- 退化模型如下图所示：

![在这里插入图片描述](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302112340811.png)

- 实现的步骤如下：
    - 将模糊，降采样和噪声复杂化（实用化）。模糊：采用两种模糊，分别是各向同性高斯模糊和各向异性高斯模糊；降采样：nearest、bilinear、bicubic以及up-down scaling；噪声：3D高斯噪声、JPEG噪声、相机噪声。
    - 随机打乱模糊，降采样和噪声的顺序



#### [√] 2.4 超分网络

---

- **超分模型并不是BSRGAN的核心**，现有的超分辨率模型均可以作为训练退化模型生成的图像的选择。

- 选择了ESRGAN作为基线模型

    ，并做了几点改动:

    - 由于本文的目的是：在未知退化前提下，解决更广义的盲图像超分。训练数据方面采用DIV2K、Flickr2K、WED以及源自FFHQ的2000人脸图
    - 采用了更大的图像块72×72
    - 损失方面采用了L1、VGG感知、PatchGAN三个损失的组合，组合系数1,1,0.1

- 在训练超参方面，优化器为Adam，batch=48，固定学习率1e-5，**整个训练在4块V100上大约花费10天**



## [√] 3.手工设计的退化模型复现

---

- 退化模型相关代码已经用paddle复现了，放在`BSRGAN/utils/utils_blindsr.py`文件中，**接近800行**，所以不在notebook中展示，感兴趣自行查看源码
- 本项目这部分展示如何使用退化模型，将高分辨率生成低分辨率图像

```python
%cd /home/aistudio/BSRGAN/
```

```python
/home/aistudio/BSRGAN
```

```python
from utils.utils_blindsr import *
import os

imgPath = r"../work/GT/TJ.jpg" # 高分辨率图像，同济大学樱花
save_dir = r"../work/LQ" # 保存图像的文件夹
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

img = imread_uint(imgPath, 3) 
img = uint2single(img)
sf = 4 #尺度
    
for i in range(10):
    img_lq, img_hq = degradation_bsrgan(img, sf=sf, lq_patchsize=72)
    print(i)
    lq_nearest =  cv2.resize(single2uint(img_lq), (int(sf*img_lq.shape[1]), int(sf*img_lq.shape[0])), interpolation=0)
    img_concat = np.concatenate([lq_nearest, single2uint(img_hq)], axis=1)#将LQ和HQ水平拼接，然后存储
    imsave(img_concat, os.path.join(save_dir,str(i)+'.png'))

```

> lq = LQ = low quality?
>
> hq = HQ = high quality?

```python
# 展示部分结果，左边为低分辨率图像，右边是高分辨率图像
import matplotlib.pyplot as plt
%matplotlib inline

from PIL import Image

imgPath = r"../work/LQ/9.png"
img = Image.open(imgPath)
plt.figure(figsize=(8,6))
plt.imshow(img)
plt.axis("off")
plt.show()

```

![在这里插入图片描述](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302112340813.png)

## [√] 4.直接体验BSRGAN

---

- 由于训练十分消耗时间，4张V100训练10天才行，所以本项目就不复现和训练有关的代码了，提供转换的权重，直接体验BSRGAN，感受**真实世界低分辨率图像超分重建的快感**

```python
import os
import os.path
import logging
import paddle
import numpy as np
import cv2

from utils import utils_logger
from utils import utils_image as util
from models.network_rrdbnet import RRDBNet as net
```

```python
 def bsrgan_predictor(model_path, L_path):
    save_results = True
    sf = 4
    utils_logger.logger_info('blind_sr_log', log_path='blind_sr_log.log')
    logger = logging.getLogger('blind_sr_log')
    # --------------------------------
    # define network and load model
    # --------------------------------
    model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=sf)  # define network

    model.set_state_dict(paddle.load(model_path))
    model.eval()
    testset_L = os.path.basename(L_path)
    testsets = os.path.dirname(L_path)

    E_path = os.path.join(testsets, testset_L+'_results_x'+str(sf))
    util.mkdir(E_path)

    logger.info('{:>16s} : {:s}'.format('Input Path', L_path))
    logger.info('{:>16s} : {:s}'.format('Output Path', E_path))
    idx = 0
    model_name = os.path.basename(model_path)
    for img in util.get_image_paths(L_path):

        # --------------------------------
        # (1) img_L
        # --------------------------------
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->4d} --> {:<s} --> x{:<d}--> {:<s}'.format(idx, model_name, sf, img_name+ext))

        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)

        h_input, w_input = img.shape[0:2]
# img: numpy
        img = img.astype(np.float32)
        if np.max(img) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        img = img / max_range

        if len(img.shape) == 2:  # gray image
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32)
        img = paddle.Tensor(np.transpose(img, (2, 0, 1)))
        img = img.unsqueeze(0)

        # ------------------- process image (without the alpha channel) ------------------- #
        with paddle.no_grad():
            output = model(img)
        output_img = output.squeeze().numpy().clip(0, 1)
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))

        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
            # ------------------------------ return ------------------------------ #
        if max_range == 65535:  # 16-bit image
            output_img = (output_img * 65535.0).round().astype(np.uint16)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        else:
            output_img = (output_img * 255.0).round().astype(np.uint8)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        if save_results:
            util.imsave(output_img, os.path.join(E_path, img_name+'_'+model_name+'.png'))

```

- 指定低分辨率图像所在文件夹，以及BSRGAN权重所在路径，即可以预测
- 预测图像保存在低分辨率图像所在文件夹的同一根目录下

```python
# 若想预测自己的低分辨率图像，修改img_dir即可
img_dir = r"./testsets/RealSRSet" # 待预测的低分辨率图像所在文件夹
model_path = r"./model_zoo/BSRGAN.pdparams"

bsrgan_predictor(model_path, img_dir)

```

```python
# 可视化展示结果
import matplotlib.pyplot as plt
%matplotlib inline

from PIL import Image

lq_path = r"./testsets/RealSRSet/frog.png"
sr_path = r"./testsets/RealSRSet_results_x4/frog_BSRGAN.pdparams.png"
SR = Image.open(sr_path)
LQ = Image.open(lq_path)

plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
plt.imshow(LQ)
plt.title('LQ')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(SR)
plt.title('SR')
plt.axis('off')
plt.show()

```

![image-20230211233449099](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302112340814.png)



## [√] 5.总结

---

- 本项目展示了BSRGAN用于训练数据的退化模型，而退化模型也是BSRGAN成功的关键
- 提供了BSRGAN的paddle模型权重，不需要训练即可体验针对真实图像退化的盲超分辨率重建模型















