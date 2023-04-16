---
title: 020 - 文章阅读笔记：目前主流的attention方法都有哪些？ - 知乎
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537694.png
tags:
  - 注意力机制
  - CNN
  - 深度学习
  - 计算机视觉
  - attention
categories:
  - 深度学习技术栈
  - 超分辨率重建
  - 文章学习
abbrlink: 2627212261
date: 2023-01-11 13:52:17
---

> 转载自：
>
> 目前主流的attention方法都有哪些？ - 知乎 https://www.zhihu.com/question/68482809
>
> 计算机视觉中的注意力机制 - 知乎 - 数学人生 - 张戎（√） - https://zhuanlan.zhihu.com/p/56501461
>
> 发布于 2019-02-12 11:18

# [√] 计算机视觉中的注意力机制

---

## [√] 引言

---

在机器翻译（Machine Translation）或者自然语言处理（Natural Language Processing）领域，以前都是使用数理统计的方法来进行分析和处理。

近些年来，随着 AlphaGo 的兴起，除了在游戏AI领域，深度学习在计算机视觉领域，机器翻译和自然语言处理领域也有着巨大的用武之地。

在 2016 年，随着深度学习的进一步发展，seq2seq 的训练模式和翻译模式已经开始进入人们的视野。

除此之外，在端到端的训练方法中，除了需要海量的业务数据之外，在网络结构中加入一些重要的模块也是非常必要的。

在此情形下，基于循环神经网咯（Recurrent Neural Network）的注意力机制（Attention Mechanism）进入了人们的视野。

除了之前提到的机器翻译和自然语言处理领域之外，**计算机视觉**中的注意力机制也是十分有趣的，本文将会简要介绍一下计算机视觉领域中的注意力方法。

在此事先声明一下，笔者并不是从事这几个领域的，可能在撰写文章的过程中会有些理解不到位的地方，请各位读者指出其中的不足。

![image-20230111140133730](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537680.png)



## [√] 注意力机制

---

顾名思义，**注意力机制**是本质上是为了模仿人类观察物品的方式。

通常来说，人们在看一张图片的时候，除了从整体把握一幅图片之外，也会更加关注图片的某个局部信息，例如局部桌子的位置，商品的种类等等。

在翻译领域，每次人们翻译一段话的时候，通常都是从句子入手，但是在阅读整个句子的时候，肯定就需要关注词语本身的信息，以及词语前后关系的信息和上下文的信息。在自然语言处理方向，如果要进行情感分类的话，在某个句子里面，肯定会涉及到表达情感的词语，包括但不限于“高兴”，“沮丧”，“开心”等关键词。

而这些句子里面的其他词语，则是上下文的关系，并不是它们没有用，而是它们所起的作用没有那些表达情感的关键词大。



在以上描述下，注意力机制其实包含**两个部分**：

1. 注意力机制需要决定整段输入的哪个部分需要更加关注；
2. 从关键的部分进行特征提取，得到重要的信息。



通常来说，在机器翻译或者自然语言处理领域，人们阅读和理解一句话或者一段话其实是有着一定的先后顺序的，并且按照语言学的语法规则来进行阅读理解。在图片分类领域，人们看一幅图也是按照先整体再局部，或者先局部再整体来看的。再看局部的时候，尤其是手写的手机号，门牌号等信息，都是有先后顺序的。为了模拟人脑的思维方式和理解模式，循环神经网络（RNN）在处理这种具有明显先后顺序的问题上有着独特的优势，因此，Attention 机制通常都会应用在循环神经网络上面。



虽然，按照上面的描述，机器翻译，自然语言处理，计算机视觉领域的注意力机制差不多，但是其实仔细推敲起来，这三者的注意力机制是有明显区别的。

1. 在机器翻译领域，翻译人员需要把已有的一句话翻译成另外一种语言的一句话。例如把一句话从英文翻译到中文，把中文翻译到法语。在这种情况下，输入语言和输出语言的词语之间的先后顺序其实是相对固定的，是具有一定的语法规则的；
2. 在视频分类或者情感识别领域，视频的先后顺序是由时间戳和相应的片段组成的，输入的就是一段视频里面的关键片段，也就是一系列具有先后顺序的图片的组合。NLP 中的情感识别问题也是一样的，语言本身就具有先后顺序的特点；
3. 图像识别，物体检测领域与前面两个有本质的不同。因为物体检测其实是在一幅图里面挖掘出必要的物体结构或者位置信息，在这种情况下，它的输入就是一幅图片，并没有非常明显的先后顺序，而且从人脑的角度来看，由于个体的差异性，很难找到一个通用的观察图片的方法。由于每个人都有着自己观察的先后顺序，因此很难统一成一个整体。

在这种情况下，机器翻译和自然语言处理领域使用基于 RNN 的 Attention 机制就变得相对自然，而计算机视觉领域领域则需要必要的改造才能够使用 Attention 机制。

> alec：
>
> - NLP中，语言是有相对顺序的，因此使用基于RNN的注意力机制变得自然。
> - 在CV中，则需要必要的改造才能使用attention机制。

![image-20230111141013512](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537681.png)



## [√] 基于 RNN 的注意力机制

---

通常来说，RNN 等深度神经网络可以进行端到端的训练和预测，在机器翻译领域和或者文本识别领域有着独特的优势。对于端到端的 RNN 来说，有一个更简洁的名字叫做 sequence to sequence，简写就是 seq2seq。顾名思义，输入层是一句话，输出层是另外一句话，中间层包括编码和解码两个步骤。

> alec：
>
> - 对于端到端的 RNN 来说，有一个更简洁的名字叫做 sequence to sequence，简写就是 seq2seq。

而基于 RNN 的注意力机制指的是，对于 seq2seq 的诸多问题，在输入层和输出层之间，也就是词语（Items）与词语之间，存在着某种隐含的联系。例如：“中国” -> “China”，“Excellent” -> “优秀的”。在这种情况下，每次进行机器翻译的时候，模型需要了解当前更加关注某个词语或者某几个词语，只有这样才能够在整句话中进行必要的提炼。在这些初步的思考下，基于 RNN 的 Attention 机制就是：

1. 建立一个**编码**（Encoder）和**解码**（Decoder）的非线性模型，神经网络的参数足够多，能够存储足够的信息；
2. 除了关注句子的整体信息之外，每次翻译下一个词语的时候，需要对不同的词语赋予不同的权重，在这种情况下，再解码的时候，就可以同时考虑到整体的信息和局部的信息。

![image-20230111141430952](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537682.png)

## [√] 注意力机制的种类

---

从初步的调研情况来看，注意力机制有两种方法，一种是基于**强化学习**（Reinforcement Learning）来做的，另外一种是基于**梯度下降**（Gradient Decent）来做的。

强化学习的机制是通过收益函数（Reward）来激励，让模型更加关注到某个局部的细节。梯度下降法是通过目标函数以及相应的优化函数来做的。无论是 NLP 还是 CV 领域，都可以考虑这些方法来添加注意力机制。

![image-20230111141549591](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537683.png)



## [√] 计算机视觉领域的 Attention 部分论文整理

---

下面将会简单的介绍几篇近期阅读的计算机视觉领域的关于注意力机制的文章。

#### [√] Look Closer to See Better：Recurrent Attention Convolutional Neural Network for Fine-grained Image Recognition

---

在图像识别领域，通常都会遇到给图片中的鸟类进行分类，包括种类的识别，属性的识别等内容。为了区分不同的鸟，除了从整体来对图片把握之外，更加关注的是一个局部的信息，也就是鸟的样子，包括头部，身体，脚，颜色等内容。至于周边信息，例如花花草草之类的，则显得没有那么重要，它们只能作为一些参照物。因为不同的鸟类会停留在树木上，草地上，关注树木和草地的信息对鸟类的识别并不能够起到至关重要的作用。所以，在图像识别领域引入注意力机制就是一个非常关键的技术，让深度学习模型更加关注某个局部的信息。

![image-20230111141903094](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537684.png)

在这篇文章里面，作者们提出了一个基于 CNN 的注意力机制，叫做 **recurrent attention convolutional neural network**（RA-CNN），该模型递归地分析局部信息，从局部的信息中提取必要的特征。同时，在 RA-CNN 中的子网络（sub-network）中存在分类结构，也就是说从不同区域的图片里面，都能够得到一个对鸟类种类划分的概率。除此之外，还引入了 attention 机制，让整个网络结构不仅关注整体信息，还关注局部信息，也就是所谓的 Attention Proposal Sub-Network（APN）。这个 APN 结构是从整个图片（full-image）出发，迭代式地生成子区域，并且对这些子区域进行必要的预测，并将子区域所得到的预测结果进行必要的整合，从而得到整张图片的分类预测概率。

> alec：
>
> - APN对图片中的子区域进行预测，预测概率。

![image-20230111142032114](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537685.png)

RA-CNN 的特点是进行一个端到端的优化，并不需要提前标注 box，区域等信息就能够进行鸟类的识别和图像种类的划分。在数据集上面，该论文不仅在鸟类数据集（CUB Birds）上面进行了实验，也在狗类识别（Stanford Dogs）和车辆识别（Stanford Cars）上进行了实验，并且都取得了不错的效果。

![image-20230111142103002](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537686.png)

从深度学习的网络结构来看，RA-CNN 的输入时是整幅图片（Full Image），输出的时候就是分类的概率。而提取图片特征的方法通常来说都是使用卷积神经网络（CNN）的结构，然后把 Attention 机制加入到整个网络结构中。从下图来看，一开始，整幅图片从上方输入，然后判断出一个分类概率；然后中间层输出一个坐标值和尺寸大小，其中坐标值表示的是子图的中心点，尺寸大小表示子图的尺寸。在这种基础上，下一幅子图就是从坐标值和尺寸大小得到的图片，第二个网络就是在这种基础上构建的；再迭代持续放大图片，从而不停地聚焦在图片中的某些关键位置。不同尺寸的图片都能够输出不同的分类概率，再将其分类概率进行必要的融合，最终的到对整幅图片的鸟类识别概率。



因此，在整篇论文中，有几个关键点需要注意：



1. 分类概率的计算，也就是最终的 **loss** 函数的设计；
2. 从上一幅图片到下一幅图片的坐标值和尺寸大小。



只要获得了这些指标，就可以把整个 RA-CNN 网络搭建起来。



大体来说，第一步就是给定了一幅输入图片 X ， 需要提取它的特征，可以记录为 Wc∗X ，这里的 ∗ 指的是卷积等各种各样的操作。所以得到的概率分布情况其实就是 p(X)=f(Wc∗X) ， f 指的是从 CNN 的特征层到全连接层的函数，外层使用了 Softmax 激活函数来计算鸟类的概率。

第二步就是计算下一个 box 的坐标 (tx,ty) 和尺寸大小 tℓ ，其中 tx,ty 分别指的是横纵坐标，正方形的边长其实是 2∗tℓ 。用数学公式来记录这个流程就是 [tx,ty,tℓ]=g(Wc∗X) 。在坐标值的基础上，我们可以得到以下四个值，分别表示 x,y 两个坐标轴的上下界：

![image-20230111143907675](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537687.png)

局部注意力和放大策略（**Attention Localization and Amplification**）指的是：从上面的方法中拿到坐标值和尺寸，然后把图像进行必要的放大。为了提炼局部的信息，其实就需要在整张图片 X 的基础上加上一个面具（Mask）。所谓面具，指的是在原始图片的基础上进行点乘 0 或者 1 的操作，把一些数据丢失掉，把一些数据留下。在图片领域，就是把周边的信息丢掉，把鸟的信息留下。但是，有的时候，如果直接进行 0 或者 1 的硬编码，会显得网络结构不够连续或者光滑，因此就有其他的替代函数。

在激活函数里面，逻辑回归函数（Logistic Regression）是很常见的。其实通过逻辑回归函数，我们可以构造出近似的阶梯函数或者面具函数。

![image-20230111144126610](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537688.png)

对于逻辑回归函数 σ(x)=1/(1+e−kx) 而言，当 k 足够大的时候， σ(x)≈1 当 x≥0 ； σ(x)≈0 当 x<0 。此时的逻辑回归函数近似于一个阶梯函数。如果假设 x0<x1 ，那么 σ(x−x0)−σ(x−x1) 就是光滑一点的阶梯函数， σ(x−x0)−σ(x−x1)≈0 当 x<x0 or x>x1 ； σ(x−x0)−σ(x−x1)≈1 当 x0≤x≤x1 。

因此，基于以上的分析和假设，我们可以构造如下的函数： Xattr=X⊙M(tx,ty,tℓ), 其中， Xattr 表示图片需要关注的区域， M(⋅) 函数就是 M(tx,ty,tℓ)=[σ(x−tx(tℓ))−σ(x−tx(br))]⋅[σ(y−ty(tℓ))−σ(y−ty(br))], 这里的 σ 函数对应了一个足够大的 k 值。

当然，从一张完整的图片到小图片，在实际操作的时候，需要把小图片继续放大，在放大的过程中，可以考虑使用双线性插值算法来扩大。也就是说：

X(i,j)amp=∑α,β=01|1−α−{i/λ}|⋅|1−β−{j/λ}|⋅X(m,n)att,

![image-20230111144320203](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537689.png)

其中 m=[i/λ]+α,n=[j/λ]+β ， λ 表示上采样因子， [⋅],{⋅} 分别表示一个实数的正数部分和小数部分。

在分类（Classification）和排序（Ranking）部分，RA-CNN 也有着自己的方法论。在损失函数（Loss Function）里面有两个重要的部分，第一个部分就是三幅图片的 LOSS 函数相加，也就是所谓的 classification loss， Y(s) 表示预测类别的概率， Y 表示真实的类别。除此之外，另外一个部分就是排序的部分， Lrank(pt(s),pt(s+1))=max{0,pt(s)−pt+1(s+1)+margin}, 其中 p(s) 表示在第 s 个尺寸下所得到的类别 t 的预测概率，并且最大值函数强制了该深度学习模型在训练中可以保证 pt(s+1)>pt(s)+margin ，也就是说，局部预测的概率值应该高于整体的概率值。

![image-20230111144716892](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537690.png)

> alec：
>
> - 总体的损失等于分类损失和排序损失

![image-20230111144801821](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537691.png)

在这种 Attention 机制下，可以使用训练好的 conv5_4 或者 VGG-19 来进行**特征的提取**。在图像领域，location 的位置是需要通过训练而得到的，因为每张图片的鸟的位置都有所不同。进一步通过数学计算可以得到， tℓ 会随着网络而变得越来越小，也就是一个层次递进的关系，越来越关注到局部信息的提取。简单来看，

![image-20230111144848707](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537692.png)

因此， tℓ 在迭代的过程中会越来越小，也就是说关注的区域会越来越集中。

RA-CNN 的实验效果如下：

![image-20230111145207316](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537693.png)

#### [√] Multiple Granularity Descriptors for Fine-grained Categorization

---

这篇文中同样做了鸟类的分类工作，与 RA-CNN 不同之处在于它使用了层次的结构，因为鸟类的区分是按照一定的层次关系来进行的，粗糙来看，有科 -> 属 -> 种三个层次结构。

![image-20230111145528537](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537694.png)

因此，在设计网络结构的过程中，需要有并行的网络结构，分别对应科，属，种三个层次。从前往后的顺序是**检测网络**（Detection Network），**区域发现**（Region Discovery），**描述网络**（Description Network）。并行的结构是 Family-grained CNN + Family-grained Descriptor，Genus-grained CNN + Genus-grained Descriptor，Species-grained CNN + Species-grained Descriptor。而在区域发现的地方，作者使用了 energy 的思想，让神经网络分别聚焦在图片中的不同部分，最终的到鸟类的预测结果。

![image-20230111152123992](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537695.png)

![image-20230111152203603](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537696.png)

#### [√![image-20230111152453230](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537697.png)] Recurrent Models of Visual Attention

---

在计算机视觉中引入注意力机制，DeepMind 的这篇文章 recurrent models of visual attention 发表于 2014 年。在这篇文章中，作者使用了基于强化学习方法的注意力机制，并且使用收益函数来进行模型的训练。从网络结构来看，不仅从整体来观察图片，也从局部来提取必要的信息。

![image-20230111152319453](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537698.png)

![image-20230111152322577](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537699.png)

![image-20230111152332404](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537700.png)

整体来看，其网络结构是 RNN，上一个阶段得到的信息和坐标会被传递到下一个阶段。这个网络只在最后一步进行分类的概率判断，这是与 RA-CNN 不同之处。这是为了模拟人类看物品的方式，人类并非会一直把注意力放在整张图片上，而是按照某种潜在的顺序对图像进行扫描。Recurrent Models of Visual Attention 本质上是把图片按照某种时间序列的形式进行输入，一次处理原始图片的一部分信息，并且在处理信息的过程中，需要根据过去的信息和任务选择下一个合适的位置进行处理。这样就可以不需要进行事先的位置标记和物品定位了。

![image-20230111152455616](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537701.png)

正如上图所示，enc 指的是对图片进行编码， ri(1) 表示解码的过程， xi 表示图片的一个子区域。而 ys 表示对图片的预测概率或者预测标签。

#### [] Multiple Object Recognition with Visual Attention

---

这篇文章同样是 DeepMind 的论文，与 Recurrent Models of Visual Attention 不同之处在于，它是一个两层的 RNN 结构，并且在最上层把原始图片进行输入。其中 enc 是编码网络， ri(1) 是解码网络， ri(2) 是注意力网络，输出概率在解码网络的最后一个单元输出。

![image-20230111152628852](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537702.png)

在门牌识别里面，该网络是按照从左到右的顺序来进行图片扫描的，这与人类识别物品的方式极其相似。除了门牌识别之外，该论文也对手写字体进行了识别，同样取得了不错的效果。

![image-20230111152911622](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537703.png)

实验效果如下：

![image-20230111152920615](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202301111537704.png)









## [√] 总结

---

本篇 Blog 初步介绍了计算机视觉中的 Attention 机制，除了这些方法之外，应该还有一些更巧妙的方法，希望各位读者多多指教。

## [√] 参考文献

---

1. Look Closer to See Better：Recurrent Attention Convolutional Neural Network for Fine-grained Image Recognition，CVPR，2017.
2. Recurrent Models of Visual Attention，NIPS，2014
3. GitHub 代码：Recurrent-Attention-CNN，[https://github.com/Jianlong-Fu/Recurrent-Attention-CNN](https://link.zhihu.com/?target=https%3A//github.com/Jianlong-Fu/Recurrent-Attention-CNN)
4. Multiple Granularity Descriptors for Fine-grained Categorization，ICCV，2015
5. Multiple Object Recognition with Visual Attention，ICRL，2015
6. Understanding LSTM Networks，Colah's Blog，2015，[http://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://link.zhihu.com/?target=http%3A//colah.github.io/posts/2015-08-Understanding-LSTMs/)
7. Survey on the attention based RNN model and its applications in computer vision，2016

