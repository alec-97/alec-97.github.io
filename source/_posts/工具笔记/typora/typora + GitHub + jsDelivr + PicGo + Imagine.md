---
title: typora + GitHub + jsDelivr + PicGo
categories:
  - 工具笔记
  - typora
abbrlink: 3859538395
---

> 参考文章：
>
> - [PicGo图床与Typora（PicGo+Typora+GitHub的完整设置）（√）](https://zhuanlan.zhihu.com/p/168729465)
> - [GitHub + jsDelivr + PicGo + Imagine 打造稳定快速、高效免费图床（√）](https://www.cnblogs.com/sitoi/p/11848816.html)

1.github创建仓库

github创建仓库



2.获取github账号token

- 点击头像，选中头像列表中的Settings
- 进入Settings,点击Developer Settings
- 点击Personal access tokens 过后再点击 Generate new token
- 在Note中取一个名字，选中repo这个框过后直接点击完成（Generate token）
- 生成token，记住这个令牌一定要复制保存，如果没有保存的删除重新来一遍



3.下载picgo

- github官网下载picgo



4.配置picgo图床

- 点击左边图床设计，选择GitHub图床，具体配置如下
- 设定仓库名，填写：**GitHub名/库名**
- 分支，**默认填master**
- 设定Token，**刚才保存的token令牌**
- 指定存储路径，**默认填img/**
- 点击确定和设为默认图床

![image-20221221205516662](https://raw.githubusercontent.com/alec-97/alec-s-images-cloud/master/img2/image-20221221205516662.png)



5.设置picgo

- 打开`开机自启`
- 打开`时间戳重命名`
- 关闭`上传后自动复制url`



