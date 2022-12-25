---
title: 0 - hexo博客搭建记录
categories:
  - 工具笔记
  - hexo
abbrlink: 1312067499
---

> 参考文章：[超详细Hexo+Github博客搭建小白教程 - 知乎 - 字节跳动 AI Lab NLP算法工程师 - godweiyang](https://zhuanlan.zhihu.com/p/35668237)

## 1 - 安装Node.js

## 2 - 安装Git

## 3 - 本地安装、运行hexo

### 3.1 - 新建文件夹

在合适的地方新建一个文件夹，用来存放自己的博客文件，比如我的博客文件都存放在`D:\坚果云\blog`目录下。



在该目录下右键点击`Git Bash Here`，打开git的控制台窗口。

### 3.2 - 安装hexo并验证

定位到该目录下，输入`npm i hexo-cli -g`安装Hexo。会有几个报错，无视它就行。

安装完后输入`hexo -v`验证是否安装成功。

```shell
npm i hexo-cli -g

hexo -v
```



### 3.3 - 初始化网站+安装必备组件

输入`hexo init`初始化文件夹，接着输入`npm install`安装必备的组件。

```shell
hexo init

npm install
```



3.4 - 本地运行，观察效果

删除网页静态文件缓存 + 生成新的静态文件 + 在本地4000端口运行程序

```shell
#c = clean, g = generate, s = server
hexo c&&hexo g&&hexo s
```

在浏览器打开网页：http://localhost:4000/

按`ctrl+c`关闭本地服务器。