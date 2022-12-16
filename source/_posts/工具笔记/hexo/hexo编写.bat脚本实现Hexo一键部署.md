---
title: hexo编写.bat脚本实现Hexo一键部署
date: '2022年11月15日16:24:52'
tags:
  - 软件使用技巧
  - hexo
  - 效率
categories:
  - 工具笔记
  - hexo
---

使用hexo发布文章的时候，每次需要在hexo根目录打开git，然后依次执行`hexo clean, hexo g, hexo d`,不太方便。



后来将此过程简化为直接输入`hexo clean&&hexo g&&hexo d`



再后来，实现编写bat脚本，一键运行。

```shell
@echo off
hexo clean&&hexo g&&gulp&&hexo d
```

其中，@echo off表示不显示后续命令行及当前命令行。

