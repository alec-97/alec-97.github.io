---
title: 4 - hexo为文章添加随机封面
categories:
  - 工具笔记
  - hexo
abbrlink: 915427997
---

在主题配置文件中添加：

```yaml
#---------------------------
# 文章页
# Post Page
#---------------------------
post:

  # 文章在首页的默认封面图，当没有指定 index_img 时会使用该图片，若两者都为空则不显示任何图片
  # Path of the default post cover when `index_img` is not set. If both are empty, no image will be displayed
  default_index_img: 
  - https://img.xjh.me/random_img.php?type=bg&ctype=nature&return=302
  # https://tuapi.eees.cc/api.php?category=meinv
  # https://img.xjh.me/random_img.php?type=bg&ctype=nature&return=302
  # https://img.xjh.me/random_img.php
  # https://img.xjh.me/random_img.php?type=bg
```

