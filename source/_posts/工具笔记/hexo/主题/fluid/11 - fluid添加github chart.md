---
title: 11 - fluid添加github chart
index_img: >-
  https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202302051234392.png
tags:
  - fluid
  - hexo
categories:
  - 工具笔记
  - hexo
  - 主题
  - fluid
abbrlink: 4079693730
date: 2023-02-05 12:19:03
---

> 参考文章：
>
> https://www.zywvvd.com/notes/hexo/theme/fluid/fluid-github-chart/fluid-github-chart/

### 1 - 什么是github chart？

github chart是在github中用于显示提交频次的绿色日历。

我们可以通过Github API提取数据生成图标，但是已经有前辈做了相关的开源方法。

### 2 - ghchart

- 2016rshah 大佬提供了现成的 API 可以直接生成贡献图表
- 官网地址：[https://ghchart.rshah.org/](https://qq52o.me/go/aHR0cHM6Ly9naGNoYXJ0LnJzaGFoLm9yZy8=)
- 源码在 Github 上开源，仓库地址：[https://github.com/2016rshah/githubchart-api](https://qq52o.me/go/aHR0cHM6Ly9naXRodWIuY29tLzIwMTZyc2hhaC9naXRodWJjaGFydC1hcGk=)



### 3 - 使用方法

访问链接：`https://ghchart.rshah.org/<github-user-name>` 即可获取指定用户的贡献图表，因为信息都是公开的，所以谁都可以直接拿到

![image-20230205122434459](D:\坚果云\Alec - backup files\typora pictures\image-20230205122434459.png)

还可以选择一个颜色作为主题颜色，比如浅蓝 `26a397`

> https://ghchart.rshah.org/26a397/alec-97

![image-20230205122458737](D:\坚果云\Alec - backup files\typora pictures\image-20230205122458737.png)

### 4 - 添加到 Fluid 主题博客中

- 比如我想将 Github Chart 添加到 `归档页` 中
- 那么就需要修改 `fluid\layout\archive.ejs` 文件，添加如下代码

```html
<div  style="padding: 2vh 0 5vh 0" >
  <img src="https://ghchart.rshah.org/alec-97" alt="My Github Chart" width=100% position='relative'>
</div>
```















