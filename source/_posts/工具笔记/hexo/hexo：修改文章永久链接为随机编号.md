---
title: hexo：修改文章永久链接为随机编号
categories:
  - 工具笔记
  - hexo
abbrlink: 1841524034
---

> 参考文章：
>
> [https://blog.51cto.com/u_13640625/3032262（√）](https://blog.51cto.com/u_13640625/3032262)

1.安装abbrlink插件

在博客根目录（执行hexo命令的地方）安装插件：

```shell
npm install hexo-abbrlink --save
```

2.编辑站点配置文件

```yaml
#permalink: :year/:month/:day/:title/
#permalink_defaults:
permalink: posts/:abbrlink/
abbrlink:
  alg: crc32 #support crc16(default) and crc32
  rep: dec   #support dec(default) and hex
```

> 设置示例：
>
> ```yaml
> crc16 & hex
> https://post.zz173.com/posts/66c8.html
> 
> crc16 & dec
> https://post.zz173.com/posts/65535.html
> 
> crc32 & hex
> https://post.zz173.com/posts/8ddf18fb.html
> 
> crc32 & dec
> https://post.zz173.com/posts/1690090958.html
> ```
>
> 