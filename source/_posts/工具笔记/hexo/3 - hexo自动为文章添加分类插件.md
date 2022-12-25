---
title: 3 - hexo自动为文章添加分类插件
categories:
  - 工具笔记
  - hexo
abbrlink: 3085432397
---

## [√] 步骤

---

想要hexo根据`_posts`中的文件夹自动为文章生成分类

#### [√] 安装

---

```shell
npm install hexo-auto-category --save
```

#### [√] 配置站点文件

---

添加：

```yaml
# Generate categories from directory-tree
# Dependencies: https://github.com/xu-song/hexo-auto-category
# depth: the depth of directory-tree you want to generate, should > 0
auto_category:
 enable: true
 depth:

```

如果只想生成第一级目录分类，可以设置depth属性，比如：

```yaml
auto_category:
 enable: true
 depth: 1
```



#### [√] 效果

---

图为自动生成的分类

![image-20221225143714692](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202212251437607.png)









## [√] 参考

---

https://blog.csdn.net/Cryu_xuan/article/details/104232173（√）